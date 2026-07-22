"""
Multi-Agent Kubernetes Operations System.

Architecture:
  Orchestrator (LangGraph StateGraph) routes between specialist nodes:

    LLM Agents (reasoning only):
      1. Detection Agent   — scans cluster, identifies unhealthy resources
      2. Diagnosis Agent   — investigates root cause (describe, logs, events)
      3. Recommendation Agent — proposes fix commands with cluster access
      4. Summary Agent     — produces final human-readable report

    Deterministic nodes (Python, no LLM):
      5. Guardrail Filter  — strips invalid/read-only recommendations
      6. Approval Gate     — presents fixes, collects human approval
      7. Executor          — runs approved commands via subprocess (shell=False)
      8. Verifier          — confirms fixes worked via kubectl checks

Features:
  - LangSmith tracing (auto + @traceable decorators)
  - Shared memory across nodes via InMemorySaver checkpointer
  - Per-node model routing (Haiku for detection/summary, Sonnet for diagnosis/recommendation)
  - Discovery cache with TTL + post-mutation invalidation
  - Poll-based verification (replaces hardcoded sleep)
  - Append-only agent_logs via Annotated reducer
  - Retry with exponential backoff on LLM calls

Setup:
  pip install -U langchain langchain-anthropic langgraph langsmith
  export ANTHROPIC_API_KEY="your-key"
  export LANGSMITH_TRACING=true
  export LANGSMITH_API_KEY="ls__your-key"
  export LANGSMITH_PROJECT="k8s-multi-agent-ops"

Run:
  python multi_agent.py
  python multi_agent.py "Find all crashing pods and fix them"
"""

import json
import logging
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import operator

from langchain.agents import create_agent                     # LangChain 1.0+
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langsmith import traceable

# =========================================================================
# Logging
# =========================================================================

logger = logging.getLogger("k8s_agent")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("  [%(node)s] %(message)s"))
logger.addHandler(_handler)


class _NodeFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "node"):
            record.node = "system"
        return True

_handler.addFilter(_NodeFilter())


def _log(node: str, msg: str):
    logger.info(msg, extra={"node": node})


# =========================================================================
# LangSmith configuration
# =========================================================================

def _configure_langsmith():
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "k8s-multi-agent-ops")


WORKDIR = Path.cwd()
IS_WINDOWS = platform.system() == "Windows"

# =========================================================================
# Model registry — different models for different task complexity
# =========================================================================
# Detection & Summary  → fast + cheap (Haiku)  — scanning, formatting
# Diagnosis & Recommendation → precise (Sonnet) — root-cause, fix generation

MODEL_REGISTRY: dict[str, dict] = {
    "detection":      {"model": "claude-haiku-4-5-20251001", "temperature": 0.2, "max_tokens": 4096},
    "diagnosis":      {"model": "claude-sonnet-4-6",         "temperature": 0.2, "max_tokens": 8192},
    "recommendation": {"model": "claude-sonnet-4-6",         "temperature": 0.1, "max_tokens": 8192},
    "summary":        {"model": "claude-haiku-4-5-20251001", "temperature": 0.1, "max_tokens": 4096},
}


def make_llm(node_name: str) -> ChatAnthropic:
    """Create an LLM configured for the given node's task complexity."""
    cfg = MODEL_REGISTRY.get(node_name, MODEL_REGISTRY["detection"])
    return ChatAnthropic(
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
    )


# =========================================================================
# Command execution — shell=False, argument lists
# =========================================================================

def _parse_command(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=not IS_WINDOWS)
    except ValueError:
        return command.split()


def _run_args(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            args, shell=False, capture_output=True,
            text=True, timeout=timeout, cwd=WORKDIR,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return -1, "", f"Command not found: {args[0]}"
    except OSError as e:
        return -1, "", f"OS error: {e}"


def _run_command(command: str) -> tuple[int, str, str]:
    args = _parse_command(command)
    return _run_args(args)


# =========================================================================
# Discovery cache — TTL + post-mutation invalidation
# =========================================================================

_cache: dict[str, tuple[float, str]] = {}
CACHE_TTL = 300

_CACHEABLE = [
    "kubectl get namespaces", "kubectl get namespace", "kubectl get ns",
    "kubectl get nodes", "kubectl get node", "kubectl api-resources",
]


def _cache_key(command: str) -> str | None:
    s = command.strip()
    for p in _CACHEABLE:
        if s == p or s.startswith(p + " "):
            return p
    return None


def _get_cached(key: str) -> str | None:
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
        del _cache[key]
    return None


def _set_cache(key: str, data: str) -> None:
    _cache[key] = (time.time(), data)


def _invalidate_cache() -> None:
    """Invalidate all cached data after mutations."""
    _cache.clear()


# =========================================================================
# Guardrails
# =========================================================================

ALLOWED_BINARIES = {"kubectl", "helm"}

READONLY_BLOCKED = (
    "kubectl delete", "kubectl apply", "kubectl scale",
    "kubectl edit", "kubectl rollout undo", "kubectl rollout restart",
    "kubectl exec", "kubectl cp", "kubectl patch", "kubectl replace",
    "kubectl set", "kubectl create", "kubectl annotate", "kubectl label",
    "kubectl cordon", "kubectl uncordon", "kubectl taint",
)

ALWAYS_BLOCKED = ("kubectl delete", "kubectl exec", "kubectl cp")


def _check_binary(command: str) -> str | None:
    try:
        base = Path(_parse_command(command)[0]).name
    except (ValueError, IndexError):
        return "BLOCKED: could not parse command."
    if base not in ALLOWED_BINARIES:
        return f"BLOCKED: '{base}' not in allow-list."
    return None


def _is_readonly_safe(command: str) -> str | None:
    lowered = command.lower().strip()
    if "|" in command or "$(" in command or "`" in command:
        return "BLOCKED: pipes and subshells not supported in tools."
    for p in READONLY_BLOCKED:
        if p in lowered:
            return f"BLOCKED: '{p}' — read-only agent cannot run mutations."
    return _check_binary(command)


def _is_mutation_allowed(command: str) -> str | None:
    lowered = command.lower().strip()
    for p in ALWAYS_BLOCKED:
        if p in lowered:
            return f"BLOCKED: '{p}' is never allowed."
    if "|" in command or "$(" in command or "`" in command:
        return "BLOCKED: pipes and subshells not allowed."
    return _check_binary(command)


# =========================================================================
# JSON injection for kubectl get
# =========================================================================

_GET_PATTERN = re.compile(r"^kubectl\s+get\s+(?!-)", re.IGNORECASE)
_SKIP_JSON = re.compile(r"-o\s+(json|yaml|wide|jsonpath|custom-columns|name)", re.IGNORECASE)
_TOP_PATTERN = re.compile(r"^kubectl\s+top\s+", re.IGNORECASE)


def _maybe_inject_json(command: str) -> tuple[str, bool]:
    s = command.strip()
    if "|" in s or _TOP_PATTERN.match(s) or _SKIP_JSON.search(s):
        return s, False
    if _GET_PATTERN.match(s):
        return s + " -o json", True
    return s, False


# =========================================================================
# Output summarizers (identical logic, shared across tools)
# =========================================================================

def _safe_parse_items(raw: str) -> tuple[list | None, str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, raw
    if isinstance(data, list):
        return data, raw
    if isinstance(data, dict):
        if "items" in data:
            return data["items"], raw
        return [data], raw
    return None, raw


def _summarize_pods(raw: str) -> str:
    items, raw = _safe_parse_items(raw)
    if items is None:
        return raw
    if not items:
        return json.dumps({"total": 0, "pods": []})
    summary: dict = {"total": len(items), "healthy": 0, "unhealthy": []}
    for pod in items:
        name = pod.get("metadata", {}).get("name", "unknown")
        ns = pod.get("metadata", {}).get("namespace", "unknown")
        phase = pod.get("status", {}).get("phase", "Unknown")
        css = pod.get("status", {}).get("containerStatuses", [])
        restarts = sum(c.get("restartCount", 0) for c in css)
        ready = all(c.get("ready", False) for c in css) if css else False
        waits = []
        for c in css:
            w = c.get("state", {}).get("waiting", {})
            if w.get("reason"):
                waits.append(w["reason"])
        for c in pod.get("status", {}).get("initContainerStatuses", []):
            w = c.get("state", {}).get("waiting", {})
            if w.get("reason"):
                waits.append(f"init:{w['reason']}")
        term = None
        for c in css:
            t = c.get("lastState", {}).get("terminated", {})
            if t.get("reason"):
                term = t["reason"]
        ok = phase in ("Running", "Succeeded") and ready and not waits and restarts < 5
        if ok:
            summary["healthy"] += 1
        else:
            info: dict = {"name": name, "namespace": ns, "phase": phase,
                          "ready": ready, "restarts": restarts}
            if waits:
                info["waitingReasons"] = waits
            if term:
                info["lastTerminatedReason"] = term
            summary["unhealthy"].append(info)
    return json.dumps(summary, indent=2)


def _summarize_nodes(raw: str) -> str:
    items, raw = _safe_parse_items(raw)
    if items is None:
        return raw
    nodes = []
    for n in items:
        name = n.get("metadata", {}).get("name", "unknown")
        conds = n.get("status", {}).get("conditions", [])
        ready, issues = "Unknown", []
        for c in conds:
            if c.get("type") == "Ready":
                ready = c.get("status", "Unknown")
            elif c.get("status") == "True" and c.get("type") != "Ready":
                issues.append(c["type"])
        ni = n.get("status", {}).get("nodeInfo", {})
        nodes.append({"name": name, "ready": ready,
                      "kubelet": ni.get("kubeletVersion", ""),
                      "issues": issues or None})
    return json.dumps({"total": len(nodes), "nodes": nodes}, indent=2)


def _summarize_namespaces(raw: str) -> str:
    items, raw = _safe_parse_items(raw)
    if items is None:
        return raw
    return json.dumps({
        "total": len(items),
        "namespaces": [{"name": i.get("metadata", {}).get("name", ""),
                        "status": i.get("status", {}).get("phase", "")}
                       for i in items]
    }, indent=2)


def _summarize_deployments(raw: str) -> str:
    items, raw = _safe_parse_items(raw)
    if items is None:
        return raw
    bad = []
    for d in items:
        desired = d.get("spec", {}).get("replicas", 0)
        st = d.get("status", {})
        rdy = st.get("readyReplicas", 0)
        avail = st.get("availableReplicas", 0)
        if rdy != desired or avail != desired:
            bad.append({"name": d.get("metadata", {}).get("name", ""),
                        "namespace": d.get("metadata", {}).get("namespace", ""),
                        "desired": desired, "ready": rdy, "available": avail})
    return json.dumps({"total": len(items), "healthy": len(items) - len(bad),
                       "unhealthy": bad}, indent=2)


def _summarize_events(raw: str) -> str:
    items, raw = _safe_parse_items(raw)
    if items is None:
        return raw
    warns, normal = [], 0
    for e in items:
        if e.get("type", "Normal") != "Normal":
            warns.append({"reason": e.get("reason", ""),
                          "message": e.get("message", "")[:200],
                          "object": e.get("involvedObject", {}).get("name", ""),
                          "count": e.get("count", 1),
                          "lastSeen": e.get("lastTimestamp", "")})
        else:
            normal += 1
    warns.sort(key=lambda x: x.get("lastSeen", ""), reverse=True)
    return json.dumps({"normalCount": normal, "warnings": warns[:20]}, indent=2)


def _summarize_services(raw: str) -> str:
    items, raw = _safe_parse_items(raw)
    if items is None:
        return raw
    svcs = []
    for s in items:
        spec = s.get("spec", {})
        svcs.append({"name": s.get("metadata", {}).get("name", ""),
                     "type": spec.get("type", ""),
                     "clusterIP": spec.get("clusterIP", ""),
                     "ports": [{"port": p.get("port"), "targetPort": p.get("targetPort")}
                               for p in spec.get("ports", [])]})
    return json.dumps({"total": len(svcs), "services": svcs}, indent=2)


def _summarize_generic(raw: str) -> str:
    items, raw = _safe_parse_items(raw)
    if items is None:
        return raw
    return json.dumps({
        "total": len(items),
        "items": [{"name": i.get("metadata", {}).get("name", ""),
                   "namespace": i.get("metadata", {}).get("namespace", "")}
                  for i in items]
    }, indent=2)


def _summarize_describe(raw: str) -> str:
    lines = raw.splitlines()
    result: dict = {}
    events: list[str] = []
    conditions: list[str] = []
    in_ev = in_cond = False
    for line in lines:
        s = line.strip()
        if s.startswith("Status:"): result["status"] = s.split(":", 1)[1].strip()
        elif s.startswith("Restart Count:"):
            try: result["restartCount"] = int(s.split(":", 1)[1].strip())
            except ValueError: pass
        elif s.startswith("Reason:"): result["reason"] = s.split(":", 1)[1].strip()
        elif s.startswith("Message:"): result["message"] = s.split(":", 1)[1].strip()
        elif s.startswith("Exit Code:"):
            try: result["exitCode"] = int(s.split(":", 1)[1].strip())
            except ValueError: pass
        elif s.startswith("Node:"): result["node"] = s.split(":", 1)[1].strip()
        if s.startswith("Conditions:"): in_cond, in_ev = True, False; continue
        elif s.startswith("Events:"): in_ev, in_cond = True, False; continue
        elif s and not s.startswith(" ") and ":" in s: in_ev = in_cond = False
        if in_ev and s and not s.startswith("Type"): events.append(s[:150])
        if in_cond and s and not s.startswith("Type"): conditions.append(s[:150])
    if events: result["events"] = events[-10:]
    if conditions: result["conditions"] = conditions
    return json.dumps(result, indent=2) if result else raw[:3000]


def _summarize_logs(raw: str) -> str:
    lines = raw.splitlines()
    err_pat = re.compile(r"error|exception|fatal|panic|traceback|oom|killed", re.IGNORECASE)
    warn_pat = re.compile(r"warn", re.IGNORECASE)
    errors, warns = [], []
    for line in lines:
        s = line.strip()
        if not s: continue
        if err_pat.search(s): errors.append(s[:200])
        elif warn_pat.search(s): warns.append(s[:200])
    def dedup(items, limit):
        seen, out = set(), []
        for i in items:
            k = i[:80]
            if k not in seen: seen.add(k); out.append(i)
        return out[:limit]
    return json.dumps({
        "totalLines": len(lines),
        "errorCount": len(errors), "warningCount": len(warns),
        "uniqueErrors": dedup(errors, 15), "uniqueWarnings": dedup(warns, 10),
    }, indent=2)


def _matches(cmd, patterns): return any(p in cmd for p in patterns)

def _has_json_output(cmd, injected):
    return injected or bool(re.search(r"-o[= ]json\b", cmd, re.IGNORECASE))

def _detect_and_summarize(command: str, stdout: str, injected: bool) -> str:
    c = command.lower().strip()
    if _has_json_output(c, injected):
        if _matches(c, ["get pods", "get pod", "get po"]): return _summarize_pods(stdout)
        elif _matches(c, ["get nodes", "get node", "get no "]): return _summarize_nodes(stdout)
        elif _matches(c, ["get namespaces", "get namespace", "get ns"]): return _summarize_namespaces(stdout)
        elif _matches(c, ["get deployments", "get deployment", "get deploy"]): return _summarize_deployments(stdout)
        elif _matches(c, ["get events", "get event", "get ev "]): return _summarize_events(stdout)
        elif _matches(c, ["get svc", "get services", "get service"]): return _summarize_services(stdout)
        else: return _summarize_generic(stdout)
    if _matches(c, ["describe pod", "describe pods"]): return _summarize_describe(stdout)
    elif _matches(c, ["logs ", "log "]): return _summarize_logs(stdout)
    elif _matches(c, ["describe "]): return _summarize_describe(stdout)
    return stdout[:5000] if len(stdout) > 5000 else stdout


# =========================================================================
# LLM tools (read-only)
# =========================================================================

@tool
def scan_cluster(command: str) -> str:
    """Run a read-only kubectl command to discover cluster resources.
    Returns structured JSON summaries. Mutations are blocked."""
    err = _is_readonly_safe(command)
    if err: return err
    ckey = _cache_key(command)
    if ckey:
        cached = _get_cached(ckey)
        if cached: return cached
    actual, injected = _maybe_inject_json(command)
    rc, stdout, stderr = _run_command(actual)
    if rc != 0:
        return json.dumps({"success": False, "exitCode": rc, "error": stderr.strip()[:1000]})
    if not stdout.strip():
        return json.dumps({"success": True, "result": "No resources found"})
    result = _detect_and_summarize(command, stdout, injected)
    if ckey: _set_cache(ckey, result)
    return result


@tool
def investigate(command: str) -> str:
    """Run a read-only kubectl command to investigate specific resources.
    Returns extracted key fields, errors, and warnings. Mutations blocked."""
    err = _is_readonly_safe(command)
    if err: return err
    actual, injected = _maybe_inject_json(command)
    rc, stdout, stderr = _run_command(actual)
    if rc != 0:
        return json.dumps({"success": False, "exitCode": rc, "error": stderr.strip()[:1000]})
    if not stdout.strip():
        return json.dumps({"success": True, "result": "No output"})
    return _detect_and_summarize(command, stdout, injected)


# =========================================================================
# Deterministic execution + verification
# =========================================================================

@traceable(name="execute_command", run_type="tool", metadata={"mutating": True})
def execute_command(command: str) -> dict:
    err = _is_mutation_allowed(command)
    if err: return {"command": command, "success": False, "error": err}
    rc, stdout, stderr = _run_command(command)
    _invalidate_cache()  # bust cache after mutation
    if rc != 0:
        return {"command": command, "success": False, "exitCode": rc,
                "error": stderr.strip()[:1000] or "Command failed"}
    return {"command": command, "success": True,
            "output": (stdout.strip() or "(no output)")[:3000]}


@traceable(name="verify_fix", run_type="tool")
def verify_fix(command: str) -> dict:
    cmd_lower = command.lower()
    ns = "default"
    ns_match = re.search(r"-n\s+(\S+)", command)
    if ns_match: ns = ns_match.group(1)
    if "patch pod" in cmd_lower or ("set image" in cmd_lower and "pod/" in cmd_lower):
        m = re.search(r"(?:patch\s+pod[/\s])(\S+)", command)
        if m: return _verify_pod(m.group(1), ns)
    if "set image" in cmd_lower or "set resources" in cmd_lower:
        m = re.search(r"(?:deployment[/\s])(\S+)", command)
        if m: return _verify_deployment(m.group(1), ns)
    if "rollout restart" in cmd_lower:
        m = re.search(r"(?:deployment[/\s])(\S+)", command)
        if m: return _verify_rollout(m.group(1), ns)
    if "scale" in cmd_lower:
        m = re.search(r"(?:deployment[/\s])(\S+)", command)
        if m: return _verify_deployment(m.group(1), ns)
    return {"verified": True, "method": "command_succeeded",
            "note": "No specific verification for this command type"}


def _verify_pod(pod_name, namespace):
    rc, stdout, _ = _run_args(["kubectl","get","pod",pod_name,"-n",namespace,
        "-o","jsonpath={.status.phase} {.status.containerStatuses[0].ready}"])
    if rc != 0: return {"verified": False, "note": "Could not check pod status"}
    parts = stdout.strip().split()
    phase = parts[0] if parts else "Unknown"
    ready = parts[1] if len(parts) > 1 else "false"
    is_ok = phase == "Running" and ready == "true"
    return {"verified": is_ok, "method": "kubectl_get_pod", "phase": phase, "ready": ready,
            "note": "Pod is healthy" if is_ok else f"Pod still unhealthy: phase={phase}, ready={ready}"}


def _verify_deployment(dep_name, namespace):
    rc, stdout, _ = _run_args(["kubectl","get","deployment",dep_name,"-n",namespace,
        "-o","jsonpath={.status.readyReplicas}/{.spec.replicas}"])
    if rc != 0: return {"verified": False, "note": "Could not check deployment status"}
    parts = stdout.strip().split("/")
    ready = parts[0] if parts else "0"
    desired = parts[1] if len(parts) > 1 else "0"
    is_ok = ready == desired and ready != "0"
    return {"verified": is_ok, "method": "kubectl_get_deployment",
            "ready": ready, "desired": desired, "note": f"Deployment {ready}/{desired} ready"}


def _verify_rollout(dep_name, namespace):
    rc, stdout, stderr = _run_args(["kubectl","rollout","status",f"deployment/{dep_name}",
        "-n",namespace,"--timeout=30s"])
    is_ok = rc == 0
    return {"verified": is_ok, "method": "kubectl_rollout_status",
            "output": (stdout if is_ok else stderr).strip()[:500],
            "note": "Rollout completed" if is_ok else "Rollout not yet complete"}


def _wait_and_verify(command, max_wait=60, interval=5):
    result = {"verified": False, "note": "Verification not started"}
    for _ in range(1, max_wait // interval + 1):
        time.sleep(interval)
        result = verify_fix(command)
        if result.get("verified"): return result
    result["note"] = f"Not verified after {max_wait}s: {result.get('note', '')}"
    return result


# =========================================================================
# Agent state — append-only agent_logs
# =========================================================================

class AgentState(TypedDict):
    task: str
    phase: str
    issues: list[dict]
    diagnoses: list[dict]
    recommendations: list[dict]
    approved_commands: list[str]
    actions_taken: list[dict]
    summary: str
    agent_logs: Annotated[list[str], operator.add]


# =========================================================================
# JSON extraction
# =========================================================================

def _extract_json(text):
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        try: return json.loads(fence.group(1).strip())
        except json.JSONDecodeError: pass
    try: return json.loads(text.strip())
    except json.JSONDecodeError: pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try: return json.loads(brace.group())
        except json.JSONDecodeError: pass
    return None

def _extract_json_strict(text, required_key):
    parsed = _extract_json(text)
    if parsed is None:
        raise ValueError(f"LLM response was not valid JSON:\n{text[:500]}")
    if required_key not in parsed:
        raise ValueError(f"Missing key '{required_key}' in: {list(parsed.keys())}")
    return parsed


# =========================================================================
# Retry wrapper
# =========================================================================

def _invoke_with_retry(agent, messages, config=None, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            return agent.invoke(messages, config=config or {})
        except Exception as e:
            if attempt == max_retries: raise
            wait = 2 ** attempt
            _log("retry", f"Attempt {attempt} failed ({type(e).__name__}: {e}), retrying in {wait}s...")
            time.sleep(wait)


# =========================================================================
# Shared checkpointer — single InMemorySaver for all sub-agents
# =========================================================================

_shared_checkpointer = InMemorySaver()


# =========================================================================
# Node 1: Detection Agent (Haiku — fast scan)
# =========================================================================

@traceable(name="detection_agent", run_type="chain",
           tags=["detection"], metadata={"node": "detection"})
def detection_node(state: AgentState) -> dict:
    _log("detection", "Scanning cluster for issues...")
    llm = make_llm("detection")
    agent = create_agent(
        model=llm, tools=[scan_cluster],
        system_prompt="""\
You are a Kubernetes Detection Agent. Your ONLY job is to scan the cluster
and identify unhealthy resources. You do NOT diagnose or fix anything.

The scan_cluster tool returns structured JSON summaries.

Workflow:
1. Run: kubectl get namespaces
2. For each namespace: kubectl get pods -n <ns>
3. Also check: kubectl get nodes, kubectl get deployments -A
4. Collect all unhealthy resources.

Output format — respond with ONLY a JSON object:
{
  "issues_found": true/false,
  "issues": [
    {"resource_type":"pod","name":"...","namespace":"...","status":"CrashLoopBackOff","details":"restarts: 17"}
  ],
  "healthy_summary": "152 pods healthy across 8 namespaces, 3 nodes ready"
}

Be thorough but fast. Check all namespaces.""",
        checkpointer=_shared_checkpointer,
        name="detection_agent",
    )
    result = _invoke_with_retry(
        agent,
        {"messages": [{"role": "user", "content": f"Scan the cluster for: {state['task']}"}]},
        config={"configurable": {"thread_id": f"detection-{id(state)}"}},
    )
    response = result["messages"][-1].content
    try:
        issues = _extract_json_strict(response, "issues_found")
    except ValueError as e:
        _log("detection", f"JSON parse failed: {e}")
        issues = {"issues_found": False, "issues": [], "healthy_summary": "Parse error."}
    if issues.get("issues"):
        issue_list = issues["issues"]
        healthy = issues.get("healthy_summary", "")
        _log("detection", f"Found {len(issue_list)} issue(s)")
        return {"phase": "diagnosis", "issues": issue_list,
                "agent_logs": [f"Detection: found {len(issue_list)} issues. {healthy}"]}
    else:
        _log("detection", "Cluster is healthy")
        msg = issues.get("healthy_summary", "No issues detected.")
        return {"phase": "complete", "issues": [], "summary": msg,
                "agent_logs": ["Detection: cluster is healthy."]}


# =========================================================================
# Node 2: Diagnosis Agent (Sonnet — precision)
# =========================================================================

@traceable(name="diagnosis_agent", run_type="chain",
           tags=["diagnosis"], metadata={"node": "diagnosis"})
def diagnosis_node(state: AgentState) -> dict:
    issues = state.get("issues", [])
    if not issues: return {"phase": "complete", "diagnoses": []}
    _log("diagnosis", f"Investigating {len(issues)} issue(s)...")
    llm = make_llm("diagnosis")
    agent = create_agent(
        model=llm, tools=[investigate],
        system_prompt="""\
You are a Kubernetes Diagnosis Agent. You receive a list of unhealthy
resources and must determine the root cause AND extract concrete details.

For each issue:
1. kubectl describe pod <name> -n <namespace>
2. kubectl logs <name> -n <namespace> --tail=100
3. If crashed: kubectl logs <name> -n <namespace> --previous --tail=100
4. kubectl get events -n <namespace>

CRITICAL: Extract SPECIFIC details:
- Exact container name(s), image name+tag
- Owning controller kind and name
- For ImagePullBackOff: exact error from events
- For OOMKilled: current memory limit
- For CrashLoopBackOff: exit code and last log errors

Output format — respond with ONLY a JSON object:
{
  "diagnoses": [
    {
      "resource": "pod/api-server", "namespace": "production",
      "root_cause": "OOMKilled — exceeds 256Mi",
      "evidence": ["Exit code 137", "restartCount: 17"],
      "severity": "high",
      "container_name": "api", "image": "myregistry.io/api:v2.3.1",
      "owner_kind": "Deployment", "owner_name": "api-server"
    }
  ]
}
Be precise — no placeholders.""",
        checkpointer=_shared_checkpointer,
        name="diagnosis_agent",
    )
    result = _invoke_with_retry(
        agent,
        {"messages": [{"role": "user", "content": f"Investigate:\n{json.dumps(issues, indent=2)}"}]},
        config={"configurable": {"thread_id": f"diagnosis-{id(state)}"}},
    )
    try:
        parsed = _extract_json_strict(result["messages"][-1].content, "diagnoses")
        diagnoses = parsed["diagnoses"]
    except ValueError as e:
        _log("diagnosis", f"JSON parse failed: {e}"); diagnoses = []
    _log("diagnosis", f"Completed {len(diagnoses)} diagnosis(es)")
    return {"phase": "recommendation", "diagnoses": diagnoses,
            "agent_logs": [f"Diagnosis: {len(diagnoses)} root cause(s) identified."]}


# =========================================================================
# Node 3: Recommendation Agent (Sonnet — precision)
# =========================================================================

@traceable(name="recommendation_agent", run_type="chain",
           tags=["recommendation"], metadata={"node": "recommendation"})
def recommendation_node(state: AgentState) -> dict:
    diagnoses = state.get("diagnoses", [])
    if not diagnoses: return {"phase": "complete", "recommendations": []}
    actionable = [d for d in diagnoses if d.get("severity") in ("high", "medium", "critical")]
    if not actionable:
        _log("recommendation", "No actionable issues")
        return {"phase": "complete", "recommendations": [],
                "agent_logs": ["Recommendation: no actionable issues."]}
    _log("recommendation", f"Generating fixes for {len(actionable)} issue(s)...")
    llm = make_llm("recommendation")
    agent = create_agent(
        model=llm, tools=[investigate],
        system_prompt="""\
You are a Kubernetes Recommendation Agent. Propose concrete kubectl commands.

RULES:
1. NEVER propose read-only commands as fixes.
2. NEVER use placeholders. Every command must be copy-paste ready.
3. Use the investigate tool to look up values BEFORE proposing.
4. Prefer least disruptive fixes.
5. Allowed verbs: apply, scale, rollout restart/undo, patch, set, create, annotate, label.
6. NEVER propose: kubectl delete, exec, cp, pipes, subshells.
7. Do NOT wrap flag values in single quotes.

Output format — respond with ONLY a JSON object:
{
  "recommendations": [
    {
      "issue": "pod/api-server OOMKilled in production",
      "fix_command": "kubectl set resources deployment/api-server -n production --limits=memory=512Mi",
      "explanation": "Double memory limit from 256Mi to 512Mi",
      "risk": "low",
      "alternative": "kubectl rollout restart deployment/api-server -n production"
    }
  ]
}""",
        checkpointer=_shared_checkpointer,
        name="recommendation_agent",
    )
    result = _invoke_with_retry(
        agent,
        {"messages": [{"role": "user", "content": f"Propose fixes:\n{json.dumps(actionable, indent=2)}"}]},
        config={"configurable": {"thread_id": f"recommendation-{id(state)}"}},
    )
    try:
        parsed = _extract_json_strict(result["messages"][-1].content, "recommendations")
        recs = parsed["recommendations"]
    except ValueError as e:
        _log("recommendation", f"JSON parse failed: {e}"); recs = []
    _log("recommendation", f"Generated {len(recs)} recommendation(s)")
    return {"phase": "guardrail", "recommendations": recs,
            "agent_logs": [f"Recommendation: {len(recs)} proposed fix(es)."]}


# =========================================================================
# Node 3b: Guardrail Filter (deterministic)
# =========================================================================

@traceable(name="guardrail_filter", run_type="tool", tags=["guardrail"])
def guardrail_filter_node(state: AgentState) -> dict:
    recs = state.get("recommendations", [])
    executable = []
    for rec in recs:
        cmd = rec.get("fix_command", "")
        if rec.get("no_fix_needed") or not cmd.strip(): continue
        cl = cmd.lower().strip()
        if any(cl.startswith(r) for r in ["kubectl get","kubectl describe","kubectl logs","kubectl top"]):
            _log("guardrail", f"FILTERED read-only: {cmd[:60]}"); continue
        if any(b in cl for b in ["kubectl delete","kubectl exec","kubectl cp"]):
            _log("guardrail", f"FILTERED blocked: {cmd[:60]}"); continue
        if "|" in cmd or "$(" in cmd or "`" in cmd:
            _log("guardrail", f"FILTERED pipe/subshell: {cmd[:60]}"); continue
        if "<" in cmd and ">" in cmd:
            _log("guardrail", f"FILTERED placeholder: {cmd[:60]}"); continue
        executable.append(rec)
    filtered = len(recs) - len(executable)
    _log("guardrail", f"{len(executable)} executable" + (f" ({filtered} filtered)" if filtered else ""))
    return {"phase": "approval", "recommendations": executable,
            "agent_logs": [f"Guardrail: {len(executable)} passed, {filtered} filtered."]}


# =========================================================================
# Node 4: Approval Gate (human-in-the-loop)
# =========================================================================

@traceable(name="approval_gate", run_type="tool", tags=["approval","human-in-the-loop"])
def approval_node(state: AgentState) -> dict:
    recs = state.get("recommendations", [])
    if not recs:
        print("\n  [approval] No fixes to approve.")
        return {"phase": "complete", "approved_commands": []}
    print("\n" + "=" * 60)
    print("PROPOSED FIXES — APPROVAL REQUIRED")
    print("=" * 60)
    for i, rec in enumerate(recs, 1):
        print(f"\n  Fix #{i}:")
        print(f"    Issue:   {rec.get('issue', 'Unknown')}")
        print(f"    Command: {rec.get('fix_command', 'N/A')}")
        print(f"    Reason:  {rec.get('explanation', 'N/A')}")
        print(f"    Risk:    {rec.get('risk', 'unknown')}")
        if rec.get("alternative"): print(f"    Alt:     {rec['alternative']}")
    print(f"\n  Options: all | 1,3 | none")
    choice = input("\n  Approve which fixes? > ").strip().lower()
    approved: list[str] = []
    if choice == "all":
        approved = [r["fix_command"] for r in recs if r.get("fix_command")]
    elif choice != "none" and choice:
        try:
            for idx in [int(x.strip()) - 1 for x in choice.split(",")]:
                if 0 <= idx < len(recs) and recs[idx].get("fix_command"):
                    approved.append(recs[idx]["fix_command"])
        except ValueError:
            print("  [approval] Invalid input, skipping.")
    if approved:
        print(f"\n  [approval] Approved {len(approved)} command(s)")
        return {"phase": "remediation", "approved_commands": approved,
                "agent_logs": [f"Approval: {len(approved)} command(s) approved."]}
    else:
        print("\n  [approval] None approved.")
        return {"phase": "complete", "approved_commands": [],
                "agent_logs": ["Approval: user declined all fixes."]}


# =========================================================================
# Node 5: Executor + Verifier (deterministic)
# =========================================================================

@traceable(name="remediation_executor", run_type="tool", tags=["executor"])
def remediation_node(state: AgentState) -> dict:
    approved = state.get("approved_commands", [])
    if not approved: return {"phase": "complete", "actions_taken": []}
    print(f"\n  [executor] Running {len(approved)} approved command(s)...\n")
    actions = []
    for i, command in enumerate(approved, 1):
        print(f"    [{i}/{len(approved)}] {command}")
        result = execute_command(command)
        success = result.get("success", False)
        if success: print(f"      ✓ OK: {result.get('output', '')[:80]}")
        else: print(f"      ✗ FAIL: {result.get('error', 'Unknown')[:80]}")
        verification = {}
        if success:
            verification = _wait_and_verify(command, max_wait=30, interval=5)
            status = "✓" if verification.get("verified") else "⚠"
            print(f"      {status} Verify: {verification.get('note', '')}")
        actions.append({"command": command, "success": success,
                        "output": result.get("output", result.get("error", "")),
                        "verification": verification})
    ok = sum(1 for a in actions if a["success"])
    verified = sum(1 for a in actions if a.get("verification", {}).get("verified"))
    print(f"\n  [executor] Done: {ok}/{len(actions)} succeeded, {verified}/{ok} verified")
    return {"phase": "complete", "actions_taken": actions,
            "agent_logs": [f"Executor: {ok}/{len(actions)} succeeded, {verified}/{ok} verified."]}


# =========================================================================
# Node 6: Summary Agent (Haiku — fast formatting)
# =========================================================================

@traceable(name="summary_agent", run_type="chain", tags=["summary"])
def summary_node(state: AgentState) -> dict:
    llm = make_llm("summary")
    context = json.dumps({
        "task": state.get("task", ""),
        "issues_found": state.get("issues", []),
        "diagnoses": state.get("diagnoses", []),
        "recommendations": state.get("recommendations", []),
        "actions_taken": state.get("actions_taken", []),
        "agent_logs": state.get("agent_logs", []),
    }, indent=2, default=str)
    response = llm.invoke([
        SystemMessage(content="You are a Kubernetes Operations Summary Agent. Produce a clear, concise summary: what was checked, issues found, root causes, actions taken + verification, remaining items."),
        HumanMessage(content=f"Summarize:\n{context}"),
    ])
    return {"summary": response.content, "phase": "done"}


# =========================================================================
# Orchestrator
# =========================================================================

def _route_after_detection(state): return "diagnosis" if state.get("issues") else "summary"
def _route_after_approval(state): return "remediation" if state.get("approved_commands") else "summary"

def build_orchestrator() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("detection", detection_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("guardrail", guardrail_filter_node)
    graph.add_node("approval", approval_node)
    graph.add_node("remediation", remediation_node)
    graph.add_node("summary", summary_node)
    graph.set_entry_point("detection")
    graph.add_conditional_edges("detection", _route_after_detection,
                                {"diagnosis": "diagnosis", "summary": "summary"})
    graph.add_edge("diagnosis", "recommendation")
    graph.add_edge("recommendation", "guardrail")
    graph.add_edge("guardrail", "approval")
    graph.add_conditional_edges("approval", _route_after_approval,
                                {"remediation": "remediation", "summary": "summary"})
    graph.add_edge("remediation", "summary")
    graph.add_edge("summary", END)
    return graph.compile()


# =========================================================================
# Entry point
# =========================================================================

def main():
    _configure_langsmith()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not task:
        print("Multi-Agent Kubernetes Operations System")
        print("=" * 50)
        print('\nExamples:')
        print('  python multi_agent.py "Find all crashing pods and fix them"')
        print('  python multi_agent.py "Check cluster health and remediate issues"')
        print()
        task = input("Enter your task: ").strip()
        

        if not task: sys.exit("No task entered.")
    print(f"\n{'=' * 60}")
    print(f"Task: {task}")
    print(f"{'=' * 60}")
    models_used = {k: v["model"] for k, v in MODEL_REGISTRY.items()}
    print(f"\nModels: {json.dumps(models_used, indent=2)}")
    print(f"\nPipeline: Detection → Diagnosis → Recommendation → Guardrail → Approval → Executor → Summary\n")
    orchestrator = build_orchestrator()
    initial_state: AgentState = {
        "task": task, "phase": "detection",
        "issues": [], "diagnoses": [], "recommendations": [],
        "approved_commands": [], "actions_taken": [],
        "summary": "", "agent_logs": [],
    }
    result = orchestrator.invoke(
        initial_state,
        config={"run_name": f"k8s-ops: {task[:50]}",
                "tags": ["production-run"],
                "metadata": {"task": task, "models": models_used},
                "recursion_limit": 50},
    )
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60 + "\n")
    print(result.get("summary", "No summary generated."))
    print("\n" + "-" * 60)
    print("AGENT EXECUTION LOG")
    print("-" * 60)
    for log in result.get("agent_logs", []):
        print(f"  • {log}")
    print()

if __name__ == "__main__":
    main()
