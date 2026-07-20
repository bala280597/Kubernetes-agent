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
      5. Approval Gate     — presents fixes, collects human approval
      6. Executor          — runs approved commands via subprocess (shell=False)
      7. Verifier          — confirms fixes worked via kubectl checks

Setup:
  pip install -U langgraph langchain-anthropic langchain-core
  export ANTHROPIC_API_KEY="your-key"

Run:
  python multi_agent.py
  python multi_agent.py "Find all crashing pods and fix them"
  python multi_agent.py "Check cluster health and remediate issues"
"""

import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

WORKDIR = Path.cwd()
IS_WINDOWS = platform.system() == "Windows"

# =========================================================================
# Command execution — shell=False, argument lists
# =========================================================================

def _parse_command(command: str) -> list[str]:
    """Parse a command string into an argument list.
    Uses POSIX splitting on Linux/Mac, Windows-safe splitting on Windows."""
    try:
        return shlex.split(command, posix=not IS_WINDOWS)
    except ValueError:
        return command.split()


def _run_args(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Execute a command as an argument list (shell=False).
    Returns (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            args, shell=False, capture_output=True,
            text=True, timeout=timeout, cwd=WORKDIR,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after 120s"
    except FileNotFoundError:
        return -1, "", f"Command not found: {args[0]}"
    except OSError as e:
        return -1, "", f"OS error: {e}"


def _run_command(command: str) -> tuple[int, str, str]:
    """Parse a command string and execute via shell=False."""
    args = _parse_command(command)
    return _run_args(args)


# =========================================================================
# Discovery cache
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

MUTATION_ALLOWED = (
    "kubectl apply", "kubectl scale", "kubectl rollout restart",
    "kubectl rollout undo", "kubectl patch", "kubectl set",
    "kubectl create", "kubectl annotate", "kubectl label",
    "helm repo", "helm install", "helm upgrade", "helm rollback",
)


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
    """Check if a mutation command is structurally valid (binary + verb).
    Does NOT check approval — executor runs only approved commands."""
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
# Output summarizers
# =========================================================================

def _safe_parse_items(raw: str) -> tuple[list | None, str]:
    """Parse kubectl JSON output, handling all response shapes.
    Returns (items_list, raw) — items is None if parsing fails."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, raw
    if isinstance(data, list):
        return data, raw
    if isinstance(data, dict):
        if "items" in data:
            return data["items"], raw
        # Single resource (e.g. kubectl get pod <name> -o json)
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
        if s.startswith("Status:"):
            result["status"] = s.split(":", 1)[1].strip()
        elif s.startswith("Restart Count:"):
            try:
                result["restartCount"] = int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith("Reason:"):
            result["reason"] = s.split(":", 1)[1].strip()
        elif s.startswith("Message:"):
            result["message"] = s.split(":", 1)[1].strip()
        elif s.startswith("Exit Code:"):
            try:
                result["exitCode"] = int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith("Node:"):
            result["node"] = s.split(":", 1)[1].strip()
        if s.startswith("Conditions:"):
            in_cond, in_ev = True, False
            continue
        elif s.startswith("Events:"):
            in_ev, in_cond = True, False
            continue
        elif s and not s.startswith(" ") and ":" in s:
            in_ev = in_cond = False
        if in_ev and s and not s.startswith("Type"):
            events.append(s[:150])
        if in_cond and s and not s.startswith("Type"):
            conditions.append(s[:150])
    if events:
        result["events"] = events[-10:]
    if conditions:
        result["conditions"] = conditions
    return json.dumps(result, indent=2) if result else raw[:3000]


def _summarize_logs(raw: str) -> str:
    lines = raw.splitlines()
    err_pat = re.compile(r"error|exception|fatal|panic|traceback|oom|killed", re.IGNORECASE)
    warn_pat = re.compile(r"warn", re.IGNORECASE)
    errors, warns = [], []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if err_pat.search(s):
            errors.append(s[:200])
        elif warn_pat.search(s):
            warns.append(s[:200])

    def dedup(items: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        out = []
        for i in items:
            k = i[:80]
            if k not in seen:
                seen.add(k)
                out.append(i)
        return out[:limit]

    return json.dumps({
        "totalLines": len(lines),
        "errorCount": len(errors), "warningCount": len(warns),
        "uniqueErrors": dedup(errors, 15),
        "uniqueWarnings": dedup(warns, 10),
    }, indent=2)


def _matches(cmd: str, patterns: list[str]) -> bool:
    return any(p in cmd for p in patterns)


def _has_json_output(cmd: str, injected: bool) -> bool:
    """Check if this command produces full JSON output (not jsonpath/custom-columns)."""
    if injected:
        return True
    # Match -o json or -o=json but NOT -o jsonpath
    return bool(re.search(r"-o[= ]json\b", cmd, re.IGNORECASE))


def _detect_and_summarize(command: str, stdout: str, injected: bool) -> str:
    c = command.lower().strip()
    if _has_json_output(c, injected):
        if _matches(c, ["get pods", "get pod", "get po"]):
            return _summarize_pods(stdout)
        elif _matches(c, ["get nodes", "get node", "get no "]):
            return _summarize_nodes(stdout)
        elif _matches(c, ["get namespaces", "get namespace", "get ns"]):
            return _summarize_namespaces(stdout)
        elif _matches(c, ["get deployments", "get deployment", "get deploy"]):
            return _summarize_deployments(stdout)
        elif _matches(c, ["get events", "get event", "get ev "]):
            return _summarize_events(stdout)
        elif _matches(c, ["get svc", "get services", "get service"]):
            return _summarize_services(stdout)
        else:
            return _summarize_generic(stdout)
    if _matches(c, ["describe pod", "describe pods"]):
        return _summarize_describe(stdout)
    elif _matches(c, ["logs ", "log "]):
        return _summarize_logs(stdout)
    elif _matches(c, ["describe "]):
        return _summarize_describe(stdout)
    return stdout[:5000] if len(stdout) > 5000 else stdout


# =========================================================================
# LLM tools (read-only, for Detection / Diagnosis / Recommendation agents)
# =========================================================================

@tool
def scan_cluster(command: str) -> str:
    """Run a read-only kubectl command to discover cluster resources.
    Use for: get namespaces, get pods, get nodes, get deployments,
    get statefulsets, get daemonsets, get jobs, get svc, get hpa.
    Returns structured JSON summaries. Mutations are blocked."""
    err = _is_readonly_safe(command)
    if err:
        return err
    ckey = _cache_key(command)
    if ckey:
        cached = _get_cached(ckey)
        if cached:
            return cached
    actual, injected = _maybe_inject_json(command)
    rc, stdout, stderr = _run_command(actual)
    if rc != 0:
        return json.dumps({"success": False, "exitCode": rc,
                           "error": stderr.strip()[:1000]})
    if not stdout.strip():
        return json.dumps({"success": True, "result": "No resources found"})
    result = _detect_and_summarize(command, stdout, injected)
    if ckey:
        _set_cache(ckey, result)
    return result


@tool
def investigate(command: str) -> str:
    """Run a read-only kubectl command to investigate specific resources.
    Use for: describe pod, logs, get events, get pod -o jsonpath,
    describe deployment, describe node, top pods, top nodes.
    Returns extracted key fields, errors, and warnings. Mutations blocked."""
    err = _is_readonly_safe(command)
    if err:
        return err
    actual, injected = _maybe_inject_json(command)
    rc, stdout, stderr = _run_command(actual)
    if rc != 0:
        return json.dumps({"success": False, "exitCode": rc,
                           "error": stderr.strip()[:1000]})
    if not stdout.strip():
        return json.dumps({"success": True, "result": "No output"})
    return _detect_and_summarize(command, stdout, injected)


# =========================================================================
# Deterministic execution + verification (NO LLM)
# =========================================================================

def execute_command(command: str) -> dict:
    """Execute a single kubectl command deterministically.
    Returns a structured result dict. No LLM involved."""
    err = _is_mutation_allowed(command)
    if err:
        return {"command": command, "success": False, "error": err}
    rc, stdout, stderr = _run_command(command)
    if rc != 0:
        return {"command": command, "success": False, "exitCode": rc,
                "error": stderr.strip()[:1000] or "Command failed"}
    return {"command": command, "success": True,
            "output": (stdout.strip() or "(no output)")[:3000]}


def verify_fix(command: str) -> dict:
    """Deterministic verification: check if a fix worked by inspecting
    the target resource. No LLM — pure kubectl + Python parsing."""
    cmd_lower = command.lower()
    ns = "default"
    ns_match = re.search(r"-n\s+(\S+)", command)
    if ns_match:
        ns = ns_match.group(1)

    if "patch pod" in cmd_lower or ("set image" in cmd_lower and "pod/" in cmd_lower):
        pod_match = re.search(r"(?:patch\s+pod[/\s])(\S+)", command)
        if pod_match:
            return _verify_pod(pod_match.group(1), ns)

    if "set image" in cmd_lower or "set resources" in cmd_lower:
        dep_match = re.search(r"(?:deployment[/\s])(\S+)", command)
        if dep_match:
            return _verify_deployment(dep_match.group(1), ns)

    if "rollout restart" in cmd_lower:
        dep_match = re.search(r"(?:deployment[/\s])(\S+)", command)
        if dep_match:
            return _verify_rollout(dep_match.group(1), ns)

    if "scale" in cmd_lower:
        dep_match = re.search(r"(?:deployment[/\s])(\S+)", command)
        if dep_match:
            return _verify_deployment(dep_match.group(1), ns)

    return {"verified": True, "method": "command_succeeded",
            "note": "No specific verification for this command type"}


def _verify_pod(pod_name: str, namespace: str) -> dict:
    """Check if a pod is now healthy."""
    rc, stdout, _ = _run_args([
        "kubectl", "get", "pod", pod_name, "-n", namespace,
        "-o", "jsonpath={.status.phase} {.status.containerStatuses[0].ready}"
    ])
    if rc != 0:
        return {"verified": False, "note": "Could not check pod status"}
    parts = stdout.strip().split()
    phase = parts[0] if parts else "Unknown"
    ready = parts[1] if len(parts) > 1 else "false"
    is_ok = phase == "Running" and ready == "true"
    return {"verified": is_ok, "method": "kubectl_get_pod",
            "phase": phase, "ready": ready,
            "note": "Pod is healthy" if is_ok
                    else f"Pod still unhealthy: phase={phase}, ready={ready}"}


def _verify_deployment(dep_name: str, namespace: str) -> dict:
    """Check if a deployment is fully available."""
    rc, stdout, _ = _run_args([
        "kubectl", "get", "deployment", dep_name, "-n", namespace,
        "-o", "jsonpath={.status.readyReplicas}/{.spec.replicas}"
    ])
    if rc != 0:
        return {"verified": False, "note": "Could not check deployment status"}
    parts = stdout.strip().split("/")
    ready = parts[0] if parts else "0"
    desired = parts[1] if len(parts) > 1 else "0"
    is_ok = ready == desired and ready != "0"
    return {"verified": is_ok, "method": "kubectl_get_deployment",
            "ready": ready, "desired": desired,
            "note": f"Deployment {ready}/{desired} ready"}


def _verify_rollout(dep_name: str, namespace: str) -> dict:
    """Check rollout status of a deployment."""
    rc, stdout, stderr = _run_args([
        "kubectl", "rollout", "status", f"deployment/{dep_name}",
        "-n", namespace, "--timeout=30s",
    ])
    is_ok = rc == 0
    return {"verified": is_ok, "method": "kubectl_rollout_status",
            "output": (stdout if is_ok else stderr).strip()[:500],
            "note": "Rollout completed" if is_ok else "Rollout not yet complete"}


# =========================================================================
# Shared LLM factory
# =========================================================================

def make_llm(temperature: float = 0.2):
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=temperature,
        max_tokens=4096,
    )


# =========================================================================
# Agent state
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
    agent_logs: list[str]


# =========================================================================
# Helper: extract JSON from LLM response
# =========================================================================

def _extract_json(text: str) -> dict | None:
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass
    return None


# =========================================================================
# Node 1: Detection Agent (LLM)
# =========================================================================

def detection_node(state: AgentState) -> dict:
    from langchain.agents import create_agent

    print("\n  [detection] Scanning cluster for issues...")
    llm = make_llm()
    agent = create_agent(
        model=llm, tools=[scan_cluster],
        system_prompt="""\
You are a Kubernetes Detection Agent. Your ONLY job is to scan the cluster
and identify unhealthy resources. You do NOT diagnose or fix anything.

The scan_cluster tool returns structured JSON summaries. Healthy resources
are counted; unhealthy ones are listed with details.

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
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": f"Scan the cluster for: {state['task']}"}]}
    )
    response = result["messages"][-1].content
    issues = _extract_json(response)

    if issues and issues.get("issues"):
        issue_list = issues["issues"]
        healthy = issues.get("healthy_summary", "")
        print(f"  [detection] Found {len(issue_list)} issue(s)")
        return {"phase": "diagnosis", "issues": issue_list,
                "agent_logs": state.get("agent_logs", []) + [
                    f"Detection: found {len(issue_list)} issues. {healthy}"]}
    else:
        print("  [detection] Cluster is healthy")
        msg = issues.get("healthy_summary", "No issues detected.") if issues else "No issues detected."
        return {"phase": "complete", "issues": [], "summary": msg,
                "agent_logs": state.get("agent_logs", []) + [
                    "Detection: cluster is healthy."]}


# =========================================================================
# Node 2: Diagnosis Agent (LLM)
# =========================================================================

def diagnosis_node(state: AgentState) -> dict:
    from langchain.agents import create_agent

    issues = state.get("issues", [])
    if not issues:
        return {"phase": "complete", "diagnoses": []}

    print(f"\n  [diagnosis] Investigating {len(issues)} issue(s)...")
    llm = make_llm()
    agent = create_agent(
        model=llm, tools=[investigate],
        system_prompt="""\
You are a Kubernetes Diagnosis Agent. You receive a list of unhealthy
resources and must determine the root cause AND extract concrete details.

The investigate tool runs read-only kubectl commands.

For each issue:
1. kubectl describe pod <name> -n <namespace>
2. kubectl logs <name> -n <namespace> --tail=100
3. If crashed: kubectl logs <name> -n <namespace> --previous --tail=100
4. kubectl get events -n <namespace>

CRITICAL: Extract these SPECIFIC details for each issue:
- Exact container name(s)
- Exact image name and tag currently configured
- Owning controller (Deployment, StatefulSet, DaemonSet, Job, or standalone)
- Owning controller's exact name
- For ImagePullBackOff: exact error from events (404, 401, etc.)
- For OOMKilled: current memory limit
- For CrashLoopBackOff: exit code and last log errors

Use these to get specifics:
- kubectl get pod <name> -n <ns> -o jsonpath={.metadata.ownerReferences}
- kubectl get pod <name> -n <ns> -o jsonpath={.spec.containers[*].name}
- kubectl get pod <name> -n <ns> -o jsonpath={.spec.containers[*].image}

Output format — respond with ONLY a JSON object:
{
  "diagnoses": [
    {
      "resource": "pod/api-server", "namespace": "production",
      "root_cause": "OOMKilled — container exceeds 256Mi memory limit",
      "evidence": ["Exit code 137", "restartCount: 17"],
      "severity": "high",
      "container_name": "api", "image": "myregistry.io/api:v2.3.1",
      "owner_kind": "Deployment", "owner_name": "api-server"
    }
  ]
}

Be precise. Extract ALL concrete values — no placeholders.""",
    )

    result = agent.invoke(
        {"messages": [{"role": "user",
                       "content": f"Investigate:\n{json.dumps(issues, indent=2)}"}]}
    )
    parsed = _extract_json(result["messages"][-1].content)
    diagnoses = parsed.get("diagnoses", []) if parsed else []
    print(f"  [diagnosis] Completed {len(diagnoses)} diagnosis(es)")
    return {"phase": "recommendation", "diagnoses": diagnoses,
            "agent_logs": state.get("agent_logs", []) + [
                f"Diagnosis: {len(diagnoses)} root cause(s) identified."]}


# =========================================================================
# Node 3: Recommendation Agent (LLM + read-only tools)
# =========================================================================

def recommendation_node(state: AgentState) -> dict:
    from langchain.agents import create_agent

    diagnoses = state.get("diagnoses", [])
    if not diagnoses:
        return {"phase": "complete", "recommendations": []}

    actionable = [d for d in diagnoses if d.get("severity") in ("high", "medium", "critical")]
    if not actionable:
        print(f"\n  [recommendation] No actionable issues")
        return {"phase": "complete", "recommendations": [],
                "agent_logs": state.get("agent_logs", []) + [
                    "Recommendation: no actionable issues."]}

    print(f"\n  [recommendation] Generating fixes for {len(actionable)} issue(s)...")
    llm = make_llm(temperature=0.1)
    agent = create_agent(
        model=llm, tools=[investigate],
        system_prompt="""\
You are a Kubernetes Recommendation Agent. Propose concrete kubectl commands
to fix each issue. You have read-only cluster access via the investigate tool.

CRITICAL RULES:
1. NEVER propose read-only commands (get, describe, logs, top) as fixes.
2. NEVER use placeholders like <name> or <container>. Every command must be
   copy-paste ready with real values from the cluster.
3. Use the investigate tool to look up any value you need BEFORE proposing.
4. Prefer least disruptive fixes (restart > delete, scale > replace).
5. For Succeeded Jobs or info-severity, return "no_fix_needed": true.
6. Allowed verbs ONLY: apply, scale, rollout restart, rollout undo,
   patch, set, create, annotate, label.
7. NEVER propose these (they are ALWAYS BLOCKED):
   - kubectl delete (ANY resource)
   - kubectl exec
   - kubectl cp
   - Commands with pipes (|) or subshells ($())
   If the only fix needs delete/exec, set "no_fix_needed": true and
   explain what the user should do manually.
8. Do NOT wrap flag values in single quotes. Write --type=json not --type='json'.

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
    )

    result = agent.invoke(
        {"messages": [{"role": "user",
                       "content": f"Propose fixes:\n{json.dumps(actionable, indent=2)}"}]}
    )
    parsed = _extract_json(result["messages"][-1].content)
    recs = parsed.get("recommendations", []) if parsed else []

    # Hard filter: strip anything non-executable
    executable = []
    for rec in recs:
        cmd = rec.get("fix_command", "")
        if rec.get("no_fix_needed") or not cmd.strip():
            continue
        cl = cmd.lower().strip()
        if any(cl.startswith(r) for r in ["kubectl get", "kubectl describe", "kubectl logs", "kubectl top"]):
            print(f"  [recommendation] FILTERED read-only: {cmd[:60]}")
            continue
        if any(b in cl for b in ["kubectl delete", "kubectl exec", "kubectl cp"]):
            print(f"  [recommendation] FILTERED blocked: {cmd[:60]}")
            continue
        if "|" in cmd or "$(" in cmd or "`" in cmd:
            print(f"  [recommendation] FILTERED pipe/subshell: {cmd[:60]}")
            continue
        if "<" in cmd and ">" in cmd:
            print(f"  [recommendation] FILTERED placeholder: {cmd[:60]}")
            continue
        executable.append(rec)

    filtered = len(recs) - len(executable)
    print(f"  [recommendation] {len(executable)} executable fix(es)"
          + (f" ({filtered} filtered)" if filtered else ""))
    return {"phase": "approval", "recommendations": executable,
            "agent_logs": state.get("agent_logs", []) + [
                f"Recommendation: {len(executable)} executable fix(es)."]}


# =========================================================================
# Node 4: Human Approval Gate (deterministic)
# =========================================================================

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
        if rec.get("alternative"):
            print(f"    Alt:     {rec['alternative']}")

    print(f"\n  Options:")
    print(f"    all  — approve all")
    print(f"    1,3  — approve by number")
    print(f"    none — skip to summary")

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
                "agent_logs": state.get("agent_logs", []) + [
                    f"Approval: {len(approved)} command(s) approved."]}
    else:
        print("\n  [approval] None approved.")
        return {"phase": "complete", "approved_commands": [],
                "agent_logs": state.get("agent_logs", []) + [
                    "Approval: user declined all fixes."]}


# =========================================================================
# Node 5: Executor + Verifier (deterministic Python, NO LLM)
# =========================================================================

def remediation_node(state: AgentState) -> dict:
    """Deterministic executor: loops over approved commands, runs each
    via subprocess (shell=False), then verifies the result.
    No LLM is used — this is a simple Python for-loop."""

    approved = state.get("approved_commands", [])
    if not approved:
        return {"phase": "complete", "actions_taken": []}

    print(f"\n  [executor] Running {len(approved)} approved command(s)...\n")

    actions = []
    for i, command in enumerate(approved, 1):
        print(f"    [{i}/{len(approved)}] {command}")

        # --- Execute ---
        result = execute_command(command)
        success = result.get("success", False)

        if success:
            print(f"      ✓ OK: {result.get('output', '')[:80]}")
        else:
            print(f"      ✗ FAIL: {result.get('error', 'Unknown')[:80]}")

        # --- Verify ---
        verification = {}
        if success:
            time.sleep(3)  # brief wait for k8s reconciliation
            verification = verify_fix(command)
            status = "✓" if verification.get("verified") else "⚠"
            print(f"      {status} Verify: {verification.get('note', '')}")

        actions.append({
            "command": command,
            "success": success,
            "output": result.get("output", result.get("error", "")),
            "verification": verification,
        })

    ok = sum(1 for a in actions if a["success"])
    verified = sum(1 for a in actions if a.get("verification", {}).get("verified"))
    print(f"\n  [executor] Done: {ok}/{len(actions)} succeeded, {verified}/{ok} verified")

    return {"phase": "complete", "actions_taken": actions,
            "agent_logs": state.get("agent_logs", []) + [
                f"Executor: {ok}/{len(actions)} succeeded, {verified}/{ok} verified."]}


# =========================================================================
# Node 6: Summary Agent (LLM)
# =========================================================================

def summary_node(state: AgentState) -> dict:
    llm = make_llm(temperature=0.1)
    context = json.dumps({
        "task": state.get("task", ""),
        "issues_found": state.get("issues", []),
        "diagnoses": state.get("diagnoses", []),
        "recommendations": state.get("recommendations", []),
        "actions_taken": state.get("actions_taken", []),
        "agent_logs": state.get("agent_logs", []),
    }, indent=2, default=str)

    response = llm.invoke([
        SystemMessage(content="""\
You are a Kubernetes Operations Summary Agent. Produce a clear summary.

Structure:
1. What was checked
2. Issues found
3. Root causes
4. Actions taken and results (including verification status)
5. Remaining items

Be concise. Use plain language."""),
        HumanMessage(content=f"Summarize:\n{context}"),
    ])
    return {"summary": response.content, "phase": "done"}


# =========================================================================
# Orchestrator
# =========================================================================

def _route_after_detection(state: AgentState) -> Literal["diagnosis", "summary"]:
    return "diagnosis" if state.get("issues") else "summary"


def _route_after_approval(state: AgentState) -> Literal["remediation", "summary"]:
    return "remediation" if state.get("approved_commands") else "summary"


def build_orchestrator() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("detection", detection_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("approval", approval_node)
    graph.add_node("remediation", remediation_node)
    graph.add_node("summary", summary_node)
    graph.set_entry_point("detection")
    graph.add_conditional_edges("detection", _route_after_detection,
                                {"diagnosis": "diagnosis", "summary": "summary"})
    graph.add_edge("diagnosis", "recommendation")
    graph.add_edge("recommendation", "approval")
    graph.add_conditional_edges("approval", _route_after_approval,
                                {"remediation": "remediation", "summary": "summary"})
    graph.add_edge("remediation", "summary")
    graph.add_edge("summary", END)
    return graph.compile()


# =========================================================================
# Entry point
# =========================================================================

def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not task:
        print("Multi-Agent Kubernetes Operations System")
        print("=" * 50)
        print("\nExample tasks:\n")
        print('  python multi_agent.py "Find all crashing pods and fix them"')
        print('  python multi_agent.py "Check cluster health and remediate issues"')
        print('  python multi_agent.py "Diagnose why pods in production are failing"')
        print()
        task = input("Enter your task: ").strip()
        if not task:
            sys.exit("No task entered.")

    print(f"\n{'=' * 60}")
    print(f"Task: {task}")
    print(f"{'=' * 60}")
    print(f"\nPipeline: Detection → Diagnosis → Recommendation → Approval → Executor → Verifier → Summary\n")

    orchestrator = build_orchestrator()
    initial_state: AgentState = {
        "task": task, "phase": "detection",
        "issues": [], "diagnoses": [], "recommendations": [],
        "approved_commands": [], "actions_taken": [],
        "summary": "", "agent_logs": [],
    }

    result = orchestrator.invoke(initial_state)

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