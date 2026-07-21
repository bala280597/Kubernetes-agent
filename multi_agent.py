"""
Multi-Agent Kubernetes Operations System — v3.

Architecture:
  Orchestrator (LangGraph StateGraph) with [9] LLM Supervisor routing:

    [9] Supervisor Node (Sonnet — dynamic routing):
      - Decides next agent based on current state
      - Can skip diagnosis for known patterns (via Mem0)
      - Can re-route on partial failures
      - Replaces static conditional edges from v1/v2

    LLM Agents (reasoning only):
      1. Detection Pipeline (parallel fan-out):
         1a. Namespace Lister — deterministic
         1b. Namespace Scanner — parallel per-NS (Haiku)
         1c. Detection Merger — collects results
      2. Diagnosis Agent   — investigates root cause (Sonnet)
      3. Recommendation Agent — proposes fix commands (Sonnet)
         3a. Reflexion Verifier — self-correction loop
      4. Summary Agent     — final report (Haiku)

    Deterministic nodes (Python, no LLM):
      5. Guardrail Filter
      6. Approval Gate (human-in-the-loop)
      7. Executor + Verifier
      8. [10] Eval Collector — captures data for offline evaluation

All improvements (v1 → v2 → v3):
  [1]  Parallel fan-out detection  — namespaces scanned concurrently via Send()
  [2]  Pydantic structured output  — typed models replace regex JSON extraction
  [3]  Reflexion loop              — recommendation self-check + critique retry
  [4]  Mem0 persistent memory      — cross-session incident learning
  [6]  PostgreSQL checkpointer     — durable, resumable workflows (prod)
  [7]  OpenTelemetry tracing       — vendor-neutral observability (OTel + LangSmith)
  [9]  Supervisor pattern          — LLM-driven dynamic routing
  [10] Evaluation framework        — golden-case evals + eval data collection

Setup:
  pip install -U langchain langchain-anthropic langgraph langsmith pydantic mem0ai
  # [6] For production PostgreSQL checkpointer:
  pip install -U "langgraph-checkpoint-postgres[async]" psycopg[binary]
  # [7] For OpenTelemetry (optional, works alongside LangSmith):
  pip install -U opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

  export ANTHROPIC_API_KEY="your-key"
  export LANGSMITH_TRACING=true
  export LANGSMITH_API_KEY="ls__your-key"
  export LANGSMITH_PROJECT="k8s-multi-agent-ops-v3"
  export MEM0_API_KEY="your-mem0-key"          # optional: enables incident memory
  export POSTGRES_URI="postgresql://..."       # optional: enables durable checkpoints
  export OTEL_EXPORTER_OTLP_ENDPOINT="..."     # optional: enables OTel export

Run:
  python multi_agent_v3.py
  python multi_agent_v3.py "Find all crashing pods and fix them"

Evaluate:
  python -m pytest tests/test_agent_evals.py -v
"""

import asyncio
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
from langgraph.constants import Send
from langgraph.graph import END, StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field

# =========================================================================
# [7] OpenTelemetry — vendor-neutral tracing (optional)
# =========================================================================
# Works alongside LangSmith. When OTEL_EXPORTER_OTLP_ENDPOINT is set,
# traces are exported to the configured collector (Langfuse, Jaeger, etc.).
# Falls back gracefully when OTel packages aren't installed.

_OTEL_READY = False

def _configure_otel() -> bool:
    """Initialize OpenTelemetry tracing if packages and endpoint are available."""
    global _OTEL_READY
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        resource = Resource.create({
            "service.name": "k8s-multi-agent-ops",
            "service.version": "3.0",
            "deployment.environment": os.getenv("DEPLOY_ENV", "development"),
        })
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _OTEL_READY = True
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  [otel] Init failed: {e}", file=sys.stderr)
        return False


def _get_tracer():
    """Get an OTel tracer if available, else a no-op."""
    if not _OTEL_READY:
        return None
    try:
        from opentelemetry import trace
        return trace.get_tracer("k8s-multi-agent-ops", "3.0")
    except Exception:
        return None


def _otel_span(name: str, attributes: dict | None = None):
    """Context manager for OTel spans. No-op if OTel is not configured."""
    tracer = _get_tracer()
    if tracer is None:
        from contextlib import nullcontext
        return nullcontext()
    span = tracer.start_span(name)
    if attributes:
        for k, v in attributes.items():
            span.set_attribute(k, str(v) if not isinstance(v, (str, int, float, bool)) else v)
    return trace.use_span(span, end_on_exit=True)


# =========================================================================
# [4] Mem0 — optional persistent incident memory
# =========================================================================

try:
    from mem0 import MemoryClient as _Mem0Client
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False

_mem0_client = None


def _init_mem0() -> bool:
    """Initialize Mem0 client if API key is set. Returns True if ready."""
    global _mem0_client
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key or not _MEM0_AVAILABLE:
        return False
    try:
        _mem0_client = _Mem0Client(api_key=api_key)
        return True
    except Exception as e:
        _log("mem0", f"Init failed: {e}")
        return False


def _mem0_search(query: str, limit: int = 5) -> list[dict]:
    """Search Mem0 for past incident context. Returns empty list if unavailable."""
    if _mem0_client is None:
        return []
    try:
        results = _mem0_client.search(
            query, user_id="k8s-ops-agent", limit=limit
        )
        return results.get("results", results) if isinstance(results, dict) else results
    except Exception as e:
        _log("mem0", f"Search failed: {e}")
        return []


def _mem0_store(text: str, metadata: dict | None = None) -> None:
    """Store an incident resolution in Mem0 for future reference."""
    if _mem0_client is None:
        return
    try:
        _mem0_client.add(
            text,
            user_id="k8s-ops-agent",
            metadata=metadata or {},
        )
    except Exception as e:
        _log("mem0", f"Store failed: {e}")


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
# LangSmith configuration (kept — works alongside OTel)
# =========================================================================

def _configure_langsmith():
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "k8s-multi-agent-ops-v3")


WORKDIR = Path.cwd()
IS_WINDOWS = platform.system() == "Windows"

# =========================================================================
# Model registry — different models for different task complexity
# =========================================================================
# [9] Added: "supervisor" uses Sonnet for intelligent routing decisions

MODEL_REGISTRY: dict[str, dict] = {
    "detection":      {"model": "claude-haiku-4-5-20251001", "temperature": 0.2, "max_tokens": 4096},
    "diagnosis":      {"model": "claude-sonnet-4-6",         "temperature": 0.2, "max_tokens": 8192},
    "recommendation": {"model": "claude-sonnet-4-6",         "temperature": 0.1, "max_tokens": 8192},
    "summary":        {"model": "claude-haiku-4-5-20251001", "temperature": 0.1, "max_tokens": 4096},
    "supervisor":     {"model": "claude-sonnet-4-6",         "temperature": 0.0, "max_tokens": 1024},
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
# Output summarizers
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
    _invalidate_cache()
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
# [2] Pydantic structured output models
# =========================================================================

class IssueDetail(BaseModel):
    resource_type: str = Field(description="e.g. pod, node, deployment")
    name: str = Field(description="Resource name")
    namespace: str = Field(default="", description="Namespace (empty for cluster-scoped)")
    status: str = Field(description="Current status, e.g. CrashLoopBackOff, NotReady")
    details: str = Field(default="", description="Extra context, e.g. restarts: 17")

class DetectionResult(BaseModel):
    issues_found: bool = Field(description="True if any unhealthy resources detected")
    issues: list[IssueDetail] = Field(default_factory=list)
    healthy_summary: str = Field(default="", description="Summary of healthy resources")

class DiagnosisDetail(BaseModel):
    resource: str = Field(description="e.g. pod/api-server")
    namespace: str = Field(default="default")
    root_cause: str = Field(description="Root cause, e.g. OOMKilled — exceeds 256Mi")
    evidence: list[str] = Field(default_factory=list)
    severity: Literal["critical", "high", "medium", "low"] = Field(default="medium")
    container_name: str = Field(default="")
    image: str = Field(default="")
    owner_kind: str = Field(default="")
    owner_name: str = Field(default="")

class DiagnosisResult(BaseModel):
    diagnoses: list[DiagnosisDetail] = Field(default_factory=list)

class RecommendationDetail(BaseModel):
    issue: str = Field(description="What this fixes")
    fix_command: str = Field(description="Exact kubectl command, no placeholders")
    explanation: str = Field(description="Why this fix works")
    risk: Literal["low", "medium", "high"] = Field(default="medium")
    alternative: str = Field(default="")
    no_fix_needed: bool = Field(default=False)

class RecommendationResult(BaseModel):
    recommendations: list[RecommendationDetail] = Field(default_factory=list)


# =========================================================================
# [9] Supervisor routing model — Pydantic structured output
# =========================================================================

# Valid nodes the supervisor can route to
SUPERVISOR_TARGETS = Literal[
    "diagnosis",
    "recommendation",
    "guardrail",
    "approval",
    "summary",
]


class SupervisorDecision(BaseModel):
    """Structured output from the Supervisor LLM for routing decisions."""
    next_node: SUPERVISOR_TARGETS = Field(
        description="Which agent/node to invoke next"
    )
    reasoning: str = Field(
        description="One-sentence justification for the routing decision"
    )
    skip_reason: str = Field(
        default="",
        description="If skipping a node, explain why (e.g. 'known pattern from memory')"
    )


# =========================================================================
# Agent state
# =========================================================================

class AgentState(TypedDict):
    task: str
    phase: str
    # [1] Parallel fan-out fields
    namespaces: list[str]
    ns_issues: Annotated[list[dict], operator.add]
    # Core pipeline fields
    issues: list[dict]
    diagnoses: list[dict]
    recommendations: list[dict]
    approved_commands: list[str]
    actions_taken: list[dict]
    summary: str
    agent_logs: Annotated[list[str], operator.add]
    # [3] Reflexion loop fields
    reflexion_count: int
    reflexion_critique: str
    # [4] Mem0 context
    memory_context: str
    # [9] Supervisor state
    supervisor_next: str
    supervisor_reasoning: str
    # [10] Eval data collection
    eval_trace: Annotated[list[dict], operator.add]


class NamespaceScanState(TypedDict):
    namespace: str
    task: str
    ns_issues: list[dict]
    agent_logs: Annotated[list[str], operator.add]


# =========================================================================
# [6] Checkpointer factory — Postgres for prod, InMemory for dev
# =========================================================================

def _create_checkpointer():
    """Create the appropriate checkpointer based on environment.

    If POSTGRES_URI is set, returns an AsyncPostgresSaver for durable,
    resumable workflows. Otherwise falls back to InMemorySaver for local dev.
    """
    pg_uri = os.getenv("POSTGRES_URI")
    if pg_uri:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            _log("system", f"Using PostgreSQL checkpointer: {pg_uri[:30]}...")
            saver = PostgresSaver.from_conn_string(pg_uri)
            saver.setup()  # creates tables if needed
            return saver, "postgres"
        except ImportError:
            _log("system", "PostgresSaver not installed — pip install langgraph-checkpoint-postgres")
        except Exception as e:
            _log("system", f"PostgreSQL checkpointer failed: {e}")

    _log("system", "Using InMemorySaver (dev mode)")
    return InMemorySaver(), "memory"


async def _create_async_checkpointer():
    """Async variant — uses AsyncPostgresSaver for non-blocking I/O."""
    pg_uri = os.getenv("POSTGRES_URI")
    if pg_uri:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            _log("system", f"Using AsyncPostgresSaver: {pg_uri[:30]}...")
            saver = AsyncPostgresSaver.from_conn_string(pg_uri)
            await saver.setup()
            return saver, "postgres-async"
        except ImportError:
            _log("system", "AsyncPostgresSaver not installed")
        except Exception as e:
            _log("system", f"Async PostgreSQL failed: {e}")

    return InMemorySaver(), "memory"


# Create checkpointer at module level (sync for simplicity)
_shared_checkpointer, _checkpointer_type = _create_checkpointer()


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
# [10] Eval data collector
# =========================================================================
# Each node appends structured eval data to state["eval_trace"].
# After a run, this can be exported for offline evaluation.

EVAL_DATA_DIR = Path(os.getenv("EVAL_DATA_DIR", "eval_data"))


def _record_eval(node_name: str, input_data: Any, output_data: Any,
                 metadata: dict | None = None) -> dict:
    """Create a structured eval record for offline analysis."""
    return {
        "node": node_name,
        "timestamp": time.time(),
        "input_summary": _truncate_for_eval(input_data),
        "output_summary": _truncate_for_eval(output_data),
        "metadata": metadata or {},
    }


def _truncate_for_eval(data: Any, max_len: int = 2000) -> Any:
    """Truncate data for eval storage without losing structure."""
    if isinstance(data, str):
        return data[:max_len] if len(data) > max_len else data
    if isinstance(data, list):
        return data[:10]  # keep first 10 items
    if isinstance(data, dict):
        return {k: _truncate_for_eval(v, 500) for k, v in list(data.items())[:20]}
    return data


def _save_eval_trace(task: str, eval_trace: list[dict], result: dict) -> Path | None:
    """Save eval trace to disk for offline analysis."""
    if not eval_trace:
        return None
    try:
        EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_task = re.sub(r"[^\w\s-]", "", task)[:40].strip().replace(" ", "_")
        filepath = EVAL_DATA_DIR / f"eval_{ts}_{safe_task}.json"
        record = {
            "task": task,
            "timestamp": time.time(),
            "trace": eval_trace,
            "final_summary": result.get("summary", ""),
            "total_issues": len(result.get("issues", [])),
            "total_diagnoses": len(result.get("diagnoses", [])),
            "total_recommendations": len(result.get("recommendations", [])),
            "actions_taken": len(result.get("actions_taken", [])),
            "reflexion_retries": result.get("reflexion_count", 0),
            "agent_logs": result.get("agent_logs", []),
        }
        filepath.write_text(json.dumps(record, indent=2, default=str))
        _log("eval", f"Saved eval trace: {filepath}")
        return filepath
    except Exception as e:
        _log("eval", f"Failed to save eval trace: {e}")
        return None


# =========================================================================
# [1] Node 1a: Namespace Lister (deterministic)
# =========================================================================

@traceable(name="list_namespaces", run_type="tool", tags=["detection"])
def list_namespaces_node(state: AgentState) -> dict:
    _log("list_ns", "Listing cluster namespaces...")
    rc, stdout, stderr = _run_args(
        ["kubectl", "get", "namespaces", "-o", "json"]
    )
    if rc != 0:
        _log("list_ns", f"Failed: {stderr[:200]}")
        return {
            "namespaces": ["default"],
            "agent_logs": ["Namespace list failed, falling back to 'default'."],
            "eval_trace": [_record_eval("list_namespaces", "kubectl get ns", {"fallback": True})],
        }
    try:
        data = json.loads(stdout)
        ns_list = [
            item["metadata"]["name"]
            for item in data.get("items", [])
            if item.get("status", {}).get("phase") == "Active"
        ]
    except (json.JSONDecodeError, KeyError):
        ns_list = ["default"]

    _log("list_ns", f"Found {len(ns_list)} active namespace(s)")
    return {
        "namespaces": ns_list,
        "agent_logs": [f"Namespace lister: {len(ns_list)} active namespaces."],
        "eval_trace": [_record_eval("list_namespaces", "kubectl get ns",
                                    {"count": len(ns_list), "namespaces": ns_list})],
    }


# =========================================================================
# [1] Node 1b: Per-Namespace Scanner (parallel via Send)
# =========================================================================

@traceable(name="scan_namespace", run_type="chain",
           tags=["detection", "parallel"])
def scan_namespace_node(state: NamespaceScanState) -> dict:
    ns = state["namespace"]
    task = state.get("task", "")
    _log("ns_scan", f"Scanning namespace: {ns}")

    llm = make_llm("detection")
    structured_llm = llm.with_structured_output(DetectionResult)

    pod_data = scan_cluster.invoke(f"kubectl get pods -n {ns}")
    deploy_data = scan_cluster.invoke(f"kubectl get deployments -n {ns}")

    try:
        result: DetectionResult = structured_llm.invoke([
            SystemMessage(content="""\
You are a Kubernetes Detection Agent scanning ONE namespace.
Analyze the pod and deployment data below. Identify unhealthy resources.
Only report genuinely unhealthy items (CrashLoopBackOff, ImagePullBackOff,
Pending, OOMKilled, not-ready, missing replicas, etc.)."""),
            HumanMessage(content=f"Task: {task}\nNamespace: {ns}\n\nPod data:\n{pod_data}\n\nDeployment data:\n{deploy_data}"),
        ])
    except Exception as e:
        _log("ns_scan", f"LLM parse failed for {ns}: {e}")
        return {"ns_issues": [], "agent_logs": [f"NS scan {ns}: parse error — {e}"]}

    issues_dicts = [issue.model_dump() for issue in result.issues]
    for issue in issues_dicts:
        if not issue.get("namespace"):
            issue["namespace"] = ns

    count = len(issues_dicts)
    _log("ns_scan", f"  {ns}: {count} issue(s) found")
    return {"ns_issues": issues_dicts, "agent_logs": [f"NS scan {ns}: {count} issue(s)."]}


# =========================================================================
# [1] Fan-out dispatcher + Detection Merger
# =========================================================================

def _fan_out_namespaces(state: AgentState) -> list[Send]:
    namespaces = state.get("namespaces", ["default"])
    task = state.get("task", "")
    _log("fan_out", f"Dispatching {len(namespaces)} parallel namespace scans")
    return [
        Send("scan_namespace", {"namespace": ns, "task": task,
                                "ns_issues": [], "agent_logs": []})
        for ns in namespaces
    ]


@traceable(name="detection_merger", run_type="tool", tags=["detection"])
def detection_merger_node(state: AgentState) -> dict:
    ns_issues = state.get("ns_issues", [])
    _log("merger", "Checking node health...")
    node_data = scan_cluster.invoke("kubectl get nodes")
    try:
        node_info = json.loads(node_data)
        for node in node_info.get("nodes", []):
            if node.get("ready") != "True":
                ns_issues.append({
                    "resource_type": "node", "name": node.get("name", "unknown"),
                    "namespace": "",
                    "status": f"NotReady (ready={node.get('ready')})",
                    "details": f"kubelet {node.get('kubelet', 'unknown')}",
                })
    except (json.JSONDecodeError, TypeError):
        pass

    total = len(ns_issues)
    _log("merger", f"Merged results: {total} total issue(s)")
    # [9] Route to supervisor instead of hardcoded conditional
    return {
        "phase": "supervisor",
        "issues": ns_issues,
        "agent_logs": [f"Detection complete: {total} issue(s) across all namespaces."],
        "eval_trace": [_record_eval("detection_merger", {"ns_issues_count": total},
                                    {"total": total, "issues": ns_issues[:5]})],
    }


# =========================================================================
# [9] Supervisor Node — LLM-driven dynamic routing
# =========================================================================

SUPERVISOR_SYSTEM_PROMPT = """\
You are the Kubernetes Operations Supervisor. You decide which agent to invoke next.

Available agents:
  - diagnosis    : investigate root causes of detected issues
  - recommendation : propose fix commands for diagnosed issues
  - guardrail    : validate recommendations before approval
  - approval     : present fixes to human for approval
  - summary      : generate final report (use when pipeline is complete or no issues)

Current pipeline state is provided. Make your routing decision based on:
1. What phase the pipeline is in and what data is available
2. Whether Mem0 memory provides enough context to skip diagnosis for known patterns
3. Whether there are issues that still need processing

Rules:
- If no issues were found → route to "summary"
- If issues exist but no diagnoses yet → route to "diagnosis"
- If diagnoses exist but no recommendations → route to "recommendation"
- If recommendations passed reflexion check → route to "guardrail"
- If recommendations passed guardrail → route to "approval"
- If all actions are complete → route to "summary"
- You MAY skip "diagnosis" and go straight to "recommendation" if memory_context
  contains clear past resolutions for the exact same issue pattern (same pod name,
  same root cause). Explain this in skip_reason.
- You MAY route to "summary" early if all issues are low-severity and informational."""


@traceable(name="supervisor", run_type="chain",
           tags=["supervisor", "routing"], metadata={"node": "supervisor"})
def supervisor_node(state: AgentState) -> dict:
    """LLM-driven routing: decides which node to invoke next."""
    _log("supervisor", "Evaluating pipeline state for routing...")

    llm = make_llm("supervisor")
    structured_llm = llm.with_structured_output(SupervisorDecision)

    # Build a compact state summary for the supervisor
    state_summary = {
        "phase": state.get("phase", "unknown"),
        "issues_count": len(state.get("issues", [])),
        "diagnoses_count": len(state.get("diagnoses", [])),
        "recommendations_count": len(state.get("recommendations", [])),
        "approved_count": len(state.get("approved_commands", [])),
        "actions_taken_count": len(state.get("actions_taken", [])),
        "reflexion_count": state.get("reflexion_count", 0),
        "has_memory_context": bool(state.get("memory_context", "")),
        "recent_logs": state.get("agent_logs", [])[-5:],
    }

    # Include issue summaries if present (truncated)
    if state.get("issues"):
        state_summary["issue_summaries"] = [
            f"{i.get('name', '?')}: {i.get('status', '?')}"
            for i in state["issues"][:5]
        ]

    # Include memory context snippet if available
    if state.get("memory_context"):
        state_summary["memory_context_snippet"] = state["memory_context"][:500]

    try:
        decision: SupervisorDecision = structured_llm.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=f"Current state:\n{json.dumps(state_summary, indent=2)}"),
        ])
        next_node = decision.next_node
        reasoning = decision.reasoning
        skip = decision.skip_reason
    except Exception as e:
        _log("supervisor", f"LLM routing failed: {e} — using fallback logic")
        # Deterministic fallback mirrors v2 static routing
        next_node, reasoning, skip = _supervisor_fallback(state)

    _log("supervisor", f"→ {next_node} ({reasoning})")
    if skip:
        _log("supervisor", f"  Skip reason: {skip}")

    return {
        "supervisor_next": next_node,
        "supervisor_reasoning": reasoning,
        "agent_logs": [f"Supervisor: → {next_node} ({reasoning})" +
                       (f" [skip: {skip}]" if skip else "")],
        "eval_trace": [_record_eval("supervisor", state_summary,
                                    {"next": next_node, "reasoning": reasoning,
                                     "skip": skip})],
    }


def _supervisor_fallback(state: AgentState) -> tuple[str, str, str]:
    """Deterministic fallback routing when supervisor LLM fails."""
    issues = state.get("issues", [])
    diagnoses = state.get("diagnoses", [])
    recs = state.get("recommendations", [])
    approved = state.get("approved_commands", [])
    actions = state.get("actions_taken", [])

    if not issues:
        return "summary", "No issues found", ""
    if not diagnoses:
        return "diagnosis", "Issues need root cause analysis", ""
    if not recs:
        return "recommendation", "Diagnoses need fix proposals", ""
    if state.get("phase") == "guardrail" or state.get("phase") == "reflexion_passed":
        return "guardrail", "Recommendations ready for validation", ""
    if state.get("phase") == "approval_ready":
        return "approval", "Guardrail passed, awaiting human approval", ""
    if actions:
        return "summary", "Actions complete, generate report", ""
    return "summary", "Fallback — generating report", ""


def _route_supervisor(state: AgentState) -> str:
    """Route based on supervisor decision."""
    return state.get("supervisor_next", "summary")


# =========================================================================
# Node 2: Diagnosis Agent (Sonnet)
# =========================================================================

@traceable(name="diagnosis_agent", run_type="chain",
           tags=["diagnosis"], metadata={"node": "diagnosis"})
def diagnosis_node(state: AgentState) -> dict:
    issues = state.get("issues", [])
    if not issues: return {"phase": "supervisor", "diagnoses": []}
    _log("diagnosis", f"Investigating {len(issues)} issue(s)...")

    memory_context = ""
    if _mem0_client:
        _log("diagnosis", "Searching Mem0 for past incidents...")
        search_terms = " ".join(
            f"{i.get('name', '')} {i.get('status', '')}" for i in issues[:3]
        )
        memories = _mem0_search(search_terms, limit=5)
        if memories:
            memory_lines = []
            for m in memories:
                text = m.get("memory", m.get("text", "")) if isinstance(m, dict) else str(m)
                if text:
                    memory_lines.append(f"  - {text}")
            if memory_lines:
                memory_context = "Past incident resolutions:\n" + "\n".join(memory_lines)
                _log("diagnosis", f"Found {len(memory_lines)} relevant past incident(s)")

    llm = make_llm("diagnosis")
    agent = create_agent(
        model=llm, tools=[investigate],
        system_prompt=f"""\
You are a Kubernetes Diagnosis Agent. You receive a list of unhealthy
resources and must determine the root cause AND extract concrete details.

For each issue:
1. kubectl describe pod <n> -n <namespace>
2. kubectl logs <n> -n <namespace> --tail=100
3. If crashed: kubectl logs <n> -n <namespace> --previous --tail=100
4. kubectl get events -n <namespace>

CRITICAL: Extract SPECIFIC details:
- Exact container name(s), image name+tag
- Owning controller kind and name
- For ImagePullBackOff: exact error from events
- For OOMKilled: current memory limit
- For CrashLoopBackOff: exit code and last log errors

{memory_context}

Output format — respond with ONLY a JSON object:
{{
  "diagnoses": [
    {{
      "resource": "pod/api-server", "namespace": "production",
      "root_cause": "OOMKilled — exceeds 256Mi",
      "evidence": ["Exit code 137", "restartCount: 17"],
      "severity": "high",
      "container_name": "api", "image": "myregistry.io/api:v2.3.1",
      "owner_kind": "Deployment", "owner_name": "api-server"
    }}
  ]
}}
Be precise — no placeholders.""",
        checkpointer=_shared_checkpointer,
        name="diagnosis_agent",
    )
    result = _invoke_with_retry(
        agent,
        {"messages": [{"role": "user", "content": f"Investigate:\n{json.dumps(issues, indent=2)}"}]},
        config={"configurable": {"thread_id": f"diagnosis-{id(state)}"}},
    )

    raw_response = result["messages"][-1].content
    try:
        parsed = DiagnosisResult.model_validate_json(raw_response)
        diagnoses = [d.model_dump() for d in parsed.diagnoses]
    except Exception:
        try:
            text = raw_response
            fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if fence:
                text = fence.group(1).strip()
            raw_dict = json.loads(text)
            parsed = DiagnosisResult.model_validate(raw_dict)
            diagnoses = [d.model_dump() for d in parsed.diagnoses]
        except Exception as e:
            _log("diagnosis", f"Structured parse failed: {e}")
            diagnoses = []

    _log("diagnosis", f"Completed {len(diagnoses)} diagnosis(es)")
    return {
        "phase": "supervisor",
        "diagnoses": diagnoses,
        "memory_context": memory_context,
        "agent_logs": [f"Diagnosis: {len(diagnoses)} root cause(s) identified."],
        "eval_trace": [_record_eval("diagnosis",
                                    {"issues_count": len(issues)},
                                    {"diagnoses_count": len(diagnoses),
                                     "severities": [d.get("severity") for d in diagnoses]})],
    }


# =========================================================================
# Node 3: Recommendation Agent (Sonnet)
# =========================================================================

@traceable(name="recommendation_agent", run_type="chain",
           tags=["recommendation"], metadata={"node": "recommendation"})
def recommendation_node(state: AgentState) -> dict:
    diagnoses = state.get("diagnoses", [])
    if not diagnoses: return {"phase": "supervisor", "recommendations": []}
    actionable = [d for d in diagnoses if d.get("severity") in ("high", "medium", "critical")]
    if not actionable:
        _log("recommendation", "No actionable issues")
        return {"phase": "supervisor", "recommendations": [],
                "agent_logs": ["Recommendation: no actionable issues."]}

    reflexion_count = state.get("reflexion_count", 0)
    critique = state.get("reflexion_critique", "")
    if reflexion_count > 0:
        _log("recommendation", f"Reflexion retry #{reflexion_count} with critique")

    _log("recommendation", f"Generating fixes for {len(actionable)} issue(s)...")

    critique_section = ""
    if critique:
        critique_section = f"""

IMPORTANT — YOUR PREVIOUS ATTEMPT WAS REJECTED:
{critique}

Fix ALL issues listed above before responding. Do NOT repeat the same mistakes."""

    llm = make_llm("recommendation")
    agent = create_agent(
        model=llm, tools=[investigate],
        system_prompt=f"""\
You are a Kubernetes Recommendation Agent. Propose concrete kubectl commands.

RULES:
1. NEVER propose read-only commands as fixes.
2. NEVER use placeholders like <value>, {{value}}, YOUR_IMAGE, etc.
3. Use the investigate tool to look up values BEFORE proposing.
4. Prefer least disruptive fixes.
5. Allowed verbs: apply, scale, rollout restart/undo, patch, set, create, annotate, label.
6. NEVER propose: kubectl delete, exec, cp, pipes, subshells.
7. Do NOT wrap flag values in single quotes.
{critique_section}

Output format — respond with ONLY a JSON object:
{{
  "recommendations": [
    {{
      "issue": "pod/api-server OOMKilled in production",
      "fix_command": "kubectl set resources deployment/api-server -n production --limits=memory=512Mi",
      "explanation": "Double memory limit from 256Mi to 512Mi",
      "risk": "low",
      "alternative": "kubectl rollout restart deployment/api-server -n production"
    }}
  ]
}}""",
        checkpointer=_shared_checkpointer,
        name="recommendation_agent",
    )
    result = _invoke_with_retry(
        agent,
        {"messages": [{"role": "user", "content": f"Propose fixes:\n{json.dumps(actionable, indent=2)}"}]},
        config={"configurable": {"thread_id": f"recommendation-{id(state)}-r{reflexion_count}"}},
    )

    raw_response = result["messages"][-1].content
    try:
        parsed = RecommendationResult.model_validate_json(raw_response)
        recs = [r.model_dump() for r in parsed.recommendations]
    except Exception:
        try:
            text = raw_response
            fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if fence:
                text = fence.group(1).strip()
            raw_dict = json.loads(text)
            parsed = RecommendationResult.model_validate(raw_dict)
            recs = [r.model_dump() for r in parsed.recommendations]
        except Exception as e:
            _log("recommendation", f"Structured parse failed: {e}")
            recs = []

    _log("recommendation", f"Generated {len(recs)} recommendation(s)")
    return {
        "phase": "reflexion_check",
        "recommendations": recs,
        "agent_logs": [f"Recommendation: {len(recs)} proposed fix(es) (attempt {reflexion_count + 1})."],
        "eval_trace": [_record_eval("recommendation",
                                    {"actionable": len(actionable), "attempt": reflexion_count + 1},
                                    {"recs_count": len(recs),
                                     "commands": [r.get("fix_command", "")[:80] for r in recs]})],
    }


# =========================================================================
# [3] Reflexion Verifier
# =========================================================================

MAX_REFLEXION_RETRIES = 2


@traceable(name="reflexion_verifier", run_type="chain",
           tags=["reflexion", "self-check"])
def reflexion_check_node(state: AgentState) -> dict:
    recs = state.get("recommendations", [])
    reflexion_count = state.get("reflexion_count", 0)

    if not recs:
        _log("reflexion", "No recommendations to verify")
        return {"phase": "supervisor", "reflexion_count": reflexion_count}

    issues: list[str] = []
    for i, rec in enumerate(recs):
        cmd = rec.get("fix_command", "")
        issue_ref = rec.get("issue", f"rec#{i+1}")

        placeholder_patterns = [
            (r"<[^>]+>", "angle-bracket placeholder"),
            (r"\{[^}]+\}", "curly-brace placeholder"),
            (r"YOUR_\w+", "YOUR_ placeholder"),
            (r"CHANGE_ME", "CHANGE_ME placeholder"),
            (r"xxx|XXX", "xxx placeholder"),
        ]
        for pattern, desc in placeholder_patterns:
            if re.search(pattern, cmd):
                issues.append(f"[{issue_ref}] {desc} in command: {cmd[:80]}")

        if not cmd.strip() and not rec.get("no_fix_needed"):
            issues.append(f"[{issue_ref}] empty fix_command")

        cmd_lower = cmd.lower().strip()
        if any(cmd_lower.startswith(b) for b in ["kubectl delete", "kubectl exec", "kubectl cp"]):
            issues.append(f"[{issue_ref}] blocked operation: {cmd[:60]}")

        if "|" in cmd or "$(" in cmd or "`" in cmd:
            issues.append(f"[{issue_ref}] pipe/subshell: {cmd[:60]}")

        readonly_verbs = ["kubectl get ", "kubectl describe ", "kubectl logs ", "kubectl top "]
        if any(cmd_lower.startswith(rv) for rv in readonly_verbs):
            issues.append(f"[{issue_ref}] read-only command as fix: {cmd[:60]}")

        if cmd.strip():
            try:
                binary = Path(shlex.split(cmd)[0]).name
                if binary not in ALLOWED_BINARIES:
                    issues.append(f"[{issue_ref}] disallowed binary '{binary}'")
            except (ValueError, IndexError):
                issues.append(f"[{issue_ref}] unparseable command: {cmd[:60]}")

    if not issues:
        _log("reflexion", f"All {len(recs)} recommendation(s) passed self-check")
        return {
            "phase": "supervisor",
            "reflexion_count": reflexion_count,
            "reflexion_critique": "",
            "agent_logs": [f"Reflexion: all {len(recs)} recommendations valid (attempt {reflexion_count + 1})."],
            "eval_trace": [_record_eval("reflexion", {"recs": len(recs)}, {"passed": True})],
        }

    _log("reflexion", f"Found {len(issues)} issue(s) in recommendations")

    if reflexion_count >= MAX_REFLEXION_RETRIES:
        _log("reflexion", f"Max retries ({MAX_REFLEXION_RETRIES}) reached — passing through")
        return {
            "phase": "supervisor",
            "reflexion_count": reflexion_count,
            "reflexion_critique": "",
            "agent_logs": [f"Reflexion: {len(issues)} issue(s) after {reflexion_count + 1} attempts — passing."],
            "eval_trace": [_record_eval("reflexion", {"recs": len(recs)},
                                        {"passed": False, "max_retries_hit": True, "issues": issues})],
        }

    critique = "The following problems were found in your recommendations:\n"
    critique += "\n".join(f"  - {iss}" for iss in issues)
    critique += "\n\nYou MUST fix every issue above. Use the investigate tool to look up real values."

    _log("reflexion", f"Routing back to recommendation (retry {reflexion_count + 1})")
    return {
        "phase": "recommendation",
        "reflexion_count": reflexion_count + 1,
        "reflexion_critique": critique,
        "recommendations": [],
        "agent_logs": [f"Reflexion: {len(issues)} issue(s) found — retrying (attempt {reflexion_count + 2})."],
        "eval_trace": [_record_eval("reflexion", {"recs": len(recs)},
                                    {"passed": False, "retry": reflexion_count + 1, "issues": issues})],
    }


# =========================================================================
# Guardrail Filter (deterministic)
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
    return {"phase": "supervisor", "recommendations": executable,
            "agent_logs": [f"Guardrail: {len(executable)} passed, {filtered} filtered."]}


# =========================================================================
# Approval Gate (human-in-the-loop)
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
# Executor + Verifier
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

    # [4] Store successful resolutions in Mem0
    for action in actions:
        if action["success"] and action.get("verification", {}).get("verified"):
            cmd = action["command"]
            matching_diag = None
            for diag in state.get("diagnoses", []):
                if diag.get("resource", "") in cmd or diag.get("owner_name", "") in cmd:
                    matching_diag = diag
                    break
            if matching_diag:
                resolution_text = (
                    f"Issue: {matching_diag.get('resource', 'unknown')} in "
                    f"{matching_diag.get('namespace', 'unknown')} — "
                    f"Root cause: {matching_diag.get('root_cause', 'unknown')} — "
                    f"Fix: {cmd} — Verified: True"
                )
                _mem0_store(resolution_text, metadata={
                    "namespace": matching_diag.get("namespace", ""),
                    "resource": matching_diag.get("resource", ""),
                    "root_cause": matching_diag.get("root_cause", ""),
                    "fix_command": cmd,
                    "type": "incident_resolution",
                })
                _log("mem0", f"Stored resolution: {matching_diag.get('resource', '')}")

    return {
        "phase": "complete", "actions_taken": actions,
        "agent_logs": [f"Executor: {ok}/{len(actions)} succeeded, {verified}/{ok} verified."],
        "eval_trace": [_record_eval("executor",
                                    {"approved": len(approved)},
                                    {"succeeded": ok, "verified": verified})],
    }


# =========================================================================
# Summary Agent (Haiku)
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
        "memory_context": state.get("memory_context", ""),
        "reflexion_attempts": state.get("reflexion_count", 0),
        "supervisor_decisions": state.get("supervisor_reasoning", ""),
    }, indent=2, default=str)
    response = llm.invoke([
        SystemMessage(content="""\
You are a Kubernetes Operations Summary Agent. Produce a clear, concise summary:
what was checked, issues found, root causes, actions taken + verification,
remaining items. Note if past incident memory or supervisor routing was used."""),
        HumanMessage(content=f"Summarize:\n{context}"),
    ])
    return {"summary": response.content, "phase": "done"}


# =========================================================================
# [9] Orchestrator — Supervisor-driven dynamic routing
# =========================================================================
# The supervisor node sits at the center. After detection and after each
# agent completes, control returns to the supervisor for the next routing
# decision. This replaces v2's static conditional edges.

def _route_after_reflexion(state):
    """Reflexion routes back to recommendation on failure, else to supervisor."""
    phase = state.get("phase", "supervisor")
    if phase == "recommendation":
        return "recommendation"
    return "supervisor"

def _route_after_approval(state):
    return "remediation" if state.get("approved_commands") else "summary"


def build_orchestrator() -> Any:
    graph = StateGraph(AgentState)

    # [1] Parallel detection pipeline
    graph.add_node("list_namespaces", list_namespaces_node)
    graph.add_node("scan_namespace", scan_namespace_node)
    graph.add_node("detection_merger", detection_merger_node)

    # [9] Supervisor hub
    graph.add_node("supervisor", supervisor_node)

    # Specialist agents
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("reflexion_check", reflexion_check_node)
    graph.add_node("guardrail", guardrail_filter_node)
    graph.add_node("approval", approval_node)
    graph.add_node("remediation", remediation_node)
    graph.add_node("summary", summary_node)

    # --- Edges ---

    # Detection pipeline (unchanged — parallel fan-out)
    graph.set_entry_point("list_namespaces")
    graph.add_conditional_edges("list_namespaces", _fan_out_namespaces)
    graph.add_edge("scan_namespace", "detection_merger")

    # Detection merger → supervisor (first routing decision)
    graph.add_edge("detection_merger", "supervisor")

    # [9] Supervisor routes to the appropriate next node
    graph.add_conditional_edges("supervisor", _route_supervisor, {
        "diagnosis": "diagnosis",
        "recommendation": "recommendation",
        "guardrail": "guardrail",
        "approval": "approval",
        "summary": "summary",
    })

    # After each specialist, return to supervisor for next decision
    graph.add_edge("diagnosis", "supervisor")
    graph.add_edge("recommendation", "reflexion_check")

    # Reflexion: retry recommendation or return to supervisor
    graph.add_conditional_edges("reflexion_check", _route_after_reflexion,
                                {"recommendation": "recommendation",
                                 "supervisor": "supervisor"})

    graph.add_edge("guardrail", "supervisor")

    # Approval and remediation are terminal — go straight to summary
    graph.add_conditional_edges("approval", _route_after_approval,
                                {"remediation": "remediation", "summary": "summary"})
    graph.add_edge("remediation", "summary")
    graph.add_edge("summary", END)

    return graph.compile(checkpointer=_shared_checkpointer)


# =========================================================================
# Entry point
# =========================================================================

def main():
    _configure_langsmith()

    # [7] Initialize OpenTelemetry
    otel_ready = _configure_otel()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    # [4] Initialize Mem0
    mem0_ready = _init_mem0()

    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not task:
        print("Multi-Agent Kubernetes Operations System — v3")
        print("=" * 50)
        print("\nAll improvements:")
        print("  [1]  Parallel namespace scanning (fan-out)")
        print("  [2]  Pydantic structured output (typed LLM responses)")
        print("  [3]  Reflexion loop (self-correcting recommendations)")
        print(f"  [4]  Mem0 incident memory ({'enabled' if mem0_ready else 'disabled — set MEM0_API_KEY'})")
        print(f"  [6]  Checkpointer: {_checkpointer_type}" +
              (" (set POSTGRES_URI for prod)" if _checkpointer_type == "memory" else ""))
        print(f"  [7]  OpenTelemetry ({'enabled' if otel_ready else 'disabled — set OTEL_EXPORTER_OTLP_ENDPOINT'})")
        print("  [9]  Supervisor routing (LLM-driven dynamic decisions)")
        print("  [10] Eval data collection (auto-saved to eval_data/)")
        print('\nExamples:')
        print('  python multi_agent_v3.py "Find all crashing pods and fix them"')
        print('  python multi_agent_v3.py "Check cluster health and remediate issues"')
        print()
        task = input("Enter your task: ").strip()
        if not task: sys.exit("No task entered.")

    print(f"\n{'=' * 60}")
    print(f"Task: {task}")
    print(f"{'=' * 60}")
    models_used = {k: v["model"] for k, v in MODEL_REGISTRY.items()}
    print(f"\nModels: {json.dumps(models_used, indent=2)}")
    print(f"Checkpointer: {_checkpointer_type}")
    print(f"Mem0:   {'connected' if mem0_ready else 'not configured'}")
    print(f"OTel:   {'exporting' if otel_ready else 'not configured'}")
    print(f"\nPipeline: List NS → [Parallel Scans] → Merge → "
          f"⟨Supervisor ⇄ Agents⟩ → Approval → Executor → Summary\n")

    orchestrator = build_orchestrator()
    initial_state: AgentState = {
        "task": task, "phase": "list_namespaces",
        "namespaces": [], "ns_issues": [],
        "issues": [], "diagnoses": [], "recommendations": [],
        "approved_commands": [], "actions_taken": [],
        "summary": "", "agent_logs": [],
        "reflexion_count": 0, "reflexion_critique": "",
        "memory_context": "",
        "supervisor_next": "", "supervisor_reasoning": "",
        "eval_trace": [],
    }

    # [7] Wrap entire run in OTel span if available
    run_config = {
        "run_name": f"k8s-ops-v3: {task[:50]}",
        "tags": ["production-run", "v3"],
        "metadata": {"task": task, "models": models_used,
                      "mem0": mem0_ready, "otel": otel_ready,
                      "checkpointer": _checkpointer_type},
        "configurable": {"thread_id": f"k8s-ops-{int(time.time())}"},
        "recursion_limit": 100,
    }

    result = orchestrator.invoke(initial_state, config=run_config)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60 + "\n")
    print(result.get("summary", "No summary generated."))
    print("\n" + "-" * 60)
    print("AGENT EXECUTION LOG")
    print("-" * 60)
    for log in result.get("agent_logs", []):
        print(f"  • {log}")

    # [10] Save eval trace to disk
    eval_path = _save_eval_trace(task, result.get("eval_trace", []), result)
    if eval_path:
        print(f"\n  Eval trace saved: {eval_path}")
    print()


if __name__ == "__main__":
    main()
