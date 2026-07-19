"""
Autonomous Kubernetes agent — LangGraph + Claude API (Optimized).

Key optimizations over the original:
  - Compact capability manifest instead of verbose README injection (~90% prompt reduction)
  - Automatic -o json injection for kubectl get commands
  - Python-side parsing and summarization of kubectl output
  - Structured JSON responses instead of raw stdout
  - Command-type detection with tailored summarizers
  - Discovery caching (namespaces, nodes) with TTL
  - Unhealthy-only filtering for pod/event queries
  - Log compression (error/warning extraction)

Setup:
  pip install -U langgraph langchain-anthropic
  export ANTHROPIC_API_KEY="your-key"

Run:
  python agent.py
  python agent.py "Find all crashing pods across namespaces"
  python agent.py "Check rollout status of api-server in production"
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain.agents import create_agent

WORKDIR = Path.cwd()

# --------------------------------------------------------------------------
# 1. Discovery cache — avoid repeated namespace/node lookups
# --------------------------------------------------------------------------

_cache: dict[str, tuple[float, str]] = {}
CACHE_TTL = 300  # 5 minutes


def _cache_key(command: str) -> str | None:
    """Return a cache key if this command is cacheable, else None."""
    cacheable = [
        "kubectl get namespaces",
        "kubectl get namespace",
        "kubectl get ns",
        "kubectl get nodes",
        "kubectl get node",
        "kubectl api-resources",
    ]
    stripped = command.strip()
    for pattern in cacheable:
        if stripped == pattern or stripped.startswith(pattern + " "):
            return pattern
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


# --------------------------------------------------------------------------
# 2. Guardrails — what the agent may and may NOT execute
# --------------------------------------------------------------------------

ALLOWED_BINARIES = {
    "kubectl", "helm",
    "grep", "jq", "cat", "echo", "head", "tail", "wc", "sort", "uniq",
}

BLOCKED_PATTERNS = (
    "kubectl delete",
    "kubectl apply",
    "kubectl scale",
    "kubectl edit",
    "kubectl rollout undo",
    "kubectl exec",
    "kubectl cp",
    "kubectl patch",
    "kubectl replace",
    "kubectl set",
    "rm -rf",
    "shutdown",
    "reboot",
)


def _command_allowed(command: str) -> str | None:
    """Return an error string if the command is blocked, else None."""
    lowered = command.lower().strip()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lowered:
            return (
                f"BLOCKED: '{pattern}' is a mutating command. "
                f"Report the command you WOULD run and ask the user to confirm."
            )
    try:
        first = shlex.split(command)[0]
    except (ValueError, IndexError):
        return "BLOCKED: could not parse command."

    base = Path(first).name
    if base not in ALLOWED_BINARIES:
        return f"BLOCKED: '{base}' not in allow-list: {', '.join(sorted(ALLOWED_BINARIES))}"
    return None


# --------------------------------------------------------------------------
# 3. JSON injection — auto-add -o json to kubectl get commands
# --------------------------------------------------------------------------

_GET_PATTERN = re.compile(
    r"^kubectl\s+get\s+(?!-)", re.IGNORECASE
)

# Commands where -o json doesn't apply or is already specified
_SKIP_JSON_INJECTION = re.compile(
    r"-o\s+(json|yaml|wide|jsonpath|custom-columns|name)", re.IGNORECASE
)

# kubectl top doesn't support -o json
_TOP_PATTERN = re.compile(r"^kubectl\s+top\s+", re.IGNORECASE)


def _maybe_inject_json(command: str) -> tuple[str, bool]:
    """If this is a plain kubectl get, add -o json. Returns (command, was_injected)."""
    stripped = command.strip()

    # Don't inject for piped commands (user wants specific filtering)
    if "|" in stripped:
        return stripped, False

    # Don't inject for kubectl top
    if _TOP_PATTERN.match(stripped):
        return stripped, False

    # Don't inject if already has an output format
    if _SKIP_JSON_INJECTION.search(stripped):
        return stripped, False

    # Inject -o json for plain kubectl get
    if _GET_PATTERN.match(stripped):
        return stripped + " -o json", True

    return stripped, False


# --------------------------------------------------------------------------
# 4. Output summarizers — parse JSON, return compact structured data
# --------------------------------------------------------------------------

def _summarize_pods(raw_json: str) -> str:
    """Parse kubectl get pods -o json and return compact summary."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json  # fallback to raw

    items = data.get("items", [])
    if not items:
        return json.dumps({"total": 0, "pods": []}, indent=2)

    summary = {"total": len(items), "healthy": 0, "unhealthy": []}

    for pod in items:
        name = pod.get("metadata", {}).get("name", "unknown")
        namespace = pod.get("metadata", {}).get("namespace", "unknown")
        phase = pod.get("status", {}).get("phase", "Unknown")

        container_statuses = pod.get("status", {}).get("containerStatuses", [])
        restart_count = sum(cs.get("restartCount", 0) for cs in container_statuses)
        ready = all(cs.get("ready", False) for cs in container_statuses) if container_statuses else False

        # Detect waiting reasons (CrashLoopBackOff, ImagePullBackOff, etc.)
        waiting_reasons = []
        for cs in container_statuses:
            waiting = cs.get("state", {}).get("waiting", {})
            if waiting.get("reason"):
                waiting_reasons.append(waiting["reason"])

        # Also check init container statuses
        init_statuses = pod.get("status", {}).get("initContainerStatuses", [])
        for cs in init_statuses:
            waiting = cs.get("state", {}).get("waiting", {})
            if waiting.get("reason"):
                waiting_reasons.append(f"init:{waiting['reason']}")

        # Determine last terminated reason
        last_terminated_reason = None
        for cs in container_statuses:
            terminated = cs.get("lastState", {}).get("terminated", {})
            if terminated.get("reason"):
                last_terminated_reason = terminated["reason"]

        is_healthy = (
            phase in ("Running", "Succeeded")
            and ready
            and not waiting_reasons
            and restart_count < 5
        )

        if is_healthy:
            summary["healthy"] += 1
        else:
            pod_info: dict = {
                "name": name,
                "namespace": namespace,
                "phase": phase,
                "ready": ready,
                "restarts": restart_count,
            }
            if waiting_reasons:
                pod_info["waitingReasons"] = waiting_reasons
            if last_terminated_reason:
                pod_info["lastTerminatedReason"] = last_terminated_reason
            summary["unhealthy"].append(pod_info)

    return json.dumps(summary, indent=2)


def _summarize_nodes(raw_json: str) -> str:
    """Parse kubectl get nodes -o json and return compact summary."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    items = data.get("items", [])
    nodes = []
    for node in items:
        name = node.get("metadata", {}).get("name", "unknown")
        conditions = node.get("status", {}).get("conditions", [])

        ready = "Unknown"
        issues = []
        for cond in conditions:
            if cond.get("type") == "Ready":
                ready = cond.get("status", "Unknown")
            elif cond.get("status") == "True" and cond.get("type") != "Ready":
                issues.append(cond["type"])  # e.g. MemoryPressure, DiskPressure

        node_info = node.get("status", {}).get("nodeInfo", {})
        nodes.append({
            "name": name,
            "ready": ready,
            "kubeletVersion": node_info.get("kubeletVersion", ""),
            "os": node_info.get("osImage", ""),
            "issues": issues if issues else None,
        })

    return json.dumps({"total": len(nodes), "nodes": nodes}, indent=2)


def _summarize_namespaces(raw_json: str) -> str:
    """Parse kubectl get namespaces -o json and return names + status."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    items = data.get("items", [])
    namespaces = [
        {
            "name": ns.get("metadata", {}).get("name", "unknown"),
            "status": ns.get("status", {}).get("phase", "Unknown"),
        }
        for ns in items
    ]
    return json.dumps({"total": len(namespaces), "namespaces": namespaces}, indent=2)


def _summarize_deployments(raw_json: str) -> str:
    """Parse kubectl get deployments -o json and return compact summary."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    items = data.get("items", [])
    deployments = []
    for dep in items:
        name = dep.get("metadata", {}).get("name", "unknown")
        namespace = dep.get("metadata", {}).get("namespace", "unknown")
        spec_replicas = dep.get("spec", {}).get("replicas", 0)
        status = dep.get("status", {})
        ready = status.get("readyReplicas", 0)
        available = status.get("availableReplicas", 0)
        updated = status.get("updatedReplicas", 0)

        is_healthy = (ready == spec_replicas and available == spec_replicas)
        if not is_healthy:
            deployments.append({
                "name": name,
                "namespace": namespace,
                "desired": spec_replicas,
                "ready": ready,
                "available": available,
                "updated": updated,
            })

    healthy_count = len(items) - len(deployments)
    return json.dumps({
        "total": len(items),
        "healthy": healthy_count,
        "unhealthy": deployments,
    }, indent=2)


def _summarize_events(raw_json: str) -> str:
    """Parse kubectl get events -o json — return only warnings and errors."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    items = data.get("items", [])
    warnings = []
    normal_count = 0

    for evt in items:
        evt_type = evt.get("type", "Normal")
        if evt_type != "Normal":
            warnings.append({
                "type": evt_type,
                "reason": evt.get("reason", ""),
                "message": evt.get("message", "")[:200],
                "object": evt.get("involvedObject", {}).get("name", ""),
                "count": evt.get("count", 1),
                "lastSeen": evt.get("lastTimestamp", ""),
            })
        else:
            normal_count += 1

    # Sort by lastSeen descending, keep top 20
    warnings.sort(key=lambda e: e.get("lastSeen", ""), reverse=True)
    warnings = warnings[:20]

    return json.dumps({
        "normalEventCount": normal_count,
        "warnings": warnings,
    }, indent=2)


def _summarize_services(raw_json: str) -> str:
    """Parse kubectl get svc -o json."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    items = data.get("items", [])
    services = []
    for svc in items:
        spec = svc.get("spec", {})
        ports = [
            {"port": p.get("port"), "targetPort": p.get("targetPort"), "protocol": p.get("protocol")}
            for p in spec.get("ports", [])
        ]
        services.append({
            "name": svc.get("metadata", {}).get("name", ""),
            "namespace": svc.get("metadata", {}).get("namespace", ""),
            "type": spec.get("type", ""),
            "clusterIP": spec.get("clusterIP", ""),
            "ports": ports,
        })
    return json.dumps({"total": len(services), "services": services}, indent=2)


def _summarize_generic(raw_json: str) -> str:
    """For any other kubectl get resource — extract names and key metadata."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    items = data.get("items", [])
    results = []
    for item in items:
        meta = item.get("metadata", {})
        entry = {
            "name": meta.get("name", "unknown"),
            "namespace": meta.get("namespace", ""),
        }
        # Include status if present
        status = item.get("status", {})
        if isinstance(status, dict) and "phase" in status:
            entry["phase"] = status["phase"]
        results.append(entry)

    return json.dumps({"total": len(results), "items": results}, indent=2)


def _summarize_describe(raw_text: str) -> str:
    """Extract key fields from kubectl describe output."""
    lines = raw_text.splitlines()

    result: dict = {}
    events: list[str] = []
    in_events = False
    conditions: list[str] = []
    in_conditions = False

    for line in lines:
        stripped = line.strip()

        # Extract key fields
        if stripped.startswith("Status:"):
            result["status"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Restart Count:"):
            try:
                result["restartCount"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                result["restartCount"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Reason:"):
            result["reason"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Message:"):
            result["message"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Exit Code:"):
            try:
                result["exitCode"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("QoS Class:"):
            result["qosClass"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Node:"):
            result["node"] = stripped.split(":", 1)[1].strip()

        # Track sections
        if stripped.startswith("Conditions:"):
            in_conditions = True
            in_events = False
            continue
        elif stripped.startswith("Events:"):
            in_events = True
            in_conditions = False
            continue
        elif stripped and not stripped.startswith(" ") and ":" in stripped:
            if in_events or in_conditions:
                in_events = False
                in_conditions = False

        if in_events and stripped and not stripped.startswith("Type"):
            # Capture event lines (compact)
            events.append(stripped[:150])
        if in_conditions and stripped and not stripped.startswith("Type"):
            conditions.append(stripped[:150])

    if events:
        result["events"] = events[-10:]  # last 10 events
    if conditions:
        result["conditions"] = conditions

    if not result:
        # Fallback: return truncated raw text
        return raw_text[:3000]

    return json.dumps(result, indent=2)


def _summarize_logs(raw_text: str) -> str:
    """Extract errors, exceptions, and warnings from log output."""
    lines = raw_text.splitlines()

    error_pattern = re.compile(r"error|exception|fatal|panic|traceback|oom|killed", re.IGNORECASE)
    warn_pattern = re.compile(r"warn", re.IGNORECASE)

    errors = []
    warnings = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if error_pattern.search(stripped):
            errors.append(stripped[:200])
        elif warn_pattern.search(stripped):
            warnings.append(stripped[:200])

    # Deduplicate while preserving order
    seen_errors: set[str] = set()
    unique_errors = []
    for e in errors:
        key = e[:80]
        if key not in seen_errors:
            seen_errors.add(key)
            unique_errors.append(e)

    seen_warnings: set[str] = set()
    unique_warnings = []
    for w in warnings:
        key = w[:80]
        if key not in seen_warnings:
            seen_warnings.add(key)
            unique_warnings.append(w)

    summary = {
        "totalLines": len(lines),
        "errorCount": len(errors),
        "warningCount": len(warnings),
        "uniqueErrors": unique_errors[:15],
        "uniqueWarnings": unique_warnings[:10],
    }

    return json.dumps(summary, indent=2)


# --------------------------------------------------------------------------
# 5. Command-type detection and routing
# --------------------------------------------------------------------------

def _detect_and_summarize(command: str, raw_output: str, was_json_injected: bool) -> str:
    """Detect the command type and apply the appropriate summarizer."""
    cmd_lower = command.lower().strip()

    # Only summarize if we successfully got JSON output
    if was_json_injected or "-o json" in cmd_lower or "-o=json" in cmd_lower:
        # Try to determine the resource type
        if _matches(cmd_lower, ["get pods", "get pod", "get po"]):
            return _summarize_pods(raw_output)
        elif _matches(cmd_lower, ["get nodes", "get node", "get no "]):
            return _summarize_nodes(raw_output)
        elif _matches(cmd_lower, ["get namespaces", "get namespace", "get ns"]):
            return _summarize_namespaces(raw_output)
        elif _matches(cmd_lower, ["get deployments", "get deployment", "get deploy"]):
            return _summarize_deployments(raw_output)
        elif _matches(cmd_lower, ["get events", "get event", "get ev "]):
            return _summarize_events(raw_output)
        elif _matches(cmd_lower, ["get svc", "get services", "get service"]):
            return _summarize_services(raw_output)
        else:
            return _summarize_generic(raw_output)

    # Non-JSON commands
    if _matches(cmd_lower, ["describe pod", "describe pods"]):
        return _summarize_describe(raw_output)
    elif _matches(cmd_lower, ["logs ", "log "]):
        return _summarize_logs(raw_output)
    elif _matches(cmd_lower, ["describe "]):
        return _summarize_describe(raw_output)

    # Fallback: return raw but truncated more aggressively
    if len(raw_output) > 5000:
        return raw_output[:5000] + "\n[... truncated, request specific details if needed ...]"
    return raw_output


def _matches(cmd: str, patterns: list[str]) -> bool:
    return any(p in cmd for p in patterns)


# --------------------------------------------------------------------------
# 6. The single intelligent tool
# --------------------------------------------------------------------------

@tool
def run_cli(command: str) -> str:
    """Execute a kubectl or helm command against the cluster.
    Returns structured JSON summaries for common operations.
    For kubectl get, output is automatically parsed and summarized.
    For describe/logs, key fields and errors are extracted.
    Healthy resources are counted; only unhealthy ones are listed in detail."""

    error = _command_allowed(command)
    if error:
        return error

    # Check cache for discovery commands
    ckey = _cache_key(command)
    if ckey:
        cached = _get_cached(ckey)
        if cached:
            return cached

    # Auto-inject -o json for kubectl get commands
    actual_command, was_json_injected = _maybe_inject_json(command)

    try:
        proc = subprocess.run(
            actual_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=WORKDIR,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Command timed out after 120s"})

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        return json.dumps({
            "success": False,
            "exitCode": proc.returncode,
            "error": stderr.strip()[:1000] or "Command failed with no stderr",
        }, indent=2)

    if not stdout.strip():
        return json.dumps({"success": True, "result": "No resources found"})

    # Summarize the output based on command type
    summarized = _detect_and_summarize(command, stdout, was_json_injected)

    # Cache discovery commands
    if ckey:
        _set_cache(ckey, summarized)

    return summarized


# --------------------------------------------------------------------------
# 7. Claude model
# --------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.2,
    max_tokens=4096,
)

# --------------------------------------------------------------------------
# 8. Compact system prompt — no README injection
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous Kubernetes operations agent.

You interact with the cluster through the run_cli tool, which executes
kubectl and helm commands. The tool automatically:
- Converts kubectl get output to structured JSON summaries
- Filters to show only unhealthy/notable resources (healthy ones are counted)
- Extracts key fields from describe output
- Extracts errors and warnings from logs

Available operations: namespaces, pods, deployments, statefulsets, daemonsets,
jobs, cronjobs, services, ingress, endpoints, configmaps, secrets (keys only),
PVCs, PVs, nodes, events, HPA, rollout status/history, resource usage (top).

Workflow:
1. Read the user's task.
2. Plan kubectl commands (always start read-only).
3. Call run_cli for each command, inspect the structured response.
4. Chain logically: find unhealthy → describe → logs → events.
5. Summarize: what you checked, findings, root cause, recommended actions.

Rules:
- Trust the tool's summaries; don't ask for raw output unless the summary
  is insufficient for diagnosis.
- For mutating commands (apply, delete, scale, edit, rollout undo, etc.),
  do NOT execute. Report the exact command and ask the user to confirm.
- Never expose secret values.
- Be concise in your final summary.
"""

agent = create_agent(
    model=llm,
    tools=[run_cli],
    system_prompt=SYSTEM_PROMPT,
)

# --------------------------------------------------------------------------
# 9. Entry point
# --------------------------------------------------------------------------

def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    if not task:
        print("No task provided. Example tasks:\n")
        print('  python agent.py "List all namespaces and their pod health"')
        print('  python agent.py "Find pods in CrashLoopBackOff and diagnose"')
        print('  python agent.py "Check resource usage across all nodes"')
        print('  python agent.py "Show rollout status of app-server in staging"')
        print('  python agent.py "Are there any pending PVCs?"')
        print()
        task = input("Enter your task: ").strip()
        if not task:
            sys.exit("No task entered.")

    print(f"\n[agent] Task: {task}")
    print("[agent] Starting autonomous execution...\n")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]}
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60 + "\n")
    final = result["messages"][-1]
    print(final.content)


if __name__ == "__main__":
    main()