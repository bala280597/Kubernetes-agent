"""
Benchmark: Optimized CLI-manifest agent vs MCP agent.

The CLI agent uses the optimized architecture:
  - Compact system prompt (no README injection)
  - Automatic -o json injection for kubectl get
  - Python-side parsing and structured JSON summaries
  - Unhealthy-only filtering, log compression
  - Discovery caching with TTL

The MCP agent is unchanged for a fair comparison.

Setup:
  pip install -U langgraph langchain-anthropic langchain-mcp-adapters "mcp[cli]"
  export ANTHROPIC_API_KEY="your-key"

  Place mcp_kubectl_server.py in the same folder.

Run:
  python benchmark.py
"""

import asyncio
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
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

WORKDIR = Path.cwd()
SERVER_SCRIPT = str(Path(__file__).parent / "mcp_kubectl_server.py")

# ======================================================================
# Shared model
# ======================================================================

def make_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0.2,
        max_tokens=4096,
    )


# ======================================================================
# Discovery cache
# ======================================================================

_cache: dict[str, tuple[float, str]] = {}
CACHE_TTL = 300  # 5 minutes

_CACHEABLE_COMMANDS = [
    "kubectl get namespaces",
    "kubectl get namespace",
    "kubectl get ns",
    "kubectl get nodes",
    "kubectl get node",
    "kubectl api-resources",
]


def _cache_key(command: str) -> str | None:
    stripped = command.strip()
    for pattern in _CACHEABLE_COMMANDS:
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


def _clear_cache() -> None:
    _cache.clear()


# ======================================================================
# Guardrails
# ======================================================================

ALLOWED_BINARIES = {
    "kubectl", "helm", "grep", "jq", "cat", "echo",
    "head", "tail", "wc", "sort", "uniq",
}

BLOCKED_PATTERNS = (
    "kubectl delete", "kubectl apply", "kubectl scale",
    "kubectl edit", "kubectl rollout undo", "kubectl exec",
    "kubectl cp", "kubectl patch", "kubectl replace", "kubectl set",
    "rm -rf", "shutdown", "reboot",
)


def _cmd_allowed(command: str) -> str | None:
    lowered = command.lower().strip()
    for p in BLOCKED_PATTERNS:
        if p in lowered:
            return f"BLOCKED: '{p}'"
    try:
        base = Path(shlex.split(command)[0]).name
    except (ValueError, IndexError):
        return "BLOCKED: parse error"
    if base not in ALLOWED_BINARIES:
        return f"BLOCKED: '{base}' not allowed"
    return None


# ======================================================================
# JSON injection — auto-add -o json to kubectl get commands
# ======================================================================

_GET_PATTERN = re.compile(r"^kubectl\s+get\s+(?!-)", re.IGNORECASE)
_SKIP_JSON_INJECTION = re.compile(
    r"-o\s+(json|yaml|wide|jsonpath|custom-columns|name)", re.IGNORECASE
)
_TOP_PATTERN = re.compile(r"^kubectl\s+top\s+", re.IGNORECASE)


def _maybe_inject_json(command: str) -> tuple[str, bool]:
    stripped = command.strip()
    if "|" in stripped:
        return stripped, False
    if _TOP_PATTERN.match(stripped):
        return stripped, False
    if _SKIP_JSON_INJECTION.search(stripped):
        return stripped, False
    if _GET_PATTERN.match(stripped):
        return stripped + " -o json", True
    return stripped, False


# ======================================================================
# Output summarizers
# ======================================================================

def _summarize_pods(raw_json: str) -> str:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json

    items = data.get("items", [])
    if not items:
        return json.dumps({"total": 0, "pods": []}, indent=2)

    summary: dict = {"total": len(items), "healthy": 0, "unhealthy": []}

    for pod in items:
        name = pod.get("metadata", {}).get("name", "unknown")
        namespace = pod.get("metadata", {}).get("namespace", "unknown")
        phase = pod.get("status", {}).get("phase", "Unknown")

        container_statuses = pod.get("status", {}).get("containerStatuses", [])
        restart_count = sum(cs.get("restartCount", 0) for cs in container_statuses)
        ready = all(cs.get("ready", False) for cs in container_statuses) if container_statuses else False

        waiting_reasons = []
        for cs in container_statuses:
            waiting = cs.get("state", {}).get("waiting", {})
            if waiting.get("reason"):
                waiting_reasons.append(waiting["reason"])
        for cs in pod.get("status", {}).get("initContainerStatuses", []):
            waiting = cs.get("state", {}).get("waiting", {})
            if waiting.get("reason"):
                waiting_reasons.append(f"init:{waiting['reason']}")

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
                "name": name, "namespace": namespace,
                "phase": phase, "ready": ready, "restarts": restart_count,
            }
            if waiting_reasons:
                pod_info["waitingReasons"] = waiting_reasons
            if last_terminated_reason:
                pod_info["lastTerminatedReason"] = last_terminated_reason
            summary["unhealthy"].append(pod_info)

    return json.dumps(summary, indent=2)


def _summarize_nodes(raw_json: str) -> str:
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
                issues.append(cond["type"])
        node_info = node.get("status", {}).get("nodeInfo", {})
        nodes.append({
            "name": name, "ready": ready,
            "kubeletVersion": node_info.get("kubeletVersion", ""),
            "os": node_info.get("osImage", ""),
            "issues": issues if issues else None,
        })
    return json.dumps({"total": len(nodes), "nodes": nodes}, indent=2)


def _summarize_namespaces(raw_json: str) -> str:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json
    items = data.get("items", [])
    namespaces = [
        {"name": ns.get("metadata", {}).get("name", "unknown"),
         "status": ns.get("status", {}).get("phase", "Unknown")}
        for ns in items
    ]
    return json.dumps({"total": len(namespaces), "namespaces": namespaces}, indent=2)


def _summarize_deployments(raw_json: str) -> str:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json
    items = data.get("items", [])
    unhealthy = []
    for dep in items:
        name = dep.get("metadata", {}).get("name", "unknown")
        namespace = dep.get("metadata", {}).get("namespace", "unknown")
        spec_replicas = dep.get("spec", {}).get("replicas", 0)
        status = dep.get("status", {})
        ready = status.get("readyReplicas", 0)
        available = status.get("availableReplicas", 0)
        updated = status.get("updatedReplicas", 0)
        if ready != spec_replicas or available != spec_replicas:
            unhealthy.append({
                "name": name, "namespace": namespace,
                "desired": spec_replicas, "ready": ready,
                "available": available, "updated": updated,
            })
    return json.dumps({
        "total": len(items), "healthy": len(items) - len(unhealthy),
        "unhealthy": unhealthy,
    }, indent=2)


def _summarize_events(raw_json: str) -> str:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json
    items = data.get("items", [])
    warnings = []
    normal_count = 0
    for evt in items:
        if evt.get("type", "Normal") != "Normal":
            warnings.append({
                "type": evt.get("type", ""), "reason": evt.get("reason", ""),
                "message": evt.get("message", "")[:200],
                "object": evt.get("involvedObject", {}).get("name", ""),
                "count": evt.get("count", 1),
                "lastSeen": evt.get("lastTimestamp", ""),
            })
        else:
            normal_count += 1
    warnings.sort(key=lambda e: e.get("lastSeen", ""), reverse=True)
    return json.dumps({
        "normalEventCount": normal_count, "warnings": warnings[:20],
    }, indent=2)


def _summarize_services(raw_json: str) -> str:
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
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json
    items = data.get("items", [])
    results = []
    for item in items:
        meta = item.get("metadata", {})
        entry: dict = {"name": meta.get("name", "unknown"), "namespace": meta.get("namespace", "")}
        status = item.get("status", {})
        if isinstance(status, dict) and "phase" in status:
            entry["phase"] = status["phase"]
        results.append(entry)
    return json.dumps({"total": len(results), "items": results}, indent=2)


def _summarize_describe(raw_text: str) -> str:
    lines = raw_text.splitlines()
    result: dict = {}
    events: list[str] = []
    conditions: list[str] = []
    in_events = False
    in_conditions = False

    for line in lines:
        stripped = line.strip()
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

        if stripped.startswith("Conditions:"):
            in_conditions, in_events = True, False
            continue
        elif stripped.startswith("Events:"):
            in_events, in_conditions = True, False
            continue
        elif stripped and not stripped.startswith(" ") and ":" in stripped:
            in_events = in_conditions = False

        if in_events and stripped and not stripped.startswith("Type"):
            events.append(stripped[:150])
        if in_conditions and stripped and not stripped.startswith("Type"):
            conditions.append(stripped[:150])

    if events:
        result["events"] = events[-10:]
    if conditions:
        result["conditions"] = conditions
    if not result:
        return raw_text[:3000]
    return json.dumps(result, indent=2)


def _summarize_logs(raw_text: str) -> str:
    lines = raw_text.splitlines()
    error_pat = re.compile(r"error|exception|fatal|panic|traceback|oom|killed", re.IGNORECASE)
    warn_pat = re.compile(r"warn", re.IGNORECASE)

    errors, warnings = [], []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if error_pat.search(s):
            errors.append(s[:200])
        elif warn_pat.search(s):
            warnings.append(s[:200])

    def _dedup(items: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        out = []
        for item in items:
            key = item[:80]
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out[:limit]

    return json.dumps({
        "totalLines": len(lines),
        "errorCount": len(errors), "warningCount": len(warnings),
        "uniqueErrors": _dedup(errors, 15),
        "uniqueWarnings": _dedup(warnings, 10),
    }, indent=2)


# ======================================================================
# Command-type detection and routing
# ======================================================================

def _matches(cmd: str, patterns: list[str]) -> bool:
    return any(p in cmd for p in patterns)


def _detect_and_summarize(command: str, raw_output: str, was_json_injected: bool) -> str:
    cmd_lower = command.lower().strip()

    if was_json_injected or "-o json" in cmd_lower or "-o=json" in cmd_lower:
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

    if _matches(cmd_lower, ["describe pod", "describe pods"]):
        return _summarize_describe(raw_output)
    elif _matches(cmd_lower, ["logs ", "log "]):
        return _summarize_logs(raw_output)
    elif _matches(cmd_lower, ["describe "]):
        return _summarize_describe(raw_output)

    if len(raw_output) > 5000:
        return raw_output[:5000] + "\n[... truncated ...]"
    return raw_output


# ======================================================================
# Optimized CLI tool — intelligent summarization
# ======================================================================

@tool
def run_cli_optimized(command: str) -> str:
    """Execute a kubectl or helm command against the cluster.
    Returns structured JSON summaries for common operations.
    For kubectl get, output is automatically parsed and summarized.
    For describe/logs, key fields and errors are extracted.
    Healthy resources are counted; only unhealthy ones are listed in detail."""

    err = _cmd_allowed(command)
    if err:
        return err

    # Check cache
    ckey = _cache_key(command)
    if ckey:
        cached = _get_cached(ckey)
        if cached:
            return cached

    # Auto-inject -o json
    actual_command, was_json_injected = _maybe_inject_json(command)

    try:
        proc = subprocess.run(
            actual_command, shell=True, capture_output=True,
            text=True, timeout=120, cwd=WORKDIR,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Command timed out after 120s"})

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if proc.returncode != 0:
        return json.dumps({
            "success": False, "exitCode": proc.returncode,
            "error": stderr.strip()[:1000] or "Command failed with no stderr",
        }, indent=2)

    if not stdout.strip():
        return json.dumps({"success": True, "result": "No resources found"})

    summarized = _detect_and_summarize(command, stdout, was_json_injected)

    if ckey:
        _set_cache(ckey, summarized)

    return summarized



# ======================================================================
# Metric extraction
# ======================================================================

def _extract_metrics(result: dict, approach: str, task: str,
                     tool_count: int, elapsed: float) -> dict:
    messages = result["messages"]
    inp_tok = out_tok = tool_calls = 0
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls += len(msg.tool_calls)
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            inp_tok += usage.get("input_tokens", 0)
            out_tok += usage.get("output_tokens", 0)
    return {
        "approach": approach,
        "task": task,
        "tool_count": tool_count,
        "tool_calls_made": tool_calls,
        "input_tokens": inp_tok,
        "output_tokens": out_tok,
        "total_tokens": inp_tok + out_tok,
        "elapsed_seconds": round(elapsed, 2),
        "message_count": len(messages),
        "answer_length": len(messages[-1].content) if messages else 0,
    }


# ======================================================================
# Agent runners
# ======================================================================

# --- Optimized CLI agent (compact prompt, intelligent tool) ---

CLI_OPTIMIZED_PROMPT = """\
You are an autonomous Kubernetes operations agent.

You interact with the cluster through the run_cli_optimized tool, which
executes kubectl and helm commands. The tool automatically:
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
3. Call run_cli_optimized for each command, inspect the structured response.
4. Chain logically: find unhealthy → describe → logs → events.
5. Summarize: what you checked, findings, root cause, recommended actions.

Rules:
- Trust the tool's summaries; don't ask for raw output unless insufficient.
- For mutating commands, do NOT execute. Report the command and ask to confirm.
- Never expose secret values.
- Be concise in your final summary.
"""


async def run_cli_optimized_agent(task: str) -> dict:
    """Run the optimized CLI agent and return metrics."""
    _clear_cache()  # fresh cache per task
    llm = make_llm()
    agent = create_agent(
        model=llm, tools=[run_cli_optimized], system_prompt=CLI_OPTIMIZED_PROMPT
    )
    t0 = time.perf_counter()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
    elapsed = time.perf_counter() - t0
    return _extract_metrics(result, "CLI-Optimized", task, tool_count=1, elapsed=elapsed)



# --- MCP agent (unchanged) ---

async def run_mcp_agent(task: str) -> dict:
    """Run the MCP agent and return metrics."""
    server_params = StdioServerParameters(
        command=sys.executable, args=[SERVER_SCRIPT],
    )
    system_prompt = (
        "You are an autonomous Kubernetes operations agent.\n"
        "You have MCP tools for kubectl operations. Use them to inspect the cluster.\n"
        "Start with discovery (list_namespaces, get_nodes), then drill in.\n"
        "For mutations, use suggest_mutation to propose — never execute directly.\n"
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            tool_count = len(tools)
            print(f"  [mcp] Discovered {tool_count} tools")

            llm = make_llm()
            agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
            t0 = time.perf_counter()
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": task}]}
            )
            elapsed = time.perf_counter() - t0
            return _extract_metrics(result, "MCP", task, tool_count=tool_count, elapsed=elapsed)


# ======================================================================
# Benchmark harness — CLI-Optimized vs MCP
# ======================================================================

TASKS = [
    "List all namespaces and show pod health in each",
    "Find any pods not in Running state and diagnose why",
    "Check resource usage across all nodes",
    "Show the services and ingress rules in the default namespace",
    "Get events in kube-system namespace sorted by time",
]

APPROACHES = [
    ("CLI-Optimized", run_cli_optimized_agent),
    ("MCP",           run_mcp_agent),
]


def _empty_metrics(approach: str, task: str) -> dict:
    return {
        "approach": approach, "task": task, "tool_count": 0,
        "tool_calls_made": 0, "input_tokens": 0, "output_tokens": 0,
        "total_tokens": 0, "elapsed_seconds": 0, "message_count": 0,
        "answer_length": 0,
    }


def print_comparison(all_results: dict[str, list[dict]]):
    """Print a side-by-side comparison table."""
    sep = "-" * 100
    print(f"\n{'=' * 100}")
    print(f"{'BENCHMARK RESULTS':^100}")
    print(f"{'=' * 100}\n")

    def avg(lst, key):
        vals = [m[key] for m in lst]
        return sum(vals) / len(vals) if vals else 0

    cli = all_results["CLI-Optimized"]
    mcp = all_results["MCP"]

    # Summary table
    hdr = f"  {'Metric':<30} {'CLI-Optimized':>15} {'MCP':>15} {'Diff':>15} {'Winner':>12}"
    print(hdr)
    print(sep)

    comparisons = [
        ("Avg input tokens",    "input_tokens"),
        ("Avg output tokens",   "output_tokens"),
        ("Avg total tokens",    "total_tokens"),
        ("Avg tool calls",      "tool_calls_made"),
        ("Avg elapsed (s)",     "elapsed_seconds"),
        ("Avg answer length",   "answer_length"),
        ("Avg messages",        "message_count"),
    ]

    for label, key in comparisons:
        cli_val = avg(cli, key)
        mcp_val = avg(mcp, key)
        diff = mcp_val - cli_val
        # Lower is better for tokens/time; higher is better for answer length
        if key == "answer_length":
            winner = "MCP" if mcp_val > cli_val else "CLI"
        else:
            winner = "CLI" if cli_val <= mcp_val else "MCP"
        print(f"  {label:<28} {cli_val:>15,.1f} {mcp_val:>15,.1f} {diff:>+14,.1f} {winner:>12}")

    print(sep)

    # Per-task breakdown
    print(f"\n{'PER-TASK BREAKDOWN':^100}")
    print(sep)
    for i, task in enumerate(TASKS):
        cm = cli[i]
        mm = mcp[i]
        print(f"\n  Task {i+1}: {task[:70]}")
        print(f"    {'':20} {'CLI-Opt':>12} {'MCP':>12} {'Diff':>12}")
        print(f"    {'Tool calls':<20} {cm['tool_calls_made']:>12} {mm['tool_calls_made']:>12} {mm['tool_calls_made']-cm['tool_calls_made']:>+12}")
        print(f"    {'Total tokens':<20} {cm['total_tokens']:>12,} {mm['total_tokens']:>12,} {mm['total_tokens']-cm['total_tokens']:>+12,}")
        print(f"    {'Input tokens':<20} {cm['input_tokens']:>12,} {mm['input_tokens']:>12,} {mm['input_tokens']-cm['input_tokens']:>+12,}")
        print(f"    {'Time (s)':<20} {cm['elapsed_seconds']:>12.1f} {mm['elapsed_seconds']:>12.1f} {mm['elapsed_seconds']-cm['elapsed_seconds']:>+12.1f}")

    # Context window analysis
    print(f"\n{'CONTEXT WINDOW ANALYSIS':^100}")
    print(sep)
    print(f"  CLI-Optimized: 1 tool (run_cli) + compact system prompt (no README)")
    mcp_tc = mcp[0].get("tool_count", 0) if mcp else 0
    print(f"  MCP:           {mcp_tc} tools with JSON schemas auto-injected")
    print(f"  CLI-Optimized avg input tokens: {avg(cli, 'input_tokens'):>10,.0f}")
    print(f"  MCP avg input tokens:           {avg(mcp, 'input_tokens'):>10,.0f}")
    token_diff = avg(mcp, 'input_tokens') - avg(cli, 'input_tokens')
    print(f"  MCP schema overhead (avg):      {token_diff:>+10,.0f} tokens")


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    print(f"Benchmark: CLI-Optimized vs MCP")
    print(f"Tasks: {len(TASKS)}")
    print(f"MCP Server: {SERVER_SCRIPT}")
    print(f"Model: claude-sonnet-4-6\n")

    all_results: dict[str, list[dict]] = {a: [] for a, _ in APPROACHES}

    for i, task in enumerate(TASKS):
        print(f"\n{'='*70}")
        print(f"Task {i+1}/{len(TASKS)}: {task}")
        print(f"{'='*70}")

        for approach_name, runner in APPROACHES:
            print(f"\n  --- {approach_name} ---")
            try:
                metrics = await runner(task)
                all_results[approach_name].append(metrics)
                print(f"  Done: {metrics['tool_calls_made']} calls, "
                      f"{metrics['total_tokens']:,} tokens, "
                      f"{metrics['elapsed_seconds']}s")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[approach_name].append(
                    _empty_metrics(approach_name, task)
                )

    # Print comparison
    print_comparison(all_results)

    # Export results
    output = {
        "model": "claude-sonnet-4-6",
        "tasks": TASKS,
        "results": {k: v for k, v in all_results.items()},
    }
    out_file = WORKDIR / "benchmark_results.json"
    out_file.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults exported to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())