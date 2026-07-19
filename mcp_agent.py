"""
MCP-based Kubernetes agent — LangGraph + Claude API.

This version runs the MCP server IN-PROCESS (no subprocess, no stdio),
which avoids Windows asyncio/stdio transport issues entirely.

Setup:
  pip install -U langgraph langchain-anthropic langchain-mcp-adapters fastmcp
  set ANTHROPIC_API_KEY=your-key

Run:
  python mcp_kubectl_agent.py "List all namespaces and their pod health"
"""

import asyncio
import os
import sys
import time
import platform

# Windows asyncio fix
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

# --------------------------------------------------------------------------
# Import the MCP server's tools DIRECTLY (in-process, no stdio)
# --------------------------------------------------------------------------
# Add the server's directory to path so we can import it
SERVER_DIR = str(Path(__file__).parent)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

# Import the server module — this gives us all the @mcp.tool functions
import mcp_kubectl_server as server

# --------------------------------------------------------------------------
# Convert MCP server functions to LangChain tools (manual adapter)
# --------------------------------------------------------------------------

def mcp_tools_to_langchain():
    """Extract all @mcp.tool decorated functions and wrap as LangChain tools."""
    import inspect

    # FastMCP stores tools internally — we grab the raw functions
    tool_functions = [
        server.list_namespaces,
        server.get_nodes,
        server.top_nodes,
        server.get_pods,
        server.get_all_pods_all_namespaces,
        server.get_deployments,
        server.get_all_resources,
        server.get_jobs,
        server.describe_pod,
        server.get_pod_logs,
        server.grep_pod_logs,
        server.get_events,
        server.get_events_for_pod,
        server.top_pods,
        server.get_hpa,
        server.get_configmaps,
        server.get_services,
        server.get_ingress,
        server.get_pvc,
        server.rollout_status,
        server.rollout_history,
        server.suggest_mutation,
    ]

    lc_tools = []
    for func in tool_functions:
        # StructuredTool.from_function auto-generates the schema from type hints
        tool = StructuredTool.from_function(
            func=func,
            name=func.__name__,
            description=func.__doc__ or f"Run {func.__name__}",
        )
        lc_tools.append(tool)

    return lc_tools


TOOLS = mcp_tools_to_langchain()
print(f"[init] Loaded {len(TOOLS)} tools: {[t.name for t in TOOLS]}")

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.2,
    max_tokens=4096,
)

# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous Kubernetes operations agent.

You have typed tools for kubectl operations. Each tool maps to a specific
kubectl command with validated parameters.

Workflow:
1. Read the user's task.
2. Start with discovery tools (list_namespaces, get_nodes) to understand the cluster.
3. Drill into specific namespaces with get_pods, then describe_pod/get_pod_logs for failing pods.
4. Chain tools logically — find problems first, then diagnose root causes.
5. When done, give a structured summary: what you checked, what you found,
   issues and root causes, and recommended next steps.

IMPORTANT: For any mutating action, use suggest_mutation to PROPOSE the
command. Never try to run destructive commands directly.
"""

# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
)

# --------------------------------------------------------------------------
# Runner with metrics
# --------------------------------------------------------------------------

async def run_agent(task: str) -> dict:
    t0 = time.perf_counter()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": task}]}
    )
    elapsed = time.perf_counter() - t0

    messages = result["messages"]
    total_input_tokens = 0
    total_output_tokens = 0
    tool_call_count = 0

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_count += len(msg.tool_calls)
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)

    return {
        "approach": "MCP (in-process)",
        "task": task,
        "tool_count": len(TOOLS),
        "tool_names": [t.name for t in TOOLS],
        "tool_calls_made": tool_call_count,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "message_count": len(messages),
        "final_answer": messages[-1].content,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    if not task:
        print("\nNo task provided. Examples:\n")
        print('  python mcp_kubectl_agent.py "List all namespaces and their pod health"')
        print('  python mcp_kubectl_agent.py "Find pods in CrashLoopBackOff and diagnose"')
        print('  python mcp_kubectl_agent.py "Check resource usage across all nodes"')
        print()
        task = input("Enter your task: ").strip()
        if not task:
            sys.exit("No task entered.")

    task = task.strip('"').strip("'")

    print(f"\n[agent] Task: {task}")
    print("[agent] Starting autonomous execution...\n")

    try:
        metrics = asyncio.run(run_agent(task))
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60 + "\n")
    print(metrics["final_answer"])

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"  Tools available  : {metrics['tool_count']}")
    print(f"  Tool calls made  : {metrics['tool_calls_made']}")
    print(f"  Input tokens     : {metrics['input_tokens']:,}")
    print(f"  Output tokens    : {metrics['output_tokens']:,}")
    print(f"  Total tokens     : {metrics['total_tokens']:,}")
    print(f"  Wall-clock time  : {metrics['elapsed_seconds']}s")
    print(f"  Messages         : {metrics['message_count']}")


if __name__ == "__main__":
    main()