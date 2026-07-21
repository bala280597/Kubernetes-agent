"""
Benchmark: Optimized CLI-manifest agent vs MCP agent.

Changes from original:
  - from langchain.agents import create_agent (LangChain 1.0+)
  - system_prompt= parameter (not prompt=)
  - shell=False (security fix from original shell=True)
  - LangSmith tracing via @traceable
  - recursion_limit on agent invoke
  - Per-approach LangSmith project names

Setup:
  pip install -U langchain langchain-anthropic langgraph langchain-mcp-adapters "mcp[cli]" langsmith
  export ANTHROPIC_API_KEY="your-key"
  export LANGSMITH_TRACING=true
  export LANGSMITH_API_KEY="ls__your-key"

Run:
  python benchmark.py
"""

import asyncio
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from langchain.agents import create_agent                     # LangChain 1.0+
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langsmith import traceable

WORKDIR = Path.cwd()
IS_WINDOWS = platform.system() == "Windows"
SERVER_SCRIPT = str(Path(__file__).parent / "mcp_kubectl_server.py")


def _setup_tracing(project_suffix: str):
    os.environ["LANGSMITH_PROJECT"] = f"k8s-benchmark-{project_suffix}"


def make_llm():
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2, max_tokens=4096)


# ======================================================================
# Discovery cache
# ======================================================================

_cache: dict[str, tuple[float, str]] = {}
CACHE_TTL = 300

_CACHEABLE_COMMANDS = [
    "kubectl get namespaces", "kubectl get namespace", "kubectl get ns",
    "kubectl get nodes", "kubectl get node", "kubectl api-resources",
]

def _cache_key(command):
    s = command.strip()
    for p in _CACHEABLE_COMMANDS:
        if s == p or s.startswith(p + " "): return p
    return None

def _get_cached(key):
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < CACHE_TTL: return data
        del _cache[key]
    return None

def _set_cache(key, data): _cache[key] = (time.time(), data)
def _clear_cache(): _cache.clear()


# ======================================================================
# Guardrails
# ======================================================================

ALLOWED_BINARIES = {"kubectl","helm","grep","jq","cat","echo","head","tail","wc","sort","uniq"}
BLOCKED_PATTERNS = (
    "kubectl delete","kubectl apply","kubectl scale","kubectl edit",
    "kubectl rollout undo","kubectl exec","kubectl cp","kubectl patch",
    "kubectl replace","kubectl set","rm -rf","shutdown","reboot",
)

def _cmd_allowed(command):
    lowered = command.lower().strip()
    for p in BLOCKED_PATTERNS:
        if p in lowered: return f"BLOCKED: '{p}'"
    try: base = Path(shlex.split(command)[0]).name
    except (ValueError, IndexError): return "BLOCKED: parse error"
    if base not in ALLOWED_BINARIES: return f"BLOCKED: '{base}' not allowed"
    return None


# ======================================================================
# JSON injection
# ======================================================================

_GET_PATTERN = re.compile(r"^kubectl\s+get\s+(?!-)", re.IGNORECASE)
_SKIP_JSON = re.compile(r"-o\s+(json|yaml|wide|jsonpath|custom-columns|name)", re.IGNORECASE)
_TOP_PATTERN = re.compile(r"^kubectl\s+top\s+", re.IGNORECASE)

def _maybe_inject_json(command):
    s = command.strip()
    if "|" in s or _TOP_PATTERN.match(s) or _SKIP_JSON.search(s): return s, False
    if _GET_PATTERN.match(s): return s + " -o json", True
    return s, False


# ======================================================================
# Output summarizers (same as multi_agent.py — extract to shared module for production)
# ======================================================================

def _safe_parse(raw):
    try: data = json.loads(raw)
    except json.JSONDecodeError: return None, raw
    if isinstance(data, list): return data, raw
    if isinstance(data, dict):
        if "items" in data: return data["items"], raw
        return [data], raw
    return None, raw

def _summarize_pods(raw):
    items, raw = _safe_parse(raw)
    if items is None: return raw
    if not items: return json.dumps({"total":0,"pods":[]},indent=2)
    s = {"total":len(items),"healthy":0,"unhealthy":[]}
    for pod in items:
        name=pod.get("metadata",{}).get("name","?")
        ns=pod.get("metadata",{}).get("namespace","?")
        phase=pod.get("status",{}).get("phase","?")
        css=pod.get("status",{}).get("containerStatuses",[])
        restarts=sum(c.get("restartCount",0) for c in css)
        ready=all(c.get("ready",False) for c in css) if css else False
        waits=[c.get("state",{}).get("waiting",{}).get("reason") for c in css if c.get("state",{}).get("waiting",{}).get("reason")]
        for c in pod.get("status",{}).get("initContainerStatuses",[]):
            w=c.get("state",{}).get("waiting",{}).get("reason")
            if w: waits.append(f"init:{w}")
        ok = phase in ("Running","Succeeded") and ready and not waits and restarts<5
        if ok: s["healthy"]+=1
        else:
            info={"name":name,"namespace":ns,"phase":phase,"ready":ready,"restarts":restarts}
            if waits: info["waitingReasons"]=waits
            s["unhealthy"].append(info)
    return json.dumps(s,indent=2)

def _summarize_nodes(raw):
    items, raw = _safe_parse(raw)
    if items is None: return raw
    nodes=[]
    for n in items:
        name=n.get("metadata",{}).get("name","?")
        conds=n.get("status",{}).get("conditions",[])
        ready="Unknown"; issues=[]
        for c in conds:
            if c.get("type")=="Ready": ready=c.get("status","?")
            elif c.get("status")=="True" and c.get("type")!="Ready": issues.append(c["type"])
        ni=n.get("status",{}).get("nodeInfo",{})
        nodes.append({"name":name,"ready":ready,"kubeletVersion":ni.get("kubeletVersion",""),
                      "os":ni.get("osImage",""),"issues":issues or None})
    return json.dumps({"total":len(nodes),"nodes":nodes},indent=2)

def _summarize_namespaces(raw):
    items, raw = _safe_parse(raw)
    if items is None: return raw
    return json.dumps({"total":len(items),"namespaces":[
        {"name":i.get("metadata",{}).get("name",""),"status":i.get("status",{}).get("phase","")} for i in items]},indent=2)

def _summarize_deployments(raw):
    items, raw = _safe_parse(raw)
    if items is None: return raw
    bad=[]
    for d in items:
        desired=d.get("spec",{}).get("replicas",0); st=d.get("status",{})
        rdy=st.get("readyReplicas",0); avail=st.get("availableReplicas",0)
        if rdy!=desired or avail!=desired:
            bad.append({"name":d.get("metadata",{}).get("name",""),"namespace":d.get("metadata",{}).get("namespace",""),
                        "desired":desired,"ready":rdy,"available":avail})
    return json.dumps({"total":len(items),"healthy":len(items)-len(bad),"unhealthy":bad},indent=2)

def _summarize_events(raw):
    items, raw = _safe_parse(raw)
    if items is None: return raw
    warns=[]; normal=0
    for e in items:
        if e.get("type","Normal")!="Normal":
            warns.append({"reason":e.get("reason",""),"message":e.get("message","")[:200],
                          "object":e.get("involvedObject",{}).get("name",""),"count":e.get("count",1),
                          "lastSeen":e.get("lastTimestamp","")})
        else: normal+=1
    warns.sort(key=lambda x:x.get("lastSeen",""),reverse=True)
    return json.dumps({"normalEventCount":normal,"warnings":warns[:20]},indent=2)

def _summarize_services(raw):
    items, raw = _safe_parse(raw)
    if items is None: return raw
    svcs=[{"name":s.get("metadata",{}).get("name",""),"namespace":s.get("metadata",{}).get("namespace",""),
           "type":s.get("spec",{}).get("type",""),"clusterIP":s.get("spec",{}).get("clusterIP",""),
           "ports":[{"port":p.get("port"),"targetPort":p.get("targetPort")} for p in s.get("spec",{}).get("ports",[])]}
          for s in items]
    return json.dumps({"total":len(svcs),"services":svcs},indent=2)

def _summarize_generic(raw):
    items, raw = _safe_parse(raw)
    if items is None: return raw
    return json.dumps({"total":len(items),"items":[
        {"name":i.get("metadata",{}).get("name",""),"namespace":i.get("metadata",{}).get("namespace","")} for i in items]},indent=2)

def _summarize_describe(raw):
    lines=raw.splitlines(); result={}; events=[]; conditions=[]; in_ev=in_cond=False
    for line in lines:
        s=line.strip()
        if s.startswith("Status:"): result["status"]=s.split(":",1)[1].strip()
        elif s.startswith("Restart Count:"):
            try: result["restartCount"]=int(s.split(":",1)[1].strip())
            except ValueError: pass
        elif s.startswith("Reason:"): result["reason"]=s.split(":",1)[1].strip()
        elif s.startswith("Message:"): result["message"]=s.split(":",1)[1].strip()
        if s.startswith("Conditions:"): in_cond,in_ev=True,False; continue
        elif s.startswith("Events:"): in_ev,in_cond=True,False; continue
        elif s and not s.startswith(" ") and ":" in s: in_ev=in_cond=False
        if in_ev and s and not s.startswith("Type"): events.append(s[:150])
        if in_cond and s and not s.startswith("Type"): conditions.append(s[:150])
    if events: result["events"]=events[-10:]
    if conditions: result["conditions"]=conditions
    return json.dumps(result,indent=2) if result else raw[:3000]

def _summarize_logs(raw):
    lines=raw.splitlines()
    ep=re.compile(r"error|exception|fatal|panic|traceback|oom|killed",re.IGNORECASE)
    wp=re.compile(r"warn",re.IGNORECASE)
    errors=[s[:200] for line in lines if (s:=line.strip()) and ep.search(s)]
    warns=[s[:200] for line in lines if (s:=line.strip()) and wp.search(s) and not ep.search(s)]
    def dedup(items,limit):
        seen,out=set(),[]
        for i in items:
            k=i[:80]
            if k not in seen: seen.add(k); out.append(i)
        return out[:limit]
    return json.dumps({"totalLines":len(lines),"errorCount":len(errors),"warningCount":len(warns),
                       "uniqueErrors":dedup(errors,15),"uniqueWarnings":dedup(warns,10)},indent=2)

def _matches(cmd,patterns): return any(p in cmd for p in patterns)

def _detect_and_summarize(command, raw, injected):
    c=command.lower().strip()
    if injected or "-o json" in c or "-o=json" in c:
        if _matches(c,["get pods","get pod","get po"]): return _summarize_pods(raw)
        elif _matches(c,["get nodes","get node","get no "]): return _summarize_nodes(raw)
        elif _matches(c,["get namespaces","get namespace","get ns"]): return _summarize_namespaces(raw)
        elif _matches(c,["get deployments","get deployment","get deploy"]): return _summarize_deployments(raw)
        elif _matches(c,["get events","get event","get ev "]): return _summarize_events(raw)
        elif _matches(c,["get svc","get services","get service"]): return _summarize_services(raw)
        else: return _summarize_generic(raw)
    if _matches(c,["describe pod","describe pods"]): return _summarize_describe(raw)
    elif _matches(c,["logs ","log "]): return _summarize_logs(raw)
    elif _matches(c,["describe "]): return _summarize_describe(raw)
    return raw[:5000]+"\n[... truncated ...]" if len(raw)>5000 else raw


# ======================================================================
# CLI tool — shell=False (FIXED from original shell=True)
# ======================================================================

@tool
@traceable(name="run_cli_optimized_tool", metadata={"agent": "cli"})
def run_cli_optimized(command: str) -> str:
    """Execute a kubectl or helm command. Returns structured JSON summaries."""
    err = _cmd_allowed(command)
    if err: return err
    ckey = _cache_key(command)
    if ckey:
        cached = _get_cached(ckey)
        if cached: return cached
    actual, injected = _maybe_inject_json(command)
    try:
        args = shlex.split(actual, posix=not IS_WINDOWS)
        proc = subprocess.run(args, shell=False, capture_output=True,
                              text=True, timeout=120, cwd=WORKDIR)
    except subprocess.TimeoutExpired:
        return json.dumps({"success":False,"error":"Command timed out after 120s"})
    except FileNotFoundError:
        return json.dumps({"success":False,"error":f"Command not found"})
    if proc.returncode != 0:
        return json.dumps({"success":False,"exitCode":proc.returncode,
                           "error":(proc.stderr or "").strip()[:1000] or "Command failed"},indent=2)
    stdout = proc.stdout or ""
    if not stdout.strip():
        return json.dumps({"success":True,"result":"No resources found"})
    summarized = _detect_and_summarize(command, stdout, injected)
    if ckey: _set_cache(ckey, summarized)
    return summarized


# ======================================================================
# Metrics
# ======================================================================

@traceable(name="extract_metrics", run_type="parser")
def _extract_metrics(result, approach, task, tool_count, elapsed):
    messages = result["messages"]
    inp_tok=out_tok=tool_calls=0
    for msg in messages:
        if hasattr(msg,"tool_calls") and msg.tool_calls: tool_calls+=len(msg.tool_calls)
        usage = getattr(msg,"usage_metadata",None)
        if usage: inp_tok+=usage.get("input_tokens",0); out_tok+=usage.get("output_tokens",0)
    return {"approach":approach,"task":task,"tool_count":tool_count,
            "tool_calls_made":tool_calls,"input_tokens":inp_tok,"output_tokens":out_tok,
            "total_tokens":inp_tok+out_tok,"elapsed_seconds":round(elapsed,2),
            "message_count":len(messages),"answer_length":len(messages[-1].content) if messages else 0}


# ======================================================================
# Agent runners
# ======================================================================

CLI_OPTIMIZED_PROMPT = """\
You are an autonomous Kubernetes operations agent.

You interact with the cluster through the run_cli_optimized tool, which
executes kubectl and helm commands. The tool automatically:
- Converts kubectl get output to structured JSON summaries
- Filters to show only unhealthy/notable resources
- Extracts key fields from describe output
- Extracts errors and warnings from logs

Workflow:
1. Read the user's task.
2. Plan kubectl commands (always start read-only).
3. Call run_cli_optimized for each command, inspect the structured response.
4. Chain logically: find unhealthy → describe → logs → events.
5. Summarize: what you checked, findings, root cause, recommended actions.

Rules:
- Trust the tool's summaries.
- For mutating commands, do NOT execute. Report and ask to confirm.
- Never expose secret values.
- Be concise."""


@traceable(name="cli_optimized_agent_run", run_type="chain", metadata={"approach":"CLI-Optimized"})
async def run_cli_optimized_agent(task):
    _setup_tracing("cli-optimized")
    _clear_cache()
    llm = make_llm()
    agent = create_agent(model=llm, tools=[run_cli_optimized],
                         system_prompt=CLI_OPTIMIZED_PROMPT, name="cli_agent")
    t0 = time.perf_counter()
    result = await agent.ainvoke(
        {"messages": [{"role":"user","content":task}]},
        config={"recursion_limit": 25})
    elapsed = time.perf_counter() - t0
    return _extract_metrics(result, "CLI-Optimized", task, tool_count=1, elapsed=elapsed)


@traceable(name="mcp_agent_run", run_type="chain", metadata={"approach":"MCP"})
async def run_mcp_agent(task):
    _setup_tracing("mcp")
    server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])
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
            agent = create_agent(model=llm, tools=tools,
                                 system_prompt=system_prompt, name="mcp_agent")
            t0 = time.perf_counter()
            result = await agent.ainvoke(
                {"messages": [{"role":"user","content":task}]},
                config={"recursion_limit": 25})
            elapsed = time.perf_counter() - t0
            return _extract_metrics(result, "MCP", task, tool_count=tool_count, elapsed=elapsed)


# ======================================================================
# Benchmark harness
# ======================================================================

TASKS = [
    "List all namespaces and show pod health in each",
    "Find any pods not in Running state and diagnose why",
    "Check resource usage across all nodes",
    "Show the services and ingress rules in the default namespace",
    "Get events in kube-system namespace sorted by time",
]

APPROACHES = [("CLI-Optimized", run_cli_optimized_agent), ("MCP", run_mcp_agent)]

def _empty_metrics(approach, task):
    return {"approach":approach,"task":task,"tool_count":0,"tool_calls_made":0,
            "input_tokens":0,"output_tokens":0,"total_tokens":0,
            "elapsed_seconds":0,"message_count":0,"answer_length":0}

def print_comparison(all_results):
    sep="-"*100
    print(f"\n{'='*100}\n{'BENCHMARK RESULTS':^100}\n{'='*100}\n")
    def avg(lst,key):
        vals=[m[key] for m in lst]
        return sum(vals)/len(vals) if vals else 0
    cli=all_results["CLI-Optimized"]; mcp=all_results["MCP"]
    print(f"  {'Metric':<30} {'CLI-Optimized':>15} {'MCP':>15} {'Diff':>15} {'Winner':>12}")
    print(sep)
    for label,key in [("Avg input tokens","input_tokens"),("Avg output tokens","output_tokens"),
                      ("Avg total tokens","total_tokens"),("Avg tool calls","tool_calls_made"),
                      ("Avg elapsed (s)","elapsed_seconds"),("Avg answer length","answer_length"),
                      ("Avg messages","message_count")]:
        cv=avg(cli,key); mv=avg(mcp,key); diff=mv-cv
        winner = "MCP" if key=="answer_length" and mv>cv else ("CLI" if cv<=mv else "MCP")
        print(f"  {label:<28} {cv:>15,.1f} {mv:>15,.1f} {diff:>+14,.1f} {winner:>12}")
    print(sep)
    print(f"\n{'PER-TASK BREAKDOWN':^100}\n{sep}")
    for i,task in enumerate(TASKS):
        cm=cli[i]; mm=mcp[i]
        print(f"\n  Task {i+1}: {task[:70]}")
        print(f"    {'':20} {'CLI-Opt':>12} {'MCP':>12} {'Diff':>12}")
        print(f"    {'Tool calls':<20} {cm['tool_calls_made']:>12} {mm['tool_calls_made']:>12} {mm['tool_calls_made']-cm['tool_calls_made']:>+12}")
        print(f"    {'Total tokens':<20} {cm['total_tokens']:>12,} {mm['total_tokens']:>12,} {mm['total_tokens']-cm['total_tokens']:>+12,}")
        print(f"    {'Time (s)':<20} {cm['elapsed_seconds']:>12.1f} {mm['elapsed_seconds']:>12.1f} {mm['elapsed_seconds']-cm['elapsed_seconds']:>+12.1f}")


@traceable(name="benchmark_harness", run_type="chain")
async def main():
    if not os.getenv("ANTHROPIC_API_KEY"): sys.exit("Set ANTHROPIC_API_KEY first.")
    print(f"Benchmark: CLI-Optimized vs MCP\nTasks: {len(TASKS)}\nModel: claude-sonnet-4-6\n")
    all_results = {a:[] for a,_ in APPROACHES}
    for i,task in enumerate(TASKS):
        print(f"\n{'='*70}\nTask {i+1}/{len(TASKS)}: {task}\n{'='*70}")
        for name,runner in APPROACHES:
            print(f"\n  --- {name} ---")
            try:
                metrics = await runner(task)
                all_results[name].append(metrics)
                print(f"  Done: {metrics['tool_calls_made']} calls, {metrics['total_tokens']:,} tokens, {metrics['elapsed_seconds']}s")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[name].append(_empty_metrics(name, task))
    print_comparison(all_results)
    output = {"model":"claude-sonnet-4-6","tasks":TASKS,"results":all_results}
    out_file = WORKDIR / "benchmark_results.json"
    out_file.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults exported to: {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
