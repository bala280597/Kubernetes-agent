"""
MCP Server — Kubernetes Operations (kubectl).

Setup:
  pip install fastmcp
  # OR: pip install "mcp[cli]"

Test standalone:
  python mcp_kubectl_server.py

Used by mcp_kubectl_agent.py via stdio transport.
"""

import os
import shlex
import subprocess
import sys

# --------------------------------------------------------------------------
# Handle both import paths: standalone fastmcp vs mcp SDK bundle
# --------------------------------------------------------------------------
try:
    from fastmcp import FastMCP          # standalone fastmcp package
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP   # bundled in mcp SDK
    except ImportError:
        sys.exit(
            "Neither 'fastmcp' nor 'mcp' package found.\n"
            "Install one of:\n"
            "  pip install fastmcp\n"
            "  pip install \"mcp[cli]\"\n"
        )

mcp = FastMCP(
    "kubectl-server",
    instructions=(
        "Kubernetes operations server. Provides read-only kubectl tools "
        "for inspecting cluster state, pods, logs, events, and resources."
    ),
)

TIMEOUT = 120

# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------

def _run(cmd: str) -> str:
    """Run a kubectl command and return formatted output."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 120s."
    except Exception as e:
        return f"ERROR: {e}"
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else "")
    out = out.strip() or "(no output)"
    if len(out) > 20_000:
        out = out[:20_000] + "\n[... truncated ...]"
    return f"exit_code={proc.returncode}\n{out}"


# --------------------------------------------------------------------------
# Cluster-level tools
# --------------------------------------------------------------------------

@mcp.tool()
def list_namespaces() -> str:
    """List all namespaces in the cluster."""
    return _run("kubectl get namespaces")


@mcp.tool()
def get_nodes(wide: bool = True) -> str:
    """List cluster nodes with status and resource info."""
    cmd = "kubectl get nodes"
    if wide:
        cmd += " -o wide"
    return _run(cmd)


@mcp.tool()
def top_nodes() -> str:
    """Show CPU and memory usage per node (requires metrics-server)."""
    return _run("kubectl top nodes")


# --------------------------------------------------------------------------
# Workload inspection
# --------------------------------------------------------------------------

@mcp.tool()
def get_pods(namespace: str, wide: bool = True) -> str:
    """List pods in a namespace with status, restarts, and node placement."""
    cmd = f"kubectl get pods -n {shlex.quote(namespace)}"
    if wide:
        cmd += " -o wide"
    return _run(cmd)


@mcp.tool()
def get_all_pods_all_namespaces() -> str:
    """List pods across ALL namespaces — useful for cluster-wide health check."""
    return _run("kubectl get pods -A -o wide")


@mcp.tool()
def get_deployments(namespace: str) -> str:
    """List deployments and their replica counts in a namespace."""
    return _run(f"kubectl get deployments -n {shlex.quote(namespace)}")


@mcp.tool()
def get_all_resources(namespace: str) -> str:
    """List all resource types (pods, services, deployments, etc.) in a namespace."""
    return _run(f"kubectl get all -n {shlex.quote(namespace)}")


@mcp.tool()
def get_jobs(namespace: str) -> str:
    """List Jobs and CronJobs in a namespace."""
    return _run(
        f"kubectl get jobs -n {shlex.quote(namespace)} && "
        f"kubectl get cronjobs -n {shlex.quote(namespace)}"
    )


# --------------------------------------------------------------------------
# Pod diagnostics
# --------------------------------------------------------------------------

@mcp.tool()
def describe_pod(pod_name: str, namespace: str) -> str:
    """Describe a pod — shows events, conditions, mounts, environment."""
    return _run(f"kubectl describe pod {shlex.quote(pod_name)} -n {shlex.quote(namespace)}")


@mcp.tool()
def get_pod_logs(
    pod_name: str,
    namespace: str,
    tail: int = 100,
    previous: bool = False,
    container: str = "",
) -> str:
    """Get logs from a pod. Set previous=True for crash logs."""
    cmd = f"kubectl logs {shlex.quote(pod_name)} -n {shlex.quote(namespace)} --tail={tail}"
    if previous:
        cmd += " --previous"
    if container:
        cmd += f" -c {shlex.quote(container)}"
    return _run(cmd)


@mcp.tool()
def grep_pod_logs(pod_name: str, namespace: str, pattern: str = "error|exception|fatal", tail: int = 500) -> str:
    """Search pod logs for error patterns."""
    cmd = (
        f"kubectl logs {shlex.quote(pod_name)} -n {shlex.quote(namespace)} --tail={tail} "
        f"| grep -i {shlex.quote(pattern)}"
    )
    return _run(cmd)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

@mcp.tool()
def get_events(namespace: str) -> str:
    """Get recent events in a namespace, sorted by time."""
    return _run(f"kubectl get events -n {shlex.quote(namespace)} --sort-by=.lastTimestamp")


@mcp.tool()
def get_events_for_pod(pod_name: str, namespace: str) -> str:
    """Get events for a specific pod."""
    return _run(
        f"kubectl get events -n {shlex.quote(namespace)} "
        f"--field-selector involvedObject.name={shlex.quote(pod_name)}"
    )


# --------------------------------------------------------------------------
# Resource usage
# --------------------------------------------------------------------------

@mcp.tool()
def top_pods(namespace: str) -> str:
    """Show CPU and memory usage per pod."""
    return _run(f"kubectl top pods -n {shlex.quote(namespace)}")


@mcp.tool()
def get_hpa(namespace: str) -> str:
    """Show Horizontal Pod Autoscaler status."""
    return _run(f"kubectl get hpa -n {shlex.quote(namespace)}")


# --------------------------------------------------------------------------
# Config & networking
# --------------------------------------------------------------------------

@mcp.tool()
def get_configmaps(namespace: str) -> str:
    """List ConfigMaps in a namespace."""
    return _run(f"kubectl get configmaps -n {shlex.quote(namespace)}")


@mcp.tool()
def get_services(namespace: str) -> str:
    """List Services with their ClusterIPs and ports."""
    return _run(f"kubectl get svc -n {shlex.quote(namespace)}")


@mcp.tool()
def get_ingress(namespace: str) -> str:
    """List Ingress rules in a namespace."""
    return _run(f"kubectl get ingress -n {shlex.quote(namespace)}")


@mcp.tool()
def get_pvc(namespace: str) -> str:
    """List PersistentVolumeClaims in a namespace."""
    return _run(f"kubectl get pvc -n {shlex.quote(namespace)}")


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------

@mcp.tool()
def rollout_status(deployment_name: str, namespace: str) -> str:
    """Check rollout progress of a deployment."""
    return _run(
        f"kubectl rollout status deployment/{shlex.quote(deployment_name)} "
        f"-n {shlex.quote(namespace)} --timeout=30s"
    )


@mcp.tool()
def rollout_history(deployment_name: str, namespace: str) -> str:
    """Show rollout history of a deployment."""
    return _run(
        f"kubectl rollout history deployment/{shlex.quote(deployment_name)} "
        f"-n {shlex.quote(namespace)}"
    )


# --------------------------------------------------------------------------
# Mutation proposals (safe — does NOT execute)
# --------------------------------------------------------------------------

@mcp.tool()
def suggest_mutation(command: str, reason: str) -> str:
    """Propose a mutating kubectl command (apply, delete, scale, etc.).
    This does NOT execute — returns the proposal for user review."""
    return (
        f"PROPOSED MUTATION (not executed):\n"
        f"  Command: {command}\n"
        f"  Reason:  {reason}\n\n"
        f"Please review and run this command manually if you agree."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
    else:
        mcp.run(transport="stdio")