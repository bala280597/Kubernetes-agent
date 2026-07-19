# Kubernetes CLI Capabilities

## Available Operations

- Cluster Discovery (namespaces, nodes)
- Pod Health (status, restarts, readiness)
- Pod Diagnostics (describe, logs, events)
- Deployments, StatefulSets, DaemonSets, Jobs
- Services, Ingress, Endpoints
- ConfigMaps, Secrets (names/keys only)
- Storage (PVC, PV)
- Resource Usage (CPU, memory)
- Rollout Status and History
- HPA (autoscaler)

## Rules

- Prefer read-only operations. Never execute mutating commands (apply, delete, scale, edit, rollout undo, patch, replace, set, exec, cp) without explicit user confirmation.
- Never expose secret values. Report key names only.
- Use JSON output and Python-side parsing — do not ask the LLM to parse tables.
- Filter results: return only unhealthy or notable resources; summarize healthy ones with counts.
- Return concise structured summaries, not raw command output.

## Investigation Workflow

1. Discover namespaces.
2. Check pod health per namespace — focus on non-Running pods.
3. For unhealthy pods: describe + events + logs (previous if crashed).
4. For resource pressure: check top pods + top nodes.
5. Summarize: healthy count, failing pods, root cause, recommended actions.
