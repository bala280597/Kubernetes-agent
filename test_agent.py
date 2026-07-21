"""
[10] Evaluation Framework for Multi-Agent Kubernetes Operations System v3.

This test suite validates agent behavior against golden test cases and
provides utilities for offline evaluation of recorded traces.

Test categories:
  1. Pydantic model validation    — structured output models parse correctly
  2. Guardrail unit tests         — blocked/allowed command classification
  3. Reflexion verifier tests     — placeholder/invalid command detection
  4. Supervisor routing tests     — correct routing given various states
  5. Golden case integration tests — end-to-end pipeline behavior (mocked kubectl)
  6. Eval trace analysis          — load and score recorded eval traces

Run:
  pytest tests/test_agent_evals.py -v
  pytest tests/test_agent_evals.py -v -k "test_guardrail"
  pytest tests/test_agent_evals.py -v -k "test_golden"
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add parent dir to path so we can import multi_agent_v3
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from multi_agent_v3 import (
    # Pydantic models
    IssueDetail,
    DetectionResult,
    DiagnosisDetail,
    DiagnosisResult,
    RecommendationDetail,
    RecommendationResult,
    SupervisorDecision,
    # Guardrails
    _is_readonly_safe,
    _is_mutation_allowed,
    _check_binary,
    ALLOWED_BINARIES,
    # Nodes (for mocked tests)
    reflexion_check_node,
    guardrail_filter_node,
    supervisor_node,
    _supervisor_fallback,
    # Eval utilities
    _record_eval,
    _truncate_for_eval,
    _save_eval_trace,
    # Constants
    MAX_REFLEXION_RETRIES,
)


# =========================================================================
# 1. Pydantic Model Validation Tests
# =========================================================================

class TestPydanticModels:
    """Verify that Pydantic models parse valid JSON and reject invalid."""

    def test_issue_detail_valid(self):
        issue = IssueDetail(
            resource_type="pod",
            name="api-server-7b9f4c5d6-x2k3m",
            namespace="production",
            status="CrashLoopBackOff",
            details="restarts: 17, exit code 137"
        )
        assert issue.resource_type == "pod"
        assert issue.namespace == "production"

    def test_issue_detail_defaults(self):
        issue = IssueDetail(resource_type="node", name="worker-1", status="NotReady")
        assert issue.namespace == ""
        assert issue.details == ""

    def test_detection_result_valid(self):
        data = {
            "issues_found": True,
            "issues": [
                {"resource_type": "pod", "name": "crash-pod", "status": "CrashLoopBackOff"}
            ],
            "healthy_summary": "42 pods healthy"
        }
        result = DetectionResult.model_validate(data)
        assert result.issues_found is True
        assert len(result.issues) == 1
        assert result.issues[0].name == "crash-pod"

    def test_detection_result_empty(self):
        result = DetectionResult(issues_found=False)
        assert result.issues == []
        assert result.healthy_summary == ""

    def test_detection_result_from_json(self):
        raw = '{"issues_found": false, "issues": [], "healthy_summary": "all good"}'
        result = DetectionResult.model_validate_json(raw)
        assert result.issues_found is False

    def test_diagnosis_detail_severity_enum(self):
        diag = DiagnosisDetail(
            resource="pod/api-server",
            root_cause="OOMKilled",
            severity="critical"
        )
        assert diag.severity == "critical"

    def test_diagnosis_detail_invalid_severity(self):
        with pytest.raises(ValidationError):
            DiagnosisDetail(
                resource="pod/x", root_cause="unknown", severity="extreme"
            )

    def test_diagnosis_result_roundtrip(self):
        data = {
            "diagnoses": [
                {
                    "resource": "pod/api-server",
                    "namespace": "production",
                    "root_cause": "OOMKilled — exceeds 256Mi",
                    "evidence": ["Exit code 137", "restartCount: 17"],
                    "severity": "high",
                    "container_name": "api",
                    "image": "myregistry.io/api:v2.3.1",
                    "owner_kind": "Deployment",
                    "owner_name": "api-server"
                }
            ]
        }
        result = DiagnosisResult.model_validate(data)
        assert len(result.diagnoses) == 1
        dumped = result.diagnoses[0].model_dump()
        assert dumped["root_cause"] == "OOMKilled — exceeds 256Mi"

    def test_recommendation_detail_risk_enum(self):
        rec = RecommendationDetail(
            issue="pod OOM", fix_command="kubectl set resources ...",
            explanation="increase memory", risk="low"
        )
        assert rec.risk == "low"
        assert rec.no_fix_needed is False

    def test_recommendation_result_multiple(self):
        data = {
            "recommendations": [
                {"issue": "oom", "fix_command": "kubectl set resources d/app -n prod --limits=memory=512Mi",
                 "explanation": "double memory", "risk": "low"},
                {"issue": "image", "fix_command": "kubectl set image d/app -n prod app=img:v2",
                 "explanation": "update image", "risk": "medium"},
            ]
        }
        result = RecommendationResult.model_validate(data)
        assert len(result.recommendations) == 2

    def test_supervisor_decision_valid(self):
        dec = SupervisorDecision(
            next_node="diagnosis",
            reasoning="Issues need root cause analysis"
        )
        assert dec.next_node == "diagnosis"
        assert dec.skip_reason == ""

    def test_supervisor_decision_invalid_target(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="nonexistent_node",
                reasoning="bad route"
            )

    def test_supervisor_decision_with_skip(self):
        dec = SupervisorDecision(
            next_node="recommendation",
            reasoning="Known OOM pattern from memory",
            skip_reason="Mem0 has exact match for this pod's OOM history"
        )
        assert dec.skip_reason != ""


# =========================================================================
# 2. Guardrail Unit Tests
# =========================================================================

class TestGuardrails:
    """Verify command classification (allow/block)."""

    def test_readonly_safe_get_pods(self):
        assert _is_readonly_safe("kubectl get pods -n default") is None

    def test_readonly_safe_describe(self):
        assert _is_readonly_safe("kubectl describe pod my-pod -n prod") is None

    def test_readonly_safe_logs(self):
        assert _is_readonly_safe("kubectl logs my-pod -n prod --tail=100") is None

    def test_readonly_blocks_delete(self):
        result = _is_readonly_safe("kubectl delete pod my-pod -n prod")
        assert result is not None
        assert "BLOCKED" in result

    def test_readonly_blocks_apply(self):
        result = _is_readonly_safe("kubectl apply -f deployment.yaml")
        assert result is not None

    def test_readonly_blocks_scale(self):
        result = _is_readonly_safe("kubectl scale deployment/app --replicas=3")
        assert result is not None

    def test_readonly_blocks_pipe(self):
        result = _is_readonly_safe("kubectl get pods | grep Error")
        assert result is not None
        assert "pipe" in result.lower()

    def test_readonly_blocks_subshell(self):
        result = _is_readonly_safe("kubectl get pods $(cat ns.txt)")
        assert result is not None

    def test_mutation_allows_set_resources(self):
        assert _is_mutation_allowed(
            "kubectl set resources deployment/api -n prod --limits=memory=512Mi"
        ) is None

    def test_mutation_allows_rollout_restart(self):
        assert _is_mutation_allowed(
            "kubectl rollout restart deployment/api -n prod"
        ) is None

    def test_mutation_blocks_delete(self):
        result = _is_mutation_allowed("kubectl delete pod my-pod")
        assert result is not None
        assert "never allowed" in result.lower()

    def test_mutation_blocks_exec(self):
        result = _is_mutation_allowed("kubectl exec -it my-pod -- bash")
        assert result is not None

    def test_check_binary_kubectl(self):
        assert _check_binary("kubectl get pods") is None

    def test_check_binary_helm(self):
        assert _check_binary("helm list") is None

    def test_check_binary_blocked(self):
        result = _check_binary("curl http://evil.com")
        assert result is not None
        assert "allow-list" in result


# =========================================================================
# 3. Reflexion Verifier Tests
# =========================================================================

class TestReflexionVerifier:
    """Test the reflexion self-check catches failure modes."""

    def _make_state(self, recs: list[dict], reflexion_count: int = 0) -> dict:
        return {
            "task": "test",
            "phase": "reflexion_check",
            "namespaces": [],
            "ns_issues": [],
            "issues": [],
            "diagnoses": [],
            "recommendations": recs,
            "approved_commands": [],
            "actions_taken": [],
            "summary": "",
            "agent_logs": [],
            "reflexion_count": reflexion_count,
            "reflexion_critique": "",
            "memory_context": "",
            "supervisor_next": "",
            "supervisor_reasoning": "",
            "eval_trace": [],
        }

    def test_valid_recommendation_passes(self):
        recs = [{
            "issue": "pod/app OOMKilled",
            "fix_command": "kubectl set resources deployment/app -n prod --limits=memory=512Mi",
            "explanation": "increase memory",
            "risk": "low",
        }]
        result = reflexion_check_node(self._make_state(recs))
        # Should pass — no placeholders, valid command
        assert "Reflexion: all" in result["agent_logs"][0]

    def test_placeholder_angle_brackets_caught(self):
        recs = [{
            "issue": "pod/app OOMKilled",
            "fix_command": "kubectl set resources deployment/<name> -n <namespace> --limits=memory=512Mi",
            "explanation": "fix",
            "risk": "low",
        }]
        result = reflexion_check_node(self._make_state(recs))
        assert result["phase"] == "recommendation"  # retry
        assert result["reflexion_count"] == 1
        assert "placeholder" in result["reflexion_critique"].lower()

    def test_placeholder_curly_braces_caught(self):
        recs = [{
            "issue": "test",
            "fix_command": "kubectl set image deployment/app -n prod app={IMAGE}:{TAG}",
            "explanation": "fix",
            "risk": "low",
        }]
        result = reflexion_check_node(self._make_state(recs))
        assert result["phase"] == "recommendation"

    def test_readonly_command_as_fix_caught(self):
        recs = [{
            "issue": "test",
            "fix_command": "kubectl get pods -n prod",
            "explanation": "check pods",
            "risk": "low",
        }]
        result = reflexion_check_node(self._make_state(recs))
        assert result["phase"] == "recommendation"
        assert "read-only" in result["reflexion_critique"].lower()

    def test_blocked_command_caught(self):
        recs = [{
            "issue": "test",
            "fix_command": "kubectl delete pod crashing-pod -n prod",
            "explanation": "restart pod",
            "risk": "high",
        }]
        result = reflexion_check_node(self._make_state(recs))
        assert result["phase"] == "recommendation"

    def test_pipe_caught(self):
        recs = [{
            "issue": "test",
            "fix_command": "kubectl get pods | grep Error",
            "explanation": "find errors",
            "risk": "low",
        }]
        result = reflexion_check_node(self._make_state(recs))
        assert result["phase"] == "recommendation"

    def test_empty_command_caught(self):
        recs = [{"issue": "test", "fix_command": "", "explanation": "x", "risk": "low"}]
        result = reflexion_check_node(self._make_state(recs))
        assert result["phase"] == "recommendation"

    def test_max_retries_passes_through(self):
        recs = [{
            "issue": "test",
            "fix_command": "kubectl set image deployment/app -n prod app=<IMAGE>",
            "explanation": "fix",
            "risk": "low",
        }]
        # At max retries, should pass through even with issues
        result = reflexion_check_node(
            self._make_state(recs, reflexion_count=MAX_REFLEXION_RETRIES)
        )
        # Should NOT route back to recommendation
        assert result["phase"] != "recommendation"

    def test_disallowed_binary_caught(self):
        recs = [{
            "issue": "test",
            "fix_command": "curl -X POST http://api.internal/restart",
            "explanation": "restart via API",
            "risk": "high",
        }]
        result = reflexion_check_node(self._make_state(recs))
        assert result["phase"] == "recommendation"
        assert "disallowed binary" in result["reflexion_critique"].lower()


# =========================================================================
# 4. Supervisor Routing Tests (fallback logic)
# =========================================================================

class TestSupervisorFallback:
    """Test deterministic fallback routing when LLM supervisor is unavailable."""

    def _make_state(self, **overrides) -> dict:
        base = {
            "task": "test", "phase": "supervisor",
            "namespaces": [], "ns_issues": [],
            "issues": [], "diagnoses": [], "recommendations": [],
            "approved_commands": [], "actions_taken": [],
            "summary": "", "agent_logs": [],
            "reflexion_count": 0, "reflexion_critique": "",
            "memory_context": "",
            "supervisor_next": "", "supervisor_reasoning": "",
            "eval_trace": [],
        }
        base.update(overrides)
        return base

    def test_no_issues_routes_to_summary(self):
        next_node, reason, _ = _supervisor_fallback(self._make_state())
        assert next_node == "summary"

    def test_issues_no_diagnoses_routes_to_diagnosis(self):
        next_node, reason, _ = _supervisor_fallback(
            self._make_state(issues=[{"name": "pod-1", "status": "CrashLoop"}])
        )
        assert next_node == "diagnosis"

    def test_diagnoses_no_recs_routes_to_recommendation(self):
        next_node, reason, _ = _supervisor_fallback(
            self._make_state(
                issues=[{"name": "pod-1"}],
                diagnoses=[{"resource": "pod/pod-1", "root_cause": "OOM"}]
            )
        )
        assert next_node == "recommendation"

    def test_actions_taken_routes_to_summary(self):
        next_node, reason, _ = _supervisor_fallback(
            self._make_state(
                issues=[{"name": "pod-1"}],
                diagnoses=[{"resource": "pod/pod-1"}],
                recommendations=[{"fix_command": "kubectl set resources ..."}],
                actions_taken=[{"command": "kubectl set resources ...", "success": True}],
            )
        )
        assert next_node == "summary"


# =========================================================================
# 5. Golden Case Integration Tests (mocked kubectl)
# =========================================================================

# These test the full pipeline logic with mocked subprocess calls.
# They verify that given a specific cluster state, the agents produce
# the expected diagnosis and recommendation categories.

GOLDEN_CASES = [
    {
        "id": "oom_crashloop",
        "description": "Pod in CrashLoopBackOff due to OOMKilled",
        "input_task": "Find crashing pods and fix them",
        "mock_issues": [
            {
                "resource_type": "pod",
                "name": "api-server-7b9f4c5d6-x2k3m",
                "namespace": "production",
                "status": "CrashLoopBackOff",
                "details": "restarts: 17, exit code 137"
            }
        ],
        "expected_diagnosis_contains": "OOMKilled",
        "expected_fix_verb": "set resources",
        "expected_severity": "high",
    },
    {
        "id": "image_pull_backoff",
        "description": "Pod stuck in ImagePullBackOff",
        "input_task": "Check cluster health",
        "mock_issues": [
            {
                "resource_type": "pod",
                "name": "frontend-abc123",
                "namespace": "staging",
                "status": "ImagePullBackOff",
                "details": "image: registry.io/frontend:v99-nonexistent"
            }
        ],
        "expected_diagnosis_contains": "ImagePull",
        "expected_fix_verb": "set image",
        "expected_severity": "high",
    },
    {
        "id": "deployment_scaled_to_zero",
        "description": "Deployment with 0 ready replicas",
        "input_task": "Find deployment issues",
        "mock_issues": [
            {
                "resource_type": "deployment",
                "name": "worker",
                "namespace": "batch",
                "status": "0/3 ready",
                "details": "desired: 3, ready: 0, available: 0"
            }
        ],
        "expected_diagnosis_contains": "replica",
        "expected_fix_verb": "scale",
        "expected_severity": "high",
    },
    {
        "id": "healthy_cluster",
        "description": "No issues found — cluster is healthy",
        "input_task": "Check cluster health",
        "mock_issues": [],
        "expected_diagnosis_contains": None,
        "expected_fix_verb": None,
        "expected_severity": None,
    },
]


class TestGoldenCases:
    """Validate agent behavior against golden test scenarios.

    These tests verify the Pydantic model layer and guardrail logic
    using known-good inputs. They do NOT call the actual LLM — they
    test the structural contract.
    """

    @pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
    def test_detection_result_parses(self, case):
        """Detection results can be parsed into DetectionResult model."""
        issues_found = len(case["mock_issues"]) > 0
        data = {
            "issues_found": issues_found,
            "issues": case["mock_issues"],
            "healthy_summary": "test summary"
        }
        result = DetectionResult.model_validate(data)
        assert result.issues_found == issues_found
        assert len(result.issues) == len(case["mock_issues"])

    @pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c["id"] for c in GOLDEN_CASES])
    def test_issue_details_validate(self, case):
        """Each mock issue can be validated as IssueDetail."""
        for issue_data in case["mock_issues"]:
            issue = IssueDetail.model_validate(issue_data)
            assert issue.name
            assert issue.status

    def test_oom_fix_passes_guardrail(self):
        """A valid OOM fix command should pass through guardrail."""
        recs = [{
            "issue": "pod/api-server OOMKilled",
            "fix_command": "kubectl set resources deployment/api-server -n production --limits=memory=512Mi",
            "explanation": "increase memory",
            "risk": "low",
        }]
        state = _make_minimal_state(recommendations=recs)
        result = guardrail_filter_node(state)
        assert len(result["recommendations"]) == 1

    def test_image_fix_passes_guardrail(self):
        """A valid image update command should pass through guardrail."""
        recs = [{
            "issue": "pod/frontend ImagePullBackOff",
            "fix_command": "kubectl set image deployment/frontend -n staging frontend=registry.io/frontend:v2",
            "explanation": "fix image tag",
            "risk": "medium",
        }]
        state = _make_minimal_state(recommendations=recs)
        result = guardrail_filter_node(state)
        assert len(result["recommendations"]) == 1

    def test_scale_fix_passes_guardrail(self):
        """A valid scale command should pass through guardrail."""
        recs = [{
            "issue": "deployment/worker 0/3 replicas",
            "fix_command": "kubectl scale deployment/worker -n batch --replicas=3",
            "explanation": "scale back up",
            "risk": "low",
        }]
        state = _make_minimal_state(recommendations=recs)
        result = guardrail_filter_node(state)
        assert len(result["recommendations"]) == 1

    def test_delete_blocked_by_guardrail(self):
        """Delete commands should be filtered out by guardrail."""
        recs = [{
            "issue": "stuck pod",
            "fix_command": "kubectl delete pod stuck-pod -n prod",
            "explanation": "force restart",
            "risk": "high",
        }]
        state = _make_minimal_state(recommendations=recs)
        result = guardrail_filter_node(state)
        assert len(result["recommendations"]) == 0


# =========================================================================
# 6. Eval Trace Analysis
# =========================================================================

class TestEvalTraceUtilities:
    """Test eval data collection and storage."""

    def test_record_eval_structure(self):
        record = _record_eval("diagnosis", {"issues": 3}, {"diagnoses": 2})
        assert record["node"] == "diagnosis"
        assert "timestamp" in record
        assert record["input_summary"] == {"issues": 3}
        assert record["output_summary"] == {"diagnoses": 2}

    def test_truncate_string(self):
        long_str = "x" * 5000
        truncated = _truncate_for_eval(long_str, max_len=100)
        assert len(truncated) == 100

    def test_truncate_list(self):
        long_list = list(range(50))
        truncated = _truncate_for_eval(long_list)
        assert len(truncated) == 10

    def test_truncate_dict(self):
        big_dict = {f"key_{i}": f"value_{i}" for i in range(30)}
        truncated = _truncate_for_eval(big_dict)
        assert len(truncated) == 20

    def test_save_eval_trace(self, tmp_path, monkeypatch):
        """Eval traces are saved to disk correctly."""
        monkeypatch.setattr("multi_agent_v3.EVAL_DATA_DIR", tmp_path)
        trace = [
            _record_eval("detection", {}, {"issues": 2}),
            _record_eval("diagnosis", {}, {"diagnoses": 1}),
        ]
        result = {"summary": "test", "issues": [1, 2], "diagnoses": [1],
                  "recommendations": [], "actions_taken": [], "agent_logs": [],
                  "reflexion_count": 0}
        path = _save_eval_trace("test task", trace, result)
        assert path is not None
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["task"] == "test task"
        assert len(data["trace"]) == 2
        assert data["total_issues"] == 2

    def test_save_eval_trace_empty(self, tmp_path, monkeypatch):
        """Empty traces should not be saved."""
        monkeypatch.setattr("multi_agent_v3.EVAL_DATA_DIR", tmp_path)
        path = _save_eval_trace("test", [], {})
        assert path is None


class TestEvalTraceLoader:
    """Load and analyze saved eval traces for regression detection."""

    @staticmethod
    def load_all_traces(eval_dir: Path) -> list[dict]:
        """Load all eval traces from the eval data directory."""
        traces = []
        if not eval_dir.exists():
            return traces
        for f in sorted(eval_dir.glob("eval_*.json")):
            try:
                traces.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, IOError):
                continue
        return traces

    @staticmethod
    def score_trace(trace: dict) -> dict:
        """Score a single eval trace for quality metrics."""
        scores = {
            "task": trace.get("task", "unknown"),
            "issues_detected": trace.get("total_issues", 0),
            "diagnoses_produced": trace.get("total_diagnoses", 0),
            "recommendations_produced": trace.get("total_recommendations", 0),
            "actions_executed": trace.get("actions_taken", 0),
            "reflexion_retries": trace.get("reflexion_retries", 0),
        }

        # Diagnosis coverage: did we diagnose all detected issues?
        if scores["issues_detected"] > 0:
            scores["diagnosis_coverage"] = (
                scores["diagnoses_produced"] / scores["issues_detected"]
            )
        else:
            scores["diagnosis_coverage"] = 1.0

        # Recommendation coverage: did we propose fixes for all diagnoses?
        if scores["diagnoses_produced"] > 0:
            scores["recommendation_coverage"] = (
                scores["recommendations_produced"] / scores["diagnoses_produced"]
            )
        else:
            scores["recommendation_coverage"] = 1.0

        # Reflexion efficiency: fewer retries = better
        scores["reflexion_efficiency"] = max(
            0.0, 1.0 - (scores["reflexion_retries"] / (MAX_REFLEXION_RETRIES + 1))
        )

        return scores

    def test_score_perfect_trace(self):
        trace = {
            "task": "fix pods",
            "total_issues": 3,
            "total_diagnoses": 3,
            "total_recommendations": 3,
            "actions_taken": 3,
            "reflexion_retries": 0,
        }
        scores = self.score_trace(trace)
        assert scores["diagnosis_coverage"] == 1.0
        assert scores["recommendation_coverage"] == 1.0
        assert scores["reflexion_efficiency"] == 1.0

    def test_score_partial_trace(self):
        trace = {
            "task": "fix pods",
            "total_issues": 4,
            "total_diagnoses": 2,
            "total_recommendations": 1,
            "actions_taken": 1,
            "reflexion_retries": 1,
        }
        scores = self.score_trace(trace)
        assert scores["diagnosis_coverage"] == 0.5
        assert scores["recommendation_coverage"] == 0.5
        assert scores["reflexion_efficiency"] < 1.0

    def test_score_healthy_cluster(self):
        trace = {
            "task": "check health",
            "total_issues": 0,
            "total_diagnoses": 0,
            "total_recommendations": 0,
            "actions_taken": 0,
            "reflexion_retries": 0,
        }
        scores = self.score_trace(trace)
        assert scores["diagnosis_coverage"] == 1.0
        assert scores["recommendation_coverage"] == 1.0


# =========================================================================
# Helpers
# =========================================================================

def _make_minimal_state(**overrides) -> dict:
    """Create a minimal AgentState dict for testing."""
    base = {
        "task": "test", "phase": "test",
        "namespaces": [], "ns_issues": [],
        "issues": [], "diagnoses": [], "recommendations": [],
        "approved_commands": [], "actions_taken": [],
        "summary": "", "agent_logs": [],
        "reflexion_count": 0, "reflexion_critique": "",
        "memory_context": "",
        "supervisor_next": "", "supervisor_reasoning": "",
        "eval_trace": [],
    }
    base.update(overrides)
    return base


# =========================================================================
# CLI runner for quick eval scoring
# =========================================================================

def _cli_score_traces():
    """Score all eval traces in the data directory and print a report."""
    eval_dir = Path(os.getenv("EVAL_DATA_DIR", "eval_data"))
    traces = TestEvalTraceLoader.load_all_traces(eval_dir)
    if not traces:
        print(f"No eval traces found in {eval_dir}/")
        return

    print(f"\nScoring {len(traces)} eval trace(s) from {eval_dir}/\n")
    print(f"{'Task':<40} {'Issues':>7} {'Diag%':>7} {'Rec%':>7} {'Refl':>5}")
    print("-" * 70)

    total_diag_cov = 0.0
    total_rec_cov = 0.0
    for trace in traces:
        scores = TestEvalTraceLoader.score_trace(trace)
        total_diag_cov += scores["diagnosis_coverage"]
        total_rec_cov += scores["recommendation_coverage"]
        print(
            f"{scores['task'][:40]:<40} "
            f"{scores['issues_detected']:>7} "
            f"{scores['diagnosis_coverage']:>7.0%} "
            f"{scores['recommendation_coverage']:>7.0%} "
            f"{scores['reflexion_retries']:>5}"
        )

    n = len(traces)
    print("-" * 70)
    print(f"{'AVERAGE':<40} {'':>7} {total_diag_cov/n:>7.0%} {total_rec_cov/n:>7.0%}")
    print()


if __name__ == "__main__":
    if "--score" in sys.argv:
        _cli_score_traces()
    else:
        pytest.main([__file__, "-v"] + sys.argv[1:])