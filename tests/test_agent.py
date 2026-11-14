"""Agent planning, evidence handling, and safety refusals."""

from __future__ import annotations

import logging
from typing import Any

from agent.orchestrator import Agent
from agent.planner import plan_message
from agent.tools.base import Registry, Tool

MULTI = (
    "Find the best recent model, identify the dataset version used to train it, "
    "check the deployment status, and explain whether retraining is necessary."
)


def _ok(payload: dict[str, Any]):
    def fn(**_kwargs: Any) -> dict[str, Any]:
        return payload

    return fn


def _boom(**_kwargs: Any) -> dict[str, Any]:
    raise TimeoutError("mlflow down")


def test_dataset_question_selects_dvc_and_mlflow_tools() -> None:
    names = [step.tool for step in plan_message("Which dataset version was used for experiment 25?")]
    assert "list_datasets" in names
    assert "list_runs" in names
    assert "get_dataset_version" in names
    assert "trigger_training_dag" not in names


def test_model_comparison_selects_mlflow_tools() -> None:
    names = [step.tool for step in plan_message(
        "Compare the last five training runs and recommend the best candidate based on F1 score."
    )]
    assert "compare_runs" in names
    assert "find_best_run" in names


def test_multi_tool_plan_covers_four_systems_and_does_not_trigger() -> None:
    names = [step.tool for step in plan_message(MULTI)]
    assert names == ["find_best_run", "get_dataset_version", "diagnose_application", "get_dag_runs"]


def test_multi_tool_response_uses_evidence_and_does_not_start_training() -> None:
    registry = Registry()
    registry.add(Tool("find_best_run", _ok({
        "metric": "f1_score",
        "metric_leader": {
            "run_id": "run-a",
            "metrics": {"f1_score": 0.81},
            "quality_gate_passed": "true",
            "dataset_version": "v1.0",
            "dataset_md5": "abc",
            "git_commit": "deadbee",
        },
        "promotion_candidate": {"run_id": "run-a", "quality_gate_passed": "true"},
        "production_ready": False,
        "selection_rule": "highest metric is not production readiness",
    })))
    registry.add(Tool("get_dataset_version", _ok({
        "path": "data/raw/customers.csv",
        "version_label": "v1.0",
        "md5": "abc",
        "git_commit": "deadbee",
        "recorded": {
            "dataset_version": "v1.0",
            "git_commit": "deadbee",
            "dataset_md5": "abc",
            "matches_workspace_md5": True,
            "matches_workspace_commit": True,
        },
    })))
    registry.add(Tool("diagnose_application", _ok({
        "namespace": "mlops-inference",
        "healthy": True,
        "pods": [{"name": "api-0", "phase": "Running", "ready": True, "restarts": 0, "waiting_reason": None}],
        "unhealthy_pods": [],
        "events": [],
        "logs": {},
    })))
    registry.add(Tool("get_dag_runs", _ok({
        "dag_id": "ml_training_pipeline",
        "runs": [{"dag_run_id": "manual-1", "state": "success"}],
    })))
    response = Agent(registry).handle(MULTI)
    assert response.retraining_started is False
    assert [item.tool for item in response.tool_results] == [
        "find_best_run",
        "get_dataset_version",
        "diagnose_application",
        "get_dag_runs",
    ]
    assert "v1.0" in response.answer
    assert "run-a" in response.answer
    assert "was not started" in response.answer


def test_unavailable_tool_does_not_invent_operational_results() -> None:
    registry = Registry()
    registry.add(Tool("find_best_run", _boom))
    registry.add(Tool("get_dataset_version", _boom))
    registry.add(Tool("diagnose_application", _boom))
    registry.add(Tool("get_dag_runs", _boom))
    response = Agent(registry).handle(MULTI)
    assert response.facts == []
    assert response.retraining_started is False
    assert "No operational claim can be made" in response.answer
    assert "0.81" not in response.answer
    assert "v1.0" not in response.answer


def test_trigger_without_confirmation_does_not_run() -> None:
    called = {"count": 0}

    def trigger(**kwargs: Any) -> dict[str, Any]:
        called["count"] += 1
        if not kwargs.get("confirm"):
            raise PermissionError("confirmation_required to trigger a DAG")
        return {"triggered": True, "dag_id": "ml_training_pipeline", "dag_run_id": "r1", "state": "queued"}

    registry = Registry()
    registry.add(Tool("list_dags", _ok({"dags": []})))
    registry.add(Tool("get_dag_runs", _ok({"dag_id": "ml_training_pipeline", "runs": []})))
    registry.add(Tool("trigger_training_dag", trigger))
    response = Agent(registry).handle("Trigger the model training pipeline and report whether execution succeeded.")
    assert called["count"] == 1
    assert response.retraining_started is False
    assert any(item.status == "confirmation_required" for item in response.tool_results)


def test_unauthorized_namespace_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("APPROVED_K8S_NAMESPACE", "mlops-inference")
    from agent.tools.kubernetes_tools import restart_deployment

    registry = Registry()
    registry.add(Tool("diagnose_application", _ok({
        "namespace": "kube-system",
        "healthy": True,
        "pods": [],
        "unhealthy_pods": [],
        "events": [],
        "logs": {},
    })))
    registry.add(Tool("get_deployed_version", _ok({"namespace": "kube-system", "deployments": []})))
    registry.add(Tool("list_services", _ok({"namespace": "kube-system", "services": []})))
    registry.add(Tool("restart_deployment", restart_deployment))
    response = Agent(registry).handle("Restart the deployment in namespace kube-system. confirm")
    restart = next(item for item in response.tool_results if item.tool == "restart_deployment")
    assert restart.status == "rejected"
    assert response.retraining_started is False
    assert "approved namespace" in response.answer


def test_tool_timeout_is_reported(caplog) -> None:
    def slow(**_kwargs: Any) -> dict[str, Any]:
        import time
        time.sleep(0.2)
        return {"ok": True}

    registry = Registry()
    registry.add(Tool("list_datasets", slow, timeout_s=0.01))
    with caplog.at_level(logging.INFO, logger="mlops.agent"):
        result = registry.execute("list_datasets", {}, request_id="req-timeout")
    assert result.status == "timeout"
    assert '"tool": "list_datasets"' in caplog.text
    assert '"request_id": "req-timeout"' in caplog.text
    assert '"status": "timeout"' in caplog.text


def test_logs_scrub_secrets(caplog, monkeypatch) -> None:
    monkeypatch.setenv("AIRFLOW_PASSWORD", "super-secret-pass")
    registry = Registry()

    def leak(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("login failed password=super-secret-pass")

    registry.add(Tool("list_dags", leak))
    with caplog.at_level(logging.INFO, logger="mlops.agent"):
        result = registry.execute("list_dags", {}, request_id="req-secret")
    assert result.status == "unavailable"
    assert "super-secret-pass" not in caplog.text
    assert "duration_ms" in caplog.text


def test_ui_endpoint_returns_html() -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "MLOps AI Agent" in response.text
