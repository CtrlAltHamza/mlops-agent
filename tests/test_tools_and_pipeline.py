"""Direct tests for tool guards, manifests, the DAG definition, and training stages."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import yaml

from agent.tools import airflow_tools
from agent.tools.base import Registry, Tool
from agent.tools.kubernetes_tools import restart_deployment
from training.train import quality_gate_passed, run_stage, validate_frame

ROOT = Path(__file__).resolve().parents[1]


def test_invalid_metric_is_rejected() -> None:
    from agent.tools.mlflow_tools import find_best_run

    registry = Registry()
    registry.add(Tool("find_best_run", find_best_run))
    result = registry.execute("find_best_run", {"metric": "loss"}, request_id="req-metric")
    assert result.status == "invalid_request"


def test_airflow_trigger_allowlist_and_payload(monkeypatch) -> None:
    monkeypatch.setenv("AIRFLOW_APPROVED_DAGS", "ml_training_pipeline")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/token":
            return httpx.Response(200, json={"access_token": "token-123"})
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"dag_run_id": "manual-9", "state": "queued"})

    monkeypatch.setattr(
        airflow_tools,
        "_client",
        lambda: httpx.Client(base_url="http://airflow.test", transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(PermissionError):
        airflow_tools.trigger_training_dag(confirm=False)
    with pytest.raises(Exception, match="not in the approved list"):
        airflow_tools.trigger_training_dag(dag_id="delete_everything", confirm=True)
    created = airflow_tools.trigger_training_dag(confirm=True)
    assert created["triggered"] is True
    assert seen["body"] == {"logical_date": None, "conf": {}}
    assert seen["path"] == "/api/v2/dags/ml_training_pipeline/dagRuns"


def test_airflow_http_failure_is_unavailable(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "down"})

    monkeypatch.setattr(
        airflow_tools,
        "_client",
        lambda: httpx.Client(base_url="http://airflow.test", transport=httpx.MockTransport(handler)),
    )
    registry = Registry()
    registry.add(Tool("list_dags", airflow_tools.list_dags))
    result = registry.execute("list_dags", {}, request_id="req-airflow")
    assert result.status == "unavailable"
    assert result.data is None


def test_restart_requires_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("APPROVED_K8S_NAMESPACE", "mlops-inference")
    with pytest.raises(PermissionError, match="confirmation_required"):
        restart_deployment(confirm=False)


def test_manifests_meet_the_deployment_contract() -> None:
    deployment = yaml.safe_load((ROOT / "kubernetes" / "deployment.yaml").read_text(encoding="utf-8"))
    service = yaml.safe_load((ROOT / "kubernetes" / "service.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "kubernetes" / "configmap.yaml").read_text(encoding="utf-8"))
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert deployment["spec"]["replicas"] >= 2
    assert service["spec"]["type"] == "ClusterIP"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["resources"]["requests"]["cpu"]
    assert container["resources"]["limits"]["memory"]
    assert container["envFrom"][0]["configMapRef"]["name"] == config["metadata"]["name"]
    assert config["data"]["APP_VERSION"]


def test_dag_source_has_required_stages_retries_and_order() -> None:
    source = (ROOT / "dags" / "training_pipeline.py").read_text(encoding="utf-8")
    for task in ("validate", "preprocess", "train", "evaluate", "quality_gate", "promote"):
        assert f'op_args=["{task}"]' in source
    assert "retries=2" in source
    assert "validate >> preprocess >> train >> evaluate >> quality_gate >> promote" in source
    assert 'dag_id="ml_training_pipeline"' in source


def test_validation_failure_blocks_the_stage(tmp_path: Path) -> None:
    frame = pd.DataFrame({"customer_id": [1], "age": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_frame(frame)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"customer_id": [1], "age": [30]}).to_csv(raw / "customers.csv", index=False)
    with pytest.raises(ValueError):
        run_stage("validate", tmp_path)


def test_quality_gate_blocks_promotion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QUALITY_GATE_THRESHOLD", "0.99")
    assert quality_gate_passed({"f1_score": 0.4}, threshold=0.99) is False
    metrics = tmp_path / "metrics"
    models = tmp_path / "models"
    metrics.mkdir()
    models.mkdir()
    (metrics / "evaluation.json").write_text(
        json.dumps({"f1_score": 0.4, "quality_gate_passed": False, "threshold": 0.99}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="quality gate failed"):
        run_stage("quality_gate", tmp_path)
    with pytest.raises(RuntimeError, match="refusing to promote"):
        run_stage("promote", tmp_path)
    assert not (models / "PROMOTED.json").exists()


def test_training_pipeline_records_provenance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    monkeypatch.setenv("QUALITY_GATE_THRESHOLD", "0.60")
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (tmp_path / "data" / "VERSION").write_text("v1.0\n", encoding="utf-8")
    rows = []
    for index in range(80):
        charges = 30 + (index * 3) % 80
        tenure = index % 36
        calls = index % 8
        rows.append(
            {
                "customer_id": index + 1,
                "age": 20 + (index * 3) % 40,
                "tenure_months": tenure,
                "monthly_charges": charges,
                "contract_type": index % 3,
                "support_calls": calls,
                "churn": int((charges > 70 and tenure < 18) or calls >= 5),
            }
        )
    pd.DataFrame(rows).to_csv(raw / "customers.csv", index=False)
    result = run_stage("all", tmp_path)
    assert result["promote"]["promoted"] is True
    assert result["promote"]["dataset_version"] == "v1.0"
    assert (tmp_path / "models" / "PROMOTED.json").exists()
