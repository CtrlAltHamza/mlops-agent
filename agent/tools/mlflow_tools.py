"""MLflow experiment tools. Facts come from MlflowClient, never from the web UI."""

from __future__ import annotations

import os
from typing import Any

from mlflow.tracking import MlflowClient

from agent.envfile import allow_local_mlflow_store
from agent.schemas import InvalidRequest, ToolUnavailable
from agent.tools.base import Tool

ALLOWED_METRICS = {"f1_score", "precision", "recall", "accuracy", "roc_auc"}
SELECTION_RULE = (
    "The metric leader is the highest value of the requested metric among the recent runs. "
    "A high score does not make a model production-ready. Promotion also requires "
    "quality_gate_passed=true, a healthy deployment, and an explicit decision."
)


def _client() -> MlflowClient:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    allow_local_mlflow_store(uri)
    try:
        return MlflowClient(tracking_uri=uri)
    except Exception as exc:  # noqa: BLE001
        raise ToolUnavailable(f"MLflow client could not be created: {exc}") from exc


def _check_metric(metric: str) -> str:
    if metric not in ALLOWED_METRICS:
        raise InvalidRequest(f"metric must be one of {sorted(ALLOWED_METRICS)}")
    return metric


def _run_brief(run: Any) -> dict[str, Any]:
    params = dict(run.data.params)
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "metrics": dict(run.data.metrics),
        "params": params,
        "dataset_version": params.get("dataset_version"),
        "dataset_md5": params.get("dataset_md5"),
        "git_commit": params.get("git_commit"),
        "quality_gate_passed": params.get("quality_gate_passed"),
    }


def list_experiments() -> dict[str, Any]:
    try:
        experiments = _client().search_experiments()
    except Exception as exc:  # noqa: BLE001
        raise ToolUnavailable(f"MLflow is unavailable: {exc}") from exc
    return {
        "experiments": [
            {"experiment_id": item.experiment_id, "name": item.name, "lifecycle_stage": item.lifecycle_stage}
            for item in experiments
        ]
    }


def list_runs(experiment_name: str | None = None, max_results: int = 5) -> dict[str, Any]:
    if max_results < 1 or max_results > 50:
        raise InvalidRequest("max_results must be between 1 and 50")
    client = _client()
    try:
        if experiment_name:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ToolUnavailable(f"experiment {experiment_name} was not found")
            experiment_ids = [experiment.experiment_id]
        else:
            experiment_ids = [item.experiment_id for item in client.search_experiments()]
        if not experiment_ids:
            return {"runs": []}
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            order_by=["attributes.start_time DESC"],
            max_results=max_results,
        )
    except ToolUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ToolUnavailable(f"MLflow is unavailable: {exc}") from exc
    return {"runs": [_run_brief(run) for run in runs]}


def get_run(run_id: str) -> dict[str, Any]:
    if not run_id or len(run_id) > 64:
        raise InvalidRequest("run_id is required")
    client = _client()
    try:
        run = client.get_run(run_id)
        artifacts = [item.path for item in client.list_artifacts(run_id)]
    except Exception as exc:  # noqa: BLE001
        raise ToolUnavailable(f"MLflow run could not be read: {exc}") from exc
    brief = _run_brief(run)
    brief["artifacts"] = artifacts
    return brief


def find_best_run(metric: str = "f1_score", max_results: int = 5) -> dict[str, Any]:
    metric = _check_metric(metric)
    listed = list_runs(max_results=max_results)
    ranked = [run for run in listed["runs"] if metric in run["metrics"]]
    if not ranked:
        return {
            "metric": metric,
            "recent_runs": listed["runs"],
            "metric_leader": None,
            "promotion_candidate": None,
            "production_ready": False,
            "selection_rule": SELECTION_RULE,
        }
    leader = max(ranked, key=lambda run: run["metrics"][metric])
    passed = [run for run in ranked if str(run.get("quality_gate_passed")).lower() == "true"]
    candidate = max(passed, key=lambda run: run["metrics"][metric]) if passed else None
    return {
        "metric": metric,
        "recent_runs": ranked,
        "metric_leader": leader,
        "promotion_candidate": candidate,
        "production_ready": False,
        "selection_rule": SELECTION_RULE,
    }


def compare_runs(run_ids: list[str] | None = None, max_results: int = 5) -> dict[str, Any]:
    if run_ids:
        runs = [get_run(run_id) for run_id in run_ids[:10]]
    else:
        runs = list_runs(max_results=max_results)["runs"]
    return {
        "runs": runs,
        "note": "Comparison lists recorded metrics only. It does not promote a model.",
    }


def list_registered_models() -> dict[str, Any]:
    client = _client()
    try:
        models = client.search_registered_models()
    except Exception as exc:  # noqa: BLE001
        raise ToolUnavailable(f"MLflow registry is unavailable: {exc}") from exc
    payload = []
    for model in models:
        versions = []
        for version in model.latest_versions or []:
            versions.append(
                {
                    "version": version.version,
                    "current_stage": getattr(version, "current_stage", None),
                    "run_id": version.run_id,
                    "status": version.status,
                }
            )
        payload.append({"name": model.name, "versions": versions})
    return {"registered_models": payload}


MLFLOW_TOOLS = [
    Tool("list_experiments", list_experiments),
    Tool("list_runs", list_runs),
    Tool("get_run", get_run),
    Tool("find_best_run", find_best_run),
    Tool("compare_runs", compare_runs),
    Tool("list_registered_models", list_registered_models),
]
