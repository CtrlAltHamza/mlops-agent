"""Apache Airflow 3 client.

Authentication is POST /auth/token. Operations use the public /api/v2 routes.
Only DAG ids in AIRFLOW_APPROVED_DAGS can be triggered.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from agent.safety import approved_dags
from agent.schemas import InvalidRequest, NotAuthorized, ToolUnavailable
from agent.tools.base import Tool

DEFAULT_DAG = "ml_training_pipeline"


def _base_url() -> str:
    return os.environ.get("AIRFLOW_BASE_URL", "http://127.0.0.1:8080").rstrip("/")


def _timeout() -> float:
    return float(os.environ.get("AIRFLOW_TIMEOUT_SECONDS", "15"))


def _token(client: httpx.Client) -> str:
    username = os.environ.get("AIRFLOW_USERNAME", "airflow")
    password = os.environ.get("AIRFLOW_PASSWORD", "airflow")
    try:
        response = client.post(
            "/auth/token",
            json={"username": username, "password": password},
        )
    except httpx.HTTPError as exc:
        raise ToolUnavailable(f"Airflow authentication failed: {exc.__class__.__name__}") from exc
    if response.status_code >= 400:
        raise ToolUnavailable(f"Airflow authentication returned HTTP {response.status_code}")
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise ToolUnavailable("Airflow did not return an access token")
    return str(token)


def _client() -> httpx.Client:
    return httpx.Client(base_url=_base_url(), timeout=_timeout())


def _request(method: str, path: str, json_body: dict[str, Any] | None = None) -> Any:
    with _client() as client:
        token = _token(client)
        try:
            response = client.request(
                method,
                path,
                headers={"Authorization": f"Bearer {token}"},
                json=json_body,
            )
        except httpx.TimeoutException as exc:
            raise ToolUnavailable("timeout") from exc
        except httpx.HTTPError as exc:
            raise ToolUnavailable(f"Airflow is unavailable: {exc.__class__.__name__}") from exc
    if response.status_code >= 400:
        raise ToolUnavailable(f"Airflow returned HTTP {response.status_code} for {path}")
    if not response.content:
        return {}
    return response.json()


def list_dags() -> dict[str, Any]:
    payload = _request("GET", "/api/v2/dags")
    dags = payload.get("dags", payload if isinstance(payload, list) else [])
    return {
        "dags": [
            {
                "dag_id": item.get("dag_id"),
                "is_paused": item.get("is_paused"),
                "description": item.get("description"),
            }
            for item in dags
        ]
    }


def get_dag_runs(dag_id: str = DEFAULT_DAG, limit: int = 10) -> dict[str, Any]:
    if not dag_id:
        raise InvalidRequest("dag_id is required")
    payload = _request("GET", f"/api/v2/dags/{dag_id}/dagRuns?limit={int(limit)}")
    runs = payload.get("dag_runs", payload if isinstance(payload, list) else [])
    return {
        "dag_id": dag_id,
        "runs": [
            {
                "dag_run_id": item.get("dag_run_id"),
                "state": item.get("state"),
                "logical_date": item.get("logical_date"),
                "start_date": item.get("start_date"),
                "end_date": item.get("end_date"),
            }
            for item in runs
        ],
    }


def trigger_training_dag(dag_id: str = DEFAULT_DAG, confirm: bool = False) -> dict[str, Any]:
    if not confirm:
        raise PermissionError("confirmation_required to trigger a DAG")
    if dag_id not in approved_dags():
        raise NotAuthorized(f"DAG {dag_id} is not in the approved list")
    # Airflow 3 rejects a body that omits logical_date. null means "run now".
    created = _request(
        "POST",
        f"/api/v2/dags/{dag_id}/dagRuns",
        {"logical_date": None, "conf": {}},
    )
    return {
        "dag_id": dag_id,
        "dag_run_id": created.get("dag_run_id"),
        "state": created.get("state"),
        "triggered": True,
    }


def get_failed_tasks(dag_id: str = DEFAULT_DAG, dag_run_id: str | None = None) -> dict[str, Any]:
    if dag_run_id is None:
        runs = get_dag_runs(dag_id=dag_id, limit=5)["runs"]
        if not runs:
            return {"dag_id": dag_id, "failed_tasks": [], "note": "no DAG runs were returned"}
        dag_run_id = runs[0]["dag_run_id"]
    payload = _request("GET", f"/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances")
    instances = payload.get("task_instances", payload if isinstance(payload, list) else [])
    failed = [item for item in instances if item.get("state") in {"failed", "upstream_failed"}]
    return {
        "dag_id": dag_id,
        "dag_run_id": dag_run_id,
        "failed_tasks": [
            {
                "task_id": item.get("task_id"),
                "state": item.get("state"),
                "try_number": item.get("try_number"),
            }
            for item in failed
        ],
    }


def get_task_diagnostics(
    dag_id: str = DEFAULT_DAG,
    dag_run_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    failed = get_failed_tasks(dag_id=dag_id, dag_run_id=dag_run_id)
    tasks = failed["failed_tasks"]
    if task_id:
        tasks = [item for item in tasks if item["task_id"] == task_id] or [
            {"task_id": task_id, "state": "unknown", "try_number": 1}
        ]
    diagnostics = []
    for task in tasks:
        try_number = task.get("try_number") or 1
        log_path = (
            f"/api/v2/dags/{dag_id}/dagRuns/{failed['dag_run_id']}"
            f"/taskInstances/{task['task_id']}/logs/{try_number}"
        )
        log_text = ""
        log_error = None
        try:
            log_payload = _request("GET", log_path)
            if isinstance(log_payload, dict):
                log_text = str(log_payload.get("content") or log_payload)[:4000]
            else:
                log_text = str(log_payload)[:4000]
        except ToolUnavailable as exc:
            log_error = str(exc)
        diagnostics.append(
            {
                "task_id": task.get("task_id"),
                "state": task.get("state"),
                "try_number": try_number,
                "log_excerpt": log_text,
                "log_error": log_error,
            }
        )
    return {
        "dag_id": dag_id,
        "dag_run_id": failed.get("dag_run_id"),
        "diagnostics": diagnostics,
    }


AIRFLOW_TOOLS = [
    Tool("list_dags", list_dags),
    Tool("get_dag_runs", get_dag_runs),
    Tool("trigger_training_dag", trigger_training_dag),
    Tool("get_failed_tasks", get_failed_tasks),
    Tool("get_task_diagnostics", get_task_diagnostics),
]
