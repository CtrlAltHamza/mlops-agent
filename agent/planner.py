"""Choose tools from the user request.

The planner never executes anything. State-changing tools are included only when
the wording asks for that operation, and they still carry confirm=false unless
the user explicitly confirmed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    fill_provenance: bool = False


def confirmed(message: str) -> bool:
    return bool(re.search(r"\bconfirm(ed|ation)?\b", message.lower()))


def _namespace(message: str) -> str | None:
    match = re.search(r"\bnamespace\s+([a-z0-9-]+)", message.lower())
    return match.group(1) if match else None


def _metric(message: str) -> str:
    text = message.lower().replace("-", " ").replace("_", " ")
    if "precision" in text:
        return "precision"
    if "recall" in text:
        return "recall"
    if "accuracy" in text:
        return "accuracy"
    if "roc" in text or "auc" in text:
        return "roc_auc"
    return "f1_score"


def _wants_trigger(text: str) -> bool:
    if "whether" in text and "retrain" in text:
        return False
    phrases = (
        "trigger the",
        "trigger training",
        "start the training",
        "start training",
        "run the training pipeline",
        "kick off the training",
    )
    return any(phrase in text for phrase in phrases)


def _multi_tool(text: str) -> bool:
    asks_model = "best" in text or "model" in text
    asks_data = "dataset" in text or "data version" in text
    asks_cluster = any(word in text for word in ("deploy", "kubernetes", "cluster", "pod"))
    asks_retrain = "retrain" in text or "training" in text
    return asks_model and asks_data and asks_cluster and asks_retrain


def plan_message(message: str) -> list[Step]:
    text = message.lower()
    steps: list[Step] = []
    if _multi_tool(text):
        return [
            Step("find_best_run", {"metric": _metric(message), "max_results": 5}),
            Step("get_dataset_version", fill_provenance=True),
            Step("diagnose_application", {}),
            Step("get_dag_runs", {"limit": 5}),
        ]

    if any(word in text for word in ("dataset", "dvc", "provenance", "reproduce", "data version")):
        steps.append(Step("list_datasets"))
        if "compare" in text or "difference between" in text:
            steps.append(Step("compare_dataset_versions"))
        if any(word in text for word in ("changed", "change", "drift", "modified")):
            steps.append(Step("detect_dataset_changes"))
        if "checkout" in text or "restore" in text:
            revision = _revision(text)
            if revision:
                steps.append(Step("checkout_dataset", {"rev": revision, "confirm": confirmed(message)}))
        if any(word in text for word in ("reproduce", "checkout", "version", "experiment", "which dataset")):
            steps.append(Step("explain_reproduction"))
        if "experiment" in text or "run" in text:
            steps.append(Step("list_runs", {"max_results": 20}))
            steps.append(Step("get_dataset_version", fill_provenance=True))
        elif any(word in text for word in ("version", "provenance", "which dataset")):
            steps.append(Step("get_dataset_version"))

    if any(word in text for word in ("experiment", "mlflow", "metric", "f1", "best model", "best run", "training run")):
        if "experiment" in text and not any(step.tool == "list_runs" for step in steps):
            steps.append(Step("list_experiments"))
        if "compare" in text:
            steps.append(Step("compare_runs", {"max_results": 5}))
        if any(word in text for word in ("best", "recommend", "f1", "metric")):
            steps.append(Step("find_best_run", {"metric": _metric(message), "max_results": 5}))
        if "registry" in text or "model version" in text:
            steps.append(Step("list_registered_models"))

    if any(word in text for word in ("airflow", "dag", "pipeline", "workflow")) or _wants_trigger(text):
        steps.append(Step("list_dags"))
        steps.append(Step("get_dag_runs", {"limit": 5}))
        if any(word in text for word in ("fail", "error", "why", "diagnostic")):
            steps.append(Step("get_failed_tasks"))
            steps.append(Step("get_task_diagnostics"))
        if _wants_trigger(text):
            steps.append(Step("trigger_training_dag", {"confirm": confirmed(message)}))

    if any(word in text for word in ("kubernetes", "pod", "deployment", "cluster", "unavailable", "prediction api", "service")):
        steps.append(Step("diagnose_application", _namespace_args(message)))
        steps.append(Step("get_deployed_version", _namespace_args(message)))
        steps.append(Step("list_services", _namespace_args(message)))
        if "restart" in text or "rollout" in text:
            args = _namespace_args(message)
            args["confirm"] = confirmed(message)
            steps.append(Step("restart_deployment", args))

    if not steps:
        steps = [
            Step("list_datasets"),
            Step("find_best_run", {"metric": "f1_score", "max_results": 5}),
            Step("get_dag_runs", {"limit": 5}),
            Step("diagnose_application"),
        ]
    return _dedupe(steps)


def _namespace_args(message: str) -> dict[str, Any]:
    namespace = _namespace(message)
    if namespace is None:
        return {}
    return {"namespace": namespace}


def _revision(text: str) -> str | None:
    match = re.search(r"\b([0-9a-f]{7,40})\b", text)
    return match.group(1) if match else None


def _dedupe(steps: list[Step]) -> list[Step]:
    seen: set[str] = set()
    unique: list[Step] = []
    for step in steps:
        if step.tool in seen:
            continue
        seen.add(step.tool)
        unique.append(step)
    return unique
