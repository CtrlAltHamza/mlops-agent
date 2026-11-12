"""Run a planned tool sequence and answer only from tool evidence."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx

from agent.logging_utils import LOGGER, scrub
from agent.planner import Step, plan_message
from agent.schemas import AgentResponse, ToolResult, format_sections
from agent.tools import build_registry
from agent.tools.base import Registry


class Agent:
    def __init__(self, registry: Registry | None = None) -> None:
        self.registry = registry or build_registry()

    def handle(self, message: str, request_id: str | None = None) -> AgentResponse:
        request_id = request_id or uuid.uuid4().hex[:12]
        steps = self._select_steps(message)
        LOGGER.info(
            scrub(
                json.dumps(
                    {
                        "request_id": request_id,
                        "event": "plan",
                        "tools": [step.tool for step in steps],
                    },
                    sort_keys=True,
                )
            )
        )
        results: list[ToolResult] = []
        for step in steps:
            arguments = dict(step.arguments)
            if step.fill_provenance:
                filled = _provenance_args(results)
                if filled is None:
                    results.append(
                        ToolResult(
                            ok=False,
                            tool=step.tool,
                            status="unavailable",
                            error="no recorded dataset reference was returned by MLflow",
                        )
                    )
                    continue
                arguments.update(filled)
            results.append(
                self.registry.execute(step.tool, arguments, request_id=request_id)
            )
        return _compose(request_id, results)

    def _select_steps(self, message: str) -> list[Step]:
        planned = _llm_steps(message, self.registry.names())
        if planned:
            return planned
        return plan_message(message)


def _provenance_args(results: list[ToolResult]) -> dict[str, Any] | None:
    best = next((item for item in results if item.tool == "find_best_run" and item.ok), None)
    runs = next((item for item in results if item.tool == "list_runs" and item.ok), None)
    source = None
    if best and best.data:
        source = best.data.get("metric_leader") or best.data.get("promotion_candidate")
    if source is None and runs and runs.data:
        listed = runs.data.get("runs") or []
        source = listed[0] if listed else None
    if not source:
        return None
    return {
        "dataset_version": source.get("dataset_version"),
        "git_commit": source.get("git_commit"),
        "dataset_md5": source.get("dataset_md5"),
    }


def _compose(request_id: str, results: list[ToolResult]) -> AgentResponse:
    facts: list[str] = []
    uncertainties: list[str] = []
    actions: list[str] = []
    retraining_started = False

    for result in results:
        if result.tool == "trigger_training_dag" and result.ok and (result.data or {}).get("triggered"):
            retraining_started = True
        if not result.ok or not result.data:
            uncertainties.append(
                f"{result.tool} returned {result.status}: {result.error or 'no data'}"
            )
            if result.status == "confirmation_required":
                actions.append(
                    f"Repeat the request and include the word confirm before {result.tool} will run."
                )
            elif result.status == "rejected":
                actions.append(f"{result.tool} was rejected. No state was changed.")
            continue
        facts.extend(_facts(result.tool, result.data))

    if retraining_started:
        actions.append("A confirmed training run was submitted. This answer does not claim it succeeded.")
    else:
        actions.extend(_retrain_advice(results))

    if not facts:
        summary = (
            "No operational claim can be made. The tools for this request did not return evidence, "
            "so no dataset version, metric, pipeline state, or cluster state is inferred."
        )
    else:
        summary = (
            "The statements below are limited to tool responses collected for this request. "
            "A leading metric is not treated as production readiness."
        )
    answer = format_sections(summary, facts, uncertainties, actions)
    return AgentResponse(
        request_id=request_id,
        answer=answer,
        facts=facts,
        uncertainties=uncertainties,
        recommended_actions=actions,
        tool_results=results,
        retraining_started=retraining_started,
    )


def _facts(tool: str, data: dict[str, Any]) -> list[str]:
    if tool == "find_best_run":
        return _best_run_facts(data)
    if tool == "compare_runs":
        lines = []
        for run in data.get("runs") or []:
            lines.append(
                f"Run {run.get('run_id')} metrics={run.get('metrics')} "
                f"quality_gate_passed={run.get('quality_gate_passed')}"
            )
        if data.get("note"):
            lines.append(str(data["note"]))
        return lines
    if tool == "get_dataset_version":
        recorded = data.get("recorded") or {}
        lines = [
            f"Workspace dataset {data.get('path')} label {data.get('version_label')} "
            f"md5 {data.get('md5')} git {data.get('git_commit')}."
        ]
        if recorded:
            lines.append(
                "Recorded training provenance from the caller is "
                f"version {recorded.get('dataset_version')}, git {recorded.get('git_commit')}, "
                f"md5 {recorded.get('dataset_md5')}. "
                f"md5 matches workspace={recorded.get('matches_workspace_md5')}, "
                f"commit matches workspace={recorded.get('matches_workspace_commit')}."
            )
        return lines
    if tool == "list_datasets":
        paths = [item.get("path") for item in data.get("datasets") or []]
        return [f"Tracked dataset paths: {paths}"]
    if tool == "explain_reproduction":
        return [
            f"To reproduce {data.get('git_commit')}: {data.get('commands')}. "
            "Those commands were not executed."
        ]
    if tool == "detect_dataset_changes":
        return [f"Dataset changed={data.get('changed')}. dvc_available={data.get('dvc_available')}."]
    if tool == "compare_dataset_versions":
        return [
            f"Compared {data.get('rev_a')} blob {data.get('blob_a')} "
            f"with workspace md5 {data.get('workspace_md5')}."
        ]
    if tool == "diagnose_application":
        return _diagnosis_facts(data)
    if tool == "get_deployed_version":
        return [
            f"Deployment {item.get('name')} version {item.get('version')} "
            f"image {item.get('images')} ready {item.get('ready_replicas')}/{item.get('replicas')}."
            for item in data.get("deployments") or []
        ] or [f"No deployments were returned in namespace {data.get('namespace')}."]
    if tool == "list_services":
        return [
            f"Service {item.get('name')} type {item.get('type')} ports {item.get('ports')}."
            for item in data.get("services") or []
        ] or [f"No services were returned in namespace {data.get('namespace')}."]
    if tool == "get_dag_runs":
        runs = data.get("runs") or []
        if not runs:
            return [f"Airflow returned no runs for DAG {data.get('dag_id')}."]
        latest = runs[0]
        return [
            f"Latest DAG run {latest.get('dag_run_id')} for {data.get('dag_id')} "
            f"is {latest.get('state')}."
        ]
    if tool == "list_dags":
        names = [item.get("dag_id") for item in data.get("dags") or []]
        return [f"Airflow DAGs: {names}"]
    if tool == "get_failed_tasks":
        failed = data.get("failed_tasks") or []
        if not failed:
            return [f"No failed tasks were returned for run {data.get('dag_run_id')}."]
        return [f"Failed task {item.get('task_id')} state {item.get('state')}." for item in failed]
    if tool == "get_task_diagnostics":
        lines = []
        for item in data.get("diagnostics") or []:
            excerpt = (item.get("log_excerpt") or "")[:240]
            lines.append(
                f"Task {item.get('task_id')} state {item.get('state')} log excerpt: {excerpt or item.get('log_error')}"
            )
        return lines or ["Airflow returned no task diagnostics."]
    if tool == "trigger_training_dag":
        return [
            f"Submitted DAG {data.get('dag_id')} run {data.get('dag_run_id')} state {data.get('state')}."
        ]
    if tool == "restart_deployment":
        return [
            f"Restart requested for {data.get('deployment')} in {data.get('namespace')} at {data.get('at')}."
        ]
    if tool == "list_registered_models":
        names = [item.get("name") for item in data.get("registered_models") or []]
        return [f"Registered models: {names}"]
    if tool == "list_experiments":
        names = [item.get("name") for item in data.get("experiments") or []]
        return [f"MLflow experiments: {names}"]
    if tool == "list_runs":
        return [f"MLflow returned {len(data.get('runs') or [])} runs."]
    return [f"{tool} returned keys {sorted(data)}."]


def _best_run_facts(data: dict[str, Any]) -> list[str]:
    metric = data.get("metric")
    leader = data.get("metric_leader")
    candidate = data.get("promotion_candidate")
    lines = [str(data.get("selection_rule"))]
    if leader is None:
        lines.append(f"No recent run recorded {metric}. There is no metric leader.")
        return lines
    lines.append(
        f"Metric leader {leader.get('run_id')} has {metric}={leader.get('metrics', {}).get(metric)} "
        f"and quality_gate_passed={leader.get('quality_gate_passed')}. "
        f"Recorded dataset version {leader.get('dataset_version')}, "
        f"git {leader.get('git_commit')}, md5 {leader.get('dataset_md5')}."
    )
    if candidate is None:
        lines.append("No compared run has quality_gate_passed=true, so none is a promotion candidate.")
    else:
        lines.append(
            f"Promotion candidate {candidate.get('run_id')} passed the quality gate. "
            "That still does not by itself make it production-ready."
        )
    lines.append("production_ready was reported false by the selection tool.")
    return lines


def _diagnosis_facts(data: dict[str, Any]) -> list[str]:
    lines = [
        f"Namespace {data.get('namespace')} healthy={data.get('healthy')}. "
        f"Unhealthy pods: {[pod.get('name') for pod in data.get('unhealthy_pods') or []]}."
    ]
    for pod in data.get("pods") or []:
        lines.append(
            f"Pod {pod.get('name')} phase {pod.get('phase')} ready {pod.get('ready')} "
            f"restarts {pod.get('restarts')} waiting {pod.get('waiting_reason')}."
        )
    for event in (data.get("events") or [])[:5]:
        lines.append(f"Event {event.get('reason')}: {event.get('message')}")
    for name, log in (data.get("logs") or {}).items():
        lines.append(f"Log excerpt for {name}: {str(log)[:240]}")
    return lines


def _retrain_advice(results: list[ToolResult]) -> list[str]:
    """Recommend another training run only from recorded evidence. Never start one here."""
    reasons: list[str] = []
    for result in results:
        if not result.ok or not result.data:
            continue
        if result.tool == "find_best_run":
            leader = result.data.get("metric_leader") or {}
            passed = str(leader.get("quality_gate_passed")).lower() == "true"
            if leader and not passed:
                reasons.append("the metric leader does not have quality_gate_passed=true")
            candidate = result.data.get("promotion_candidate")
            if leader and candidate is None:
                reasons.append("no recent run passed the quality gate")
        if result.tool == "get_failed_tasks" and result.data.get("failed_tasks"):
            reasons.append("Airflow reported failed tasks in a recent training run")
        if result.tool == "get_dag_runs":
            runs = result.data.get("runs") or []
            if runs and runs[0].get("state") in {"failed", "upstream_failed"}:
                reasons.append(f"the latest DAG run is {runs[0].get('state')}")
    if not reasons:
        return [
            "Retraining was not started. No failed quality gate, failed training run, "
            "freshness policy, or supplied degradation result is present in the tool evidence."
        ]
    unique = list(dict.fromkeys(reasons))
    return [
        "Retraining is recommended because " + "; ".join(unique) + ". It was not started."
    ]


def _llm_steps(message: str, tool_names: list[str]) -> list[Step] | None:
    if os.environ.get("LLM_PLANNER", "false").lower() != "true":
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return JSON only: {\"calls\": [{\"tool\": str, \"arguments\": object}]}. "
                    f"Tools: {tool_names}. Do not invent tool names. "
                    "Do not set confirm=true unless the user message contains confirm."
                ),
            },
            {"role": "user", "content": message},
        ],
    }
    try:
        response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=prompt,
            timeout=20.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = json.loads(content)
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, TypeError):
        return None
    steps: list[Step] = []
    allowed = set(tool_names)
    for call in payload.get("calls") or []:
        tool = call.get("tool")
        if tool not in allowed:
            continue
        arguments = dict(call.get("arguments") or {})
        if tool in {"trigger_training_dag", "restart_deployment", "checkout_dataset"}:
            arguments["confirm"] = bool(arguments.get("confirm")) and _user_confirmed(message)
        steps.append(Step(tool, arguments, fill_provenance=tool == "get_dataset_version"))
    return steps or None


def _user_confirmed(message: str) -> bool:
    from agent.planner import confirmed

    return confirmed(message)
