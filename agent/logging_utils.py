"""JSON logs for tool calls. Secrets are scrubbed before anything is written."""

from __future__ import annotations

import json
import logging
import os
import re

LOGGER = logging.getLogger("mlops.agent")

_SECRET_RE = re.compile(
    r"(?i)(password|token|authorization|secret|api_key)(['\"]?\s*[:=]\s*)(\S+)"
)


def scrub(text: str) -> str:
    cleaned = _SECRET_RE.sub(r"\1\2***", text)
    for name in ("AIRFLOW_PASSWORD", "OPENAI_API_KEY", "MLFLOW_TRACKING_TOKEN"):
        value = os.environ.get(name)
        if value:
            cleaned = cleaned.replace(value, "***")
    return cleaned


def log_tool_event(
    *,
    request_id: str,
    tool: str,
    status: str,
    duration_ms: float,
    error: str | None = None,
) -> None:
    payload = {
        "request_id": request_id,
        "tool": tool,
        "status": status,
        "duration_ms": round(duration_ms, 2),
    }
    if error:
        payload["error"] = scrub(error)
    LOGGER.info(scrub(json.dumps(payload, sort_keys=True)))
