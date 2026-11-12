"""Allowlists for operations that change external state."""

from __future__ import annotations

import os
import re

_REV_RE = re.compile(r"^[A-Za-z0-9._]{1,64}$")


def approved_dags() -> set[str]:
    raw = os.environ.get("AIRFLOW_APPROVED_DAGS", "ml_training_pipeline")
    return {item.strip() for item in raw.split(",") if item.strip()}


def approved_namespace() -> str:
    return os.environ.get("APPROVED_K8S_NAMESPACE", "mlops-inference")


def validate_revision(rev: str) -> str:
    if not _REV_RE.fullmatch(rev):
        raise ValueError("revision is not an approved git reference")
    return rev


def ensure_confirmed(confirm: bool, action: str) -> None:
    if not confirm:
        raise PermissionError(f"confirmation_required for {action}")


def ensure_namespace(namespace: str) -> str:
    allowed = approved_namespace()
    if namespace != allowed:
        raise PermissionError(
            f"namespace {namespace} is outside the approved namespace {allowed}"
        )
    return namespace
