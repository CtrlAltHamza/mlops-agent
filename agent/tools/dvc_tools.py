"""DVC and Git provenance tools.

User text never becomes a shell command. Each function builds its own argument list.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from agent.safety import validate_revision
from agent.schemas import InvalidRequest, ToolUnavailable
from agent.tools.base import Tool

ROOT = Path(os.environ.get("MLOPS_PROJECT_ROOT", Path.cwd()))
DATASET = Path("data/raw/customers.csv")
VERSION_FILE = Path("data/VERSION")


def _root() -> Path:
    return Path(os.environ.get("MLOPS_PROJECT_ROOT", Path.cwd()))


def _run(argv: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=_root(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ToolUnavailable(f"{argv[0]} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolUnavailable("timeout") from exc


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_head() -> str:
    proc = _run(["git", "rev-parse", "HEAD"])
    if proc.returncode != 0:
        raise ToolUnavailable(proc.stderr.strip() or "git rev-parse failed")
    return proc.stdout.strip()


def _version_label() -> str:
    path = _root() / VERSION_FILE
    if not path.exists():
        return "unversioned"
    return path.read_text(encoding="utf-8").strip()


def list_datasets() -> dict[str, Any]:
    root = _root()
    tracked: list[dict[str, str]] = []
    pipeline = root / "dvc.yaml"
    if pipeline.exists():
        document = yaml.safe_load(pipeline.read_text(encoding="utf-8")) or {}
        for stage, spec in (document.get("stages") or {}).items():
            for dep in spec.get("deps") or []:
                if str(dep).startswith("data/"):
                    tracked.append({"stage": stage, "path": str(dep), "source": "dvc.yaml"})
    search_roots = [root / "data", root]
    seen: set[str] = set()
    for folder in search_roots:
        if not folder.exists():
            continue
        pointers = folder.glob("*.dvc") if folder == root else folder.rglob("*.dvc")
        for pointer in pointers:
            rel = str(pointer.relative_to(root))
            if rel in seen or ".dvc" in pointer.parts:
                continue
            seen.add(rel)
            tracked.append({"stage": "", "path": rel, "source": "dvc-file"})
    if not tracked and not (root / DATASET).exists():
        raise ToolUnavailable("no DVC-tracked dataset was found in the project")
    if not tracked:
        tracked.append({"stage": "raw", "path": str(DATASET), "source": "workspace"})
    return {"datasets": tracked}


def get_dataset_version(
    dataset_version: str | None = None,
    git_commit: str | None = None,
    dataset_md5: str | None = None,
) -> dict[str, Any]:
    """Return the workspace dataset identity and, when supplied, a recorded run reference."""
    path = _root() / DATASET
    if not path.exists():
        raise ToolUnavailable(f"{DATASET} is not in the workspace")
    workspace_md5 = _md5(path)
    workspace_commit = _git_head()
    label = _version_label()
    payload: dict[str, Any] = {
        "path": str(DATASET),
        "version_label": label,
        "md5": workspace_md5,
        "git_commit": workspace_commit,
        "dvc_pipeline": (_root() / "dvc.yaml").exists(),
        "reproduce": [
            f"git checkout {workspace_commit}",
            "dvc checkout",
            "dvc repro",
        ],
        "commands_executed": False,
    }
    if any(value is not None for value in (dataset_version, git_commit, dataset_md5)):
        payload["recorded"] = {
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "dataset_md5": dataset_md5,
            "matches_workspace_md5": dataset_md5 == workspace_md5 if dataset_md5 else None,
            "matches_workspace_commit": git_commit == workspace_commit if git_commit else None,
            "source": "caller-supplied provenance, usually MLflow params",
        }
    return payload


def compare_dataset_versions(rev_a: str = "HEAD", rev_b: str | None = None) -> dict[str, Any]:
    try:
        left = validate_revision(rev_a)
        right = validate_revision(rev_b) if rev_b else None
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc
    left_blob = _blob(left, DATASET)
    current_md5 = _md5(_root() / DATASET) if (_root() / DATASET).exists() else None
    result: dict[str, Any] = {
        "rev_a": left,
        "blob_a": left_blob,
        "workspace_md5": current_md5,
        "version_label": _version_label(),
    }
    if right:
        result["rev_b"] = right
        result["blob_b"] = _blob(right, DATASET)
        result["same_blob"] = result["blob_a"] == result["blob_b"]
    return result


def detect_dataset_changes() -> dict[str, Any]:
    git_status = _run(["git", "status", "--porcelain", "--", "data"])
    if git_status.returncode != 0:
        raise ToolUnavailable(git_status.stderr.strip() or "git status failed")
    changed = bool(git_status.stdout.strip())
    dvc_available = False
    dvc_output = ""
    try:
        dvc_proc = _run(["dvc", "status"])
    except ToolUnavailable as exc:
        dvc_output = str(exc)
    else:
        dvc_output = (dvc_proc.stdout or dvc_proc.stderr).strip()
        dvc_available = dvc_proc.returncode == 0
        if dvc_available and dvc_output and "up to date" not in dvc_output.lower():
            changed = True
    return {
        "changed": changed,
        "git_porcelain": git_status.stdout.strip(),
        "dvc_available": dvc_available,
        "dvc_status": dvc_output,
    }


def explain_reproduction(git_commit: str | None = None) -> dict[str, Any]:
    commit = git_commit or _git_head()
    try:
        validate_revision(commit)
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc
    return {
        "git_commit": commit,
        "version_label": _version_label(),
        "commands": [
            f"git checkout {commit}",
            "dvc checkout",
            "dvc repro",
        ],
        "commands_executed": False,
        "note": "Reproduction commands are described only. Nothing was checked out.",
    }


def checkout_dataset(rev: str, confirm: bool = False) -> dict[str, Any]:
    if not confirm:
        raise PermissionError("confirmation_required to checkout a dataset revision")
    try:
        revision = validate_revision(rev)
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc
    git_proc = _run(["git", "checkout", revision, "--", "data/raw/customers.csv", "data/VERSION"])
    if git_proc.returncode != 0:
        raise ToolUnavailable(git_proc.stderr.strip() or "git checkout failed")
    dvc_proc = _run(["dvc", "checkout"])
    return {
        "rev": revision,
        "git_checkout": git_proc.stdout.strip() or "ok",
        "dvc_checkout": (dvc_proc.stdout or dvc_proc.stderr).strip(),
        "dvc_ok": dvc_proc.returncode == 0,
        "commands_executed": True,
    }


def _blob(rev: str, path: Path) -> str:
    proc = _run(["git", "rev-parse", f"{rev}:{path.as_posix()}"])
    if proc.returncode != 0:
        raise ToolUnavailable(proc.stderr.strip() or f"no blob for {rev}:{path}")
    return proc.stdout.strip()


DVC_TOOLS = [
    Tool("list_datasets", list_datasets),
    Tool("get_dataset_version", get_dataset_version),
    Tool("compare_dataset_versions", compare_dataset_versions),
    Tool("detect_dataset_changes", detect_dataset_changes),
    Tool("explain_reproduction", explain_reproduction),
    Tool("checkout_dataset", checkout_dataset),
]
