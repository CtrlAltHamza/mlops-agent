"""Load a local .env file without overriding variables already set in the process."""

from __future__ import annotations

import os
from pathlib import Path


def allow_local_mlflow_store(tracking_uri: str) -> None:
    """Current MLflow releases reject a directory store unless this opt-in is set."""
    remote_prefixes = ("http://", "https://", "sqlite:", "postgresql:", "mysql:", "mssql:")
    if tracking_uri.lower().startswith(remote_prefixes):
        return
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def load_dotenv(path: str | Path = ".env") -> None:
    file = Path(path)
    if not file.exists():
        return
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
