"""Train a small churn classifier and record provenance in MLflow.

Stages are separate so Airflow can retry training without repeating a failed
quality gate, and so a failed gate cannot promote a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from agent.envfile import allow_local_mlflow_store
from training.schema import FEATURES, REQUIRED_COLUMNS, TARGET


def project_root() -> Path:
    return Path(os.environ.get("MLOPS_PROJECT_ROOT", Path.cwd()))


def validate_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if len(frame) < 20:
        raise ValueError("dataset has fewer than 20 rows")
    if frame[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("dataset contains nulls in required columns")
    labels = set(frame[TARGET].astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"churn labels must be 0 or 1, found {sorted(labels)}")


def quality_gate_passed(metrics: dict[str, Any], threshold: float | None = None) -> bool:
    limit = threshold if threshold is not None else float(os.environ.get("QUALITY_GATE_THRESHOLD", "0.60"))
    return float(metrics["f1_score"]) >= limit


def run_stage(stage: str, root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    if os.environ.get("INJECT_VALIDATION_FAILURE", "").lower() == "true" and stage == "validate":
        raise RuntimeError("injected validation failure")
    handlers = {
        "validate": _validate,
        "preprocess": _preprocess,
        "train": _train,
        "evaluate": _evaluate,
        "quality_gate": _quality_gate,
        "promote": _promote,
    }
    if stage == "all":
        payload: dict[str, Any] = {}
        for name in ("validate", "preprocess", "train", "evaluate", "quality_gate", "promote"):
            payload[name] = handlers[name](root)
        return payload
    if stage not in handlers:
        raise ValueError(f"unknown stage {stage}")
    return handlers[stage](root)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "raw": root / "data" / "raw" / "customers.csv",
        "version": root / "data" / "VERSION",
        "interim": root / "data" / "interim" / "validated.csv",
        "train": root / "data" / "processed" / "train.csv",
        "test": root / "data" / "processed" / "test.csv",
        "model": root / "models" / "model.joblib",
        "metrics": root / "metrics" / "evaluation.json",
        "promoted": root / "models" / "PROMOTED.json",
    }


def _validate(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    frame = pd.read_csv(paths["raw"])
    validate_frame(frame)
    paths["interim"].parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["interim"], index=False)
    return {"rows": int(len(frame)), "path": str(paths["interim"])}


def _preprocess(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    frame = pd.read_csv(paths["interim"])
    validate_frame(frame)
    train_frame, test_frame = train_test_split(
        frame,
        test_size=0.25,
        random_state=42,
        stratify=frame[TARGET].astype(int),
    )
    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(paths["train"], index=False)
    test_frame.to_csv(paths["test"], index=False)
    return {"train_rows": int(len(train_frame)), "test_rows": int(len(test_frame))}


def _train(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    frame = pd.read_csv(paths["train"])
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(frame[FEATURES], frame[TARGET].astype(int))
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths["model"])
    return {"model": str(paths["model"])}


def _evaluate(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    model = joblib.load(paths["model"])
    test_frame = pd.read_csv(paths["test"])
    predictions = model.predict(test_frame[FEATURES])
    labels = test_frame[TARGET].astype(int)
    metrics = {
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }
    passed = quality_gate_passed(metrics)
    payload = {
        **metrics,
        "quality_gate_passed": passed,
        "threshold": float(os.environ.get("QUALITY_GATE_THRESHOLD", "0.60")),
        "dataset_version": _version_label(paths["version"]),
        "dataset_md5": _md5(paths["raw"]),
        "git_commit": _git_head(root),
    }
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    paths["metrics"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log_mlflow(payload, paths["model"])
    return payload


def _quality_gate(root: Path) -> dict[str, Any]:
    payload = json.loads(_paths(root)["metrics"].read_text(encoding="utf-8"))
    if not payload.get("quality_gate_passed"):
        raise RuntimeError(
            f"quality gate failed: f1_score={payload.get('f1_score')} "
            f"threshold={payload.get('threshold')}"
        )
    return {"quality_gate_passed": True, "f1_score": payload.get("f1_score")}


def _promote(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    if not payload.get("quality_gate_passed"):
        raise RuntimeError("refusing to promote a model that failed the quality gate")
    record = {
        "promoted": True,
        "dataset_version": payload.get("dataset_version"),
        "dataset_md5": payload.get("dataset_md5"),
        "git_commit": payload.get("git_commit"),
        "f1_score": payload.get("f1_score"),
        "model": str(paths["model"]),
    }
    paths["promoted"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _log_mlflow(payload: dict[str, Any], model_path: Path) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    allow_local_mlflow_store(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "churn-training"))
    with mlflow.start_run(run_name="churn-training"):
        mlflow.log_params(
            {
                "dataset_version": str(payload["dataset_version"]),
                "dataset_md5": str(payload["dataset_md5"]),
                "git_commit": str(payload["git_commit"]),
                "quality_gate_passed": str(payload["quality_gate_passed"]).lower(),
                "model_type": "logistic_regression",
            }
        )
        mlflow.log_metrics(
            {
                "f1_score": payload["f1_score"],
                "precision": payload["precision"],
                "recall": payload["recall"],
                "accuracy": payload["accuracy"],
            }
        )
        if model_path.exists():
            mlflow.log_artifact(str(model_path))


def _version_label(path: Path) -> str:
    if not path.exists():
        return "unversioned"
    return path.read_text(encoding="utf-8").strip() or "unversioned"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one training stage.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["validate", "preprocess", "train", "evaluate", "quality_gate", "promote", "all"],
    )
    args = parser.parse_args(argv)
    try:
        result = run_stage(args.stage)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
