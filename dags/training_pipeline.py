"""Airflow training DAG.

validate -> preprocess -> train -> evaluate -> quality_gate -> promote

A failed validation or quality gate raises, so Airflow skips promotion.
Training retries transient failures. The quality gate does not retry a real miss.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:  # Airflow 2.x compatibility for local installs
    from airflow.operators.python import PythonOperator

LOGGER = logging.getLogger("ml_training_pipeline")
ROOT = Path(os.environ.get("MLOPS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def run_stage(stage: str) -> None:
    LOGGER.info("starting stage %s in %s", stage, ROOT)
    env = os.environ.copy()
    env["MLOPS_PROJECT_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "training.train", "--stage", stage],
        cwd=ROOT,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with exit code {completed.returncode}")
    LOGGER.info("finished stage %s", stage)


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    description="Validate, preprocess, train, evaluate, and promote only after the quality gate.",
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["mlops", "training"],
) as dag:
    validate = PythonOperator(
        task_id="validate",
        python_callable=run_stage,
        op_args=["validate"],
        retries=1,
    )
    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=run_stage,
        op_args=["preprocess"],
    )
    train = PythonOperator(
        task_id="train",
        python_callable=run_stage,
        op_args=["train"],
        retries=2,
    )
    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=run_stage,
        op_args=["evaluate"],
    )
    quality_gate = PythonOperator(
        task_id="quality_gate",
        python_callable=run_stage,
        op_args=["quality_gate"],
        retries=0,
    )
    promote = PythonOperator(
        task_id="promote",
        python_callable=run_stage,
        op_args=["promote"],
        retries=0,
    )

    validate >> preprocess >> train >> evaluate >> quality_gate >> promote
