# Technical report

This report describes the MLOps agent in this repository, the permissions on each tool, the evidence each graded conversation depends on, and two failures that were injected on purpose.

## System architecture

A single agent receives a natural-language request from the CLI (`python -m agent.main`) or from `POST /chat`. The planner chooses a sequence of tools. The registry executes each tool with a timeout and writes a JSON log line containing the request id, tool name, status, and duration. The composer builds the answer from those results and keeps three sections separate: facts, uncertainties, and recommended actions.

Facts are emitted only when a tool returns `ok` and a payload. If MLflow, DVC, Airflow, or Kubernetes is down, the matching statement is an uncertainty. The answer then says that no operational claim is made. The agent does not fill in an F1 score, a dataset revision, a DAG state, or a Pod status from memory.

The four systems stay behind their own modules:

- DVC and Git identify the dataset file, its MD5, the current commit, and the commands that reproduce a revision.
- MLflow stores runs, metrics, parameters, artifacts, and registry entries. Training writes `dataset_version`, `git_commit`, and `dataset_md5` onto every run so a later question can recover provenance.
- Airflow 3 runs `ml_training_pipeline`. The client uses `POST /auth/token` and the `/api/v2` routes.
- Kubernetes serves the FastAPI inference process. The agent reads Pods, Deployments, Services, events, and logs. Restarts are a separate, confirmed action.

The training order is validate, preprocess, train, evaluate, quality gate, then promote. Promote is downstream of the quality gate. A failed validate or quality-gate task leaves promote unstarted.

`architecture.md` has the same layout in diagram form, plus the permission table.

## Tool definitions and permissions

Read tools always run. State-changing tools run only after both an allowlist check and an explicit `confirm` in the user text. A model planner cannot set `confirm=true` unless that word is in the user message. Shell commands for DVC and Git are fixed argument lists. User text is not concatenated into a shell string.

| Tool | Permission |
| --- | --- |
| `list_datasets`, `get_dataset_version`, `compare_dataset_versions`, `detect_dataset_changes`, `explain_reproduction` | Read DVC metadata and Git. Compare revisions must match a restricted character set. Reproduction commands are printed and not executed. |
| `checkout_dataset` | Requires `confirm=true`. Checks out one approved path. |
| `list_experiments`, `list_runs`, `get_run`, `find_best_run`, `compare_runs`, `list_registered_models` | Read MLflow. `find_best_run` accepts only `f1_score`, `precision`, `recall`, `accuracy`, and `roc_auc`. It always returns `production_ready=false`. The metric leader and the quality-gate candidate are reported separately. |
| `list_dags`, `get_dag_runs`, `get_failed_tasks`, `get_task_diagnostics` | Read the Airflow 3 API. |
| `trigger_training_dag` | Requires `confirm=true` and a DAG id in `AIRFLOW_APPROVED_DAGS` (default `ml_training_pipeline`). |
| `list_workloads`, `get_pod_logs`, `list_services`, `get_deployed_version`, `diagnose_application` | Read the approved namespace, default `mlops-inference`. |
| `restart_deployment` | Requires `confirm=true` and the approved namespace. Any other namespace is rejected and the API is not called. |

Logs pass through a scrubber that masks values for password, token, authorization, secret, and API key fields, and that replaces the raw `AIRFLOW_PASSWORD`, `OPENAI_API_KEY`, and `MLFLOW_TRACKING_TOKEN` if they appear.

## Evidence to capture

Screenshots belong in `screenshots/` at demonstration time. Export the five conversations with:

```powershell
$env:MLFLOW_TRACKING_URI = "file:./mlruns"
python -m agent.main --scenarios
```

That writes `evidence/demo-scenarios.json`. Each object includes the answer and the tool payloads. Capture the following from the live services as well:

- `dvc status` and `data/VERSION`, showing label `v1.0` and the dataset MD5.
- The MLflow run page for experiment `churn-training`, showing `f1_score` and the params `dataset_version`, `git_commit`, and `dataset_md5`.
- The Airflow grid for `ml_training_pipeline`, showing task order and a run state.
- `kubectl get pods,deploy,svc -n mlops-inference` and `kubectl describe` for the inference Deployment, showing two desired replicas, probes, and the ConfigMap.

## Five graded conversations

1. Dataset provenance. "Which dataset version is recorded, and how do I reproduce it?" The plan lists tracked datasets and calls `get_dataset_version` plus `explain_reproduction`. The fact section must quote the label, MD5, and Git commit returned by those tools, then the checkout commands. The commands are not run.
2. Model selection. "Compare the last five training runs and recommend the best candidate based on F1 score." The plan calls `compare_runs` and `find_best_run`. The answer names the metric leader and, separately, a promotion candidate only when `quality_gate_passed` is true. Highest F1 is not described as production-ready.
3. Pipeline execution. "Trigger the model training pipeline and report whether execution succeeded." Without the word `confirm`, `trigger_training_dag` returns `confirmation_required` and no DAG run is created. With `confirm`, the client posts to `/api/v2/dags/ml_training_pipeline/dagRuns`. The answer reports the returned `dag_run_id` and state, and does not claim success until a later status call says so.
4. Deployment diagnosis. "Why is my prediction API unavailable?" The plan calls `diagnose_application`, `get_deployed_version`, and `list_services`. The answer quotes Pod phase, ready flags, restart counts, waiting reasons, events, and log excerpts from the tool payload.
5. Multi-tool reasoning. "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary." The plan is exactly `find_best_run`, `get_dataset_version` (filled from the MLflow params), `diagnose_application`, and `get_dag_runs`. Retraining is not started. A recommendation to retrain is added only when the payload shows a failed quality gate or a failed DAG run, and that sentence says the run was not started. If none of those signals are present, the answer says retraining was not started because no failed quality gate, failed training run, freshness policy, or supplied degradation result is in the evidence.

## Failure 1: validation blocks the pipeline

Root cause: `training.schema` requires `customer_id`, `age`, `tenure_months`, `monthly_charges`, `contract_type`, `support_calls`, and `churn`. A file that only has `customer_id` and `age` fails `validate_frame` with "missing columns".

What the agent and the DAG do: `run_stage("validate")` raises `ValueError`. In Airflow the `validate` task fails, and `preprocess`, `train`, `evaluate`, `quality_gate`, and `promote` stay upstream-failed. `models/PROMOTED.json` is not written.

How to inject it:

```powershell
$env:INJECT_VALIDATION_FAILURE = "true"
python -m training.train --stage validate
```

Resolution: restore the seven columns, unset `INJECT_VALIDATION_FAILURE`, and rerun from `validate`. Promotion is still impossible until the quality gate passes.

Automated coverage: `test_validation_failure_blocks_the_stage`.

## Failure 2: quality gate and an unready API

Quality gate. Root cause: F1 is below `QUALITY_GATE_THRESHOLD`. With the threshold set to 0.99 and a recorded F1 of 0.40, `quality_gate_passed` is false. `run_stage("quality_gate")` raises, and `run_stage("promote")` raises "refusing to promote" if it is invoked anyway. `models/PROMOTED.json` is absent. The Airflow `quality_gate` task uses `retries=0`, so a real miss is not retried into a promotion. `train` still uses two retries for transient training errors.

```powershell
$env:QUALITY_GATE_THRESHOLD = "0.99"
python -m training.train --stage all
```

Resolution: lower the threshold to the project default of 0.60, or replace the model so the recorded F1 clears the bar, then rerun `quality_gate` and `promote`.

Unready API. Root cause: `kubernetes/failure-unready.yaml` keeps the image but replaces the process with a Python one-liner that prints a message and sleeps. Nothing listens on port 8000. The readiness probe calls `/ready` and the liveness probe calls `/health`, so both fail. The Deployment label version is `1.0.0-broken`. `diagnose_application` reports the Pods as not ready, includes the probe failure events, and attaches the log line from the sleeping process.

Resolution: `kubectl apply -f kubernetes/deployment.yaml` restores the image command, the `/health` and `/ready` routes, and version `1.0.0`.

Automated coverage: `test_quality_gate_blocks_promotion` and `test_manifests_meet_the_deployment_contract`. The unready manifest is the cluster-side drill and is not applied by tests.

## Known limitations

- The deterministic planner covers the graded wording. An optional OpenAI-compatible planner is used only when `LLM_PLANNER=true` and `OPENAI_API_KEY` is set, and any error falls back to the deterministic plan.
- Airflow log retrieval depends on the Airflow 3 task-log route. A deployment that disables task logs yields a diagnostic error string instead of an excerpt.
- The DVC remote in `.dvc/config` is a local directory, `../mlops-dvc-remote`. A shared remote would be a later change and must not put credentials in the repository.
- Kubernetes mutations other than a confirmed restart of `churn-inference` in `mlops-inference` are out of scope.
- No freshness window is configured. Age of a run is not treated as a reason to retrain.
- CI mocks external systems. A green GitHub Actions run does not prove the live Airflow or cluster is healthy. That is what the demonstration scenarios are for.
