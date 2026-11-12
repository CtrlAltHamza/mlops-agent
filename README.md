# MLOps agent

A conversational agent that answers operational questions from DVC, MLflow, Apache Airflow 3, and Kubernetes. Claims in an answer come from tool responses. If a tool fails, the agent says so and does not fill in metrics, dataset versions, or cluster state.

Repository: https://github.com/CtrlAltHamza/mlops-agent

## What it covers

- Dataset identity, comparison, change detection, and reproduction steps through DVC and Git.
- Experiment search, metric comparison, and registry reads through `MlflowClient`.
- DAG listing, run status, approved triggers, failed tasks, and log excerpts through the Airflow 3 `/api/v2` API.
- Pod, Deployment, and Service inspection, plus restarts that require confirmation and stay inside one namespace.
- A training DAG: validate, preprocess, train, evaluate, quality gate, promote.
- A FastAPI inference service with a two-replica Kubernetes deployment.

## Setup

Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

Use the same `MLFLOW_TRACKING_URI` for training and the agent. The sample file points at a local MLflow server. For an offline run, set it to `file:./mlruns`. Training and the MLflow tools opt in to that directory store, which current MLflow releases otherwise reject.

Do not commit `.env`. Real passwords, API keys, and kubeconfig files are gitignored.

## Train and track

```powershell
$env:MLFLOW_TRACKING_URI = "file:./mlruns"
python -m training.train --stage all
dvc repro
```

`data/VERSION` is the dataset label. Training stores that label, the dataset MD5, and `git rev-parse HEAD` as MLflow params. The quality gate refuses promotion when F1 is below `QUALITY_GATE_THRESHOLD` (default 0.60).

To reproduce a recorded dataset revision:

```powershell
git checkout <commit>
dvc checkout
dvc repro
```

The DVC remote in `.dvc/config` is a local directory, `../mlops-dvc-remote`.

## MLflow

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Or start the bundled compose file: `docker compose up mlflow`.

## Airflow 3

Use the Airflow 3.0.6 Docker Compose quick start so the API matches this client:

```powershell
curl -LfO https://airflow.apache.org/docs/apache-airflow/3.0.6/docker-compose.yaml
```

Point that stack at this repo's `dags` directory, install this project in the Airflow image or set `PYTHONPATH` to the repo, and set `MLOPS_PROJECT_ROOT` to the repo path. The DAG id is `ml_training_pipeline`.

The agent authenticates with `POST /auth/token` and calls `/api/v2`. Only DAG ids in `AIRFLOW_APPROVED_DAGS` can be triggered, and only after the user includes the word `confirm`.

A failed `validate` or `quality_gate` task raises, so `promote` does not run. `train` retries twice. `quality_gate` does not retry.

## Kubernetes

Build the inference image after training so `models/model.joblib` is present:

```powershell
docker build -t mlops-agent:1.0.0 .
kind create cluster
kind load docker-image mlops-agent:1.0.0
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

The deployment requests two replicas, a ClusterIP Service, readiness and liveness probes, CPU and memory requests and limits, and the `churn-inference-config` ConfigMap. The approved namespace is `mlops-inference`.

`kubectl apply -f kubernetes/failure-unready.yaml` replaces the process with one that does not listen, so the readiness probe fails. Use that only for the failure demo, then re-apply `kubernetes/deployment.yaml`.

## Agent

```powershell
python -m agent.main "Which dataset version is recorded, and how do I reproduce it?"
python -m agent.main "Compare the last five training runs and recommend the best candidate based on F1 score."
python -m agent.main "Why is my prediction API unavailable?"
python -m agent.main --scenarios
```

`--scenarios` writes `evidence/demo-scenarios.json`.

Trigger a pipeline only with an explicit confirmation:

```powershell
python -m agent.main "Trigger the model training pipeline and report whether execution succeeded. confirm"
```

Restart a deployment the same way. A restart aimed at any other namespace is rejected.

The chat API is `POST /chat` on the FastAPI app. Prediction is `POST /predict`. Liveness is `GET /health`. Readiness is `GET /ready` and returns 503 until the model file exists.

Set `LLM_PLANNER=true` and `OPENAI_API_KEY` to let a model propose tool calls. The deterministic planner remains the fallback. State-changing tools still ignore a model-supplied confirmation unless the user text itself contains `confirm`.

## Tests

```powershell
ruff check .
pytest
```

GitHub Actions runs the same lint and tests on pull requests and on pushes to `main`. External systems are not required. Kubernetes, MLflow, and Airflow calls in the tests are fakes or a temporary file store.

## Layout

`architecture.md` describes the runtime and the permission model. `REPORT.md` records the graded conversations and the two injected failures.

## Failure drills

Validation failure:

```powershell
$env:INJECT_VALIDATION_FAILURE = "true"
python -m training.train --stage validate
```

Quality-gate failure:

```powershell
$env:QUALITY_GATE_THRESHOLD = "0.99"
python -m training.train --stage all
```

The gate exits non-zero and does not write `models/PROMOTED.json`. In Airflow, that failed task prevents `promote`.
