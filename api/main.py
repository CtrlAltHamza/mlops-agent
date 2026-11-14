"""Inference service deployed to Kubernetes, plus the agent chat endpoint."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent.envfile import load_dotenv
from agent.orchestrator import Agent
from agent.schemas import ChatRequest
from training.schema import FEATURES

load_dotenv()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("mlops.api")

app = FastAPI(title="Churn inference", version=os.environ.get("APP_VERSION", "1.0.0"))
_AGENT: Agent | None = None


class PredictRequest(BaseModel):
    age: float
    tenure_months: float
    monthly_charges: float
    contract_type: int = Field(ge=0, le=2)
    support_calls: float


class PredictResponse(BaseModel):
    churn: int
    app_version: str


def model_path() -> str:
    return os.environ.get("MODEL_PATH", "models/model.joblib")


@lru_cache(maxsize=1)
def load_model() -> Any | None:
    path = model_path()
    if not os.path.exists(path):
        LOGGER.error("model file missing at %s; readiness will fail", path)
        return None
    return joblib.load(path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive", "app_version": app.version}


@app.get("/ready")
def ready() -> dict[str, Any]:
    if load_model() is None:
        raise HTTPException(status_code=503, detail=f"model missing at {model_path()}")
    return {"status": "ready", "model_path": model_path()}


@app.get("/version")
def version() -> dict[str, str]:
    return {"app_version": app.version, "model_path": model_path()}


HTML_PATH = Path(__file__).parent / "index.html"


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    if HTML_PATH.exists():
        return HTML_PATH.read_text(encoding="utf-8")
    return "<h1>MLOps AI Agent</h1><p><a href='/docs'>Swagger API</a></p>"


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    row = pd.DataFrame([[getattr(body, name) for name in FEATURES]], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    return PredictResponse(churn=prediction, app_version=app.version)


@app.post("/chat")
def chat(body: ChatRequest) -> dict[str, Any]:
    response = _agent().handle(body.message, body.request_id)
    return response.model_dump()


def _agent() -> Agent:
    global _AGENT
    if _AGENT is None:
        _AGENT = Agent()
    return _AGENT
