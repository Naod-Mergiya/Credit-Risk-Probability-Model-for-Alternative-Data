"""FastAPI service exposing model predictions."""

from __future__ import annotations

import joblib
from fastapi import FastAPI, HTTPException
from pathlib import Path

from src.api.pydantic_models import PredictionResponse, Transaction
from src.data_processing import engineer_features
import pandas as pd

MODEL_PATH = Path("artifacts/model.pkl")

app = FastAPI(title="Credit Risk Scoring API", version="0.1.0")


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError("Model artifact not found. Train the model first.")
    app.state.model = joblib.load(MODEL_PATH)


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(tx: Transaction):
    df = pd.DataFrame([tx.dict()])
    X, _ = engineer_features(df)
    model = app.state.model
    prob = float(model.predict_proba(X)[:, 1][0])
    return PredictionResponse(risk_probability=prob)