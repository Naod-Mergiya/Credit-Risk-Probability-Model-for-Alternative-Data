"""Pydantic request/response models for API."""

from pydantic import BaseModel, Field
from typing import List, Any


class Transaction(BaseModel):
    TransactionId: str
    Amount: float


class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., ge=0, le=1)