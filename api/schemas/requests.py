"""Pydantic request/response models."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CustomerFeaturesRequest(BaseModel):
    """A single customer's engineered feature values for scoring.

    Field names mirror the columns produced by
    src/features/build_features.py::build_customer_feature_table
    (excluding customer_id, churned, and the variable-width affinity_*
    columns, which don't fit a fixed API schema).
    """

    customer_id: int
    recency_days: float = Field(ge=0)
    frequency: float = Field(ge=0)
    monetary_total: float = Field(ge=0)
    monetary_avg: float = Field(ge=0)
    historical_clv: float = Field(ge=0)
    avg_basket_size: float = Field(ge=0)
    purchase_trend_slope: float
    purchase_interval_variance: float = Field(ge=0)
    seasonal_concentration: float = Field(ge=0, le=1)
    return_rate: float = Field(ge=0, le=1)
    discount_sensitivity: float = Field(ge=0, le=1)
    engagement_score: float = Field(ge=0, le=100)

    @field_validator("customer_id")
    @classmethod
    def customer_id_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("customer_id must be a positive integer")
        return v


class PredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    churn_predicted: bool
    predicted_clv: Optional[float] = None
    model_name: str
    model_version: str
    plain_language_explanation: Optional[str] = None
    scored_at: datetime


class BatchPredictionSummary(BaseModel):
    n_scored: int
    n_failed: int
    errors: List[str] = []


class ModelMetadataResponse(BaseModel):
    model_name: str
    model_version: str
    trained_at: Optional[str] = None
    metrics: Dict[str, float] = {}
    feature_columns: List[str] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_reachable: bool