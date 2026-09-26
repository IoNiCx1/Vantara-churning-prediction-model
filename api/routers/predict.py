"""
Prediction router: single-customer prediction, batch-scoring (CSV
upload), model-metadata, and health-check endpoints.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.db import Prediction, get_db
from api.schemas.requests import (
    BatchPredictionSummary, CustomerFeaturesRequest, HealthResponse,
    ModelMetadataResponse, PredictionResponse,
)
from src.explainability.shap_lime import plain_language_explanation
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)
router = APIRouter()

MODEL_ARTIFACTS_DIR = Path("model_artifacts")
MODEL_VERSION = "1.0.0"
DEFAULT_MODEL_NAME = "xgboost"  # fallback if no training run has completed yet

_model = None
_model_name: Optional[str] = None
FEATURE_COLUMNS = [
    "recency_days", "frequency", "monetary_total", "monetary_avg", "historical_clv",
    "avg_basket_size", "purchase_trend_slope", "purchase_interval_variance",
    "seasonal_concentration", "return_rate", "discount_sensitivity", "engagement_score",
]


def _resolve_model_name() -> str:
    """The production model is selected dynamically by run_pipeline.py
    based on the ROC-AUC comparison and recorded in best_model_name.json.
    Falls back to DEFAULT_MODEL_NAME if the pipeline hasn't been run yet.
    """
    best_model_path = MODEL_ARTIFACTS_DIR / "best_model_name.json"
    if best_model_path.exists():
        return json.loads(best_model_path.read_text())["best_model"]
    return DEFAULT_MODEL_NAME


def _load_model():
    global _model, _model_name
    if _model is None:
        _model_name = _resolve_model_name()
        model_path = MODEL_ARTIFACTS_DIR / f"{_model_name}.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found at {model_path}. Run the training "
                "pipeline (run_pipeline.py) before starting the API.",
            )
        _model = joblib.load(model_path)
        logger.info(f"Loaded model artifact '{_model_name}' from {model_path}")
    return _model


def _score_single(features: CustomerFeaturesRequest) -> PredictionResponse:
    model = _load_model()
    row = pd.DataFrame([features.model_dump(exclude={"customer_id"})])[FEATURE_COLUMNS]

    proba = float(model.predict_proba(row)[:, 1][0])
    predicted = proba >= 0.5

    explanation = None
    tree_based_models = ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier", "DecisionTreeClassifier")
    if type(model).__name__ in tree_based_models:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(row)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
            explanation = plain_language_explanation(
                shap_values[0], FEATURE_COLUMNS, row.iloc[0]
            )
        except Exception as e:  # noqa: BLE001 — explanation is best-effort, never blocks scoring
            logger.warning(f"Could not compute SHAP explanation for customer {features.customer_id}: {e}")
    else:
        logger.info(
            f"Model type {type(model).__name__} is not tree-based; skipping SHAP "
            "TreeExplainer."
        )

    return PredictionResponse(
        customer_id=features.customer_id,
        churn_probability=proba,
        churn_predicted=predicted,
        predicted_clv=None,
        model_name=_model_name or DEFAULT_MODEL_NAME,
        model_version=MODEL_VERSION,
        plain_language_explanation=explanation,
        scored_at=datetime.now(timezone.utc),
    )


@router.post("/predict", response_model=PredictionResponse)
def predict_single(features: CustomerFeaturesRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    """Single-customer churn prediction with plain-language explanation."""
    result = _score_single(features)

    db.add(
        Prediction(
            customer_id=result.customer_id,
            churn_probability=result.churn_probability,
            predicted_clv=result.predicted_clv,
            model_name=result.model_name,
            model_version=result.model_version,
        )
    )
    db.commit()
    return result


@router.post("/predict/batch", response_model=BatchPredictionSummary)
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db)) -> BatchPredictionSummary:
    """Batch scoring via CSV upload. Expects one row per customer with the
    same columns as CustomerFeaturesRequest."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    n_scored, errors = 0, []
    for _, row in df.iterrows():
        try:
            features = CustomerFeaturesRequest(**row.to_dict())
            result = _score_single(features)
            db.add(
                Prediction(
                    customer_id=result.customer_id,
                    churn_probability=result.churn_probability,
                    predicted_clv=result.predicted_clv,
                    model_name=result.model_name,
                    model_version=result.model_version,
                )
            )
            n_scored += 1
        except Exception as e:  # noqa: BLE001 — one bad row shouldn't fail the whole batch
            errors.append(f"Row with customer_id={row.get('customer_id', '?')}: {e}")

    db.commit()
    return BatchPredictionSummary(n_scored=n_scored, n_failed=len(errors), errors=errors)


@router.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata() -> ModelMetadataResponse:
    """Model metadata endpoint."""
    metrics_path = MODEL_ARTIFACTS_DIR / "training_metadata.json"
    metrics = {}
    trained_at = None
    if metrics_path.exists():
        meta = json.loads(metrics_path.read_text())
        metrics = meta.get("metrics", {})
        trained_at = meta.get("trained_at")

    return ModelMetadataResponse(
        model_name=_resolve_model_name(),
        model_version=MODEL_VERSION,
        trained_at=trained_at,
        metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
    )


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """Health-check endpoint."""
    model_path = MODEL_ARTIFACTS_DIR / f"{_resolve_model_name()}.joblib"
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok" if (model_path.exists() and db_ok) else "degraded",
        model_loaded=model_path.exists(),
        database_reachable=db_ok,
    )