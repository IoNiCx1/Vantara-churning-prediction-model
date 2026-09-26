from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import partial_dependence


def compute_shap_values(model: Any, X: pd.DataFrame, model_type: str = "tree"):
    
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    return explainer, shap_values


def global_feature_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    
    importance = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def select_representative_customers(
    y_proba: np.ndarray, customer_ids: pd.Series
) -> Dict[str, Any]:
    
    idx_low = int(np.argmin(y_proba))
    idx_high = int(np.argmax(y_proba))
    idx_borderline = int(np.argmin(np.abs(y_proba - 0.5)))
    return {
        "low_risk": {"customer_id": customer_ids.iloc[idx_low], "index": idx_low, "proba": float(y_proba[idx_low])},
        "high_risk": {"customer_id": customer_ids.iloc[idx_high], "index": idx_high, "proba": float(y_proba[idx_high])},
        "borderline": {"customer_id": customer_ids.iloc[idx_borderline], "index": idx_borderline, "proba": float(y_proba[idx_borderline])},
    }


def build_lime_explainer(X_train: pd.DataFrame, feature_names: List[str]) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["retained", "churned"],
        mode="classification",
        random_state=42,
    )


def explain_with_lime(
    explainer: LimeTabularExplainer, model: Any, instance: np.ndarray, num_features: int = 10
):
    
    return explainer.explain_instance(instance, model.predict_proba, num_features=num_features)


def compute_partial_dependence(model: Any, X: pd.DataFrame, features: List[str]):
    
    return partial_dependence(model, X, features=features, kind="average")


def plain_language_explanation(
    shap_row: np.ndarray, feature_names: List[str], feature_values: pd.Series, top_n: int = 3
) -> str:
   
    contributions = pd.Series(shap_row, index=feature_names).sort_values(key=np.abs, ascending=False)
    top_features = contributions.head(top_n)

    readable_names = {
        "recency_days": "days since their last purchase",
        "frequency": "how often they order",
        "monetary_total": "their total spend",
        "return_rate": "how often they return items",
        "engagement_score": "their overall engagement score",
        "discount_sensitivity": "how much they rely on discounts",
        "seasonal_concentration": "how seasonal their buying pattern is",
        "purchase_trend_slope": "whether their ordering is speeding up or slowing down",
    }

    clauses = []
    for feat, value in top_features.items():
        direction = "increases" if value > 0 else "decreases"
        readable = readable_names.get(feat, feat.replace("_", " "))
        actual_value = feature_values.get(feat, "N/A")
        clauses.append(f"{readable} ({actual_value}) {direction} churn risk")

    return "This customer's churn risk is driven mainly by: " + "; ".join(clauses) + "."