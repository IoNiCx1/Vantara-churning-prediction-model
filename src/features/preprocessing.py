"""
Preprocessing: encoding, scaling, and persisted artifacts.

Encoders and scalers are fit on the training split only, then persisted
as artifacts so the exact same transformation is applied at inference
time in the live scoring API — never refit on new data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature columns that are scale-sensitive (used by Logistic Regression,
# KNN, SVM, and neural networks). Tree-based models use the unscaled table.
SCALE_SENSITIVE_COLUMNS = [
    "recency_days", "frequency", "monetary_total", "monetary_avg",
    "avg_basket_size", "purchase_trend_slope", "purchase_interval_variance",
    "seasonal_concentration", "return_rate", "discount_sensitivity",
    "engagement_score",
]


@dataclass
class PreprocessingArtifacts:
    scaler: StandardScaler
    feature_columns: List[str]
    scale_sensitive_columns: List[str]


def stratified_split(
    features: pd.DataFrame,
    target_col: str = "churned",
    train_size: float = 0.70,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 stratified train/val/test split."""
    test_size = 1 - train_size - val_size
    train, temp = train_test_split(
        features,
        test_size=(1 - train_size),
        stratify=features[target_col],
        random_state=random_state,
    )
    relative_val = val_size / (val_size + test_size)
    val, test = train_test_split(
        temp,
        test_size=(1 - relative_val),
        stratify=temp[target_col],
        random_state=random_state,
    )
    return train, val, test


def get_feature_columns(features: pd.DataFrame, exclude: List[str]) -> List[str]:
    return [c for c in features.columns if c not in exclude]


def fit_scaler(
    train_df: pd.DataFrame, scale_sensitive_columns: List[str] = SCALE_SENSITIVE_COLUMNS
) -> StandardScaler:
    """Fit StandardScaler on the training split only."""
    cols = [c for c in scale_sensitive_columns if c in train_df.columns]
    scaler = StandardScaler()
    scaler.fit(train_df[cols])
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler, scale_sensitive_columns: List[str] = SCALE_SENSITIVE_COLUMNS
) -> pd.DataFrame:
    cols = [c for c in scale_sensitive_columns if c in df.columns]
    out = df.copy()
    out[cols] = scaler.transform(out[cols])
    return out


def save_artifacts(artifacts: PreprocessingArtifacts, path: str | Path) -> None:
    joblib.dump(artifacts, path)


def load_artifacts(path: str | Path) -> PreprocessingArtifacts:
    return joblib.load(path)