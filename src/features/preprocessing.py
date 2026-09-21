from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from typing import List,Tuple

import joblib 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

from sklearn.proprocessing import StandardScaler

SCALE_SENSITIVE_COLUMNS = [
    "recency_days","frequency","monetary_total","monetary_avg",
    "avg_basket_size","purchase_trend_slope","purchase_interval_variance",
    "seasonal_concentration","return_rate","discount_sensitivity",
    "engagement_score"
]


@dataclass
class PreprocessingArtifacts:
    scaler:StandardScaler
    feature_columns:List[str]
    scale_sensitive_columns:List[str]

def stratified_split(
    features:pd.DataFrame,
    target_col:str = "churned",
    train_size:float = .70,
    val_size:float = .15,
    random_state = 42,
)-> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    