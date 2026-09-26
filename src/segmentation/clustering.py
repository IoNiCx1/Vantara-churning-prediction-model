
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

def find_optimal_k(
    X: np.ndarray, k_range: range = range(2, 11)
) -> Tuple[int, pd.DataFrame]:
    
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
        rows.append({"k": k, "inertia": km.inertia_, "silhouette_score": sil})
    diagnostics = pd.DataFrame(rows)
    best_k = int(diagnostics.loc[diagnostics["silhouette_score"].idxmax(), "k"])
    return best_k, diagnostics


def run_kmeans(X: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 10) -> np.ndarray:
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(X)


def clustering_quality(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    valid = labels != -1  # exclude DBSCAN noise points from scoring
    if valid.sum() < 2 or len(set(labels[valid])) < 2:
        return {"silhouette_score": float("nan"), "davies_bouldin_index": float("nan")}
    return {
        "silhouette_score": silhouette_score(X[valid], labels[valid]),
        "davies_bouldin_index": davies_bouldin_score(X[valid], labels[valid]),
    }


def profile_segments(
    features: pd.DataFrame, labels: np.ndarray, cluster_col: str = "cluster"
) -> pd.DataFrame:
    
    df = features.copy()
    df[cluster_col] = labels

    profile_cols = [
        "recency_days", "frequency", "monetary_total", "engagement_score",
        "return_rate", "seasonal_concentration",
    ]
    profile_cols = [c for c in profile_cols if c in df.columns]

    summary = df.groupby(cluster_col)[profile_cols].agg(["mean", "median", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    summary["business_label"] = summary.apply(
        lambda row: _label_segment(row, cluster_col), axis=1
    )
    return summary


def _label_segment(row: pd.Series, cluster_col: str) -> str:
    
    if row.get(cluster_col) == -1:
        return "Unclassified / noise (DBSCAN)"

    recency = row.get("recency_days_mean", np.nan)
    frequency = row.get("frequency_mean", np.nan)
    monetary = row.get("monetary_total_mean", np.nan)
    seasonality = row.get("seasonal_concentration_mean", np.nan)

    if pd.isna(recency) or pd.isna(frequency) or pd.isna(monetary):
        return "Unclassified"

    return _rule_based_label(recency, frequency, monetary, seasonality)


def _rule_based_label(recency: float, frequency: float, monetary: float, seasonality: float) -> str:
    
    if recency > 180 and monetary > 0:
        return "Lapsed high-value" if monetary > 500 else "Lapsed low-value"
    if frequency > 10 and monetary > 500:
        return "Loyal high-value"
    if frequency > 10 and monetary < 200:
        return "High-frequency low-value"
    if recency < 30 and frequency <= 2:
        return "New / recently acquired"
    if seasonality > 0.5:
        return "Seasonal gift shopper"
    return "Steady mid-value shopper"