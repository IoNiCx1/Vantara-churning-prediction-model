"""
Classical machine learning models.

Trains Logistic Regression, Decision Tree, Random Forest, XGBoost,
LightGBM, and KNN for churn classification, with cross-validated
hyperparameter search on the training set only (the test set is touched
exactly once, at final evaluation).

Every run is logged (params, metrics, training time) to a CSV experiment
log.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

RANDOM_SEED = 42
CV_FOLDS = 5


def _cv_search(estimator, param_grid: dict, X_train, y_train) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(
        estimator, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    return search


def _classification_metrics(y_true, y_pred, y_proba) -> Dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_logistic_regression(X_train, y_train) -> GridSearchCV:
    """Interpretable baseline; requires scaled features."""
    grid = {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"], "max_iter": [1000]}
    return _cv_search(LogisticRegression(random_state=RANDOM_SEED), grid, X_train, y_train)


def train_decision_tree(X_train, y_train) -> GridSearchCV:
    """Baseline non-linear, human-readable model."""
    grid = {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10]}
    return _cv_search(DecisionTreeClassifier(random_state=RANDOM_SEED), grid, X_train, y_train)


def train_random_forest(X_train, y_train) -> GridSearchCV:
    """Robust churn classifier; tuned via cross-validated search."""
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 5],
    }
    return _cv_search(
        RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
        grid, X_train, y_train,
    )


def train_xgboost(X_train, y_train) -> GridSearchCV:
    """Primary production candidate."""
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed. `pip install xgboost`.")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "n_estimators": [200],
    }
    return _cv_search(
        XGBClassifier(
            random_state=RANDOM_SEED, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
        grid, X_train, y_train,
    )


def train_lightgbm(X_train, y_train) -> GridSearchCV:
    """Comparison against XGBoost, native categorical handling."""
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed. `pip install lightgbm`.")
    grid = {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31],
        "n_estimators": [200],
    }
    return _cv_search(
        LGBMClassifier(random_state=RANDOM_SEED, class_weight="balanced", verbosity=-1),
        grid, X_train, y_train,
    )


def train_knn(X_train, y_train) -> GridSearchCV:
    """Distance-based comparison baseline; requires scaled features."""
    grid = {"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]}
    return _cv_search(KNeighborsClassifier(), grid, X_train, y_train)


CLASSICAL_MODEL_TRAINERS = {
    "logistic_regression": (train_logistic_regression, "scaled"),
    "decision_tree": (train_decision_tree, "unscaled"),
    "random_forest": (train_random_forest, "unscaled"),
    "xgboost": (train_xgboost, "unscaled"),
    "lightgbm": (train_lightgbm, "unscaled"),
    "knn": (train_knn, "scaled"),
}


def train_all_classical_models(
    X_train_scaled: pd.DataFrame,
    X_train_unscaled: pd.DataFrame,
    y_train: pd.Series,
    X_test_scaled: pd.DataFrame,
    X_test_unscaled: pd.DataFrame,
    y_test: pd.Series,
    model_artifacts_dir: str | Path,
    log_path: str | Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Train every classical model, evaluate once on the held-out test set,
    persist artifacts, and append a row per model to the experiment log."""
    model_artifacts_dir = Path(model_artifacts_dir)
    model_artifacts_dir.mkdir(parents=True, exist_ok=True)

    fitted = {}
    log_rows: List[Dict[str, Any]] = []

    for name, (trainer_fn, feature_set) in CLASSICAL_MODEL_TRAINERS.items():
        X_train = X_train_scaled if feature_set == "scaled" else X_train_unscaled
        X_test = X_test_scaled if feature_set == "scaled" else X_test_unscaled

        try:
            start = time.time()
            search = trainer_fn(X_train, y_train)
            train_time = time.time() - start
        except ImportError as e:
            log_rows.append({"model": name, "status": "skipped", "reason": str(e)})
            continue

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        metrics = _classification_metrics(y_test, y_pred, y_proba)

        joblib.dump(best_model, model_artifacts_dir / f"{name}.joblib")
        fitted[name] = best_model

        log_rows.append(
            {
                "model": name,
                "status": "trained",
                "best_params": search.best_params_,
                "cv_best_roc_auc": search.best_score_,
                "train_time_sec": round(train_time, 2),
                **metrics,
            }
        )

    log_df = pd.DataFrame(log_rows)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    return fitted, log_df


def train_clv_regressor(X_train, y_train, X_test, y_test) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Random Forest regressor predicting future/realized CLV."""
    grid = {"n_estimators": [200, 400], "max_depth": [5, 10, None]}
    search = _cv_search(RandomForestRegressor(random_state=RANDOM_SEED), grid, X_train, y_train)
    best = search.best_estimator_
    y_pred = best.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
    }
    return best, metrics
