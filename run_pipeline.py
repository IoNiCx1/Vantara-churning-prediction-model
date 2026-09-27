"""
End-to-end pipeline orchestrator.

Runs the full path from raw data to trained, evaluated, and persisted
models with a single command:

    python run_pipeline.py

If data/raw/online_retail_II.xlsx is absent, a synthetic dataset matching
the real schema is generated automatically so the pipeline is runnable
out of the box; swap in the real file for production results.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.clean import run_cleaning_pipeline
from src.data.load import load_raw_transactions
from src.data.synthetic import write_synthetic_workbook
from src.data.validate import run_validation
from src.features.build_features import build_customer_feature_table
from src.features.preprocessing import (
    apply_scaler, fit_scaler, get_feature_columns, save_artifacts,
    stratified_split, PreprocessingArtifacts,
)
from src.models.train_classical import train_all_classical_models
from src.segmentation.clustering import find_optimal_k, profile_segments, run_dbscan, run_kmeans
from src.utils.config import load_config
from src.utils.logging_setup import get_logger

logger = get_logger("run_pipeline")

NON_FEATURE_COLUMNS = ["customer_id", "churned"]

# Product-affinity columns are variable-width (top-N product codes observed
# in the training data), so they make a poor fit for a stable API request
# schema. They're used for segmentation below, but excluded from the
# classifier's feature set so CustomerFeaturesRequest stays a fixed schema.
AFFINITY_PREFIX = "affinity_"


def main() -> None:
    cfg = load_config()
    raw_path = Path(cfg["paths"]["raw_data"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    model_dir = Path("model_artifacts")
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data collection --------------------------------------------------
    if not raw_path.exists():
        logger.warning(
            f"{raw_path} not found — generating a synthetic dataset for a runnable "
            "demo. Replace with the real Online Retail II file for production use."
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        write_synthetic_workbook(str(raw_path), n_customers=500)

    logger.info("Loading raw transactions")
    raw = load_raw_transactions(raw_path, cfg["sheets"])

    # 2. Validation ---------------------------------------------------------
    logger.info("Running data validation")
    validation_result = run_validation(raw, cfg)
    validation_result.raise_if_failed()
    logger.info("Validation passed")

    # 3. Cleaning -----------------------------------------------------------
    logger.info("Running cleaning pipeline")
    cleaned = run_cleaning_pipeline(
        raw,
        cfg["cleaning"]["non_product_stockcodes"],
        cfg["cleaning"]["outlier_iqr_multiplier"],
    )
    logger.info(f"Removed {cleaned.attrs.get('n_duplicates_removed', 0)} duplicate rows")

    # 4. Feature engineering -------------------------------------------------
    cutoff_date = cleaned["invoice_date"].max() - pd.Timedelta(
        days=cfg["features"]["churn_definition_days"]
    )
    logger.info(f"Building customer feature table at cutoff_date={cutoff_date}")
    features = build_customer_feature_table(
        cleaned, cutoff_date, cfg["features"]["churn_definition_days"]
    )
    features.to_parquet(processed_dir / "customer_features.parquet")
    logger.info(f"Feature table shape: {features.shape}")

    if features["churned"].nunique() < 2:
        logger.warning(
            "Only one churn class present in the feature table — skipping "
            "classifier training. Check your churn_definition_days or dataset size."
        )
        return

    # 5. Preprocessing --------------------------------------------------------
    logger.info("Splitting train/val/test and fitting preprocessing artifacts")
    train_df, val_df, test_df = stratified_split(features, target_col="churned")
    feature_cols = [
        c for c in get_feature_columns(features, exclude=NON_FEATURE_COLUMNS)
        if not c.startswith(AFFINITY_PREFIX)
    ]

    scaler = fit_scaler(train_df)
    save_artifacts(
        PreprocessingArtifacts(scaler=scaler, feature_columns=feature_cols, scale_sensitive_columns=[]),
        model_dir / "preprocessing_artifacts.joblib",
    )

    X_train_unscaled = train_df[feature_cols]
    X_test_unscaled = test_df[feature_cols]
    X_train_scaled = apply_scaler(X_train_unscaled, scaler)
    X_test_scaled = apply_scaler(X_test_unscaled, scaler)
    y_train, y_test = train_df["churned"], test_df["churned"]

    # 6. Classical model training ---------------------------------------------
    logger.info("Training classical ML models (this may take a few minutes)")
    fitted_models, comparison_log = train_all_classical_models(
        X_train_scaled, X_train_unscaled, y_train,
        X_test_scaled, X_test_unscaled, y_test,
        model_artifacts_dir=model_dir,
        log_path=processed_dir / "classical_model_comparison.csv",
    )
    logger.info("Model comparison:\n" + comparison_log[
        [c for c in ["model", "status", "roc_auc", "recall", "f1"] if c in comparison_log.columns]
    ].to_string())

    trained_rows = comparison_log[comparison_log["status"] == "trained"]
    best_model_name = (
        trained_rows.sort_values("roc_auc", ascending=False)["model"].iloc[0]
        if not trained_rows.empty else None
    )
    if best_model_name:
        (model_dir / "best_model_name.json").write_text(json.dumps({"best_model": best_model_name}))
        logger.info(f"Best model by ROC-AUC: {best_model_name}")

    # 7. Segmentation -----------------------------------------------------------
    logger.info("Running customer segmentation")
    segmentation_cols = [c for c in get_feature_columns(features, exclude=NON_FEATURE_COLUMNS)]
    segmentation_scaler = fit_scaler(features, scale_sensitive_columns=segmentation_cols)
    X_all_scaled = apply_scaler(features[segmentation_cols], segmentation_scaler, segmentation_cols)
    best_k, k_diagnostics = find_optimal_k(X_all_scaled.values, k_range=range(2, 8))
    kmeans_model, kmeans_labels = run_kmeans(X_all_scaled.values, best_k)

    kmeans_profiles = profile_segments(features, kmeans_labels)
    kmeans_profiles.to_csv(processed_dir / "segment_profiles.csv", index=False)
    logger.info(f"K-Means: k={best_k}, profiles written to segment_profiles.csv")

    # 8. Persist run metadata ----------------------------------------------------
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_customers": int(len(features)),
        "cutoff_date": str(cutoff_date),
        "best_model": best_model_name,
        "kmeans_k": best_k,
        "metrics": (
            trained_rows.set_index("model").loc[best_model_name][
                ["accuracy", "precision", "recall", "f1", "roc_auc"]
            ].to_dict()
            if best_model_name else {}
        ),
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Pipeline complete. Run `uvicorn api.main:app --reload` to serve predictions, "
                "or `streamlit run frontend/dashboard.py` for the dashboard.")


if __name__ == "__main__":
    main()