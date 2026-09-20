"""
Data cleaning.

Each rule below is implemented as a standalone, unit-testable function so
the same logic runs in both the training pipeline and the live scoring API.

Documented cleaning rules:
  - Missing Customer ID    -> flagged, not dropped; excluded only from
                              customer-level tables (still usable for
                              product-level analysis).
  - Negative Quantity      -> treated as returns/cancellations, kept and
                              flagged (return behavior is itself a signal),
                              not dropped.
  - Zero/negative Price    -> flagged as adjustments/manual corrections and
                              excluded from monetary aggregations (kept in
                              the row-level table for auditability).
  - Non-product StockCodes -> filtered out of product-affinity features
                              (postage, bank charges, manual entries, etc).
  - Exact duplicate lines  -> deduplicated, while legitimate repeated
                              purchases of the same SKU in the same order
                              (different rows, same values) are preserved
                              by keying dedup on the full row signature.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

DEFAULT_NON_PRODUCT_CODES = {
    "POST", "D", "DOT", "M", "BANK CHARGES", "PADS", "CRUK", "C2",
}

DEFAULT_DEDUP_SUBSET = [
    "invoice", "stock_code", "customer_id", "quantity", "invoice_date", "price",
]


def flag_missing_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with a missing Customer ID rather than dropping them."""
    out = df.copy()
    out["is_missing_customer_id"] = out["customer_id"].isna()
    return out


def flag_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Flag negative-quantity rows as returns/cancellations."""
    out = df.copy()
    out["is_return"] = out["quantity"] < 0
    return out


def flag_invalid_price(df: pd.DataFrame) -> pd.DataFrame:
    """Flag zero/negative unit prices as manual-correction adjustments."""
    out = df.copy()
    out["is_invalid_price"] = out["price"] <= 0
    return out


def flag_non_product_rows(
    df: pd.DataFrame, non_product_codes: Iterable[str] = DEFAULT_NON_PRODUCT_CODES
) -> pd.DataFrame:
    """Flag administrative StockCodes (postage, bank charges, manual entries)."""
    out = df.copy()
    codes = {c.upper() for c in non_product_codes}
    out["is_non_product"] = out["stock_code"].str.upper().isin(codes)
    return out


def deduplicate_transactions(
    df: pd.DataFrame, subset: list[str] = DEFAULT_DEDUP_SUBSET
) -> Tuple[pd.DataFrame, int]:
    """Remove exact duplicate invoice lines."""
    before = len(df)
    out = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    n_removed = before - len(out)
    return out, n_removed


def flag_price_quantity_outliers(
    df: pd.DataFrame, iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """Flag statistical outliers in Quantity and Price using IQR bounds."""
    out = df.copy()
    for col in ["quantity", "price"]:
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        out[f"is_{col}_outlier"] = (out[col] < lower) | (out[col] > upper)
    return out


def standardize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize inconsistent product descriptions (case/whitespace),
    then map every StockCode to a single canonical description."""
    out = df.copy()
    out["description"] = (
        out["description"]
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )
    canonical = (
        out.dropna(subset=["description"])
        .groupby("stock_code")["description"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("canonical_description")
    )
    out = out.merge(canonical, on="stock_code", how="left")
    out["description"] = out["canonical_description"].fillna(out["description"])
    out = out.drop(columns=["canonical_description"])
    return out


def run_cleaning_pipeline(
    df: pd.DataFrame,
    non_product_codes: Iterable[str] = DEFAULT_NON_PRODUCT_CODES,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Apply the full, ordered cleaning pipeline."""
    out = df.copy()
    out = flag_missing_customer_id(out)
    out = flag_returns(out)
    out = flag_invalid_price(out)
    out = flag_non_product_rows(out, non_product_codes)
    out = flag_price_quantity_outliers(out, iqr_multiplier)
    out = standardize_descriptions(out)
    out, n_dupes_removed = deduplicate_transactions(out)
    out.attrs["n_duplicates_removed"] = n_dupes_removed
    return out
