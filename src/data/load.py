"""
Data collection.

Loads both sheets of the Online Retail II workbook, standardizes column
names/types, and combines them into a single chronologically ordered
transaction table before any cleaning logic is applied.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

# Canonical column names used throughout the pipeline (snake_case).
COLUMN_RENAME_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "CustomerID": "customer_id",
    "Country": "country",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_RENAME_MAP)
    missing = set(COLUMN_RENAME_MAP.values()) - set(df.columns) - {"customer_id"}
    if "customer_id" not in df.columns:
        missing.add("customer_id")
    if missing:
        raise ValueError(f"Source sheet is missing expected columns: {sorted(missing)}")
    return df


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["invoice"] = df["invoice"].astype("string")
    df["stock_code"] = df["stock_code"].astype("string")
    df["description"] = df["description"].astype("string")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float64")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    df["country"] = df["country"].astype("string")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    return df


def load_raw_transactions(
    xlsx_path: str | Path,
    sheet_names: List[str] = ("Year 2009-2010", "Year 2010-2011"),
) -> pd.DataFrame:
    """Load both sheets of the source workbook and combine chronologically."""
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {xlsx_path}. Download it (UCI 'Online "
            "Retail II', id=502) and place it there."
        )

    frames = []
    for sheet in sheet_names:
        sheet_df = pd.read_excel(xlsx_path, sheet_name=sheet)
        sheet_df = _standardize_columns(sheet_df)
        sheet_df["source_sheet"] = sheet
        frames.append(sheet_df)

    combined = pd.concat(frames, ignore_index=True)
    combined = _coerce_dtypes(combined)
    combined = combined.sort_values("invoice_date", kind="mergesort").reset_index(drop=True)
    return combined
