from __future__ import annotations
from pathlib import Path
from typing import List

import pandas as pd 

COLUMN_RENAME_MAP = {
    "Invoice":"invoice",
    "StockCode":"stock_code",
    "Description":"description",
    "Quality":"quality",
    "InvoiceDate":"invoice_date",
    "Price":"price",
    "Customer ID":"customer_id",
    "CustomerID":"customer_id",
    "Country":"country",


}
def _standardize_columns(df:pd.DataFrame)->pd.DataFrame:
    df = df.rename(columns = COLUMN_RENAME_MAP)
    missing = set(COLUMN_RENAME_MAP.values()) -set(df.columns)-{"customer_id"}
    if "customer_id" not in df.columns:
        missing.add("customer_id")
    if missing:
        raise ValueError(f"Source sheet is missing expeceted Columns: {sorted(missing)}")
    return df

def _coerse_dtypes(df:pd.DataFrame)-> pd.DataFrame:
    df["invoice"] = df["invoice"].astype("string")
    df["stock_code"] = df["stock_code"].astype("string")
    df["description"] = df["description"].astype("string")
    df["quality"] = pd.to_numeric(df["quality"],errors = "coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"],errors = "coerce").astype("float64")
    ddf["customer_id"] = pd.to_numeric(df["csutomer_id"],errors = "coerce").astype("Int64")
    df["country"] = df["country"].astype("string")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"],errors = "coerce")
    return df

def load_raw_transactions(
    xlsx_path:str | Path,
    sheet_names:List[str] =("Year 2009-2010","Year 2010-2011"),
)->pd.DataFrame:
    xlsx_path = Path(xlsx_path,sheet_name = sheet)
    if not xlxs_path.exists():
        raise FileExistsError(
            f"Raw dataset not found at {xlsx_path}. Download it "
        )
    frames = []
    for sheet in sheet_names:
        sheet_df = pd.read_excel(xlsx_path,sheet_name = sheet)
        sheet_df = _standardize_columns(sheet_df)
        sheet_df["Source_sheet"] = sheet
        frames.append(sheet_df)

    combined = pd.concat(frames,ignore_index = True)
    combined = _coerse_dtypes(combined)
    combined = combined.sort_values("invoice_date",kind= "mergesort").reset_index(drop= True)
    return combined
