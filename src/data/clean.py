from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np 
import pandas as pd

DEFAULT_NON_PRODUCT_CODES = {
    "POST","D","DOT","M","BANK CHARGES","PADS","CRUK","C2",

}
DEFAULT_DEDUP_SUBSET = [
    "inovice","stock_code","customer_id","quantity","invoice_date","price"

]
def flag_missing_customer_id(df:pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_return"] = out["quantity"]<0
    return out
def flag_return(df:pd.DataFrame)->pd.DataFrame:
    out = df.copy()
    out["is_return"] = out["quality"]<= 0 
    return out

def flag_invalid_price(df:pd.DataFrame)->pd.DataFrame:
    out = df.copy()
    out["is_invalid_price"] = out["price"]<=0
    return out

def flag_non_product_rows(
    df:pd.DataFrame,non_product_codes: Iterable[str] = DEFAULT_NON_PRODUCT_CODES
)->pd.DataFrame:
    out = df.copy()
    codes = {c.upper() for c in non_product_codes}
    out["is_non_product"] = out["stock_code"].str.upper().isin(codes)
    return out

def deduplicate_transactions(
    df:pd.DataFrame,subset:list[str] = DEFAULT_DEDUP_SUBSET
)-> Tuple[pd.DataFrame,int]:
    before = len(df)
    out  = df.drop_duplicates(subset = subset,keep = "first").reset_index(drop = True)
    n_removed = before -len(out)
    return out,n_removed
def flag_price_quantity_outliners(df:pd.DataFrame,iqr_mulitpler:float = 1.5)-> pd.DataFrame:
    out = df.copy()
    for col in ["quantity","price"]:
        q1 = out[col].quantile(0.25)
        q3 - out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 -iqr_mulitpler *iqr
        upper = q3 +iqr_mulitpler *iqr
        out[f"is_{col}_outliner"] = (out[col]<lower) | (out[col]> upper)
    return out
def standardize_descriptions(df:pd.DataFrame)->pd.DataFrame:
    out = df.copy()
    out["description"] = (
        out["description"]
        .str.strip()
        .str.upper()
        .str.replace(r"\s+"," ",regex = True)

    )
    canonical = (
        out.dropna(subset = ["description"])
        .groupby("stock_code")["description"]
        .agg(lambda s:s.value_counts().idmax())
        .rename("canonical_description")

    )
    out = out.merge(canonical,on = "stock_code",how = "left")
    out["description"] = out["canonical_description"].fillna(out["description"])
    out = out.drop(columns = ["canonical_description"])
    return out
def run_cleaning_pipeline(
    df:pd.DataFrame,
    non_product_codes:Iterable[str] = DEFAULT_NON_PRODUCT_CODES,
    iqr_mulitpler:float = 1.5

)->pd.DataFrame:
    out = df.copy()
    out = flag_missing_customer_id(out)
    out = flag_invalid_price(out)
    out = flag_non_product_rows(out,non_product_codes)
    out = flag_price_quantity_outliners(out,iqr_mulitpler)
    out = standardize_descriptions(out)
    out, n_dupes_removed = deduplicate_transactions(out)
    out.attrs["n_duplicates_removed"] = n_dupes_removed
    return out

