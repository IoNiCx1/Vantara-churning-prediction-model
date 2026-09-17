from  __future__ import annotations
from dataclasses import dataclass,field
from typing import Dict,List

import pandas as pd

@dataclass
class ValidationResult:
    passed : bool
    errors:List[str] = field(default_factory = list)
    warnings:List[str] = field(default_factory = list)
    def raise_if_failed(self)->None:
        if not self.passed:
            raise ValueError(
                "Data validation failed:\n-"+"\n-".join(self.errors)
            )
    def validate_schema(df:pd.DataFrame,required_columns:List[str])->List[str]:
        errors = []
        missing = [ c for c in required_columns if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns:{missing}")
        return errors
    
    def validate_null_rates(df:pd.DataFrame,max_null_rate:Dict[str,float])->List[str]:
        errors = []
        for col,threshold in maxx_null_rate.items():
            if col not in df.columns:
                continue
            rate = df[col].isna().mean()
            if rate>threshold:
                errors.append(
                    f"NUll rate for '{col}'is {rate:.2%},exceeds threeshold {threeshold:.2%}"
                )
            return errors
    def validate_date_range(df:pd.DataFrame,date_col:str,min_date:str,max_date:str):
        errors = []
        if date_col not in df.columns:
            return [f"Date Column'{date_col}' not found"]
        min_ts,max_ts = pd.Timestamp(min_date),pd.Timestamp(max_date)
        out_of_range = df[(df[date_col]<min_ts)|(df[date_col]>max_ts)]
        if len(out_of_range)>0:
            errors.append(
                f"{len(out_of_range)} rows have '{date_col}' outside expected range"
                f"[{min_date},{max_date}]"
            )
        return errors
    def run_valdation(df:pd.DataFrame,config:dict)->ValidationResult:
        val_cfg = config["validation"]
        raw_to_snake = {
            "Invoice":"invoice",
            "StockCode":"stock_code",
            "Description":"description",
            "Quality":"quality",
            "InvoiceDate":"invoice_date",
            "Price":"price",
            "Customer ID":"customer_id",
            "Country":"country"
        }
        required_cols = [raw_to_snake.get(c,c) for c in val_cfg["requirement_columns"]]
        null_rate_cfg = {raw_to_snake.get(k,k): v for k, v in val_cfg.get("max_null_rate",{}).item()}

        errors:List[str]=[]
        errors += validate_schema(df,required_cols)
        if not errors:
            errors+=valdate_null_rates(df,null_rate_cfg)
            errors += validation_date_range(
                df,"invoice_date",val_cfg["date_range"]["min"],val_cfg["date_range"]["max"]
            )
        return ValidationResult(passed=len(errors)==0,errors = errors)