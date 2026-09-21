
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _customer_level_rows(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Rows usable for customer-level modeling as of cutoff_date:
    strictly before cutoff, has a customer id, and is not a non-product line.
    Returns (negative quantity) ARE kept deliberately — treated as signal,
    not dropped.
    """
    mask = (
        (df["invoice_date"] < cutoff_date)
        & (~df["is_missing_customer_id"])
        & (~df.get("is_non_product", False))
    )
    return df.loc[mask].copy()


def compute_rfm(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Recency, Frequency, Monetary — the core churn predictors."""
    hist = _customer_level_rows(df, cutoff_date)
    valid_sales = hist.loc[~hist["is_invalid_price"]].copy()
    valid_sales["line_total"] = valid_sales["quantity"] * valid_sales["price"]

    last_purchase = hist.groupby("customer_id")["invoice_date"].max()
    recency_days = (cutoff_date - last_purchase).dt.days

    frequency = hist.groupby("customer_id")["invoice"].nunique()

    monetary_total = valid_sales.groupby("customer_id")["line_total"].sum()
    monetary_avg = valid_sales.groupby("customer_id")["line_total"].mean()

    rfm = pd.DataFrame(
        {
            "recency_days": recency_days,
            "frequency": frequency,
            "monetary_total": monetary_total,
            "monetary_avg": monetary_avg,
        }
    )
    return rfm.fillna({"monetary_total": 0.0, "monetary_avg": 0.0})


def compute_historical_clv(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.Series:
    """Historical CLV = total realized spend up to cutoff_date."""
    hist = _customer_level_rows(df, cutoff_date)
    valid_sales = hist.loc[~hist["is_invalid_price"]].copy()
    valid_sales["line_total"] = valid_sales["quantity"] * valid_sales["price"]
    clv = valid_sales.groupby("customer_id")["line_total"].sum()
    clv.name = "historical_clv"
    return clv


def compute_basket_size(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.Series:
    """Average number of distinct line items per invoice."""
    hist = _customer_level_rows(df, cutoff_date)
    lines_per_invoice = hist.groupby(["customer_id", "invoice"]).size()
    basket = lines_per_invoice.groupby("customer_id").mean()
    basket.name = "avg_basket_size"
    return basket


def compute_purchase_trend(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.Series:
    """Slope of order count over time (per 30-day bucket)."""
    hist = _customer_level_rows(df, cutoff_date)
    if hist.empty:
        return pd.Series(name="purchase_trend_slope", dtype=float)

    hist = hist.copy()
    hist["period"] = ((cutoff_date - hist["invoice_date"]).dt.days // 30)
    counts = (
        hist.groupby(["customer_id", "period"])["invoice"]
        .nunique()
        .rename("orders")
        .reset_index()
    )

    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return 0.0
        x = -group["period"].values.astype(float)
        y = group["orders"].values.astype(float)
        if np.std(x) == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    slope = counts.groupby("customer_id").apply(_slope)
    slope.name = "purchase_trend_slope"
    return slope


def compute_purchase_interval_variance(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.Series:
    """Variance of days between consecutive purchases."""
    hist = _customer_level_rows(df, cutoff_date)
    invoice_dates = (
        hist.groupby(["customer_id", "invoice"])["invoice_date"].min().reset_index()
    )
    invoice_dates = invoice_dates.sort_values(["customer_id", "invoice_date"])
    invoice_dates["gap_days"] = (
        invoice_dates.groupby("customer_id")["invoice_date"].diff().dt.days
    )
    variance = invoice_dates.groupby("customer_id")["gap_days"].var()
    variance.name = "purchase_interval_variance"
    return variance.fillna(0.0)


def compute_seasonal_concentration(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.Series:
    """Concentration of spend across calendar months (0 = evenly spread,
    1 = all spend in a single month)."""
    hist = _customer_level_rows(df, cutoff_date)
    valid_sales = hist.loc[~hist["is_invalid_price"]].copy()
    valid_sales["line_total"] = valid_sales["quantity"] * valid_sales["price"]
    valid_sales["month"] = valid_sales["invoice_date"].dt.month

    monthly = valid_sales.groupby(["customer_id", "month"])["line_total"].sum()
    totals = monthly.groupby("customer_id").sum()
    shares_sq = (monthly / totals.reindex(monthly.index.get_level_values("customer_id")).values) ** 2
    concentration = shares_sq.groupby("customer_id").sum()
    concentration.name = "seasonal_concentration"
    return concentration.fillna(0.0)


def compute_product_affinity(
    df: pd.DataFrame, cutoff_date: pd.Timestamp, top_n_categories: int = 10
) -> pd.DataFrame:
    """Share of a customer's spend in each of the top-N most common StockCodes."""
    hist = _customer_level_rows(df, cutoff_date)
    valid_sales = hist.loc[~hist["is_invalid_price"]].copy()
    valid_sales["line_total"] = valid_sales["quantity"] * valid_sales["price"]

    top_codes = (
        valid_sales.groupby("stock_code")["line_total"].sum().nlargest(top_n_categories).index
    )
    pivot = (
        valid_sales[valid_sales["stock_code"].isin(top_codes)]
        .groupby(["customer_id", "stock_code"])["line_total"]
        .sum()
        .unstack(fill_value=0.0)
    )
    pivot.columns = [f"affinity_{c}" for c in pivot.columns]
    row_sums = pivot.sum(axis=1)
    affinity_shares = pivot.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return affinity_shares


def compute_return_rate(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.Series:
    """Share of line items that are returns."""
    hist = _customer_level_rows(df, cutoff_date)
    rate = hist.groupby("customer_id")["is_return"].mean()
    rate.name = "return_rate"
    return rate.fillna(0.0)


def compute_discount_sensitivity(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.Series:
    """Share of a customer's line items priced in the bottom decile for
    their product — a proxy for markdown-driven buying."""
    hist = _customer_level_rows(df, cutoff_date)
    valid_sales = hist.loc[~hist["is_invalid_price"]].copy()
    if valid_sales.empty:
        return pd.Series(name="discount_sensitivity", dtype=float)

    decile_threshold = valid_sales.groupby("stock_code")["price"].transform(
        lambda s: s.quantile(0.1)
    )
    valid_sales["is_markdown_price"] = valid_sales["price"] <= decile_threshold
    sensitivity = valid_sales.groupby("customer_id")["is_markdown_price"].mean()
    sensitivity.name = "discount_sensitivity"
    return sensitivity.fillna(0.0)


def compute_engagement_score(rfm: pd.DataFrame) -> pd.Series:
    """Composite 0-100 score from percentile-ranked R, F, M.
    recency is inverted (lower days-since-purchase = higher score)."""
    r_rank = 1 - rfm["recency_days"].rank(pct=True)
    f_rank = rfm["frequency"].rank(pct=True)
    m_rank = rfm["monetary_total"].rank(pct=True)
    score = 100 * (0.4 * r_rank + 0.3 * f_rank + 0.3 * m_rank)
    score.name = "engagement_score"
    return score


def compute_churn_label(
    df: pd.DataFrame, cutoff_date: pd.Timestamp, churn_window_days: int = 90
) -> pd.Series:
    """Churn label: 1 if the customer makes NO purchase in the window
    (cutoff_date, cutoff_date + churn_window_days]. Computed strictly from
    data AFTER cutoff_date — never merge this with feature computation.

    Only customers with at least one purchase before cutoff_date are
    labeled (a customer with no history cannot be scored for churn).
    """
    horizon_end = cutoff_date + pd.Timedelta(days=churn_window_days)
    future = df[
        (df["invoice_date"] > cutoff_date)
        & (df["invoice_date"] <= horizon_end)
        & (~df["is_missing_customer_id"])
    ]
    purchased_in_window = set(future["customer_id"].unique())

    hist_customers = set(
        _customer_level_rows(df, cutoff_date)["customer_id"].unique()
    )
    label = pd.Series(
        {cid: 0 if cid in purchased_in_window else 1 for cid in hist_customers},
        name="churned",
    )
    return label


def build_customer_feature_table(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    churn_window_days: int = 90,
    include_label: bool = True,
) -> pd.DataFrame:
    """Assemble the full customer-level feature table for a given cutoff date."""
    rfm = compute_rfm(df, cutoff_date)
    clv = compute_historical_clv(df, cutoff_date)
    basket = compute_basket_size(df, cutoff_date)
    trend = compute_purchase_trend(df, cutoff_date)
    interval_var = compute_purchase_interval_variance(df, cutoff_date)
    seasonality = compute_seasonal_concentration(df, cutoff_date)
    affinity = compute_product_affinity(df, cutoff_date)
    return_rate = compute_return_rate(df, cutoff_date)
    discount_sens = compute_discount_sensitivity(df, cutoff_date)

    features = rfm.join(
        [clv, basket, trend, interval_var, seasonality, affinity, return_rate, discount_sens],
        how="left",
    )
    features["engagement_score"] = compute_engagement_score(rfm)
    features = features.fillna(0.0)

    if include_label:
        label = compute_churn_label(df, cutoff_date, churn_window_days)
        features = features.join(label, how="inner")

    features.index.name = "customer_id"
    return features.reset_index()
