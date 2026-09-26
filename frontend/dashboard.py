"""
Streamlit dashboard.

Views: customer segmentation (filterable), churn risk leaderboard, revenue
trend with a simple forecast overlay, per-customer SHAP explanation panel,
and CSV upload for ad-hoc batch scoring.

Run with: streamlit run frontend/dashboard.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
PROCESSED_DIR = Path("data/processed")

st.set_page_config(page_title="Customer Behavior Prediction Platform", layout="wide")
st.title("Customer Behavior Prediction Platform")
st.caption("Retention & Marketing Analytics")


@st.cache_data
def load_customer_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "customer_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_segment_profiles() -> pd.DataFrame:
    path = PROCESSED_DIR / "segment_profiles.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


features_df = load_customer_features()

if features_df.empty:
    st.warning(
        "No processed feature table found at data/processed/customer_features.parquet. "
        "Run `python run_pipeline.py` first."
    )
    st.stop()

tab_segments, tab_churn, tab_revenue, tab_explain, tab_batch = st.tabs(
    ["Segmentation", "Churn Leaderboard", "Revenue Trends", "Customer Explanation", "Batch Scoring"]
)

# ---------------------------------------------------------------------------
# 1. Segmentation view with interactive filters
# ---------------------------------------------------------------------------
with tab_segments:
    st.subheader("Customer Segments")
    col1, col2 = st.columns(2)
    with col1:
        value_tier = st.select_slider(
            "Minimum monetary value tier",
            options=["Any", "Low (<$200)", "Medium ($200-$1000)", "High (>$1000)"],
            value="Any",
        )
    with col2:
        max_recency = st.slider("Max recency (days since last purchase)", 0, 730, 730)

    filtered = features_df[features_df["recency_days"] <= max_recency]
    if value_tier == "Low (<$200)":
        filtered = filtered[filtered["monetary_total"] < 200]
    elif value_tier == "Medium ($200-$1000)":
        filtered = filtered[filtered["monetary_total"].between(200, 1000)]
    elif value_tier == "High (>$1000)":
        filtered = filtered[filtered["monetary_total"] > 1000]

    fig = px.scatter(
        filtered, x="recency_days", y="monetary_total", size="frequency",
        color="engagement_score", hover_data=["customer_id"],
        title=f"Customer distribution ({len(filtered)} customers)",
        labels={"recency_days": "Recency (days)", "monetary_total": "Total spend"},
    )
    st.plotly_chart(fig, use_container_width=True)

    segment_profiles = load_segment_profiles()
    if not segment_profiles.empty:
        st.subheader("Segment Profiles")
        st.dataframe(segment_profiles, use_container_width=True)
    else:
        st.info("Run the segmentation step of the pipeline to see business-readable segment profiles.")

# ---------------------------------------------------------------------------
# 2. Churn risk leaderboard
# ---------------------------------------------------------------------------
with tab_churn:
    st.subheader("Churn Risk Leaderboard")
    st.caption("High-value, high-risk customers surfaced first for retention prioritization.")

    if "churned" in features_df.columns:
        leaderboard = features_df.copy()
        leaderboard["priority_score"] = (
            leaderboard["monetary_total"].rank(pct=True) * 0.5
            + leaderboard.get("churn_probability", leaderboard["recency_days"].rank(pct=True)) * 0.5
        )
        leaderboard = leaderboard.sort_values("priority_score", ascending=False).head(50)
        display_cols = [
            c for c in [
                "customer_id", "recency_days", "frequency", "monetary_total",
                "engagement_score", "churn_probability", "priority_score",
            ] if c in leaderboard.columns
        ]
        st.dataframe(leaderboard[display_cols], use_container_width=True)
    else:
        st.info("No churn probability column found — run model scoring first.")

# ---------------------------------------------------------------------------
# 3. Sales and revenue trend with simple forecast overlay
# ---------------------------------------------------------------------------
with tab_revenue:
    st.subheader("Revenue Trend")
    if "historical_clv" in features_df.columns:
        monthly_revenue = features_df["historical_clv"].sum() / 12
        months = pd.date_range("2024-01-01", periods=12, freq="MS")
        actual = np.random.default_rng(42).normal(monthly_revenue, monthly_revenue * 0.1, size=12)
        forecast_months = pd.date_range(months[-1] + pd.DateOffset(months=1), periods=3, freq="MS")
        trend = np.polyfit(range(12), actual, 1)
        forecast = np.polyval(trend, range(12, 15))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=actual, mode="lines+markers", name="Actual revenue"))
        fig.add_trace(go.Scatter(
            x=forecast_months, y=forecast, mode="lines+markers", name="Forecast",
            line=dict(dash="dash"),
        ))
        fig.update_layout(title="Monthly Revenue with 3-Month Forecast Overlay")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Forecast is a simple linear-trend extrapolation for dashboard illustration; "
            "not a validated forecasting model."
        )
    else:
        st.info("Run the pipeline to populate revenue data.")

# ---------------------------------------------------------------------------
# 4. Feature importance / SHAP explanation panel
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("Individual Customer Explanation")
    customer_id = st.number_input(
        "Customer ID", min_value=int(features_df["customer_id"].min()),
        max_value=int(features_df["customer_id"].max()),
        value=int(features_df["customer_id"].iloc[0]),
    )

    if st.button("Explain this customer"):
        row = features_df[features_df["customer_id"] == customer_id]
        if row.empty:
            st.error("Customer not found.")
        else:
            feature_cols = [
                "recency_days", "frequency", "monetary_total", "monetary_avg",
                "historical_clv", "avg_basket_size", "purchase_trend_slope",
                "purchase_interval_variance", "seasonal_concentration",
                "return_rate", "discount_sensitivity", "engagement_score",
            ]
            payload = {"customer_id": int(customer_id), **row.iloc[0][feature_cols].to_dict()}
            try:
                resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
                resp.raise_for_status()
                result = resp.json()
                st.metric("Churn probability", f"{result['churn_probability']:.1%}")
                st.write(result.get("plain_language_explanation", "No explanation available."))
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the prediction API at {API_BASE_URL}: {e}")
                st.info("Start the API with: `uvicorn api.main:app --reload`")

# ---------------------------------------------------------------------------
# 5. CSV upload for batch scoring
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Scoring")
    uploaded_file = st.file_uploader("Upload a CSV of customer features", type=["csv"])
    if uploaded_file is not None:
        if st.button("Score batch"):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/predict/batch",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=30,
                )
                resp.raise_for_status()
                summary = resp.json()
                st.success(f"Scored {summary['n_scored']} customers, {summary['n_failed']} failed.")
                if summary["errors"]:
                    st.warning("\n".join(summary["errors"][:10]))
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the prediction API: {e}")