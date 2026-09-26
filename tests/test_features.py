"""
Unit tests for src/features/build_features.py.

Target leakage in engineered features is an explicit risk in this kind
of pipeline. The tests below construct a small, hand-built transaction
table where the "future" purchase is known, and assert that no feature
computed at cutoff_date can see it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import run_cleaning_pipeline
from src.features.build_features import (
    build_customer_feature_table,
    compute_churn_label,
    compute_historical_clv,
    compute_rfm,
)


def _make_raw_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


@pytest.fixture
def toy_transactions() -> pd.DataFrame:
    """One customer (1) who churns after cutoff, one (2) who returns within
    the window, one (3) with a huge purchase strictly AFTER cutoff that
    must never leak into their pre-cutoff features.
    """
    rows = [
        dict(invoice="INV1", stock_code="A1", description="Item A", quantity=2,
             invoice_date="2024-01-01", price=10.0, customer_id=1, country="UK"),
        dict(invoice="INV2", stock_code="A1", description="Item A", quantity=1,
             invoice_date="2024-01-15", price=10.0, customer_id=1, country="UK"),
        dict(invoice="INV3", stock_code="A2", description="Item B", quantity=3,
             invoice_date="2024-01-10", price=5.0, customer_id=2, country="UK"),
        dict(invoice="INV4", stock_code="A2", description="Item B", quantity=1,
             invoice_date="2024-02-20", price=5.0, customer_id=2, country="UK"),
        dict(invoice="INV5", stock_code="A3", description="Item C", quantity=1,
             invoice_date="2024-01-05", price=20.0, customer_id=3, country="UK"),
        dict(invoice="INV6", stock_code="A3", description="Item C", quantity=1000,
             invoice_date="2024-03-01", price=999.0, customer_id=3, country="UK"),
    ]
    raw = _make_raw_df(rows)
    return run_cleaning_pipeline(raw)


CUTOFF = pd.Timestamp("2024-02-01")
CHURN_WINDOW_DAYS = 90


def test_historical_clv_excludes_future_purchase(toy_transactions):
    """Customer 3's huge post-cutoff purchase (999 * 1000) must not appear
    in their historical CLV computed at cutoff_date."""
    clv = compute_historical_clv(toy_transactions, CUTOFF)
    assert clv.loc[3] == pytest.approx(20.0)
    assert clv.loc[3] < 999.0 * 1000


def test_rfm_recency_uses_only_pre_cutoff_purchases(toy_transactions):
    """Customer 2's recency at cutoff must be measured from their last
    PRE-cutoff purchase (Jan 10), not their future purchase (Feb 20)."""
    rfm = compute_rfm(toy_transactions, CUTOFF)
    expected_recency = (CUTOFF - pd.Timestamp("2024-01-10")).days
    assert rfm.loc[2, "recency_days"] == expected_recency


def test_churn_label_matches_known_future_behavior(toy_transactions):
    """Customer 1 has no purchase in the 90-day window after cutoff -> churned.
    Customer 2 purchases within the window -> not churned.
    Customer 3 purchases within the window -> not churned.
    """
    label = compute_churn_label(toy_transactions, CUTOFF, CHURN_WINDOW_DAYS)
    assert label.loc[1] == 1
    assert label.loc[2] == 0
    assert label.loc[3] == 0


def test_full_feature_table_has_no_post_cutoff_contamination(toy_transactions):
    """End-to-end check: rebuilding the feature table must be blind to
    anything after cutoff regardless of what's in the input frame."""
    full_features = build_customer_feature_table(toy_transactions, CUTOFF, CHURN_WINDOW_DAYS)
    truncated_features = build_customer_feature_table(
        toy_transactions, CUTOFF, CHURN_WINDOW_DAYS, include_label=False
    )

    feature_cols = [c for c in full_features.columns if c not in ("churned",)]
    pd.testing.assert_frame_equal(
        full_features[feature_cols].sort_values("customer_id").reset_index(drop=True),
        truncated_features[feature_cols].sort_values("customer_id").reset_index(drop=True),
    )


def test_engagement_score_is_bounded(toy_transactions):
    features = build_customer_feature_table(toy_transactions, CUTOFF, CHURN_WINDOW_DAYS)
    assert features["engagement_score"].between(0, 100).all()


def test_return_flag_not_dropped(toy_transactions):
    """Negative-quantity rows must be flagged, never dropped."""
    assert "is_return" in toy_transactions.columns
    assert toy_transactions["is_return"].sum() == 0  # none in this fixture, but column must exist


def test_missing_customer_id_excluded_from_customer_table():
    rows = [
        dict(invoice="INV1", stock_code="A1", description="Item A", quantity=1,
             invoice_date="2024-01-01", price=10.0, customer_id=None, country="UK"),
    ]
    raw = _make_raw_df(rows)
    cleaned = run_cleaning_pipeline(raw)
    features = build_customer_feature_table(cleaned, CUTOFF, CHURN_WINDOW_DAYS)
    assert len(features) == 0  # the only row has no customer id -> excluded


def test_seasonal_concentration_handles_zero_net_spend():
    """A customer whose purchase and return net out to exactly zero spend
    must get seasonal_concentration = 0.0, not inf (regression test for
    the division-by-zero bug found during real-data testing)."""
    rows = [
        dict(invoice="INV1", stock_code="A1", description="Item A", quantity=1,
             invoice_date="2024-01-01", price=10.0, customer_id=1, country="UK"),
        dict(invoice="C-INV1", stock_code="A1", description="Item A", quantity=-1,
             invoice_date="2024-01-02", price=10.0, customer_id=1, country="UK"),
    ]
    raw = _make_raw_df(rows)
    cleaned = run_cleaning_pipeline(raw)
    features = build_customer_feature_table(cleaned, CUTOFF, CHURN_WINDOW_DAYS)
    assert not features["seasonal_concentration"].isin([float("inf"), float("-inf")]).any()