"""
Synthetic transaction generator — behaviorally realistic.

The real UCI Online Retail II dataset isn't available yet, so this
generates a schema-accurate AND behaviorally realistic synthetic dataset,
so the rest of the pipeline can be built and tested end-to-end. Swap in
the real file later by just changing config.yaml's paths.raw_data.

Each customer gets a persistent behavioral archetype (purchase cadence,
spend level, return propensity, discount affinity, seasonality) and a
per-order hazard of going permanently dormant. That's what makes churn
genuinely learnable from RFM features, instead of being random noise.

The known real-dataset quirks (missing Customer ID, returns, zero/negative
price, non-product StockCodes, description casing variance, exact
duplicates) are injected on top, at the row level, as noise orthogonal to
customer behavior — matching how those issues work in the real data too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

RNG_SEED = 42

COUNTRIES = [
    "United Kingdom", "Germany", "France", "EIRE", "Spain",
    "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia",
]

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 4.50),
    ("71053", "WHITE METAL LANTERN", 6.90),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 3.20),
    ("22423", "REGENCY CAKESTAND 3 TIER", 14.50),
    ("47566", "PARTY BUNTING", 8.30),
    ("85099B", "JUMBO BAG RED RETROSPOT", 2.10),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 9.90),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 5.40),
    ("22197", "SMALL POPCORN HOLDER", 1.80),
    ("20725", "LUNCH BAG RED RETROSPOT", 2.50),
]

NON_PRODUCT_CODES = ["POST", "D", "M", "BANK CHARGES", "DOT"]

OBS_START = pd.Timestamp("2009-12-01")
OBS_END = pd.Timestamp("2011-12-08")


@dataclass
class Archetype:
    """A persistent customer behavior profile.

    mean_gap_days: average days between consecutive orders.
    gap_dispersion: >1 = bursty/irregular, <1 = clockwork regular.
    return_propensity: probability an order also generates a return line.
    discount_affinity: 0-1, shifts realized unit price downward.
    seasonal_boost_months: calendar months where this archetype orders
        more often (holiday gift shoppers).
    hazard_per_order: probability, after each order, that the customer
        goes permanently dormant — this is what creates genuine,
        learnable churn signal.
    share: fraction of the customer base assigned this archetype.
    """
    name: str
    mean_gap_days: float
    gap_dispersion: float
    return_propensity: float
    discount_affinity: float
    seasonal_boost_months: tuple
    hazard_per_order: float
    share: float


ARCHETYPES: List[Archetype] = [
    Archetype("loyal_high_value", 14, 0.6, 0.03, 0.1, (), 0.01, 0.15),
    Archetype("steady_mid_value", 28, 0.8, 0.05, 0.2, (), 0.03, 0.30),
    Archetype("occasional_low_value", 60, 1.2, 0.04, 0.3, (), 0.05, 0.20),
    Archetype("discount_hunter", 35, 1.0, 0.06, 0.8, (), 0.04, 0.10),
    Archetype("seasonal_gift_shopper", 90, 0.5, 0.08, 0.15, (10, 11, 12), 0.02, 0.10),
    Archetype("at_risk_departing", 25, 0.9, 0.10, 0.25, (), 0.18, 0.10),
    Archetype("new_customer", 35, 1.0, 0.05, 0.2, (), 0.05, 0.05),
]


def _assign_archetypes(n_customers: int, rng: np.random.Generator) -> List[Archetype]:
    shares = np.array([a.share for a in ARCHETYPES])
    shares = shares / shares.sum()
    idx = rng.choice(len(ARCHETYPES), size=n_customers, p=shares)
    return [ARCHETYPES[i] for i in idx]


def _generate_customer_orders(
    archetype: Archetype, acquisition_date: pd.Timestamp, rng: np.random.Generator
) -> List[pd.Timestamp]:
    """Simulate one customer's order dates via a renewal process with a
    per-order dormancy hazard."""
    dates = []
    current = acquisition_date
    while current < OBS_END:
        seasonal_multiplier = 0.5 if (
            archetype.seasonal_boost_months and current.month not in archetype.seasonal_boost_months
        ) else 1.0
        gap = rng.exponential(archetype.mean_gap_days * archetype.gap_dispersion) / seasonal_multiplier
        current = current + pd.Timedelta(days=max(gap, 1))
        if current >= OBS_END:
            break
        dates.append(current)
        if rng.random() < archetype.hazard_per_order:
            break  # customer goes dormant — no more orders generated
    return dates


def _generate_order_lines(
    order_date: pd.Timestamp, customer_id: int, archetype: Archetype, rng: np.random.Generator
) -> List[dict]:
    """Generate 1-4 line items for a single order."""
    n_lines = rng.integers(1, 5)
    product_idx = rng.choice(len(PRODUCTS), size=n_lines, replace=True)
    invoice = f"5{rng.integers(10000, 99999)}"

    lines = []
    for pi in product_idx:
        code, desc, base_price = PRODUCTS[pi]
        price_multiplier = 1.0 - archetype.discount_affinity * rng.uniform(0.2, 0.6)
        price = round(max(base_price * price_multiplier, 0.5), 2)
        quantity = max(1, int(rng.poisson(3)))
        lines.append(dict(
            Invoice=invoice, StockCode=code, Description=desc, Quantity=quantity,
            InvoiceDate=order_date, Price=price, CustomerID=customer_id,
        ))

    if rng.random() < archetype.return_propensity:
        returned = lines[rng.integers(0, len(lines))]
        return_date = min(order_date + pd.Timedelta(days=int(rng.integers(1, 14))), OBS_END - pd.Timedelta(hours=1))
        lines.append(dict(
            Invoice=f"C{invoice}", StockCode=returned["StockCode"], Description=returned["Description"],
            Quantity=-min(returned["Quantity"], max(1, int(rng.poisson(1)))),
            InvoiceDate=return_date,
            Price=returned["Price"], CustomerID=customer_id,
        ))
    return lines


def generate_synthetic_transactions(
    n_customers: int = 500,
    n_rows: int = 20_000,  # kept for API-shape compatibility; not a strict cap
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Generate a behaviorally realistic synthetic transaction-line dataset."""
    rng = np.random.default_rng(seed)
    archetypes = _assign_archetypes(n_customers, rng)
    customer_ids = np.arange(12000, 12000 + n_customers)

    all_lines = []
    for cid, archetype in zip(customer_ids, archetypes):
        if archetype.name == "new_customer":
            acquisition_date = OBS_END - pd.Timedelta(days=int(rng.integers(30, 120)))
        else:
            acquisition_offset_days = int(rng.integers(0, (OBS_END - OBS_START).days - 30))
            acquisition_date = OBS_START + pd.Timedelta(days=acquisition_offset_days)

        order_dates = _generate_customer_orders(archetype, acquisition_date, rng)
        for order_date in order_dates:
            all_lines.extend(_generate_order_lines(order_date, int(cid), archetype, rng))

    df = pd.DataFrame(all_lines)
    df["Country"] = rng.choice(COUNTRIES, size=len(df), p=_country_weights())

    # --- Inject the documented data-quality quirks, at the row level ---
    missing_mask = rng.random(len(df)) < 0.25
    df.loc[missing_mask, "CustomerID"] = np.nan

    zero_price_mask = rng.random(len(df)) < 0.01
    df.loc[zero_price_mask, "Price"] = 0.0

    n_admin = max(1, len(df) // 200)
    admin_idx = rng.choice(len(df), size=n_admin, replace=False)
    df.loc[admin_idx, "StockCode"] = rng.choice(NON_PRODUCT_CODES, size=n_admin)
    df.loc[admin_idx, "Description"] = df.loc[admin_idx, "StockCode"]
    df.loc[admin_idx, "CustomerID"] = np.nan

    mangle_mask = rng.random(len(df)) < 0.3
    df.loc[mangle_mask, "Description"] = df.loc[mangle_mask, "Description"].apply(
        lambda d: _mangle_description(d, rng)
    )

    dup_sample = df.sample(n=max(1, len(df) // 500), random_state=seed)
    df = pd.concat([df, dup_sample], ignore_index=True)

    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    return df[["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
               "Price", "CustomerID", "Country"]]


def _mangle_description(desc: str, rng: np.random.Generator) -> str:
    choice = rng.integers(0, 3)
    if choice == 0:
        return desc.title()
    if choice == 1:
        return f"  {desc}  "
    return desc.replace(" ", "  ")


def _country_weights() -> np.ndarray:
    w = np.array([0.85, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
    return w / w.sum()


def write_synthetic_workbook(path: str, **kwargs) -> None:
    """Write a two-sheet xlsx matching the real file's sheet names."""
    df = generate_synthetic_transactions(**kwargs)
    cutoff = pd.Timestamp("2010-12-01")
    df1 = df[df["InvoiceDate"] < cutoff]
    df2 = df[df["InvoiceDate"] >= cutoff]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        df2.to_excel(writer, sheet_name="Year 2010-2011", index=False)


if __name__ == "__main__":
    write_synthetic_workbook("data/raw/online_retail_II.xlsx", n_customers=500)
    print("Synthetic dataset written to data/raw/online_retail_II.xlsx") 