"""Feature engineering shared by the notebooks, training script and the API.

Kept dependency-light (numpy + pandas only) so backend/ can import an exact
copy and the deployed model sees features built the same way as in training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "ipo_mainboard.csv"

# Subscription multiples, sizes and prices are heavily right-skewed: log1p them.
LOG1P = [
    "qib_x", "nii_x", "bnii_x", "snii_x", "retail_x", "employee_x", "total_x",
    "issue_size_cr", "anchor_portion_cr", "price_band_upper", "lot_size",
    "min_investment", "revenue_cr", "mcap_post_cr",
]
SIGNED_LOG = ["pat_cr"]  # profit can be negative
PASSTHROUGH = [
    "fresh_issue_pct", "anchor_pct_of_issue", "retail_reserved_pct",
    "listing_month", "issue_days_open",
    "nifty_ret_7d", "nifty_ret_30d", "vix_close",
    "ipo_heat_30d", "prev_gain_mean_90d", "prev_count_90d",
    "revenue_growth_pct", "pat_margin_pct", "roe_pct", "debt_equity",
    "pe_post", "promoter_post_pct", "promoter_dilution_pct",
]
CATEGORICAL = ["exchange", "sector"]

# listing_year is in the dataset but NOT a model feature: with a time split the
# test years (2024+) are outside the training range, and trees would just treat
# them as "the latest year seen", which is not information.
# days_close_to_listing is dropped for the same reason: SEBI moved listing
# from T+6 to T+3 in Dec 2023, so train (~10 days) and test (~3-5 days) don't
# overlap at all (KS = 0.99 in notebook 01). It encodes the era, not demand.
# ofs_pct = 100 - fresh_issue_pct and issue_price ~= price_band_upper, so both
# are dropped as redundant.

CATEGORIES = {
    "exchange": ["BSE,NSE", "NSE", "BSE", "Other"],
    "sector": [
        "Basic Materials", "Communication Services", "Consumer Cyclical",
        "Consumer Defensive", "Energy", "Financial Services", "Healthcare",
        "Industrials", "Real Estate", "Technology", "Utilities", "Unknown",
    ],
}


def numeric_names() -> list[str]:
    return [f"log_{c}" for c in LOG1P] + [f"slog_{c}" for c in SIGNED_LOG] + PASSTHROUGH


def feature_names() -> list[str]:
    return numeric_names() + CATEGORICAL


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw (registry) columns -> model matrix. Missing values stay NaN."""
    X = pd.DataFrame(index=raw.index)
    for c in LOG1P:
        v = pd.to_numeric(raw.get(c), errors="coerce") if c in raw else np.nan
        X[f"log_{c}"] = np.log1p(np.clip(v, 0, None))
    for c in SIGNED_LOG:
        v = pd.to_numeric(raw.get(c), errors="coerce") if c in raw else np.nan
        X[f"slog_{c}"] = np.sign(v) * np.log1p(np.abs(v))
    for c in PASSTHROUGH:
        X[c] = pd.to_numeric(raw[c], errors="coerce") if c in raw else np.nan
    ex = raw["exchange"].fillna("Other") if "exchange" in raw else pd.Series("Other", index=raw.index)
    ex = ex.where(ex.isin(CATEGORIES["exchange"]), "Other")
    X["exchange"] = pd.Categorical(ex, categories=CATEGORIES["exchange"])
    sec = raw["sector"].fillna("Unknown") if "sector" in raw else pd.Series("Unknown", index=raw.index)
    sec = sec.where(sec.isin(CATEGORIES["sector"]), "Unknown")
    X["sector"] = pd.Categorical(sec, categories=CATEGORIES["sector"])
    return X[feature_names()]


def load(path: Path = PROCESSED) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_date", "close_date", "listing_date"])
    return df.sort_values("close_date").reset_index(drop=True)


def splits(df: pd.DataFrame):
    """Time-based split: train <= 2022, val = 2023, test >= 2024 (by listing year)."""
    return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]


def xy(df: pd.DataFrame, target: str = "listing_gain_pct"):
    return engineer(df), df[target].astype(float)


def sklearn_preprocessor(scale: bool = True, impute: str = "median"):
    """ColumnTransformer for models that can't take NaN or categoricals."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_steps = [("impute", SimpleImputer(strategy=impute, add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric_names()),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
