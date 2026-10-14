"""Loads the exported XGBoost models and serves predictions + SHAP explanations.

SHAP values come from XGBoost's own exact TreeSHAP (pred_contribs=True), so the
container doesn't need the shap package.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import features as F

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

LABELS = {
    "log_qib_x": "QIB subscription", "log_nii_x": "NII subscription", "log_bnii_x": "bNII subscription",
    "log_snii_x": "sNII subscription", "log_retail_x": "Retail subscription",
    "log_employee_x": "Employee subscription", "log_total_x": "Total subscription",
    "log_issue_size_cr": "Issue size", "log_anchor_portion_cr": "Anchor book size",
    "log_price_band_upper": "Price band (upper)", "log_lot_size": "Lot size",
    "log_min_investment": "Min. investment", "log_revenue_cr": "Revenue", "log_mcap_post_cr": "Market cap",
    "slog_pat_cr": "Profit after tax", "fresh_issue_pct": "Fresh issue %",
    "anchor_pct_of_issue": "Anchor % of issue", "retail_reserved_pct": "Retail quota %",
    "listing_month": "Listing month", "issue_days_open": "Days open", "nifty_ret_7d": "Nifty 7-day return",
    "nifty_ret_30d": "Nifty 30-day return", "vix_close": "India VIX",
    "ipo_heat_30d": "IPOs closed in last 30d", "prev_gain_mean_90d": "Recent IPOs' avg listing gain",
    "prev_count_90d": "IPOs listed in last 90d", "revenue_growth_pct": "Revenue growth",
    "pat_margin_pct": "PAT margin", "roe_pct": "ROE", "debt_equity": "Debt/Equity", "pe_post": "Issue P/E",
    "promoter_post_pct": "Promoter holding (post)", "promoter_dilution_pct": "Promoter dilution",
    "exchange": "Exchange", "sector": "Sector",
}
RAW_OF = {f"log_{c}": c for c in F.LOG1P} | {f"slog_{c}": c for c in F.SIGNED_LOG}


class Model:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.reg = xgb.Booster()
        self.reg.load_model(model_dir / "xgb_reg.json")
        self.clf = xgb.Booster()
        self.clf.load_model(model_dir / "xgb_clf.json")
        self.past = json.loads((model_dir / "past_ipos.json").read_text())

    def frame(self, payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = dict(payload)
        if raw.get("anchor_portion_cr") is not None and raw.get("issue_size_cr"):
            raw["anchor_pct_of_issue"] = raw["anchor_portion_cr"] / raw["issue_size_cr"] * 100
        raw_df = pd.DataFrame([raw])
        return raw_df, F.engineer(raw_df)

    def _dm(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X, enable_categorical=True)

    def out_of_range(self, raw: dict) -> list[str]:
        out = []
        for k, r in self.meta["ranges"].items():
            v = raw.get(k)
            if v is not None and (v < r["min"] or v > r["max"]):
                out.append(k)
        return out

    def predict(self, payload: dict) -> dict:
        raw_df, X = self.frame(payload)
        dm = self._dm(X)
        gain = float(self.reg.predict(dm)[0])
        prob = float(self.clf.predict(dm)[0])
        mae = self.meta["test_metrics"]["MAE"]
        cut = self.meta.get("clf_threshold", 0.5)
        notes = [f"Typical error on 2024+ test IPOs is about +/-{mae:.0f} percentage points (MAE)."]
        if (gain > 0) != (prob >= cut):
            notes.append("The gain model and the up/down model disagree on direction, so treat this one as low confidence.")
        elif abs(prob - cut) < 0.03:
            notes.append(f"The up/down score is right at the decision cut-off ({cut:.2f}), so the direction is close to a coin flip.")
        missing = [k for k in ("qib_x", "nii_x", "retail_x") if payload.get(k) is None]
        if missing:
            notes.append("Missing subscription inputs (" + ", ".join(missing) + "): the model fills these the way it learned "
                         "from older IPOs, which can swing the estimate by many points. Enter them for a reliable number.")
        oor = self.out_of_range(raw_df.iloc[0].to_dict())
        if oor:
            notes.append("Inputs outside the range seen in training: " + ", ".join(oor) + ". Extrapolation is unreliable.")
        return {
            "predicted_gain_pct": round(gain, 2),
            "prob_positive": round(prob, 4),
            "decision_threshold": round(cut, 4),
            "predicted_direction": "positive" if prob >= cut else "flat_or_negative",
            "confidence_note": " ".join(notes),
            "model_version": self.meta["version"],
        }

    def explain(self, payload: dict, top: int = 12) -> dict:
        raw_df, X = self.frame(payload)
        contrib = self.reg.predict(self._dm(X), pred_contribs=True)[0]
        names = list(X.columns)
        base = float(contrib[-1])
        items = []
        for name, s in zip(names, contrib[:-1]):
            raw_name = RAW_OF.get(name, name)
            v = raw_df.iloc[0].get(raw_name)
            if isinstance(v, float) and np.isnan(v):
                v = None
            items.append({"feature": name, "label": LABELS.get(name, name), "value": v, "shap": round(float(s), 3)})
        items.sort(key=lambda d: abs(d["shap"]), reverse=True)
        return {
            "base_value": round(base, 3),
            "predicted_gain_pct": round(float(contrib.sum()), 2),
            "contributions": items[:top],
            "note": "SHAP values for the listing-gain model, in percentage points. base_value + sum of all "
                    "contributions = prediction. Only the largest are listed; missing inputs can still contribute.",
        }


@lru_cache(maxsize=1)
def get_model() -> Model:
    return Model()
