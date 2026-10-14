"""Train and export the production model.

Run:  python -m src.train

1. Fit XGBoost (regression + classification) on train (<= 2022) with early
   stopping on validation (2023) to pick the number of trees.
2. Refit on train + val with that many trees.
3. Score the untouched test set (2024+) once and export everything the API
   needs to models/: model JSON files, metadata.json, past_ipos.json.

Hyperparameters come from models/best_params.json (written by notebook 02's
Optuna search) if present, else sensible defaults.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import features as F
from .evaluate import baselines, clf_metrics, reg_metrics, youden_threshold

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
VERSION = "1.0.0"

DEFAULT_REG = dict(
    objective="reg:pseudohubererror", huber_slope=10.0, learning_rate=0.03, max_depth=3,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=0.0,
)
DEFAULT_CLF = dict(
    objective="binary:logistic", learning_rate=0.03, max_depth=3, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
)


def _params() -> tuple[dict, dict]:
    f = MODELS / "best_params.json"
    if f.exists():
        p = json.loads(f.read_text())
        return {**DEFAULT_REG, **p.get("reg", {})}, {**DEFAULT_CLF, **p.get("clf", {})}
    return DEFAULT_REG, DEFAULT_CLF


def fit_xgb(params: dict, X_tr, y_tr, X_va=None, y_va=None, n_estimators: int = 2000, kind: str = "reg"):
    cls = xgb.XGBRegressor if kind == "reg" else xgb.XGBClassifier
    use_es = X_va is not None
    m = cls(
        n_estimators=n_estimators, enable_categorical=True, tree_method="hist",
        early_stopping_rounds=100 if use_es else None, random_state=42, n_jobs=2,
        eval_metric="mae" if kind == "reg" else "auc", **params,
    )
    if use_es:
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        m.fit(X_tr, y_tr, verbose=False)
    return m


def contributions(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values from XGBoost itself (last column = bias)."""
    return booster.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    df = F.load()
    tr, va, te = F.splits(df)
    reg_p, clf_p = _params()

    Xtr, ytr = F.xy(tr)
    Xva, yva = F.xy(va)
    Xte, yte = F.xy(te)
    ctr, cva, cte = (d["listed_positive"].astype(int) for d in (tr, va, te))

    reg = fit_xgb(reg_p, Xtr, ytr, Xva, yva, kind="reg")
    clf = fit_xgb(clf_p, Xtr, ctr, Xva, cva, kind="clf")
    n_reg, n_clf = reg.best_iteration + 1, clf.best_iteration + 1
    # the classifier's probabilities sit in a narrow band above 0.5 (most IPOs list up), so a 0.5
    # cut-off calls everything positive; pick the cut-off on the validation year instead
    pva = clf.predict_proba(Xva)[:, 1]
    threshold = youden_threshold(cva, pva)
    val_metrics = {**reg_metrics(yva, reg.predict(Xva)), **clf_metrics(cva, pva, threshold)}

    trva = pd.concat([tr, va])
    Xtv, ytv = F.xy(trva)
    reg_f = fit_xgb(reg_p, Xtv, ytv, n_estimators=n_reg, kind="reg")
    clf_f = fit_xgb(clf_p, Xtv, trva["listed_positive"].astype(int), n_estimators=n_clf, kind="clf")

    pred = reg_f.predict(Xte)
    prob = clf_f.predict_proba(Xte)[:, 1]
    test_metrics = {**reg_metrics(yte, pred), **clf_metrics(cte, prob, threshold)}
    base = baselines(trva, {"test": te}).iloc[0].to_dict()

    reg_f.get_booster().save_model(MODELS / "xgb_reg.json")
    clf_f.get_booster().save_model(MODELS / "xgb_clf.json")

    # input ranges for API validation: training data range with some slack
    raw_cols = [c for c in F.LOG1P + F.SIGNED_LOG + F.PASSTHROUGH]
    ranges = {}
    for c in raw_cols:
        s = pd.to_numeric(trva[c], errors="coerce").dropna()
        if len(s):
            ranges[c] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median())}

    # prefilled example for the web form: the most recent test IPO with complete inputs
    inputs = raw_cols + F.CATEGORICAL
    complete = te.dropna(subset=["qib_x", "nii_x", "retail_x", "total_x", "anchor_portion_cr", "pe_post", "vix_close"])
    ex_row = (complete if len(complete) else te).iloc[-1]
    example = {"company": ex_row["company"], "listing_date": f"{ex_row['listing_date']:%Y-%m-%d}",
               "actual_listing_gain_pct": float(ex_row["listing_gain_pct"]),
               "inputs": {c: (None if pd.isna(ex_row.get(c)) else (ex_row[c] if isinstance(ex_row[c], str) else float(ex_row[c])))
                          for c in inputs if c not in ("anchor_pct_of_issue",)}}
    if example["inputs"].get("sector") is None:
        example["inputs"]["sector"] = "Unknown"

    meta = {
        "version": VERSION,
        "example": example,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "XGBoost (hist, native categorical + missing handling)",
        "train_rows": int(len(trva)),
        "train_period": f"{trva['listing_date'].min():%Y-%m-%d} to {trva['listing_date'].max():%Y-%m-%d}",
        "test_rows": int(len(te)),
        "test_period": f"{te['listing_date'].min():%Y-%m-%d} to {te['listing_date'].max():%Y-%m-%d}",
        "n_trees": {"reg": int(n_reg), "clf": int(n_clf)},
        "clf_threshold": threshold,
        "params": {"reg": reg_p, "clf": clf_p},
        "features": F.feature_names(),
        "raw_inputs": raw_cols + F.CATEGORICAL,
        "categories": F.CATEGORIES,
        "ranges": ranges,
        "validation_metrics_2023": val_metrics,
        "test_metrics": test_metrics,
        "baseline_test_metrics": {k: v for k, v in base.items() if k not in ("split",)},
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))

    past = te[["ipo_id", "company", "listing_date", "sector", "issue_price", "issue_size_cr", "total_x",
               "listing_gain_pct", "listed_positive"]].copy()
    past["predicted_gain_pct"] = pred
    past["prob_positive"] = prob
    past["listing_date"] = past["listing_date"].dt.strftime("%Y-%m-%d")
    past["sector"] = past["sector"].fillna("Unknown")
    (MODELS / "past_ipos.json").write_text(past.to_json(orient="records", double_precision=4))

    print(json.dumps({"n_trees": meta["n_trees"], "val": val_metrics, "test": test_metrics,
                      "baseline": meta["baseline_test_metrics"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
