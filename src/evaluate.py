"""Metrics and naive baselines, so every model is compared against something dumb."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)


def reg_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def clf_metrics(y, prob, threshold: float = 0.5) -> dict:
    y, prob = np.asarray(y, int), np.asarray(prob, float)
    hard = (prob >= threshold).astype(int)
    out = {
        "ROC_AUC": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "F1": float(f1_score(y, hard, zero_division=0)),
        "Precision": float(precision_score(y, hard, zero_division=0)),
        "Recall": float(recall_score(y, hard, zero_division=0)),
        "Accuracy": float(accuracy_score(y, hard)),
    }
    return out


def best_total_x_threshold(total_x: pd.Series, positive: pd.Series) -> float:
    """Dumb rule 'positive if total_x > t', with t chosen on TRAINING data by F1."""
    best_t, best_f1 = 1.0, -1.0
    for t in np.unique(np.round(np.quantile(total_x.dropna(), np.linspace(0, 0.95, 60)), 2)):
        f1 = f1_score(positive.astype(int), (total_x.fillna(0) > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def baselines(train: pd.DataFrame, evals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Median-gain regressor and total_x threshold classifier, fit on train only."""
    med = float(train["listing_gain_pct"].median())
    t = best_total_x_threshold(train["total_x"], train["listed_positive"])
    # rank-based score for AUC: log total subscription
    rows = []
    for name, d in evals.items():
        r = reg_metrics(d["listing_gain_pct"], np.full(len(d), med))
        score = np.log1p(d["total_x"].fillna(0))
        c = clf_metrics(d["listed_positive"], (d["total_x"].fillna(0) > t).astype(float))
        c["ROC_AUC"] = float(roc_auc_score(d["listed_positive"].astype(int), score))
        rows.append({"split": name, "model": f"Baseline: median gain ({med:.1f}%) / total_x > {t:g}", **r, **c})
    return pd.DataFrame(rows)
