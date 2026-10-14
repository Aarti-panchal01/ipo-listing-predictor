"""FastAPI service for the IPO listing-gain predictor."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .model import get_model
from .schemas import Explanation, IPOInput, Prediction

DISCLAIMER = "Educational project. Not investment advice and not SEBI-registered advice."

app = FastAPI(
    title="Indian IPO listing-gain predictor",
    version="1.0.0",
    description="Predicts listing-day gain of Indian mainboard IPOs from pre-listing data. " + DISCLAIMER,
)

origins = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://ipo-listing-predictor.vercel.app",
).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX", r"https://ipo-listing-predictor(-[a-z0-9-]+)?\.vercel\.app"),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def root():
    return {"service": "ipo-listing-predictor", "docs": "/docs", "disclaimer": DISCLAIMER}


@app.get("/health")
def health():
    m = get_model()
    return {"status": "ok", "model_version": m.meta["version"]}


@app.get("/meta")
def meta():
    m = get_model().meta
    return {
        "model_version": m["version"],
        "trained_at": m["trained_at"],
        "model": m["model"],
        "train_period": m["train_period"],
        "test_period": m["test_period"],
        "features": m["features"],
        "inputs": IPOInput.model_json_schema()["properties"],
        "valid_ranges_training_data": m["ranges"],
        "sectors": m["categories"]["sector"],
        "exchanges": [e for e in m["categories"]["exchange"] if e != "Other"],
        "test_metrics": m["test_metrics"],
        "decision_threshold": m.get("clf_threshold", 0.5),
        "baseline_test_metrics": m["baseline_test_metrics"],
        "example": m["example"],
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=Prediction)
def predict(ipo: IPOInput):
    return get_model().predict(ipo.model_dump())


@app.post("/explain", response_model=Explanation)
def explain(ipo: IPOInput, top: int = Query(12, ge=1, le=40)):
    return get_model().explain(ipo.model_dump(), top=top)


@app.get("/past-ipos")
def past_ipos(year: Optional[int] = Query(None, ge=2024, le=2030), sector: Optional[str] = None):
    rows = get_model().past
    if year is not None:
        rows = [r for r in rows if r["listing_date"].startswith(str(year))]
    if sector:
        if sector not in get_model().meta["categories"]["sector"]:
            raise HTTPException(422, f"Unknown sector '{sector}'")
        rows = [r for r in rows if r["sector"] == sector]
    return {"count": len(rows), "decision_threshold": get_model().meta.get("clf_threshold", 0.5), "items": rows}
