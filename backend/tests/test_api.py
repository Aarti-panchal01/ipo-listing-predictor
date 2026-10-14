from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

EXAMPLE = {
    "issue_size_cr": 570.0, "price_band_upper": 448.0, "fresh_issue_pct": 56.14, "lot_size": 33,
    "min_investment": 14784, "anchor_portion_cr": 171.0, "retail_reserved_pct": 35.0,
    "exchange": "BSE,NSE", "sector": "Healthcare",
    "qib_x": 116.15, "nii_x": 66.83, "bnii_x": 73.67, "snii_x": 53.15, "retail_x": 17.64, "total_x": 56.33,
    "listing_month": 12, "issue_days_open": 5,
    "nifty_ret_7d": 0.5, "nifty_ret_30d": 7.0, "vix_close": 14.0,
    "ipo_heat_30d": 15, "prev_gain_mean_90d": 25.0, "prev_count_90d": 30,
    "revenue_cr": 935.58, "pat_cr": 67.95, "revenue_growth_pct": 16.4, "pat_margin_pct": 7.34,
    "roe_pct": 24.58, "debt_equity": 0.85, "pe_post": 36.42, "mcap_post_cr": 3098.16,
    "promoter_post_pct": 50.89, "promoter_dilution_pct": 15.07,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_meta():
    r = client.get("/meta")
    assert r.status_code == 200
    body = r.json()
    assert "Healthcare" in body["sectors"]
    assert {"MAE", "RMSE", "R2", "ROC_AUC", "F1"} <= set(body["test_metrics"])
    assert "qib_x" in body["inputs"]
    assert "SEBI" in body["disclaimer"]


def test_predict():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert -100 < body["predicted_gain_pct"] < 500
    assert 0 <= body["prob_positive"] <= 1
    assert body["predicted_direction"] in ("positive", "flat_or_negative")
    assert body["predicted_direction"] == ("positive" if body["prob_positive"] >= body["decision_threshold"] else "flat_or_negative")
    assert body["confidence_note"]


def test_predict_minimal_inputs_only_required_fields():
    minimal = {k: EXAMPLE[k] for k in ("issue_size_cr", "price_band_upper", "fresh_issue_pct", "total_x", "listing_month")}
    r = client.post("/predict", json=minimal)
    assert r.status_code == 200, r.text


def test_predict_rejects_out_of_range():
    r = client.post("/predict", json={**EXAMPLE, "retail_x": 5000})
    assert r.status_code == 422
    assert "retail_x" in r.text


def test_predict_rejects_missing_required_and_unknown_fields():
    bad = {k: v for k, v in EXAMPLE.items() if k != "total_x"}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "gmp": 40}).status_code == 422


def test_predict_rejects_anchor_bigger_than_issue():
    r = client.post("/predict", json={**EXAMPLE, "anchor_portion_cr": 900.0})
    assert r.status_code == 422
    assert "anchor_portion_cr" in r.text


def test_explain_adds_up_to_prediction():
    r = client.post("/explain", json=EXAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["contributions"]
    pred = client.post("/predict", json=EXAMPLE).json()["predicted_gain_pct"]
    assert abs(body["predicted_gain_pct"] - pred) < 0.05
    shaps = [abs(c["shap"]) for c in body["contributions"]]
    assert shaps == sorted(shaps, reverse=True)


def test_explain_top_param():
    r = client.post("/explain?top=5", json=EXAMPLE)
    assert len(r.json()["contributions"]) == 5


def test_past_ipos_and_filters():
    r = client.get("/past-ipos")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == len(body["items"]) > 0
    row = body["items"][0]
    assert {"company", "listing_gain_pct", "predicted_gain_pct", "prob_positive"} <= set(row)
    y = client.get("/past-ipos?year=2024").json()
    assert all(i["listing_date"].startswith("2024") for i in y["items"])
    assert client.get("/past-ipos?sector=NotASector").status_code == 422


def test_cors_allows_localhost_and_blocks_others():
    ok = client.options("/predict", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    bad = client.options("/predict", headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_cors_allows_production_frontend_and_blocks_lookalikes():
    ok = client.options("/predict", headers={"Origin": "https://ipo-listing-predictor-three.vercel.app", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://ipo-listing-predictor-three.vercel.app"
    fake = client.options("/predict", headers={"Origin": "https://ipo-listing-predictor-evil.vercel.app", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in fake.headers


def test_predict_warns_when_subscription_inputs_missing():
    minimal = {k: EXAMPLE[k] for k in ("issue_size_cr", "price_band_upper", "fresh_issue_pct", "total_x", "listing_month")}
    note = client.post("/predict", json=minimal).json()["confidence_note"]
    assert "Missing subscription inputs" in note and "retail_x" in note
