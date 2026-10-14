"""Smoke test for the deployed app.

Usage:
  python scripts/smoke_test.py --api https://<space>.hf.space --web https://<app>.vercel.app

Checks every API endpoint, every frontend page, that the deployed frontend
bundle points at this API, and that a browser-style prediction round trip
(CORS preflight from the frontend origin, then POST /predict and /explain)
works. Exits non-zero on the first failure. Uses only the standard library
plus requests.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import requests

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        summary()
        sys.exit(1)


def summary() -> None:
    n_pass = sum(r[0] == PASS for r in results)
    print(f"\n{n_pass}/{len(results)} checks passed")


def wake(api: str, budget_s: int = 240) -> float:
    """Free HF Spaces sleep; poll /health until the container answers."""
    t0 = time.time()
    while time.time() - t0 < budget_s:
        try:
            r = requests.get(f"{api}/health", timeout=20)
            if r.status_code == 200:
                return time.time() - t0
        except requests.RequestException:
            pass
        time.sleep(5)
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--web", required=True)
    a = ap.parse_args()
    api, web = a.api.rstrip("/"), a.web.rstrip("/")

    took = wake(api)
    check("API wakes up and /health answers", took >= 0, f"{took:.1f}s" if took >= 0 else "no answer in 240s")

    r = requests.get(f"{api}/health", timeout=30)
    check("GET /health", r.status_code == 200 and r.json().get("status") == "ok", r.text[:80])

    r = requests.get(f"{api}/meta", timeout=30)
    meta = r.json() if r.ok else {}
    check("GET /meta", r.ok and {"MAE", "ROC_AUC"} <= set(meta.get("test_metrics", {})) and meta.get("sectors"),
          f"model {meta.get('model_version')}, test MAE {meta.get('test_metrics', {}).get('MAE', float('nan')):.2f}")
    example = {k: v for k, v in meta["example"]["inputs"].items() if v is not None}

    r = requests.post(f"{api}/predict", json=example, timeout=30)
    p = r.json() if r.ok else {}
    check("POST /predict (example IPO)", r.ok and 0 <= p.get("prob_positive", -1) <= 1,
          f"{meta['example']['company']}: predicted {p.get('predicted_gain_pct')}%, P(up) {p.get('prob_positive')}, "
          f"actual {meta['example']['actual_listing_gain_pct']:.1f}%")

    r = requests.post(f"{api}/explain", json=example, timeout=30)
    e = r.json() if r.ok else {}
    check("POST /explain", r.ok and len(e.get("contributions", [])) > 0
          and abs(e.get("predicted_gain_pct", 1e9) - p["predicted_gain_pct"]) < 0.05,
          f"top driver: {e['contributions'][0]['label']} {e['contributions'][0]['shap']:+.2f} pts" if r.ok else r.text[:120])

    r = requests.post(f"{api}/predict", json={**example, "retail_x": 99999}, timeout=30)
    check("POST /predict rejects out-of-range input with 422", r.status_code == 422, r.text[:100])

    r = requests.get(f"{api}/past-ipos", timeout=30)
    past = r.json() if r.ok else {}
    check("GET /past-ipos", r.ok and past.get("count", 0) > 0, f"{past.get('count')} test-set IPOs")
    r = requests.get(f"{api}/past-ipos", params={"year": 2025}, timeout=30)
    check("GET /past-ipos?year=2025 filter", r.ok and all(i["listing_date"].startswith("2025") for i in r.json()["items"]),
          f"{r.json().get('count')} rows")

    # frontend pages
    for path, needle in [("/", "LISTING"), ("/past", "LISTING"), ("/insights", "LISTING"), ("/about", "LISTING")]:
        r = requests.get(f"{web}{path}", timeout=30)
        check(f"Frontend GET {path}", r.status_code == 200 and needle in r.text and "SEBI" in r.text,
              f"HTTP {r.status_code}, disclaimer {'present' if 'SEBI' in r.text else 'MISSING'}")
    r = requests.get(f"{web}/insights/eda.json", timeout=30)
    check("Frontend serves exported insight data", r.ok and r.json().get("n_ipos", 0) > 0, f"{r.json().get('n_ipos')} IPOs")

    # the deployed bundle must be built against THIS api
    html = requests.get(web, timeout=30).text
    chunks = re.findall(r'/_next/static/chunks/[^"]+\.js', html)
    found = False
    for c in chunks:
        if api in requests.get(web + c, timeout=30).text:
            found = True
            break
    check("Frontend bundle is configured with this API URL", found, f"searched {len(chunks)} chunks")

    # browser-style round trip: preflight from the frontend origin, then the real calls
    origin = web
    r = requests.options(f"{api}/predict", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                    "Access-Control-Request-Headers": "content-type"}, timeout=30)
    check("CORS preflight from frontend origin", r.ok and r.headers.get("access-control-allow-origin") == origin,
          f"allow-origin={r.headers.get('access-control-allow-origin')}")
    r = requests.post(f"{api}/predict", json=example, headers={"Origin": origin}, timeout=30)
    check("End-to-end: POST /predict with frontend Origin", r.ok and r.headers.get("access-control-allow-origin") == origin,
          f"predicted {r.json().get('predicted_gain_pct')}%")
    r = requests.options(f"{api}/predict", headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"}, timeout=30)
    check("CORS blocks an unknown origin", "access-control-allow-origin" not in r.headers)

    summary()


if __name__ == "__main__":
    main()
