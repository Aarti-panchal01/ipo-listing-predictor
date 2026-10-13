---
title: IPO Listing Predictor API
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predicts Indian IPO listing-day gains, explained with SHAP
---

# IPO listing-gain predictor: API

FastAPI service for an educational project that predicts the listing-day gain of Indian
mainboard IPOs from information available before listing. **Not investment advice and not
SEBI-registered advice.**

| Endpoint | What it does |
|---|---|
| `GET /health` | Liveness check |
| `GET /meta` | Feature list, input ranges, sectors, model metrics, a prefilled example |
| `POST /predict` | Predicted listing gain %, probability of listing above issue price, confidence note |
| `POST /explain` | Signed SHAP contributions for the same input (top N, `?top=12`) |
| `GET /past-ipos` | Test-set IPOs (2024+) with predicted vs actual, filter by `year` / `sector` |

Interactive docs at `/docs`. Source, notebooks and report:
https://github.com/Aarti-panchal01/ipo-listing-predictor
