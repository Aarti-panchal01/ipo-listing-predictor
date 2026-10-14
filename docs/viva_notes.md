# Viva notes: 15 likely questions, answered from the actual results

**1. What exactly are you predicting, and why the open price?**
`listing_gain_pct` = (listing-day open − issue price) / issue price × 100, plus `listed_positive` (gain > 0).
The open is the first price an allottee can sell at and is set by the pre-open auction, so it reflects
pre-listing demand most directly. NSE open is used, BSE if NSE is missing.

**2. How big is the dataset and how did you split it?**
604 mainboard IPOs from March 2012 to October 2026, 35 features. Split by listing date: train 2012-2022
(256), validation 2023 (58), test 2024+ (290). A random split would leak the future: IPOs from the same
hot month share market conditions.

**3. How do you know there's no leakage?**
Every feature is in a registry with the moment it becomes known; `leakage_check()` stops the build if a
feature is unregistered, named like a listing-day or target column, VIX is dated after the close date, a
close date is not before listing, a feature has |Spearman| > 0.95 with the target, or a GMP reading is on
or after listing day. Market-heat features only count IPOs listed before this IPO closed. The InvestorGain
tracker's GMP is updated on listing morning, so I used each IPO's own GMP history page instead.

**4. Why did you drop `days_close_to_listing`?**
SEBI moved listings to T+3 in 2023. The feature went from about 10 days to 5, with KS drift 0.99 between
train and test. It encoded the calendar, not the IPO.

**5. What's your headline result?**
Test MAE 20.1 points, against 18.6 for always predicting the median. On the average miss the model does
not beat the baseline on 2024-2026. It is better on RMSE (27.8 vs 30.1), so it gets the big listings
closer, and it ranks up vs down well (ROC-AUC 0.82, balanced accuracy 0.75 vs 0.64 for the subscription rule).

**6. Why is validation (14.8) so much better than test (20.1)?**
A regime shift. Median total subscription was about 9x in the training years and about 34x in 2023
onward, but 2025-2026 gains were modest (median 5 to 8%). The model learned "34x means hot" and
over-predicts the test years by 12.1 points on average. 2023 still followed the old pattern, so validation
looked good.

**7. fastai has the best test MAE (16.2). Why not use it?**
Because I'd be choosing the model by its test score, which turns the test set into a second validation set
and makes the reported number optimistic. fastai was the worst on cross-validation (20.9) and on 2023
(20.6). Its test win is explained by bias: it over-predicts least (+5.4 vs +12.1) because heavy
regularisation pulls it toward the average.

**8. Why does your classifier need a cut-off of 0.71 instead of 0.5?**
About 70% of IPOs list up and the tuned classifier uses a low learning rate, so its scores sit between
roughly 0.55 and 0.75. At 0.5 it calls every IPO up (balanced accuracy 0.50). The cut-off is chosen on the
2023 validation year by Youden's J (maximise true-positive rate minus false-positive rate), never on test.

**9. Grid vs random vs Optuna: which won?**
On CV Optuna was best (14.75), then grid (14.98), random (15.14), all beating my hand-picked settings
(15.86) with 270 fits each. But on 2023 the hand-picked settings were best (15.2 vs 15.8 to 16.4). The
searches fitted the 2012-2022 regime. I chose by 2023 MAE, so production uses the manual settings.

**10. Imputation or XGBoost's native missing-value handling?**
Native was best (14.59) but every strategy was within 0.35 points on 58 IPOs, so the honest conclusion is
"no difference". Native is simplest and keeps the missingness pattern (bNII/sNII only exist from 2022),
which is itself informative.

**11. What does SHAP tell you?**
Subscription is almost the whole model: total, retail, QIB, NII are the top four. QIB adds about +10
points once it passes roughly 70 to 100x; retail adds +15 to +20 above roughly 25 to 50x. Issue size and
VIX stay within ±1 point. Recent IPO-market heat slightly lowers the forecast, but those are the most
drifted features, so I don't read that as causal.

**12. Why SHAP instead of XGBoost's built-in importance?**
Built-in `weight` counts splits, so continuous features split often for tiny gains look important (VIX
12th by weight, 28th by SHAP). `gain` rewards a feature for one lucky split. SHAP is consistent, additive
per prediction (base value + contributions = prediction, which the API test checks) and has a direction.
A model-agnostic Permutation explainer agreed with TreeSHAP (Spearman 0.85).

**13. Tell me about a prediction that went badly wrong.**
Shree Ram Twistex, March 2026: predicted +37.3%, opened −34.6%. Retail 77x and NII 220x, but QIB only
3.9x; retail alone added +20 points. GMP was −27% the evening before. When retail and HNI money is euphoric
but institutions stay away, the model is overconfident.

**14. Does subscription data add anything beyond the grey-market premium?**
No. GMP taken literally has test MAE 8.8 (R² 0.78). XGBoost on GMP alone gets 9.5, and adding subscription
(10.0) or everything (10.3) makes it slightly worse. GMP is read later and already prices in demand. I keep
it out of the production model because it is unofficial, unregulated and easy to manipulate.

**15. Did blending help, and what would you do next?**
The best 2023 blend was 0.95 XGBoost + 0.05 TabNet, improving 2023 MAE by 0.03 points; on test it was
0.14 better (CI 0.07 to 0.20), real but negligible, so production is XGBoost alone. Next: subscription
relative to recent IPOs (for example total_x over the trailing 90-day median) to adapt to the new regime,
evaluated only on IPOs listed after October 2026; quarterly retraining; conformal prediction intervals.

---

Quick numbers: 604 IPOs · 256/58/290 split · 35 features · 69.2% list up · median gain 8.2%, mean 18.6% ·
test MAE 20.1 (baseline 18.6) · RMSE 27.8 (30.1) · ROC-AUC 0.82 · balanced accuracy 0.75 · cut-off 0.71 ·
GMP MAE 8.8 · bias +12.1 · Spearman(total_x, gain) 0.76.
