# Financial Advisor Bot

An academic SPY forecasting dashboard with reproducible training, technical indicators, chronological evaluation, and a Streamlit interface. The target is whether the adjusted closing price will be **higher after 10 trading sessions**. Class 0 includes both lower and unchanged prices.

**Current finding:** the tested indicator models do not establish an advantage over a simple UP-frequency baseline. The dashboard explicitly reports that limitation rather than treating model probabilities as investment confidence.

## Quick start

Use Python **3.12 or newer**; this run was tested with **Python 3.14.3**. The included artifacts were trained with scikit-learn 1.8.0; use the pinned version to load them.

```bash
git clone https://github.com/mariorinaldi1907/Financial-Advisor-Bot.git
cd Financial-Advisor-Bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate` instead. Open the local address printed by Streamlit. Run the commands from the repository directory. The app resolves its artifact paths relative to its own location.

The included snapshot and trained artifacts work without downloading data. Only load joblib/pickle files from sources you trust.

## What you can explore

- **Latest forecast:** uses the deployed model, refitted on all available mature labels, for the latest snapshot session only.
- **Historical holdout:** displays saved predictions from a model trained strictly before the evaluation period. Weekend selections use the preceding trading session.
- Adjusted SPY prices, moving averages, RSI, momentum, relative volume, and volatility.
- Candidate model scores and a downloadable CSV of historical predictions.
- A visible warning when the data snapshot is more than seven calendar days old.

Indicator context describes market conditions; it is not a local explanation of model decisions. Investment amount and risk sliders have been removed because the prototype does not model suitability or portfolio allocation.

## Refresh data and retrain

```bash
# Network required for this step only. The end date is exclusive.
python -m src.download --start 2014-01-01 --end 2026-11-23

# Offline, deterministic training from the saved snapshot.
python -m src.train

# Run correctness checks.
python -m unittest discover -s tests -v
```

Omit `--end` to use today's date in New York and exclude the current, potentially incomplete session. Yahoo data uses `auto_adjust=True` for dividend/split adjusted OHLC. A failed or empty download leaves the existing snapshot untouched. Refreshing downloads the whole history consistently; it does not append prices with different adjustment factors.

Training writes these files together:

| Artifact | Purpose |
| --- | --- |
| `data/raw/spy.csv` | Saved adjusted market snapshot, used as training input |
| `data/spy_features.csv` | Causal indicator rows, including the latest unlabeled session |
| `models/forecast.joblib` | Selected model refitted for latest-session inference, plus metadata |
| `reports/evaluation.json` | Fold scores, holdout scores, dates, feature names, versions, raw-data SHA-256 |
| `reports/holdout_predictions.csv` | Predictions made by development-only models on the holdout |

For isolated experiments, use `python -m src.train --raw PATH_TO_SNAPSHOT --output PATH_TO_EXPERIMENT`. Restart Streamlit after replacing artifacts; file modification times also invalidate its loading caches. Avoid running the app while training is replacing files.

## Model and evaluation design

1. Build indicators using information available at each session's close. No backward filling, future prices, or labels enter model inputs.
2. Define `Target(t) = 1 if Close(t+10) > Close(t), else 0`. The final 10 sessions have unknown labels and remain available for feature computation, not training.
3. Reserve the final 20% of labeled rows as the chronological holdout.
4. Remove development rows whose 10-session outcomes reach the holdout. Each development fold also has a 10-session gap, with an explicit check that training outcomes mature before validation begins.
5. Compare six fixed candidates across four expanding development folds. Select using mean **log loss**, without consulting holdout scores.
6. Evaluate all candidates on the holdout for transparency, then refit the preselected winner on all mature labels for the latest forecast. Saved historical predictions remain separate from this refit.

Candidates: training-prior baseline, original Random Forest configuration, regularized Random Forest with normalized features, its expanded-indicator variant, regularized logistic regression, and expanded logistic regression. Scaling is fitted inside each logistic-regression training fold.

### Indicators tested

The original model uses absolute moving-average levels, MACD, RSI, volatility, momentum, volume change, and raw volume. Normalized candidates use price-relative SMA/EMA gaps and MACD, plus relative volume. These reduce dependence on the historical price level.

The expanded set adds daily return, 60-session momentum, Bollinger z-score, 60-session drawdown, short/long volatility ratio, ATR relative to price, and the stochastic oscillator. RSI uses the original rolling-average formulation, not Wilder smoothing. These additions are evaluated as a bundle, not claimed individually to improve predictions.

### Included run

Snapshot: **2 January 2014 – 20 November 2026**, 3,242 sessions. Development ends 24 April 2024, with its last label maturing 8 May. Holdout: **9 May 2024 – 6 November 2026**, 627 labeled sessions.

| Candidate | Mean validation log loss ↓ | Holdout accuracy | Holdout log loss ↓ |
| --- | ---: | ---: | ---: |
| **Prior baseline (selected)** | **0.6464** | **61.56%** | **0.6682** |
| Original Random Forest, retrained | 1.2172 | 40.35% | 1.0112 |
| Normalized Random Forest | 0.7122 | 60.13% | 0.6937 |
| Expanded Random Forest | 0.7168 | 60.13% | 0.6909 |
| Logistic regression | 0.7072 | 61.40% | 0.6709 |
| Expanded logistic regression | 0.7261 | 61.40% | 0.6720 |

The baseline estimates UP probability from the training-label frequency. Its balanced accuracy and ROC AUC are both 0.5; its 61.56% holdout accuracy reflects the prevalence of UP outcomes, not market timing skill. The original-model row retrains the old configuration using the new evaluation procedure; it is not a score for the old pickle on its original test set.

No candidate demonstrated a validation advantage over the baseline. Adding indicators alone did not solve that problem. Probabilities are uncalibrated estimates. The dashboard does not provide trading-performance claims or allocation instructions.

## Repository layout

```text
app.py                              Streamlit research dashboard
src/features.py                     Shared causal features and target construction
src/download.py                     Explicit SPY snapshot refresh
src/train.py                        Purged validation, selection, evaluation, refit
tests/test_pipeline.py              Temporal leakage and feature correctness checks
data/raw/spy.csv                    Updated adjusted OHLCV snapshot
data/spy_features.csv               Generated indicator snapshot
models/forecast.joblib              Current deployment artifact
reports/evaluation.json             Detailed machine-readable run evidence
reports/holdout_predictions.csv     Historical predictions from pre-holdout fits
notebooks/FinancialAppModel_legacy.ipynb  Original notebook, outputs cleared
models/rf_model_spy.pkl              Preserved original model (unused by app)
data/spy_processed.csv               Preserved original data (unused by app)
```

The legacy notebook is retained as project history. Its original evaluation has overlapping train/test label windows and is superseded by `src.train`. It also contains notebook-specific installation commands and optional plotting/widget dependencies; it is not the supported reproduction path.

## Limits and sensible next experiments

- SPY only. QQQ, VEU, and AGG appeared in the original download notebook but were not trained or served by the app.
- Overlapping 10-session targets are statistically dependent. Row counts should not be interpreted as independent trials, and no confidence intervals or significance claims are supplied.
- Yahoo adjusted history can change with corporate actions and vendor revisions. The snapshot and hash identify this run; it is not a full point-in-time institutional dataset.
- Features require the selected session's completed close and volume. This is an end-of-day research forecast, not an intraday trading system.
- The latest refit has seen outcomes through the snapshot end. Only the separately saved holdout predictions support the reported historical evaluation.
- This holdout is now inspected. Future experiments need a new untouched forward period or nested walk-forward evaluation before claiming an improvement.
- Potential next experiments: longer historical coverage, individually ablated indicators, lagged cross-asset signals with aligned trading calendars, and release-date-aware macroeconomic data. Test each against the baseline before expanding scope.
- A useful trading study would also require a defined executable entry price, holding/rebalancing policy, transaction costs, and dependence-aware uncertainty estimates.

## Method references

- [scikit-learn TimeSeriesSplit and its gap parameter](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [yfinance download parameters and price adjustment](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)

Developed by **Mario Rinaldi**, BSc Computer Science final-year project. For academic and educational use; it does not constitute personalized financial advice.
