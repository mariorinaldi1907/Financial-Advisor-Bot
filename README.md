# Financial Advisor Bot — Market Research

[Try the public Market Outlook app](https://mario-market-outlook.streamlit.app/).

A reproducible market-outlook and downside-risk system for **22 ETFs**, with **1-, 5-, 10-, and 20-session horizons**. Shared models learn from technical features, cross-asset relationships, and delayed volatility/Treasury-index context. The beginner-friendly dashboard starts with a **Positive / Uncertain / Negative** outlook, plain explanations, price history, and past forecast accuracy. Detailed research results remain available in an optional section. Each outlook includes a plain-language research takeaway: **Consider buying**, **Wait / review**, or **Consider reducing**. Weak evidence always produces Wait / review; uncertainty is not an automatic hold instruction. These prompts do not assess whether an investment suits someone.

**Status:** research software. Temporal safeguards and richer models do not guarantee an investment edge. The current results do not establish consistent out-of-sample superiority over simple priors.

## Run

Python 3.12+; the Docker image uses Python 3.13 and the full research checks use Python 3.14.

```bash
git clone https://github.com/mariorinaldi1907/Financial-Advisor-Bot.git
cd Financial-Advisor-Bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows activation: `.venv\Scripts\activate`. The app uses the committed snapshots and saved evaluation predictions offline. Load joblib files only from trusted sources.

## Train, refresh, and track future forecasts

Training dependencies are separate from the lightweight website dependencies. Install them before training or running the full test suite.

```bash
python -m pip install -r requirements-train.txt

# Full offline research run, including chronological validation and evaluation.
python -m src.research

# Retrain one horizon. Existing forecasts are preserved in the issue ledger.
python -m src.research --horizons 20

# Refresh the free Philadelphia Fed monthly economic vintages.
python -m src.macro

# Refresh market/environment snapshots and retrain all horizons.
python -m src.universe --download --start 2007-01-01

# Re-evaluate issued forecasts when actual later prices become available.
python -m src.forward

# Correctness and dashboard checks.
python -m unittest discover -s tests -v
```

`python -m src.train` is an alias for the current full training run. `python -m src.download` refreshes data only. Downloads default to an exclusive end date of today in New York, excluding an incomplete current session. `--end YYYY-MM-DD` makes the cutoff explicit.

Snapshots are checked for duplicate and missing ETF sessions and adequate coverage. Models/report hashes must match the snapshots; the app rejects stale combinations. After refreshing, retrain all horizons before relying on the dashboard. Avoid running the app while training updates its files.

## Deploy a free website

Streamlit Community Cloud can serve the existing Python app directly from GitHub. Choose the repository, select the branch containing these changes (`market-outlook-improvements` until merged), and set the main file to `app.py`. Select Python 3.13 in advanced settings. No API keys or paid model service are required. Share the actual `streamlit.app` URL after the deployment succeeds.

The website reads saved public market snapshots; it does not train models or download fresh data at startup. Refresh and train separately, validate the results, then commit a complete matching snapshot. The saved date is shown prominently, and old data produces an Uncertain outlook.

## Run with Docker

With Docker installed:

```bash
docker compose up --build -d
# Open http://localhost:8501
# Stop the demo when finished:
docker compose down
```

If port 8501 is already occupied, use `APP_PORT=8502 docker compose up --build -d` and open port 8502. The compose configuration binds to the local computer; use a hosting provider’s HTTPS routing for a public deployment.

The image includes only website dependencies and public snapshots, runs as an unprivileged user, and checks snapshot consistency before startup. It supports a host-provided `PORT` and includes an HTTP health check. Compose runs with a read-only filesystem and temporary scratch space. GitHub Actions builds the image, renders the app using its website dependencies, and checks HTTP startup. Model weights, credentials, local environments, and training code are excluded from the image.

## Review findings and implemented safeguards

The earlier pipeline already kept logistic/Ridge preprocessing inside training pipelines, purged fitting/calibration target overlap, used rolling history, limited model depth, and fixed random seeds. The review identified these gaps:

| Finding | Refactor |
| --- | --- |
| No explicit embargo after validation | Twenty actual market sessions after the last validation outcome are embargoed; later fitting and calibration exclude those dates. |
| Validation dates near year-end could have outcomes in the next period | Both validation and evaluation slices require the entire outcome window to end within their period. |
| Environment inputs included raw index levels | Use lagged changes and rolling Z-scores referenced to the previous 252 sessions; no raw environment levels enter models. |
| Liquidity proxy depended on nominal price scale | Normalize recent illiquidity by its strictly historical rolling mean. |
| Annual evaluation was pooled in the summary | Report each period and asset separately, including accuracy, balanced accuracy, log loss, Brier score, ECE, and confusion matrices. |
| Calibration-fit diagnostics could be mistaken for evaluation | Label these as in-sample diagnostics and calculate separate out-of-sample probability scores. |
| Previously inspected dates could be presented as a fresh holdout | Mark all historical results exploratory and keep issued forward forecasts separately. |
| Numerical cancellation in self-correlation could affect tree splits across platforms | Use exact self-correlation/beta identities and eight-decimal model-feature precision before fitting and inference. |
| Historical vendor corrections are not original release vintages | Disclose this source limitation; observation delays do not remove historical revisions. |

The refactored training script is [`src/research.py`](src/research.py); causal feature generation is in [`src/context.py`](src/context.py).

### Splitting and non-stationarity

- Split by **global trading date**, keeping all assets for a date together.
- Use the preceding **five years** of mature outcomes at every refit. This is a rolling training window.
- Reserve the final **126 historical sessions** for probability and interval calibration.
- Purge fitting rows whose target outcomes reach calibration. Purge all fitting/calibration outcomes that reach validation/evaluation.
- End every scored period before its boundary, including the target's end date.
- Embargo **20 market sessions after the last scored outcome**. Advance the next scored period past that embargo and exclude embargo observations from subsequent fits/calibration.
- Fit imputation, missingness indicators, and scaling only inside each training pipeline. Tree models use their native missing-value handling and learn bins only from their fit data.
- Disable histogram boosting's random internal validation. Use fixed iteration budgets and explicit regularization.

Earlier scored outcomes may enter a later refit once they are known, as in a real walk-forward process. This does not make successive folds independent. Post-validation embargoes supplement outcome purging; they do not eliminate all financial dependence.

Validation nominal periods are 2018–2019, 2020–2021, and 2022–2023. Subsequent starts move forward to respect embargoes. A candidate is eligible only if it beats the per-asset historical prior in at least two validation periods, improves average asset-mean log loss by at least 0.0005, and worsens no fold by more than 0.002. The eligible candidate with the lowest mean loss is selected; otherwise the prior is used. These fixed research guardrails do not guarantee statistical significance or future performance. The same independent gate is used for the downside head. The 2024, 2025, and 2026 evaluation blocks refit annually using only earlier mature outcomes. Evaluation scores never choose the candidate.

### Features and model families

Market inputs include percentage-return momentum, price-relative moving averages/MACD, RSI, relative volume, normalized volatility/liquidity, drawdown, ATR relative to price, rolling correlations/beta, relative strength, calendar cycles, and historical frequencies of already realized outcomes. Fixed asset/class indicators describe the predeclared universe; their categories are not learned from future observations.

Environment observations are delayed by two completed market sessions. Stale values are removed. Volatility and Treasury-index levels become changes and backward-looking Z-scores. These transformations reduce dependence on nominal levels; they do not mathematically guarantee stationarity across regimes.

Compared families:

- Smoothed prior probability for each asset.
- Pooled logistic regression with strong L2 regularization (`C=0.01`).
- CatBoost with market inputs, and a matched CatBoost with environmental context: depth 5, 240 iterations, learning rate 0.03, L2 leaf penalty 20.
- Histogram gradient boosting: 15 leaves, minimum 100 observations per leaf, L2 penalty 20, 160 iterations, no random early-stopping split.
- Fixed equal-weight ensemble and a version blended equally with the prior.

Feature-group experiments compare the same L2 logistic estimator on a compact core (returns, momentum, volatility, drawdown, relative volume, asset IDs and matured historical frequencies), then add credit, breadth, economic vintages, or delayed environment inputs separately. A compact CatBoost is also compared. The report saves the exact columns, validation folds, and promotion decision for every candidate. This tests redundancy without selecting features using evaluation outcomes.

The credit inputs are HYG/LQD relative-return and historical Z-score proxies, not actual credit spreads. Breadth measures the fixed nine-sector ETF universe; it is not a survivorship-free index of all stocks.

A separate calibrated logistic head estimates a loss exceeding one current trailing-volatility unit over the chosen horizon. The threshold is fixed from information available on the forecast date (`20-session volatility × sqrt(horizon)`). This compares larger losses consistently across assets, but it is not the probability of any loss or a 5% loss. Per-asset prior tail-event frequencies form the downside baseline. Brier/log-loss scores are saved per evaluation period; return intervals continue to describe potential severity.

Non-baseline probability estimates receive sigmoid calibration using the separate historical calibration slice. A Ridge model estimates volatility-normalized expected return. A regularized CatBoost quantile model predicts the 10th/50th/90th percentiles; per-asset historical calibration widens the interval. Dependence and changing distributions mean nominal 80% coverage is not guaranteed.

## Data and scope

Targets: SPY, QQQ, IWM, VEU, AGG, TLT, GLD, XLF, XLK, XLE, XLV, XLI, XLU, XLP, XLY, XLB, EEM, EFA, HYG, LQD, IEF, and DBC. UUP supplies additional dollar context.

The saved Yahoo snapshot covers 2007–20 November 2026, subject to inception dates. ETF OHLC is dividend/split-adjusted. Environment inputs are Yahoo's VIX and 13-week/5-year/10-year/30-year Treasury-yield indices **as reported**. They are market-environment proxies, not direct inflation/employment releases. Inflation and payroll employment context comes from the Philadelphia Fed’s [monthly historical CPI vintages](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcpi) and [employment vintages](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/employ). For each vintage, monthly and annual growth are computed entirely within that vintage. No latest-revised column is substituted. A vintage enters the model only on the first calendar day after its month ends; values expire after 62 days without an update. This intentionally sacrifices release-day responsiveness for conservative timing. The compact CSV records observation date, vintage month, availability date, source URL and original workbook SHA-256. Refreshing data never occurs in a website request.

No paid feed, historical news archive, exact release-time/consensus-surprise feed, brokerage execution, or cloud GPU service is included. Yahoo history can be revised. Strict point-in-time guarantees require point-in-time source data, even when feature timing is correct.

## Included results

Equal-weight averages across 22 ETFs, using the embargoed 2024–2026 evaluation periods:

| Horizon | Selected model | Accuracy | Balanced accuracy | Prior accuracy |
| --- | --- | ---: | ---: | ---: |
| 1 session | logistic_credit | 53.13% | 50.06% | 52.93% |
| 5 sessions | shrink_ensemble | 55.08% | 49.53% | 55.77% |
| 10 sessions | shrink_ensemble | 55.70% | 49.12% | 57.05% |
| 20 sessions | shrink_ensemble | 59.71% | 49.14% | 60.08% |

The selected models still do not beat the priors on average probability quality at any horizon. The downside head falls back to historical tail-event frequencies at all four horizons because the fitted alternative did not pass validation. The website labels that estimate accordingly. Balanced accuracy is unavailable for a risk evaluation slice containing only one observed class. The selected models do not consistently outperform the priors. These results quantify the current limitation; richer inputs and more complex estimators are not assumed to improve accuracy. Expected-return forecasts are also compared against a zero-return forecast, and interval coverage is reported rather than assumed.

Read `data/research_results.json` for every validation fold, annual evaluation, asset, candidate, calibration diagnostic, confidence-interval calculation, and model-selection decision. The dashboard exposes annual regime stability directly. `data/research_predictions.csv` contains the saved evaluation predictions and fitting-outcome cutoffs.

These dates overlap previously inspected experiments. They are valid chronological simulations but cannot be described as a new untouched test. Daily multi-session targets overlap, and ETFs are correlated. Date-block bootstrap intervals retain all assets together; they do not correct for repeated research or selecting among multiple horizons.

### Execution study

The optional research study uses a fixed 60% UP-probability threshold and predicted return above assumed round-trip costs. Each sampled holding window is a separate round trip: enter at the next adjusted open, exit after the horizon, charge 5 basis points each way, and hold cash during gaps. Cash interest is zero. Buy/hold-window comparisons use the same round-trip convention.

Drawdown is measured at holding-period endpoints; intraperiod drawdowns are not shown. This study excludes market impact, financing, taxes, and operational slippage. It is not live trading performance or a portfolio-suitability calculation.

### Forward evidence

`data/forecast_ledger.csv` preserves the first forecast issued for each asset/date/horizon/model version, including versions issued before later refits. New entries record the exact model-file SHA-256; older unversioned entries are identified explicitly. Retraining does not rewrite those probabilities. `src.forward` scores a forecast only after its future market session exists and rejects issue timestamps outside the interval between its as-of close and the next session's open. Forward summaries exclude pending and ineligible issues, retain model versions separately, and report direction accuracy, Brier/log-loss scores against paired baseline forecasts, interval coverage, and downside scores when fields exist. Older issues without those fields are not silently assigned new baseline or risk forecasts. Forecast evidence is kept distinct from retrospective simulations. Run `python -m src.forward` after updating market observations; this cannot manufacture future evidence.

## Files

```text
app.py                         Dashboard
src/research.py                Walk-forward fitting, calibration, embargo, evaluation, refit
src/context.py                 Shared market panel and causal stationary-style features
src/train.py                   Portable model primitives and training alias
src/universe.py                Validated downloads and training entry point
src/forward.py                 Resolution of genuinely issued forecasts
src/features.py                Technical indicator primitives
src/macro.py                   Free monthly economic vintage refresh and normalization
tests/                         Leakage, preprocessing, timing, artifact, and UI checks
data/market_prices.csv         One shared ETF snapshot
data/environment.csv           One environment snapshot
data/macro_vintages.csv        Compact historical inflation/employment vintages
data/research_results.json     Per-fold/per-period evidence and configuration
data/research_predictions.csv  Saved evaluation predictions
data/forecast_ledger.csv       Issued forecasts and eventual observed outcomes
models/horizon_*.joblib        Four pooled models and calibration/return heads
```

Superseded single-ETF and neural experiment outputs have been removed from the active layout. Earlier versions remain in Git history. No duplicate feature datasets, extra training environments, or artifact folders are retained.

References: [scikit-learn time-series splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), [CatBoost objectives](https://catboost.ai/docs/en/concepts/loss-functions), [FRED historical real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

**Mario Rinaldi.** Financial modelling research; forecasts are uncertain and do not establish suitable investments or allocations.
