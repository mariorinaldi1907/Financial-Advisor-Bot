# Financial Advisor Bot

A multi-ETF research system that predicts whether an ETF's adjusted closing price will be higher after **10 trading sessions**. It combines reproducible model training, market-context features, Hugging Face forecasting models, and a Streamlit dashboard.

**Scope:** SPY, QQQ, IWM, VEU, AGG, TLT, and GLD. UUP supplies dollar context. The saved snapshot covers 2007–20 November 2026, subject to each ETF's listing date.

The project reports unsuccessful experiments as well as promising ones. Higher accuracy alone does not establish a useful trading signal: a model that always predicts UP can score well when prices usually rise.

## Run the app

Python 3.12+; the dashboard was tested with Python 3.14.3 and optional neural training with Python 3.13.

```bash
git clone https://github.com/mariorinaldi1907/Financial-Advisor-Bot.git
cd Financial-Advisor-Bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows activation: `.venv\Scripts\activate`. Open Streamlit's printed local address. The app works with the included snapshots and predictions without downloading Hugging Face weights or installing PyTorch.

The dashboard lets you switch ETFs, inspect historical predictions, view the latest forecast, compare accuracy with baselines, and explore Chronos forecast bands. Historical predictions remain separate from models refitted on later data. Weekend date selections use the preceding session.

## What is trained

### Separate models for each ETF

Each ETF is evaluated with nine candidates:

- A training-prior baseline, which estimates how often the ETF's 10-session return was positive.
- The original Random Forest configuration.
- Regularized Random Forests using normalized indicators, with and without additional indicators.
- Regularized logistic regression, with and without additional indicators.
- A contextual Random Forest and two contextual logistic regressions with different regularization strengths.

Technical features include price-relative moving-average gaps, normalized MACD, RSI, momentum, volatility, relative volume, Bollinger position, drawdown, ATR, and stochastic position. RSI retains the original rolling-average formulation rather than Wilder smoothing.

Contextual candidates also use **other ETFs'** recent returns, trends, and volatility. Equity ETFs proxy broad market conditions; AGG/TLT proxy bond-market conditions; GLD supplies gold context; UUP supplies dollar context. These are market-environment proxies, not direct measurements of inflation, unemployment, geopolitical events, or news sentiment.

### Hugging Face models and joint fine-tuning

The project benchmarks:

- `amazon/chronos-bolt-tiny`: frozen price-only forecasts.
- `amazon/chronos-2`: frozen price-only and historical-context forecasts.
- **Chronos-2 with a project-trained LoRA adapter:** jointly adapted to all seven ETF histories, using historical technical indicators and the eight market-price series.

The adapter uses 120 optimizer steps, learning rate `1e-5`, LoRA rank 8, a 256-session context, and a 10-session forecast horizon. ETF fine-tuning ends **30 September 2025**. It uses historical covariates only; future indicators or ETF prices are not supplied.

Pinned base-model revisions predate the November 2025 evaluation start. The neural model outputs price quantiles; an approximate directional probability is obtained by interpolating the forecast deciles. Its probabilities and intervals have not been calibrated for financial markets.

The adapter is experimental and is shown separately in the dashboard. A favorable retrospective score does not automatically replace the model selected using earlier validation data.

## Retrain and refresh

```bash
# Retrain all seven ETF models from the included snapshot, offline.
python -m src.universe

# Download a consistent, longer market snapshot and retrain all seven.
python -m src.universe --download --start 2007-01-01

# Train a specific ETF.
python -m src.train --ticker QQQ

# Check temporal boundaries, model inputs, and the dashboard.
python -m unittest discover -s tests -v
```

Downloads use Yahoo Finance through yfinance, with dividend/split-adjusted OHLC. The exclusive end date defaults to today in New York to avoid an incomplete current session. `--end YYYY-MM-DD` makes the cutoff explicit. Downloads are validated for all ETFs before replacing the shared snapshot; failed downloads leave it untouched.

Model and data files must match. After a data refresh, retrain all assets; the app detects a changed snapshot and asks for retraining. Restart Streamlit after replacing models. Avoid using the app while a training command is updating its files.

### Reproduce optional neural training

Install the heavier dependencies into your existing environment only when needed:

```bash
python -m pip install -r requirements-hf.txt
python -m src.fetch_foundation
python -m src.finetune
python -m src.foundation
```

`fetch_foundation` downloads only pinned model configs and safetensors. Base weights are cached outside the project in the system temporary directory. Fine-tuning uses temporary checkpoints and retains only the small adapter and training metadata in `models/`. Downloaded base weights are not committed to GitHub. CPU inference and training are supported; `src.foundation --device cuda` supports compatible CUDA installations.

Re-running the fixed training recipe is an experiment, not permission to select parameters using the already inspected test results. After refreshing beyond the included snapshot, keep fine-tuning cutoffs and evaluation dates explicit.

## How evaluation avoids future information

For the ETF classifiers, the final 20% of labeled dates is reserved as a chronological holdout. Four expanding development folds select the candidate with lowest mean log loss. A 10-session gap and explicit label-maturity checks ensure that training outcomes end before validation/test dates begin. Scaling is fitted inside each logistic-regression training fold.

The final 10 sessions have unknown direction labels and are excluded from training. Indicators are computed using past and current completed sessions only, with no backward or forward filling. Targets are shifted on the original trading-session index before removing invalid indicator rows.

Each selected classifier is then refitted on all mature labels for **latest-session inference only**. Historical holdout predictions come from its separate pre-holdout fit. The app never evaluates past dates with the later refit.

The Chronos comparison starts **3 November 2025**, after both pinned base-model revisions and the ETF adaptation cutoff. Every forecast sees only its own historical window, and cross-learning across different as-of windows is disabled. The report includes daily scores and a subset with non-overlapping 10-session outcomes. Neither is an untouched future test after this research run.

## Accuracy of the included run

The following are the **validation-selected defaults** on each ETF's held-out period, roughly December 2022–6 November 2026:

| ETF | Direction accuracy | Balanced accuracy | Selected default |
| --- | ---: | ---: | --- |
| SPY | 64.0% | 50.0% | Prior baseline |
| QQQ | 62.5% | 50.0% | Prior baseline |
| IWM | 56.4% | 50.0% | Prior baseline |
| VEU | 57.4% | 50.0% | Prior baseline |
| AGG | 54.2% | 50.0% | Prior baseline |
| TLT | 47.3% | 50.0% | Prior baseline |
| GLD | 58.8% | 50.0% | Prior baseline |

All selected defaults currently predict UP because their training priors exceed 50%. Their accuracies reflect the holdout's UP frequency; their 50% balanced accuracies demonstrate no direction discrimination. The more complex candidates did not consistently beat these baselines on the earlier validation folds. The app states **“No validated timing signal”** for these defaults.

### Fine-tuned neural model: separate later-period test

The Chronos-2 adapter was evaluated on **255 daily as-of dates per ETF, 3 November 2025–6 November 2026**. This period differs from the classifier holdout above. Compare these results only with baselines from the same period.

| ETF | Fine-tuned accuracy | Balanced accuracy | Always-UP accuracy |
| --- | ---: | ---: | ---: |
| SPY | 45.1% | 44.1% | 52.5% |
| QQQ | 47.8% | 48.0% | 49.0% |
| IWM | 45.9% | 45.5% | 51.8% |
| VEU | 52.9% | 51.6% | 53.3% |
| AGG | 51.4% | 50.6% | 51.4% |
| TLT | 47.5% | 51.1% | 40.8% |
| GLD | 56.5% | 56.8% | 49.0% |

The daily target windows overlap. The report also provides **26 non-overlapping windows per ETF**; this is a small research sample. Fine-tuning is not assumed to improve every asset, and these results do not establish a dependable trading advantage.

See `data/training_results.json` for every candidate's fold and holdout scores, and `data/foundation_results.json` for the neural comparison, interval coverage, model revisions, and adaptation recipe. Log loss and Brier score assess probability quality; lower is better. These runs do not establish investment profitability.

## Compact project layout

```text
app.py                 Dashboard
src/                   Download, feature, training, and neural-model code
tests/                 Correctness and dashboard checks
data/                  One shared market snapshot, saved predictions, evaluation JSON
models/                Per-ETF model files, LoRA adapter, and training metadata
requirements.txt       Dashboard and classical-model dependencies
requirements-hf.txt    Optional pretrained-model and fine-tuning dependencies
```

Indicators are regenerated from the shared market snapshot rather than stored as duplicate per-ETF feature datasets. Temporary training folders, copied legacy notebooks, superseded models/data, and generated artifact directories have been removed. Earlier project versions remain available in Git history. Automated checks run in GitHub Actions.

## Limits and next useful experiments

- Overlapping daily 10-session targets are dependent. Hundreds of daily predictions are not hundreds of independent trials. Non-overlapping scores also have a limited sample size; no significance or superiority claim is made.
- This is an after-close research system using the completed session's prices and volumes. It is not an intraday execution strategy.
- Yahoo adjusted history can be revised; the saved data hashes identify the snapshot but do not make it a full point-in-time institutional dataset. Pretraining corpora also cannot be exhaustively audited.
- Indicator descriptions are context, not explanations of a particular neural prediction. Personal investment amount and risk tolerance are not used to produce allocations.
- Additional macroeconomic data should use actual release dates and historical vintages. News sentiment needs timestamped historical articles and an explicitly evaluated connection to future returns; plugging in a sentiment model alone would not establish that connection.
- Future changes need a new forward evaluation period. A trading study also needs executable entry prices, holding/rebalancing rules, transaction costs, and uncertainty estimates.

References: [Chronos models and release history](https://github.com/amazon-science/chronos-forecasting), [Chronos-2 model card](https://huggingface.co/amazon/chronos-2), [scikit-learn chronological splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [yfinance download behavior](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

**Mario Rinaldi · BSc Computer Science final-year project.** For academic and educational research; this system does not provide personalized financial advice.
