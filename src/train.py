"""Run with python -m src.train. Selection uses development folds only."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import MARKET_PATH, load_market, raw_digest, snapshot_digest
from .features import HORIZON, LEGACY_FEATURES, CORE_FEATURES, EXTRA_FEATURES, build_features, add_target, build_market_context

ROOT = Path(__file__).resolve().parents[1]


def candidates(context_features=None):
    registry = {
        'prior_baseline': (DummyClassifier(strategy='prior'), CORE_FEATURES),
        'original_rf': (RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1), LEGACY_FEATURES),
        'normalized_rf': (RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=25, max_features=0.7, random_state=42, n_jobs=-1), CORE_FEATURES),
        'expanded_rf': (RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=25, max_features=0.7, random_state=42, n_jobs=-1), CORE_FEATURES + EXTRA_FEATURES),
        'logistic': (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), CORE_FEATURES),
        'expanded_logistic': (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), CORE_FEATURES + EXTRA_FEATURES),
    }
    if context_features:
        cols = CORE_FEATURES + EXTRA_FEATURES + context_features
        registry['contextual_rf'] = (RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=40, max_features=0.7, random_state=42, n_jobs=-1), cols)
        for strength in [0.01, 0.1]:
            registry[f'contextual_logistic_{strength}'] = (make_pipeline(StandardScaler(), LogisticRegression(C=strength, max_iter=3000)), cols)
    return registry


def scores(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                brier=float(brier_score_loss(y, p)),
                roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def purged_before(frame, boundary):
    """Even the last training label must mature strictly before evaluation."""
    return frame.loc[frame.Label_end < boundary]


def train(ticker='SPY', output=ROOT, prices_path=MARKET_PATH):
    raw, markets = load_market(ticker, prices_path)
    all_features = build_features(raw)
    context = build_market_context(markets, exclude=ticker)
    context_cols = list(context.columns)
    all_features = all_features.join(context, how='left')
    features = all_features.dropna(subset=LEGACY_FEATURES + CORE_FEATURES + EXTRA_FEATURES + context_cols)
    # Shift on the original trading-session index before removing invalid feature rows.
    labeled = add_target(all_features).loc[features.index].dropna(subset=['Target'])
    if len(labeled) < 800:
        raise ValueError('At least 800 labeled trading sessions are required.')
    split = int(len(labeled) * .8)
    test = labeled.iloc[split:]
    dev = purged_before(labeled.iloc[:split], test.index[0])
    registry = candidates(context_cols)
    results = {}
    for name, (estimator, cols) in registry.items():
        folds = []
        for tr, va in TimeSeriesSplit(n_splits=4, gap=HORIZON).split(dev):
            train_fold, val_fold = dev.iloc[tr], dev.iloc[va]
            assert train_fold.Label_end.max() < val_fold.index.min()
            fitted = clone(estimator).fit(train_fold[cols], train_fold.Target.astype(int))
            p = fitted.predict_proba(val_fold[cols])[:, 1]
            folds.append(dict(train_end=str(train_fold.index.max().date()),
                              label_end=str(train_fold.Label_end.max().date()),
                              validation_start=str(val_fold.index.min().date()),
                              validation_end=str(val_fold.index.max().date()),
                              **scores(val_fold.Target, p)))
        results[name] = dict(mean_log_loss=float(np.mean([f['log_loss'] for f in folds])), folds=folds)
    winner = min(results, key=lambda n: results[n]['mean_log_loss'])
    holdout = {}
    predictions = test[['Close', 'Target', 'Label_end']].copy()
    for name, (estimator, cols) in registry.items():
        fitted = clone(estimator).fit(dev[cols], dev.Target.astype(int))
        p = fitted.predict_proba(test[cols])[:, 1]
        holdout[name] = scores(test.Target, p)
        predictions[f'{name}_p_up'] = p
    # Refit the preselected model on all mature labels for latest-date inference.
    estimator, cols = registry[winner]
    deployed = clone(estimator).fit(labeled[cols], labeled.Target.astype(int))
    metadata = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    ticker=ticker, horizon=HORIZON, selected_model=winner, features=cols,
                    selection_metric='mean development-fold log loss (lower is better)',
                    sklearn_version=sklearn.__version__, raw_sha256=raw_digest(raw), snapshot_sha256=snapshot_digest(prices_path),
                    raw_start=str(raw.index.min().date()), raw_end=str(raw.index.max().date()),
                    raw_rows=len(raw), labeled_rows=len(labeled),
                    context_sha256=snapshot_digest(prices_path),
                    context_features=context_cols,
                    development_end=str(dev.index.max().date()),
                    development_label_end=str(dev.Label_end.max().date()),
                    holdout_start=str(test.index.min().date()), holdout_end=str(test.index.max().date()),
                    live_train_end=str(labeled.index.max().date()),
                    live_label_end=str(labeled.Label_end.max().date()),
                    validation=results, holdout=holdout,
                    always_up_holdout=scores(test.Target, np.ones(len(test)) * (1 - 1e-7)),
                    caveats=['Overlapping 10-day outcomes are dependent; row counts are not independent trials.',
                             'Holdout is a retrospective comparison; no trading-profit claim is made.',
                             'Probabilities are uncalibrated model estimates.',
                             'Latest-date refit must not be used to evaluate historical dates.'])
    (output / 'models').mkdir(parents=True, exist_ok=True)
    (output / 'data').mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=deployed, metadata=metadata), output / 'models' / f'{ticker.lower()}_forecast.joblib', compress=3)
    predictions.to_csv(output / 'data' / f'{ticker.lower()}_predictions.csv', index_label='Date')
    print(json.dumps(dict(ticker=ticker, selected=winner, validation_log_loss={n: r['mean_log_loss'] for n,r in results.items()}, holdout=holdout), indent=2))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=MARKET_PATH)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--ticker', default='SPY')
    args = parser.parse_args()
    metadata = train(args.ticker, args.output, args.prices)
    results_path = args.output / 'data/training_results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results[args.ticker] = metadata
    results_path.write_text(json.dumps(results, indent=2) + '\n')
