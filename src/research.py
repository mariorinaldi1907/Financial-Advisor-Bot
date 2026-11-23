"""Pooled, calibrated, multi-horizon research with date-purged walk-forward tests."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .context import ASSETS, HORIZONS, build_panel, input_hashes
from .data import ROOT
from .train import scores, PriorModel

CONFIG = dict(seed=42, horizons=HORIZONS, training_years=5, calibration_sessions=126, embargo_sessions=20,
              cv_periods=[['2018-01-01', '2020-01-01'], ['2020-01-01', '2022-01-01'], ['2022-01-01', '2024-01-01']],
              evaluation_periods=[['2024-01-01', '2025-01-01'], ['2025-01-01', '2026-01-01'], ['2026-01-01', '2027-01-01']],
              model_selection='equal-weight asset-mean validation log loss, averaged over three temporal folds',
              environment_lag_sessions=2, trade_probability=.6, one_way_cost_bps=5,
              bootstrap_blocks=20, bootstrap_draws=400,
              catboost=dict(iterations=240, depth=5, learning_rate=.03, l2_leaf_reg=20),
              hist=dict(max_iter=160, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=20))
FAMILIES = ['prior', 'logistic_market', 'cat_market', 'cat_context', 'hist_context', 'ensemble', 'shrink_ensemble']


def labeled_panel(panel, horizon):
    result = panel.copy()
    result['Forward_return'] = result[f'Forward_return_{horizon}']
    result['Label_end'] = result[f'Label_end_{horizon}']
    result['Target'] = (result.Forward_return > 0).astype(float).where(result.Forward_return.notna())
    result['Scale'] = result[f'Scale_{horizon}']
    result['Z_return'] = result.Forward_return / result.Scale
    return result


def temporal_parts(data, boundary, excluded_dates=()):
    boundary = pd.Timestamp(boundary)
    known = data.loc[(data.Label_end < boundary) & data.Target.notna() &
                     (data.index >= boundary - pd.DateOffset(years=CONFIG['training_years']))]
    known = known.loc[~known.index.isin(excluded_dates)]
    dates = known.index.unique().sort_values()
    if len(dates) < 600:
        raise ValueError('Too few historical sessions for separate fitting and calibration.')
    calibration_start = dates[-CONFIG['calibration_sessions']]
    fit = known.loc[known.Label_end < calibration_start]
    calibration = known.loc[known.index >= calibration_start]
    if fit.Label_end.max() >= calibration.index.min() or calibration.Label_end.max() >= boundary:
        raise ValueError('Overlapping training/calibration/evaluation outcomes.')
    return fit, calibration, known


def validation_slice(data, start, end):
    return data.loc[(data.index >= start) & (data.index < end) & (data.Label_end < end) & data.Target.notna()]


def walk_forward_folds(data, periods, excluded_dates=(), next_allowed=None):
    """Keep all assets on the same date together; embargo follows the last outcome."""
    sessions = data.index.unique().sort_values()
    excluded = pd.DatetimeIndex(excluded_dates)
    for nominal_start, end in periods:
        start = pd.Timestamp(nominal_start)
        if next_allowed is not None:
            start = max(start, pd.Timestamp(next_allowed))
        validation = validation_slice(data, start, end)
        if validation.empty:
            raise ValueError(f'Empty evaluation period: {start} to {end}')
        outcome_end = validation.Label_end.max()
        following = sessions[sessions > outcome_end][:CONFIG['embargo_sessions']]
        if len(following):
            position = sessions.searchsorted(following[-1], side='right')
            next_allowed = sessions[position] if position < len(sessions) else following[-1] + pd.Timedelta(days=1)
        else:
            next_allowed = outcome_end + pd.Timedelta(days=1)
        embargo = dict(start=str(following[0].date()) if len(following) else None,
                       end=str(following[-1].date()) if len(following) else None,
                       observed_sessions=len(following), requested_sessions=CONFIG['embargo_sessions'])
        yield validation, start, end, excluded, embargo, following, next_allowed
        excluded = excluded.union(following)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1-p)).reshape(-1, 1)


def expected_calibration_error(y, p):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * 10).astype(int), 9)
    return float(sum(np.mean(bins == i) * abs(y[bins == i].mean() - p[bins == i].mean())
                     for i in range(10) if (bins == i).any()))


def metrics_by_asset(frame, probability):
    output = {}
    for ticker in ASSETS:
        mask = frame.Ticker.to_numpy() == ticker
        if not mask.any():
            continue
        y, p = frame.Target.to_numpy()[mask], np.asarray(probability)[mask]
        output[ticker] = dict(rows=int(mask.sum()), ece=expected_calibration_error(y, p), **scores(y, p))
    return output


def mean_asset_loss(frame, probability):
    return float(np.mean([m['log_loss'] for m in metrics_by_asset(frame, probability).values()]))


def fit_classifiers(data, boundary, market_cols, all_cols, excluded_dates=()):
    fit, calibration, known = temporal_parts(data, boundary, excluded_dates)
    prior = PriorModel().fit(known[market_cols], known.Target.astype(int))
    models = {'prior': dict(model=prior, columns=market_cols, calibrator=None)}
    specifications = {
        'logistic_market': (make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                                         LogisticRegression(C=.01, max_iter=2000)), market_cols),
        'cat_market': (CatBoostClassifier(**CONFIG['catboost'], loss_function='Logloss',
                        random_seed=42, verbose=False, allow_writing_files=False, thread_count=4), market_cols),
        'cat_context': (CatBoostClassifier(**CONFIG['catboost'], loss_function='Logloss',
                        random_seed=42, verbose=False, allow_writing_files=False, thread_count=4), all_cols),
        # Random internal validation is explicitly disabled for the time-ordered task.
        'hist_context': (HistGradientBoostingClassifier(**CONFIG['hist'], learning_rate=.04,
                                                        early_stopping=False, random_state=42), all_cols),
    }
    calibration_scores = {}
    for name, (model, columns) in specifications.items():
        model.fit(fit[columns], fit.Target.astype(int))
        p = model.predict_proba(calibration[columns])[:, 1]
        calibrator = LogisticRegression(C=.1, max_iter=1000).fit(logit(p), calibration.Target.astype(int))
        corrected = calibrator.predict_proba(logit(p))[:, 1]
        calibration_scores[name] = dict(raw_brier=float(np.mean((p-calibration.Target.to_numpy())**2)),
            fitted_brier=float(np.mean((corrected-calibration.Target.to_numpy())**2)),
            note='Calibration-fit diagnostics, not independent test scores.')
        models[name] = dict(model=model, columns=columns, calibrator=calibrator)
    trace = dict(fit_start=str(fit.index.min().date()), fit_end=str(fit.index.max().date()),
                 fit_label_end=str(fit.Label_end.max().date()),
                 calibration_start=str(calibration.index.min().date()),
                 calibration_end=str(calibration.index.max().date()),
                 calibration_label_end=str(calibration.Label_end.max().date()),
                 evaluation_start=str(pd.Timestamp(boundary).date()), fit_rows=len(fit), calibration_rows=len(calibration),
                 calibration_fit_diagnostics=calibration_scores)
    return models, trace, fit, calibration


def probabilities(models, frame, raw=False):
    output = {}
    for name, bundle in models.items():
        p = bundle['model'].predict_proba(frame[bundle['columns']])[:, 1]
        if bundle['calibrator'] is not None and not raw:
            p = bundle['calibrator'].predict_proba(logit(p))[:, 1]
        output[name] = np.clip(p, 1e-6, 1-1e-6)
    output['ensemble'] = np.mean([output[n] for n in ['logistic_market', 'cat_context', 'hist_context']], axis=0)
    output['shrink_ensemble'] = .5 * output['ensemble'] + .5 * output['prior']
    return output


def conformal_width(errors, coverage=.8):
    values = np.sort(np.asarray(errors)[np.isfinite(errors)])
    if len(values) < 20:
        raise ValueError('At least 20 calibration observations are required.')
    k = min(len(values), int(np.ceil((len(values) + 1) * coverage)))
    return float(values[k-1])


def fit_returns(fit, calibration, all_cols):
    mean_model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), Ridge(alpha=100.))
    mean_model.fit(fit[all_cols], fit.Z_return.clip(-10, 10))
    quantile_model = CatBoostRegressor(iterations=200, depth=5, learning_rate=.035, l2_leaf_reg=20,
        loss_function='MultiQuantile:alpha=0.1,0.5,0.9', random_seed=42, verbose=False,
        allow_writing_files=False, thread_count=4)
    quantile_model.fit(fit[all_cols], fit.Z_return.clip(-10, 10))
    q = np.sort(quantile_model.predict(calibration[all_cols]), axis=1)
    error = np.maximum(np.maximum(q[:, 0] - calibration.Z_return, calibration.Z_return - q[:, 2]), 0)
    widths = {ticker: conformal_width(error[calibration.Ticker == ticker]) for ticker in ASSETS}
    return dict(mean=mean_model, quantiles=quantile_model, columns=all_cols, widths=widths)


def return_predictions(bundle, frame):
    columns = bundle['columns']
    scale = frame.Scale.to_numpy()
    point = bundle['mean'].predict(frame[columns]) * scale
    q = np.sort(bundle['quantiles'].predict(frame[columns]), axis=1)
    width = frame.Ticker.map(bundle['widths']).to_numpy()
    q[:, 0] -= width
    q[:, 2] += width
    return point, q * scale[:, None]


def block_bootstrap_difference(records, horizon):
    eps = 1e-8
    y = records.Target.to_numpy()
    def loss(p):
        p = np.clip(p, eps, 1-eps)
        return -(y*np.log(p) + (1-y)*np.log(1-p))
    delta = pd.Series(loss(records.p_up.to_numpy()) - loss(records.prior_p_up.to_numpy()), index=records.index)
    daily = delta.groupby(level=0).mean().to_numpy()
    rng = np.random.default_rng(42)
    block = max(horizon, CONFIG['bootstrap_blocks'])
    draws = []
    for _ in range(CONFIG['bootstrap_draws']):
        starts = rng.integers(0, max(1, len(daily)-block+1), size=int(np.ceil(len(daily)/block)))
        sample = np.concatenate([daily[s:s+block] for s in starts])[:len(daily)]
        draws.append(sample.mean())
    return dict(mean_log_loss_difference=float(daily.mean()),
                bootstrap_95_interval=[float(x) for x in np.quantile(draws, [.025, .975])],
                block_sessions=block, draws=len(draws), unit='dates, with all assets retained together',
                note='Exploratory dependent-data interval; not corrected for repeated research or multiple horizons.')


def execution_study(frame, horizon):
    results = {}
    cost = CONFIG['one_way_cost_bps'] / 10000
    for ticker in ASSETS:
        rows = frame.loc[frame.Ticker == ticker].sort_index().dropna(subset=['entry_open', 'exit_open']).iloc[::horizon]
        if rows.empty:
            continue
        gross = rows.exit_open.to_numpy() / rows.entry_open.to_numpy() - 1
        signal = ((rows.p_up >= CONFIG['trade_probability']) &
                  (rows.expected_return > 2*cost)).astype(float).to_numpy()
        # Every sampled holding window is a separate round trip; gaps are cash.
        strategy = signal * (gross - 2*cost)
        market = gross - 2*cost
        def summarize(returns):
            wealth = np.cumprod(1+returns)
            drawdown = wealth / np.maximum.accumulate(np.r_[1., wealth])[1:] - 1
            return dict(total_return=float(wealth[-1]-1),
                        holding_period_endpoint_drawdown=float(drawdown.min()))
        results[ticker] = dict(periods=len(rows), invested_fraction=float(signal.mean()),
                              one_way_cost_bps=CONFIG['one_way_cost_bps'], strategy=summarize(strategy),
                              buy_hold=summarize(market), cash_interest_assumption=0.)
    return results


def evaluate_horizon(panel, market_cols, all_cols, horizon):
    data = labeled_panel(panel, horizon)
    validation = {name: [] for name in FAMILIES}
    traces, fold_metrics = [], []
    excluded, next_allowed = pd.DatetimeIndex([]), None
    for validation_frame, start, end, excluded, embargo, following, next_allowed in walk_forward_folds(data, CONFIG['cv_periods']):
        models, trace, _, _ = fit_classifiers(data, start, market_cols, all_cols, excluded)
        p = probabilities(models, validation_frame)
        for name in FAMILIES:
            validation[name].append(mean_asset_loss(validation_frame, p[name]))
        trace.update(validation_end=str(validation_frame.index.max().date()), validation_last_label_end=str(validation_frame.Label_end.max().date()), period_end=end, post_validation_embargo=embargo)
        fold_metrics.append(dict(start=str(start.date()), end=end, models={n:metrics_by_asset(validation_frame,p[n]) for n in FAMILIES}))
        traces.append(trace)
        print(f'h={horizon}: validation {start}: ' + str({n:round(validation[n][-1],4) for n in FAMILIES}), flush=True)
    means = {name:float(np.mean(values)) for name,values in validation.items()}
    selected = min(means, key=means.get)
    excluded = excluded.union(following)
    records, evaluation_traces, period_metrics = [], [], []
    comparisons = {name: [] for name in FAMILIES}
    for frame, start, end, excluded, embargo, following, next_allowed in walk_forward_folds(data, CONFIG['evaluation_periods'], excluded, next_allowed):
        models, trace, fit, calibration = fit_classifiers(data, start, market_cols, all_cols, excluded)
        risk = fit_returns(fit, calibration, all_cols)
        p, raw = probabilities(models, frame), probabilities(models, frame, raw=True)
        point, q = return_predictions(risk, frame)
        record = frame[['Ticker', 'Target', 'Forward_return', 'Label_end']].copy()
        record['horizon'], record['model'] = horizon, selected
        record['p_up'], record['raw_p_up'], record['prior_p_up'] = p[selected], raw[selected], p['prior']
        record['expected_return'], record['lower_return'], record['median_return'], record['upper_return'] = point, q[:,0], q[:,1], q[:,2]
        record['entry_open'], record['exit_open'] = frame[f'Entry_open_{horizon}'], frame[f'Exit_open_{horizon}']
        record['training_label_end'] = trace['calibration_label_end']
        records.append(record)
        for name in FAMILIES:
            comparisons[name].append(pd.DataFrame(dict(Ticker=frame.Ticker, Target=frame.Target, probability=p[name]), index=frame.index))
        trace.update(validation_end=str(frame.index.max().date()), validation_last_label_end=str(frame.Label_end.max().date()), period_end=end, post_validation_embargo=embargo)
        period_metrics.append(dict(start=str(start.date()), end=end, model=selected,
            models={n:metrics_by_asset(frame,p[n]) for n in FAMILIES},
            mean_asset_log_loss={n:mean_asset_loss(frame,p[n]) for n in FAMILIES}))
        evaluation_traces.append(trace)
        print(f'h={horizon}: evaluated {start}; selected={selected}', flush=True)
    records = pd.concat(records).sort_index(kind='stable')
    metrics = metrics_by_asset(records, records.p_up)
    baseline = metrics_by_asset(records, records.prior_p_up)
    for ticker,m in metrics.items():
        rows = records.loc[records.Ticker == ticker]
        m['baseline_accuracy'] = baseline[ticker]['accuracy']
        m['baseline_log_loss'] = baseline[ticker]['log_loss']
        m['always_up_accuracy'] = float(rows.Target.mean())
        m['raw_brier'] = float(np.mean((rows.raw_p_up - rows.Target)**2))
        m['expected_return_mae'] = float(np.mean(abs(rows.expected_return - rows.Forward_return)))
        m['zero_return_mae'] = float(np.mean(abs(rows.Forward_return)))
        m['interval_80_coverage'] = float(((rows.Forward_return >= rows.lower_return) & (rows.Forward_return <= rows.upper_return)).mean())
        independent = rows.iloc[::horizon]
        m['nonoverlapping_rows'] = len(independent)
        m['nonoverlapping_accuracy'] = float(((independent.p_up >= .5) == independent.Target).mean())
        m['interval_mean_width'] = float((rows.upper_return - rows.lower_return).mean())
    all_comparisons = {}
    for name,blocks in comparisons.items():
        f = pd.concat(blocks)
        all_comparisons[name] = dict(mean_asset_log_loss=mean_asset_loss(f, f.probability), by_asset=metrics_by_asset(f, f.probability))
    report = dict(horizon=horizon, selected_model=selected, validation_log_loss=means,
                  validation_folds=validation, validation_traces=traces, validation_period_metrics=fold_metrics,
                  evaluation_traces=evaluation_traces, evaluation_period_metrics=period_metrics,
                  test_start=str(records.index.min().date()), test_end=str(records.index.max().date()),
                  metrics=metrics, candidates=all_comparisons,
                  bootstrap=block_bootstrap_difference(records, horizon), execution=execution_study(records, horizon))
    # Refit with all outcomes actually known at the latest completed close.
    as_of = panel.index.max()
    excluded = excluded.union(following)
    models, live_trace, fit, calibration = fit_classifiers(data, as_of + pd.Timedelta(days=1), market_cols, all_cols, excluded)
    risk = fit_returns(fit, calibration, all_cols)
    latest = data.loc[data.index == as_of]
    p = probabilities(models, latest)
    point, q = return_predictions(risk, latest)
    latest_records = []
    for i,(_, row) in enumerate(latest.iterrows()):
        latest_records.append(dict(ticker=row.Ticker, horizon=horizon, as_of=str(as_of.date()),
            model=selected, p_up=float(p[selected][i]), expected_return=float(point[i]),
            lower_return=float(q[i,0]), median_return=float(q[i,1]), upper_return=float(q[i,2])))
    report['latest'], report['live_trace'] = latest_records, live_trace
    bundle = dict(classifiers=models, returns=risk, selected_model=selected, horizon=horizon,
                  hashes=input_hashes(), features=all_cols, excluded_dates=excluded, as_of=str(as_of.date()), live_trace=live_trace)
    joblib.dump(bundle, ROOT / 'models' / f'horizon_{horizon}.joblib', compress=3)
    return report, records


def append_forecast_ledger(latest, path=None):
    path = Path(path) if path is not None else ROOT / 'data/forecast_ledger.csv'
    current = pd.DataFrame(latest)
    current['issued_utc'] = datetime.now(timezone.utc).isoformat()
    current['market_sha256'] = input_hashes()['market']
    if path.exists() and path.stat().st_size:
        old = pd.read_csv(path)
        # Existing forecasts are immutable: retries do not rewrite their probabilities.
        keys = ['ticker', 'horizon', 'as_of']
        seen = set(map(tuple, old[keys].to_numpy()))
        current = current.loc[[tuple(row) not in seen for row in current[keys].to_numpy()]]
        current = pd.concat([old, current], ignore_index=True)
    current.to_csv(path, index=False)


def run(horizons=HORIZONS):
    panel, market_cols, all_cols = build_panel()
    results, predictions = {}, []
    # Preserve completed horizons on a resumed run, but never reuse other input snapshots.
    path = ROOT / 'data/research_results.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['input_hashes'] == input_hashes() and old.get('config') == CONFIG:
            results = old['horizons']
    with threadpool_limits(limits=4):
        for horizon in horizons:
            report, record = evaluate_horizon(panel, market_cols, all_cols, horizon)
            results[str(horizon)] = report
            predictions.append(record)
            prediction_path = ROOT / 'data/research_predictions.csv'
            if prediction_path.exists():
                previous = pd.read_csv(prediction_path, index_col=0, parse_dates=True)
                previous = previous.loc[previous.horizon != horizon]
                saved = pd.concat([previous, record])
            else:
                saved = record
            saved.sort_values(['horizon', 'Ticker'], kind='stable').to_csv(prediction_path, index_label='Date')
            output = dict(schema_version=2, created_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
                          assets={t:dict(name=n, group=g) for t,(n,g) in ASSETS.items()},
                          input_hashes=input_hashes(), feature_counts=dict(market=len(market_cols), with_environment=len(all_cols)),
                          environment_source='Yahoo VIX and Treasury yield indices, as reported; delayed two market sessions.',
                          environment_vintage_policy='Vendor snapshots, not original historical release vintages.',
                          horizons=results, caveats=[
                              'All results are exploratory retrospective research; previous experiments inspected overlapping dates.',
                              'Asset rows are correlated and multi-session target windows overlap.',
                              'Calibration and conformal adjustments use earlier dates; nominal coverage is not guaranteed under market dependence.',
                              'Execution assumes separate holding-window round trips at next adjusted open, 5 bps each way, cash gaps earn zero, no market impact; drawdown is measured only at holding-period endpoints.',
                              'No paid economic-release vintages, historical news archive, live brokerage, or paid cloud service is included.'])
            path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
            append_forecast_ledger(report['latest'])
            print(f'Completed horizon {horizon}: {report["selected_model"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizons', nargs='+', type=int, choices=HORIZONS, default=HORIZONS)
    args = parser.parse_args()
    run(args.horizons)
