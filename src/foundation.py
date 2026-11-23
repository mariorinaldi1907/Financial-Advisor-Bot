"""Optional frozen Hugging Face benchmark; never silently replaces trained models.

Run with the optional environment: python -m src.foundation
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_market, raw_digest, snapshot_digest
from .features import build_features, add_target, HORIZON
from .train import ROOT, scores
from .universe import TICKERS

QUANTILES = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
CONTEXT_LENGTH = 256
# Both pinned revisions predate this date. No hyperparameter search is performed.
BENCHMARK_START = '2025-11-03'
COVARIATES = ['RSI_14', 'Mom_10', 'Volatility_20', 'Relative_volume', 'SMA_gap_30']


def probability_up(quantiles, current):
    """Approximate P(price > current) from sorted deciles; tails capped at .1/.9."""
    values = np.sort(np.asarray(quantiles, dtype=float))
    if values.shape != (9,) or not np.isfinite(values).all() or not np.isfinite(current):
        raise ValueError('Expected nine finite forecast deciles and a finite current price.')
    return float(1 - np.interp(current, values, QUANTILES))


def context_at(frame, date, covariates=False):
    """Slice each example independently; later benchmark windows never enter it."""
    history = frame.loc[:date].tail(CONTEXT_LENGTH)
    if len(history) != CONTEXT_LENGTH:
        raise ValueError('Insufficient historical context.')
    target = history.Close.to_numpy(dtype=np.float32)
    if not covariates:
        return target
    cols = COVARIATES + [c for c in frame if c.startswith('Market_')]
    if not np.isfinite(history[cols].to_numpy()).all():
        raise ValueError('Missing historical covariates; no future filling is allowed.')
    return {'target': target, 'past_covariates': {c: history[c].to_numpy(dtype=np.float32) for c in cols}}


def benchmark(tickers, device='cpu'):
    os.environ.setdefault('HF_HOME', str(Path(tempfile.gettempdir()) / 'financial-advisor-hf'))
    import torch
    from chronos import BaseChronosPipeline
    from huggingface_hub import snapshot_download
    torch.set_num_threads(4)
    registry = json.loads((ROOT / 'src/foundation_models.json').read_text())
    _, markets = load_market(tickers[0])
    all_frames, all_dates, source_hashes = {}, {}, {}
    records, latest, runs = [], {}, {}
    for ticker in tickers:
        raw, _ = load_market(ticker)
        frame = add_target(build_features(raw))
        for symbol in markets:
            frame[f'Market_{symbol}'] = markets[symbol].reindex(frame.index)
        frame = frame.dropna(subset=COVARIATES + [c for c in frame if c.startswith('Market_')])
        valid = frame.loc[frame.index >= BENCHMARK_START].dropna(subset=['Target'])
        dates = list(valid.index)  # all daily as-of dates; dependence is reported explicitly
        all_frames[ticker], all_dates[ticker] = frame, dates
        source_hashes[ticker] = raw_digest(raw)
        latest[ticker] = {}
    configs = [('chronos_bolt_tiny', 'amazon/chronos-bolt-tiny', False),
               ('chronos2_price', 'amazon/chronos-2', False),
               ('chronos2_context', 'amazon/chronos-2', True)]
    adapter_path = ROOT / 'models'
    if (adapter_path / 'chronos2_adapter_config.json').exists():
        configs.append(('chronos2_finetuned_context', 'amazon/chronos-2', True))
    loaded_id, pipeline = None, None
    for name, model_id, use_covariates in configs:
        start = time.monotonic()
        if loaded_id != model_id:
            path = snapshot_download(model_id, revision=registry[model_id]['revision'],
                                     allow_patterns=['config.json', 'model.safetensors'], local_files_only=True)
            pipeline = BaseChronosPipeline.from_pretrained(path, device_map=device,
                         torch_dtype=torch.float32, trust_remote_code=False, use_safetensors=True)
            loaded_id = model_id
        if name == 'chronos2_finetuned_context':
            from peft import PeftModel
            from chronos import Chronos2Pipeline
            with tempfile.TemporaryDirectory(prefix='etf-adapter-') as temporary:
                shutil.copy2(adapter_path / 'chronos2_adapter_config.json', Path(temporary) / 'adapter_config.json')
                shutil.copy2(adapter_path / 'chronos2_adapter.safetensors', Path(temporary) / 'adapter_model.safetensors')
                pipeline = Chronos2Pipeline(model=PeftModel.from_pretrained(pipeline.model, temporary, is_trainable=False))
        for ticker in tickers:
            frame, dates = all_frames[ticker], all_dates[ticker]
            requested = dates + [frame.index[-1]]
            inputs = [context_at(frame, date, use_covariates) for date in requested]
            outputs = []
            # Cross-learning must stay disabled: different as-of windows overlap in time.
            for offset in range(0, len(inputs), 8):
                batch = inputs[offset:offset + 8]
                if model_id.endswith('bolt-tiny'):
                    tensor = torch.tensor(np.stack(batch), dtype=torch.float32)
                    q, _ = pipeline.predict_quantiles(tensor, prediction_length=HORIZON,
                                                       quantile_levels=QUANTILES.tolist())
                    outputs.extend(q.numpy())
                else:
                    q, _ = pipeline.predict_quantiles(batch, prediction_length=HORIZON,
                        quantile_levels=QUANTILES.tolist(), context_length=CONTEXT_LENGTH,
                        batch_size=64, cross_learning=False)
                    outputs.extend(a[0].numpy() for a in q)
            for date, forecast in zip(requested, outputs):
                forecast = np.sort(forecast, axis=1)
                p = probability_up(forecast[-1], frame.loc[date, 'Close'])
                if date == frame.index[-1]:
                    latest[ticker][name] = dict(as_of=str(date.date()), close=float(frame.loc[date, 'Close']),
                        p_up=p, lower=forecast[:, 0].tolist(), median=forecast[:, 4].tolist(),
                        upper=forecast[:, 8].tolist())
                else:
                    records.append(dict(ticker=ticker, model=name, Date=str(date.date()),
                        Target=int(frame.loc[date, 'Target']), p_up=p,
                        close=float(frame.loc[date, 'Close']), median_10=float(forecast[-1, 4]),
                        lower_10=float(forecast[-1, 0]), upper_10=float(forecast[-1, 8]),
                        actual_10=float(frame.loc[date, 'Future_Close'])))
            print(f'{name}: {ticker}: {len(dates)} historical windows + latest forecast', flush=True)
        runs[name] = dict(model_id=model_id, **registry[model_id], past_covariates=use_covariates,
                          seconds=time.monotonic() - start)
    predictions = pd.DataFrame(records)
    metrics, nonoverlapping_metrics = {}, {}
    for ticker in tickers:
        subset = predictions[predictions.ticker == ticker]
        metrics[ticker], nonoverlapping_metrics[ticker] = {}, {}
        for name, group in subset.groupby('model'):
            result = scores(group.Target, group.p_up.to_numpy())
            result['windows'] = len(group)
            result['interval_80_coverage'] = float(((group.actual_10 >= group.lower_10) & (group.actual_10 <= group.upper_10)).mean())
            result['median_forecast_mape'] = float((abs(group.median_10 - group.actual_10) / group.actual_10).mean())
            metrics[ticker][name] = result
            independent = group.iloc[::HORIZON]
            nonoverlapping_metrics[ticker][name] = dict(windows=len(independent), **scores(independent.Target, independent.p_up.to_numpy()))
        saved = pd.read_csv(ROOT / 'data' / f'{ticker.lower()}_predictions.csv', index_col=0, parse_dates=True)
        meta = json.loads((ROOT / 'data/training_results.json').read_text())[ticker]
        common = saved.loc[pd.DatetimeIndex(all_dates[ticker])]
        for name in [meta['selected_model'], 'prior_baseline']:
            metrics[ticker][f'trained_{name}'] = dict(windows=len(common), **scores(common.Target, common[f'{name}_p_up'].to_numpy()))
        metrics[ticker]['always_up'] = dict(windows=len(common), **scores(common.Target, np.full(len(common), 1 - 1e-7)))
    if (adapter_path / 'chronos2_training.json').exists():
        runs['chronos2_finetuned_context']['training'] = json.loads((adapter_path / 'chronos2_training.json').read_text())
    report = dict(benchmark_start=BENCHMARK_START, stride_sessions=1, context_length=CONTEXT_LENGTH,
                  horizon=HORIZON, source_hashes=source_hashes,
                  adapter_sha256=hashlib.sha256((ROOT / 'models/chronos2_adapter.safetensors').read_bytes()).hexdigest() if (ROOT / 'models/chronos2_adapter.safetensors').exists() else None,
                  context_sha256=snapshot_digest(),
                  versions={p: importlib.metadata.version(p) for p in ['torch', 'chronos-forecasting', 'transformers', 'huggingface_hub', 'peft', 'datasets']},
                  models=runs, metrics=metrics, nonoverlapping_metrics=nonoverlapping_metrics, latest=latest,
                  caveats=['Frozen variants and one LoRA-adapted variant are evaluated separately; adaptation uses data through September 2025 only.',
                           'Exploratory daily benchmark after both model revisions. Daily 10-session targets overlap; nonoverlapping metrics are also provided.',
                           'Pretraining corpora cannot be exhaustively audited; pinned weights predate evaluated outcomes.',
                           'Quantile-interpolated probabilities and intervals are not calibrated for financial markets.',
                           'No model selection or deployment replacement is performed using this report.'])
    predictions.to_csv(ROOT / 'data/foundation_predictions.csv', index=False)
    (ROOT / 'data/foundation_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers', nargs='+', choices=TICKERS, default=TICKERS)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    benchmark(args.tickers, args.device)
