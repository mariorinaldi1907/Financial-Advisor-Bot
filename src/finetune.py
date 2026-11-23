"""Adapt Chronos-2 jointly to seven ETFs using historical context only."""
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_market, raw_digest, snapshot_digest
from .features import build_features, HORIZON
from .foundation import COVARIATES, CONTEXT_LENGTH
from .train import ROOT
from .universe import TICKERS

TRAIN_END = '2025-09-30'
STEPS = 120


def finetune():
    os.environ.setdefault('HF_HOME', str(Path(tempfile.gettempdir()) / 'financial-advisor-hf'))
    import torch
    from chronos import BaseChronosPipeline
    from huggingface_hub import snapshot_download
    from transformers import set_seed
    torch.set_num_threads(4)
    set_seed(42)
    info = json.loads((ROOT / 'src/foundation_models.json').read_text())['amazon/chronos-2']
    base = snapshot_download('amazon/chronos-2', revision=info['revision'],
                             allow_patterns=['config.json', 'model.safetensors'], local_files_only=True)
    pipeline = BaseChronosPipeline.from_pretrained(base, device_map='cpu', torch_dtype=torch.float32,
                                                  trust_remote_code=False, use_safetensors=True)
    _, markets = load_market(TICKERS[0])
    inputs, evidence = [], {}
    for ticker in TICKERS:
        raw, _ = load_market(ticker)
        frame = build_features(raw)
        for symbol in markets:
            frame[f'Market_{symbol}'] = markets[symbol].reindex(frame.index)
        columns = COVARIATES + [c for c in frame if c.startswith('Market_')]
        frame = frame.loc[:TRAIN_END].dropna(subset=columns)
        inputs.append({'target': frame.Close.to_numpy(dtype=np.float32),
                       'past_covariates': {c: frame[c].to_numpy(dtype=np.float32) for c in columns}})
        evidence[ticker] = dict(rows=len(frame), start=str(frame.index.min().date()),
                                end=str(frame.index.max().date()), raw_sha256=raw_digest(raw))
    with tempfile.TemporaryDirectory(prefix='etf-finetune-') as temporary:
        pipeline.fit(inputs, prediction_length=HORIZON, finetune_mode='lora',
            context_length=CONTEXT_LENGTH, learning_rate=1e-5, num_steps=STEPS,
            batch_size=32, min_past=CONTEXT_LENGTH, output_dir=Path(temporary),
            optim='adamw_torch', logging_steps=20, disable_tqdm=True, seed=42,
            data_seed=42, report_to='none')
        source = Path(temporary) / 'finetuned-ckpt'
        output = ROOT / 'models'
        shutil.copy2(source / 'adapter_model.safetensors', output / 'chronos2_adapter.safetensors')
        shutil.copy2(source / 'adapter_config.json', output / 'chronos2_adapter_config.json')
    adapter = output / 'chronos2_adapter_config.json'
    config = json.loads(adapter.read_text())
    config['base_model_name_or_path'] = 'amazon/chronos-2'
    config['revision'] = info['revision']
    adapter.write_text(json.dumps(config, indent=2) + '\n')
    metadata = dict(base_model='amazon/chronos-2', **info, train_cutoff=TRAIN_END,
                    num_steps=STEPS, learning_rate=1e-5, batch_size=32, seed=42,
                    mode='lora', context_length=CONTEXT_LENGTH, horizon=HORIZON,
                    historical_covariates=columns, training_series=evidence,
                    context_sha256=snapshot_digest(),
                    caveat='Fixed-budget experimental adaptation. No test-period values or model selection used in fine-tuning.')
    (output / 'chronos2_training.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Adapter saved to {output}', flush=True)


if __name__ == '__main__':
    finetune()
