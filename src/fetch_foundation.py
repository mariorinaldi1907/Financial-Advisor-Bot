"""Download only pinned public safetensors and configs into the local cache."""
import json
import os
import tempfile
from .train import ROOT

if __name__ == '__main__':
    os.environ.setdefault('HF_HOME', str(Path(tempfile.gettempdir()) / 'financial-advisor-hf'))
    from huggingface_hub import snapshot_download
    registry = json.loads((ROOT / 'src/foundation_models.json').read_text())
    for model_id, info in registry.items():
        path = snapshot_download(model_id, revision=info['revision'],
                                 allow_patterns=['config.json', 'model.safetensors'])
        print(f'{model_id} → {path}')
