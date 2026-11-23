"""Integration checks against the committed snapshot and generated artifacts."""
import hashlib
import json
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def test_artifacts_reference_committed_snapshot(self):
        meta = json.loads((ROOT / 'reports/evaluation.json').read_text())
        digest = hashlib.sha256((ROOT / 'data/raw/spy.csv').read_bytes()).hexdigest()
        self.assertEqual(digest, meta['raw_sha256'])
        self.assertLess(meta['development_label_end'], meta['holdout_start'])
        for candidate in meta['validation'].values():
            for fold in candidate['folds']:
                self.assertLess(fold['label_end'], fold['validation_start'])
        winner = min(meta['validation'], key=lambda n: meta['validation'][n]['mean_log_loss'])
        self.assertEqual(winner, meta['selected_model'])

    def test_latest_and_historical_views(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 2)
        app.sidebar.radio[0].set_value('Historical holdout').run()
        self.assertFalse(app.exception)
        # A Sunday within the committed holdout must use the preceding Friday.
        meta = json.loads((ROOT / 'reports/evaluation.json').read_text())
        import pandas as pd
        dates = pd.date_range(meta['holdout_start'], meta['holdout_end'], freq='W-SUN')
        sunday = dates[-1].date()
        app.sidebar.date_input[0].set_value(sunday).run()
        self.assertFalse(app.exception)
        predictions = pd.read_csv(ROOT / 'reports/holdout_predictions.csv', index_col=0, parse_dates=True)
        expected = predictions.index[predictions.index <= pd.Timestamp(sunday)][-1].date()
        self.assertTrue(any(str(expected) in c.value for c in app.caption))
