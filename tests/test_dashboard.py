"""End-to-end checks against the committed market snapshot and models."""
import json
import unittest
from pathlib import Path
import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.data import load_market, raw_digest, snapshot_digest, feature_frame
from src.universe import TICKERS

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    def test_all_models_and_temporal_boundaries(self):
        runs = json.loads((ROOT / 'data/training_results.json').read_text())
        for ticker in TICKERS:
            with self.subTest(ticker=ticker):
                meta = runs[ticker]
                raw, _ = load_market(ticker)
                self.assertEqual(meta['ticker'], ticker)
                self.assertEqual(meta['snapshot_sha256'], snapshot_digest())
                self.assertEqual(meta['raw_sha256'], raw_digest(ticker))
                self.assertLess(meta['development_label_end'], meta['holdout_start'])
                for candidate in meta['validation'].values():
                    for fold in candidate['folds']:
                        self.assertLess(fold['label_end'], fold['validation_start'])
                winner = min(meta['validation'], key=lambda n: meta['validation'][n]['mean_log_loss'])
                self.assertEqual(winner, meta['selected_model'])
                bundle = joblib.load(ROOT / 'models' / f'{ticker.lower()}_forecast.joblib')
                features = feature_frame(ticker).dropna()
                pred = bundle['model'].predict_proba(features[meta['features']].tail(1))
                self.assertEqual(pred.shape, (1, 2))
                self.assertAlmostEqual(float(pred.sum()), 1.)

    def test_latest_and_historical_views_for_all_assets(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 5)
        app.sidebar.radio[0].set_value('Historical holdout').run()
        self.assertFalse(app.exception)
        for ticker in TICKERS[1:] + ['SPY']:
            app.sidebar.selectbox[0].set_value(ticker).run()
            self.assertFalse(app.exception)
        meta = json.loads((ROOT / 'data/training_results.json').read_text())['SPY']
        sunday = pd.date_range(meta['holdout_start'], meta['holdout_end'], freq='W-SUN')[-1].date()
        app.sidebar.date_input[0].set_value(sunday).run()
        self.assertFalse(app.exception)
        predictions = pd.read_csv(ROOT / 'data/spy_predictions.csv', index_col=0, parse_dates=True)
        expected = predictions.index[predictions.index <= pd.Timestamp(sunday)][-1].date()
        self.assertTrue(any(str(expected) in c.value for c in app.caption))
        app.sidebar.radio[0].set_value('Latest forecast').run()
        self.assertFalse(app.exception)
