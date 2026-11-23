"""Optional-model logic tests do not require torch or network access."""
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.foundation import probability_up, context_at, BENCHMARK_START
from src.features import build_market_context

ROOT = Path(__file__).resolve().parents[1]


class FoundationTests(unittest.TestCase):
    def test_quantile_probabilities(self):
        q = np.arange(1, 10.)
        self.assertAlmostEqual(probability_up(q, 5), .5)
        self.assertAlmostEqual(probability_up(q, 0), .9)
        self.assertAlmostEqual(probability_up(q, 10), .1)
        self.assertAlmostEqual(probability_up(q[::-1], 5), .5)
        with self.assertRaises(ValueError):
            probability_up(np.full(9, np.nan), 5)

    def test_context_stops_at_forecast_date(self):
        dates = pd.bdate_range('2020-01-01', periods=400)
        frame = pd.DataFrame({'Close': np.arange(400.)}, index=dates)
        first = context_at(frame, dates[300])
        frame.loc[dates[301]:, 'Close'] = 1000000
        np.testing.assert_array_equal(first, context_at(frame, dates[300]))
        self.assertEqual(first[-1], 300)
        self.assertEqual(len(first), 256)

    def test_market_context_is_causal_and_does_not_fill(self):
        dates = pd.bdate_range('2020-01-01', periods=120)
        prices = pd.DataFrame({'SPY': np.arange(120.) + 100, 'GLD': np.arange(120.) + 200}, index=dates)
        before = build_market_context(prices, exclude='SPY')
        changed = prices.copy()
        changed.iloc[100:] *= 2
        pd.testing.assert_frame_equal(before.iloc[:100], build_market_context(changed, exclude='SPY').iloc[:100])
        self.assertTrue(all('SPY' not in c for c in before))
        prices.loc[dates[80], 'GLD'] = np.nan
        self.assertTrue(np.isnan(build_market_context(prices).loc[dates[80], 'Context_GLD_return_5']))

    def test_weights_and_adaptation_predate_evaluation(self):
        registry = json.loads((ROOT / 'src/foundation_models.json').read_text())
        for info in registry.values():
            self.assertLess(info['revision_date'][:10], BENCHMARK_START)
        meta_path = ROOT / 'models/chronos2_training.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.assertLess(meta['train_cutoff'], BENCHMARK_START)
            for series in meta['training_series'].values():
                self.assertLessEqual(series['end'], meta['train_cutoff'])

    def test_saved_benchmark_targets_and_probabilities(self):
        from src.data import load_market, raw_digest
        report_path = ROOT / 'data/foundation_results.json'
        if not report_path.exists():
            self.skipTest('Foundation benchmark has not been generated.')
        report = json.loads(report_path.read_text())
        records = pd.read_csv(ROOT / 'data/foundation_predictions.csv', parse_dates=['Date'])
        self.assertFalse(records.duplicated(['ticker', 'model', 'Date']).any())
        self.assertTrue(records.p_up.between(.099999, .900001).all())
        for ticker in report['metrics']:
            raw, _ = load_market(ticker)
            self.assertEqual(report['source_hashes'][ticker], raw_digest(ticker))
            future = raw.Close.shift(-10)
            labels = (future > raw.Close).astype(int)
            for model, group in records.loc[records.ticker == ticker].groupby('model'):
                np.testing.assert_array_equal(group.Target.to_numpy(), labels.loc[group.Date].to_numpy())
                self.assertEqual(len(group), report['metrics'][ticker][model]['windows'])
                self.assertTrue((group.Date >= pd.Timestamp(BENCHMARK_START)).all())
                self.assertEqual(report['latest'][ticker][model]['as_of'], str(raw.index.max().date()))

    def test_source_hash_uses_csv_records_and_normalizes_line_endings(self):
        import tempfile
        from src.data import raw_digest
        text = ('Date,Ticker,Close,High,Low,Open,Volume\n'
                '2020-01-01,SPY,100.123456789012345,101,99,100,1000\n'
                '2020-01-01,GLD,150,151,149,150,2000\n')
        with tempfile.NamedTemporaryFile(suffix='.csv') as temporary:
            path = Path(temporary.name)
            path.write_bytes(text.encode())
            original = raw_digest('SPY', path)
            path.write_bytes(text.replace('\n', '\r\n').encode())
            self.assertEqual(original, raw_digest('SPY', path))
            path.write_text(text.replace('GLD,150', 'GLD,155'))
            self.assertEqual(original, raw_digest('SPY', path))
            path.write_text(text.replace('100.123456789012345', '100.123456789012346'))
            self.assertNotEqual(original, raw_digest('SPY', path))
