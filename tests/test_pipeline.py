import unittest
import numpy as np
import pandas as pd
from src.features import build_features, add_target, CORE_FEATURES, EXTRA_FEATURES
from src.train import purged_before


class PipelineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(.0002, .01, 350)))
        self.raw = pd.DataFrame({'Close': close, 'High': close * 1.01,
                                 'Low': close * .99, 'Volume': rng.integers(1000, 10000, 350)},
                                index=pd.bdate_range('2020-01-01', periods=350))

    def test_future_data_cannot_change_past_features(self):
        original = build_features(self.raw)
        changed = self.raw.copy()
        changed.iloc[250:, :3] *= 3
        changed.iloc[250:, 3] *= 2
        pd.testing.assert_frame_equal(original.iloc[:250], build_features(changed).iloc[:250])

    def test_latest_targets_stay_unknown(self):
        labeled = add_target(build_features(self.raw))
        self.assertTrue(labeled.Target.tail(10).isna().all())
        self.assertEqual(labeled.Target.iloc[120], int(self.raw.Close.iloc[130] > self.raw.Close.iloc[120]))
        self.assertEqual(labeled.Label_end.iloc[120], self.raw.index[130])

    def test_purge_removes_overlapping_outcomes(self):
        labeled = add_target(build_features(self.raw)).dropna(subset=['Target'])
        boundary = labeled.index[250]
        train = purged_before(labeled.iloc[:250], boundary)
        self.assertTrue((train.Label_end < boundary).all())
        self.assertEqual(len(train), 240)

    def test_duplicate_dates_rejected(self):
        with self.assertRaises(ValueError):
            build_features(pd.concat([self.raw, self.raw.tail(1)]))

    def test_normalized_features_invariant_to_price_units(self):
        original = build_features(self.raw)
        changed = self.raw.copy()
        changed[['Close', 'High', 'Low']] *= 10
        pd.testing.assert_frame_equal(original[CORE_FEATURES + EXTRA_FEATURES],
                                      build_features(changed)[CORE_FEATURES + EXTRA_FEATURES])

    def test_flat_prices_have_neutral_rsi(self):
        raw = self.raw.copy()
        raw['Close'], raw['High'], raw['Low'] = 100., 101., 99.
        self.assertEqual(build_features(raw).RSI_14.iloc[-1], 50)


if __name__ == '__main__':
    unittest.main()
