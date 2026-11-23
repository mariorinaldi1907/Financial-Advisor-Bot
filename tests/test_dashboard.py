"""End-to-end checks against the shared snapshots and saved research evidence."""
import json
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.context import ASSETS, HORIZONS, build_panel, input_hashes
from src.research import labeled_panel, probabilities, return_predictions

ROOT = Path(__file__).resolve().parents[1]


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'data/research_results.json').read_text())
        cls.predictions = pd.read_csv(ROOT / 'data/research_predictions.csv', parse_dates=['Date','Label_end'])
        cls.panel, _, _ = build_panel()

    def test_models_dates_selection_and_saved_predictions(self):
        self.assertEqual(self.report['input_hashes'], input_hashes())
        self.assertFalse(self.predictions.duplicated(['Date','Ticker','horizon']).any())
        for h in HORIZONS:
            with self.subTest(horizon=h):
                run = self.report['horizons'][str(h)]
                self.assertEqual(run['selected_model'], min(run['validation_log_loss'], key=run['validation_log_loss'].get))
                for trace in run['validation_traces'] + run['evaluation_traces'] + [run['live_trace']]:
                    self.assertLess(trace['fit_label_end'], trace['calibration_start'])
                    self.assertLess(trace['calibration_label_end'], trace['evaluation_start'])
                self.assertEqual(len(run['evaluation_period_metrics']),3)
                for period in run['evaluation_period_metrics']:
                    self.assertEqual(len(period['models'][run['selected_model']]),22)
                bundle = joblib.load(ROOT / 'models' / f'horizon_{h}.joblib')
                self.assertEqual(bundle['hashes'], input_hashes())
                data = labeled_panel(self.panel, h)
                latest = data.loc[data.index == data.index.max()]
                from src.research import temporal_parts
                fit, _, _ = temporal_parts(data, pd.Timestamp(bundle['as_of']) + pd.Timedelta(days=1), bundle['excluded_dates'])
                imputer = bundle['classifiers']['logistic_market']['model'].steps[0][1]
                cols = bundle['classifiers']['logistic_market']['columns']
                np.testing.assert_allclose(imputer.statistics_,fit[cols].median().to_numpy(),rtol=1e-8,atol=1e-10)
                pd.testing.assert_frame_equal(bundle['replay_inputs'], latest[bundle['features']],check_exact=True)
                p = probabilities(bundle['classifiers'], latest)[bundle['selected_model']]
                point, q = return_predictions(bundle['returns'], latest)
                self.assertTrue(np.isfinite(q).all())
                self.assertTrue((q[:,0] <= q[:,1]).all() and (q[:,1] <= q[:,2]).all())
                for i, asset in enumerate(latest.Ticker):
                    saved = next(r for r in run['latest'] if r['ticker'] == asset)
                    self.assertAlmostEqual(saved['p_up'], float(p[i]), places=6)
                    self.assertAlmostEqual(saved['expected_return'], float(point[i]), places=6)
                    records = self.predictions.loc[(self.predictions.Ticker == asset) & (self.predictions.horizon == h)]
                    labels = data.loc[data.Ticker == asset].loc[records.Date]
                    np.testing.assert_array_equal((labels.Forward_return.to_numpy() > 0).astype(int), records.Target.to_numpy())
                    accuracy = float(((records.p_up >= .5) == records.Target).mean())
                    self.assertAlmostEqual(accuracy, run['metrics'][asset]['accuracy'])
                    self.assertTrue((records.Label_end > records.Date).all())

    def test_dashboard_all_assets_and_horizons(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=40).run()
        self.assertFalse(app.exception)
        for asset in ASSETS:
            app.sidebar.selectbox[0].set_value(asset).run()
            self.assertFalse(app.exception)
        app.sidebar.selectbox[0].set_value('SPY').run()
        for horizon in HORIZONS:
            app.sidebar.selectbox[1].set_value(horizon).run()
            self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Historical evaluation').run()
        self.assertFalse(app.exception)
        horizon = int(app.sidebar.selectbox[1].value)
        records = self.predictions.loc[(self.predictions.Ticker == 'SPY') & (self.predictions.horizon == horizon)]
        sunday = pd.date_range(records.Date.min(), records.Date.max(), freq='W-SUN')[-1].date()
        app.sidebar.date_input[0].set_value(sunday).run()
        self.assertFalse(app.exception)
        expected = records.loc[records.Date <= pd.Timestamp(sunday), 'Date'].max().date()
        self.assertTrue(any(str(expected) in c.value for c in app.caption))
