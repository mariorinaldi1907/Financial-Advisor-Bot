import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.context import environment_features, build_panel, HORIZONS, macro_features, market_conditions
from src.research import temporal_parts, labeled_panel, validation_slice, conformal_width, execution_study, append_forecast_ledger, walk_forward_folds, CONFIG, select_candidate, feature_groups
from src.forward import resolve_forecasts, summarize_forecasts
from src.data import raw_digest


class ResearchTests(unittest.TestCase):
    def test_environment_is_delayed_and_causal(self):
        dates = pd.bdate_range('2020-01-01', periods=100)
        columns = ['VIX','TBill13WeekIndex','Treasury5YIndex','Treasury10YIndex','Treasury30YIndex']
        environment = pd.DataFrame({c:np.arange(100.) + i for i,c in enumerate(columns)}, index=dates)
        first = environment_features(environment, dates)
        self.assertTrue(first.Env_VIX_change20.iloc[:22].isna().all())
        self.assertEqual(first.Env_VIX_change20.iloc[22], environment.VIX.iloc[20] - environment.VIX.iloc[0])
        self.assertNotIn('Env_VIX', first)
        environment.iloc[70:] *= 5
        pd.testing.assert_frame_equal(first.iloc[:70], environment_features(environment, dates).iloc[:70])
        sparse = environment.iloc[:5]
        self.assertTrue(np.isnan(environment_features(sparse, dates).Env_VIX_z252.iloc[-1]))

    def test_label_boundaries_and_no_future_features(self):
        panel, market, all_features = build_panel()
        self.assertEqual(panel.Ticker.nunique(), 22)
        self.assertTrue(set(market).issubset(all_features))
        self.assertFalse(any(c.startswith(('Forward_', 'Label_', 'Entry_', 'Exit_')) or c in ['Target','Z_return'] for c in all_features))
        for h in HORIZONS:
            data = labeled_panel(panel, h)
            fit, calibration, known = temporal_parts(data, '2024-01-01')
            self.assertLess(fit.Label_end.max(), calibration.index.min())
            self.assertLess(calibration.Label_end.max(), pd.Timestamp('2024-01-01'))
            validation = validation_slice(data, '2022-01-01', '2024-01-01')
            self.assertLess(validation.Label_end.max(), pd.Timestamp('2024-01-01'))
            latest = data.loc[data.index == data.index.max()]
            self.assertTrue(latest.Target.isna().all())

    def test_conformal_quantile_has_finite_sample_correction(self):
        self.assertEqual(conformal_width(np.arange(20.)), 16.)
        with self.assertRaises(ValueError):
            conformal_width(np.arange(10.))

    def test_execution_costs_apply_to_each_round_trip(self):
        frame = pd.DataFrame({'Ticker':['SPY']*3, 'entry_open':[100.]*3, 'exit_open':[100.]*3,
                              'p_up':[.9]*3, 'expected_return':[.01]*3},
                             index=pd.bdate_range('2024-01-01', periods=3))
        result = execution_study(frame, 1)['SPY']
        self.assertEqual(result['invested_fraction'], 1.)
        self.assertAlmostEqual(result['strategy']['total_return'], (1-.001)**3-1)

    def test_forward_evaluation_rejects_late_issuance(self):
        dates = pd.to_datetime(['2026-11-20','2026-11-23','2026-11-24','2026-11-25'])
        market = pd.DataFrame({'Date':dates,'Ticker':['SPY']*4,'Close':[100.,101.,102.,103.]})
        ledger = pd.DataFrame([dict(ticker='SPY',horizon=h,as_of='2026-11-20',issued_utc=issued,p_up=.6)
                              for h,issued in [(1,'2026-11-23T03:00:00Z'),(5,'2026-11-23T03:00:00Z'),(1,'2026-11-23T16:00:00Z')]])
        result = resolve_forecasts(ledger, market)
        self.assertEqual(list(result.status), ['evaluated','pending','ineligible_issue_time'])
        self.assertEqual(result.loc[0,'correct'], 1)
        self.assertAlmostEqual(result.loc[0,'realized_return'], .01)
        np.testing.assert_array_equal(ledger.p_up, result.p_up)

    def test_source_hash_is_platform_independent(self):
        text = ('Date,Ticker,Close,High,Low,Open,Volume\n'
                '2020-01-01,SPY,100.123456789012345,101,99,100,1000\n')
        with tempfile.NamedTemporaryFile(suffix='.csv') as temporary:
            path=Path(temporary.name)
            path.write_bytes(text.encode()); original=raw_digest('SPY',path)
            path.write_bytes(text.replace('\n','\r\n').encode())
            self.assertEqual(original,raw_digest('SPY',path))
            path.write_text(text.replace('100.123456789012345','100.123456789012346'))
            self.assertNotEqual(original,raw_digest('SPY',path))


    def test_embargo_is_excluded_from_later_training(self):
        panel, _, _ = build_panel()
        data = labeled_panel(panel, 20)
        previous_after = None
        total_excluded = pd.DatetimeIndex([])
        for validation, start, end, excluded, embargo, following, next_allowed in walk_forward_folds(data, CONFIG['cv_periods']):
            if previous_after is not None:
                self.assertGreaterEqual(start, previous_after)
            self.assertLess(validation.Label_end.max(), pd.Timestamp(end))
            self.assertEqual(len(following), 20)
            fit, calibration, _ = temporal_parts(data, start, excluded)
            self.assertFalse(fit.index.isin(excluded).any())
            self.assertFalse(calibration.index.isin(excluded).any())
            total_excluded = total_excluded.union(following)
            previous_after = next_allowed
        self.assertEqual(len(total_excluded), 60)

    def test_issued_probabilities_are_not_rewritten(self):
        rows = [dict(ticker='SPY', horizon=1, as_of='2026-11-20', p_up=.6)]
        with tempfile.NamedTemporaryFile(suffix='.csv') as temporary:
            path=Path(temporary.name)
            append_forecast_ledger(rows,path)
            original=pd.read_csv(path)
            rows[0]['p_up']=.9
            append_forecast_ledger(rows,path)
            after=pd.read_csv(path)
            self.assertEqual(len(after),1)
            self.assertEqual(after.loc[0,'p_up'],.6)
            self.assertEqual(after.loc[0,'issued_utc'],original.loc[0,'issued_utc'])

    def test_promotion_requires_consistency_and_material_gain(self):
        losses = {'prior':[.69,.69,.69], 'one_lucky_fold':[.60,.70,.70],
                  'negligible':[.6899]*3, 'stable':[.688,.688,.689]}
        selected, diagnostics = select_candidate(losses)
        self.assertEqual(selected,'stable')
        self.assertFalse(diagnostics['one_lucky_fold']['eligible'])
        self.assertFalse(diagnostics['negligible']['eligible'])
        self.assertEqual(select_candidate({'prior':[.69]*3,'worse':[.7]*3})[0],'prior')
        with self.assertRaises(ValueError):
            select_candidate({'prior':[.69]*3,'broken':[.68,float('nan'),.68]})

    def test_macro_vintages_are_joined_by_availability_not_observation(self):
        rows = [dict(Date=available,indicator=name,vintage_month=vintage,observation_date=observation,
                     growth_1m=.01,growth_12m=.03)
                for name in ['inflation','employment']
                for available,vintage,observation in [('2020-02-01','2020-01-01','2019-12-01'),
                                                        ('2020-03-01','2020-02-01','2020-01-01')]]
        vintages=pd.DataFrame(rows)
        sessions=pd.bdate_range('2020-01-01','2020-06-01')
        original=macro_features(vintages,sessions)
        self.assertTrue(original.loc[:'2020-01-31','Macro_inflation_growth_1m'].isna().all())
        self.assertEqual(original.loc['2020-02-03','Macro_inflation_growth_1m'],.01)
        vintages.loc[vintages.Date == '2020-03-01','growth_1m'] = 9
        altered=macro_features(vintages,sessions)
        pd.testing.assert_frame_equal(original.loc[:'2020-02-28'],altered.loc[:'2020-02-28'])
        self.assertTrue(altered.loc['2020-06-01'].filter(like='growth').isna().all())
        vintages.loc[0,'Date']='2020-01-15'
        with self.assertRaises(ValueError):
            macro_features(vintages,sessions)

    def test_forward_summary_excludes_pending_and_pairs_comparisons(self):
        ledger=pd.DataFrame([
            dict(status='evaluated',horizon=10,model_sha256='a',outcome_date='2026-11-20',
                 actual_up=1,p_up=.8,prior_p_up=.5,realized_return=.03,lower_return=-.02,upper_return=.04,
                 downside_p=.2,downside_prior_p=.3,downside_threshold=-.02),
            dict(status='pending',horizon=10,model_sha256='a',p_up=.1),
            dict(status='ineligible_issue_time',horizon=10,model_sha256='a',p_up=.9)])
        summary=summarize_forecasts(ledger)
        self.assertEqual(len(summary['groups']),1)
        group=summary['groups'][0]
        self.assertEqual(group['rows'],1)
        self.assertEqual(group['interval']['coverage'],1.)
        self.assertAlmostEqual(group['direction']['brier'],.04)
        self.assertAlmostEqual(group['downside']['model']['brier'],.04)
        self.assertEqual(group['baseline_comparison']['rows'],1)

    def test_feature_ablation_sets_keep_baselines_and_exclude_targets(self):
        _, market, all_cols=build_panel()
        groups=feature_groups(market,all_cols)
        self.assertLess(len(groups['core']),len(market))
        self.assertEqual(set(groups['credit'])-set(groups['core']),{c for c in all_cols if c.startswith('Credit_')})
        for columns in groups.values():
            self.assertTrue(any(c.startswith('Asset_') for c in columns))
            self.assertFalse(any(c.startswith(('Forward_', 'Label_', 'Scale_')) for c in columns))
