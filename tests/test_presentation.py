import math
import unittest
from src.presentation import outlook_summary, direction_text


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {'balanced_accuracy':.65,'log_loss':.5,'baseline_log_loss':.69}
        self.evidence = {'bootstrap_95_interval':[-.04,-.01]}

    def test_high_probability_does_not_override_weak_evidence(self):
        weak = {'bootstrap_95_interval':[-.01,.03]}
        result = outlook_summary(.95,self.metrics,weak)
        self.assertEqual(result['label'],'Uncertain')
        self.assertEqual(result['takeaway'],'Wait / review')
        self.assertIn('does not give a clear reason to invest',result['meaning'])

    def test_positive_and_negative_describe_direction_only(self):
        positive = outlook_summary(.8,self.metrics,self.evidence)
        negative = outlook_summary(.2,self.metrics,self.evidence)
        self.assertEqual(positive['label'],'Positive')
        self.assertEqual(negative['label'],'Negative')
        self.assertEqual(positive['takeaway'],'Consider buying')
        self.assertEqual(negative['takeaway'],'Consider reducing')
        self.assertIn('loss',positive['reason'].lower())
        self.assertIn('estimate',negative['reason'].lower())

    def test_old_data_and_invalid_estimates_fail_closed(self):
        self.assertEqual(outlook_summary(.9,self.metrics,self.evidence,stale=True)['label'],'Uncertain')
        self.assertEqual(outlook_summary(float('nan'),self.metrics,self.evidence)['label'],'Uncertain')
        self.assertEqual(outlook_summary(.5,self.metrics,self.evidence)['label'],'Uncertain')

    def test_poor_asset_performance_blocks_positive_label(self):
        self.metrics['log_loss']=.8
        self.assertEqual(outlook_summary(.9,self.metrics,self.evidence)['label'],'Uncertain')
