import json
import tempfile
import unittest
from pathlib import Path
from src.serve import get_port, validate_snapshot


class DeploymentTests(unittest.TestCase):
    def test_shipped_demo_is_complete_and_consistent(self):
        report = validate_snapshot()
        self.assertEqual(set(report['horizons']),{'1','5','10','20'})
        self.assertEqual(len(report['assets']),22)

    def test_platform_port_validation(self):
        self.assertEqual(get_port({}),8501)
        self.assertEqual(get_port({'PORT':'10000'}),10000)
        for value in ['abc','80','70000','8501; echo no']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                get_port({'PORT':value})

    def test_changed_snapshot_is_rejected_before_server_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'data').mkdir()
            (root/'data/research_results.json').write_text(json.dumps({'input_hashes':{'market':'wrong'}}))
            (root/'data/market_prices.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'market snapshot'):
                validate_snapshot(root)
