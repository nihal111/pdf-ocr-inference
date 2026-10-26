import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.compare import compare
from eval.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[2]
RESULT = {'ocr_text': 'Hello world\n',
          'chunks': [{'id': 'chunk-000', 'embedding': [1.0, 0.0, 0.0]}]}


class FakeWorker:
    startup_ms = 1
    first_result_ms = 2

    def __init__(self, module, timeout):
        pass

    def call(self, path):
        return {'output': copy.deepcopy(RESULT), 'latency_ms': 10}

    def close(self):
        pass


class ReportTests(unittest.TestCase):
    def test_tuple_is_rejected_before_json_conversion(self):
        report = evaluate('eval.tests.adapters.tuple_vector',
                          ROOT / 'eval/fixtures/manifest.json', ROOT / 'eval/fixtures',
                          warmup=0, repeats=1, timeout=5)
        self.assertFalse(report['pass'])
        self.assertIn('embedding must be a non-empty list', str(report['cases'][0]['failures']))

    def report(self, repeats=5, minimum=.99):
        with tempfile.TemporaryDirectory() as temp, patch('eval.evaluate.Worker', FakeWorker):
            manifest = Path(temp) / 'manifest.json'
            source = json.loads((ROOT / 'eval/fixtures/manifest.json').read_text())
            source['purpose'] = 'unit-test'
            manifest.write_text(json.dumps(source))
            return evaluate('mocked.adapter', manifest, ROOT / 'eval/fixtures',
                            warmup=1, repeats=repeats, minimum=minimum,
                            environment={'hardware_id': 'unit-test', 'gpu_model': 'mock',
                                         'gpu_count': 1, 'driver_version': 'mock',
                                         'concurrency': 1})

    def test_short_run_passes_but_is_not_comparable(self):
        report = self.report(repeats=4)
        self.assertTrue(report['pass'])
        self.assertFalse(report['summary']['comparable'])

    def test_weaker_threshold_is_not_comparable(self):
        report = self.report(minimum=.98)
        self.assertTrue(report['pass'])
        self.assertFalse(report['summary']['comparable'])

    def test_complete_run_can_be_compared(self):
        report = self.report()
        self.assertTrue(report['summary']['comparable'])
        self.assertEqual(compare(report, report)['metrics']['p50_ms']['speedup'], 1)

    def test_compare_rejects_short_reports_even_if_flagged_comparable(self):
        report = self.report(repeats=4)
        report['summary']['comparable'] = True
        with self.assertRaisesRegex(ValueError, 'five measured repeats'):
            compare(report, report)


if __name__ == '__main__':
    unittest.main()
