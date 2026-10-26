import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from eval.evaluate import check, evaluate, load_cases, load_json, percentile

ROOT = Path(__file__).resolve().parents[2]
R = {'ocr_text': 'Hi\n', 'chunks': [{'id': 'c0', 'embedding': [1., 0.]}]}


class Checks(unittest.TestCase):
    def changed(self, **changes):
        return {**copy.deepcopy(R), **changes}

    def vector(self, v):
        return self.changed(chunks=[{'id': 'c0', 'embedding': v}])

    def test_pass(self):
        self.assertTrue(check(R, R)['pass'])

    def test_exact_bytes(self):
        for text in ['Hi', 'Hi\r\n', 'hi\n', 'Hi\n ']:
            with self.subTest(text=text):
                self.assertFalse(check(self.changed(ocr_text=text), R)['pass'])

    def test_unicode_normalization_not_applied(self):
        ref = self.changed(ocr_text='é')
        self.assertFalse(check(self.changed(ocr_text='e\u0301'), ref)['pass'])

    def test_surrogate_is_failure(self):
        self.assertFalse(check(self.changed(ocr_text='\ud800'), R)['pass'])

    def test_invalid_vectors(self):
        for vector in [[], [0, 0], [1], [-1, 0], [True, False], ['1', '0'],
                       [float('nan'), 0], [float('inf'), 0], [10**400, 0], [None, 0]]:
            with self.subTest(vector=vector):
                self.assertFalse(check(self.vector(vector), R)['pass'])

    def test_extreme_finite_vectors(self):
        for scalar in [1e-300, 1e300]:
            self.assertTrue(check(self.vector([scalar, 0]), R)['pass'])

    def test_ids_count_and_shape(self):
        for chunks in [None, [], [{}], [{'id':'bad','embedding':[1,0]}], R['chunks']*2]:
            self.assertFalse(check(self.changed(chunks=chunks), R)['pass'])
        self.assertFalse(check(None, R)['pass'])

    def test_every_chunk_checked(self):
        ref = self.changed(chunks=[{'id': str(i), 'embedding':[1,0]} for i in range(100)])
        got = copy.deepcopy(ref)
        got['chunks'][-1]['embedding'] = [-1,0]
        self.assertFalse(check(got, ref)['pass'])

    def test_threshold_inclusive(self):
        vector = [.99, math.sqrt(1-.99**2)]
        self.assertTrue(check(self.vector(vector), R)['pass'])
        self.assertFalse(check(self.vector([.989, math.sqrt(1-.989**2)]), R)['pass'])

    def test_invalid_reference(self):
        for bad in [None, {}, self.changed(chunks=[]), self.vector([0,0]), self.changed(chunks=R['chunks']*2)]:
            with self.assertRaises(ValueError):
                check(R, bad)

    def test_invalid_threshold(self):
        for minimum in [float('nan'), float('inf'), 0, -1, 1.01, True]:
            with self.assertRaises(ValueError):
                check(R, R, minimum)

    def test_percentiles(self):
        self.assertEqual(percentile([10, 20, 30], .5), 20)
        self.assertEqual(percentile([10, 20, 30], .95), 29)
        self.assertIsNone(percentile([], .5))


class Manifests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'x.pdf').write_bytes(b'%PDF-1.4\n')
        (self.root/'ref.json').write_text(json.dumps(R))
        self.case = {'id':'x','pdf':'x.pdf','reference':'ref.json','pages':1}

    def load(self, cases):
        manifest = self.root/'manifest.json'
        manifest.write_text(json.dumps({'cases':cases}))
        return load_cases(manifest, self.root)

    def test_valid(self):
        self.assertEqual(len(self.load([self.case])), 1)

    def test_empty_duplicate_bad_pages(self):
        for cases in [[], [self.case, self.case], [{**self.case,'pages':0}],
                      [{**self.case,'pages':True}], [{**self.case,'pages':1.5}], [{}]]:
            with self.assertRaises(ValueError):
                self.load(cases)

    def test_missing_absolute_escape_and_symlink(self):
        (self.root/'escape.pdf').symlink_to('/etc/hosts')
        for path in ['missing.pdf', '../x.pdf', '/etc/hosts', 'escape.pdf']:
            with self.assertRaises(ValueError):
                self.load([{**self.case, 'pdf':path}])

    def test_nonfinite_duplicate_json(self):
        path = self.root/'bad.json'
        for text in ['{"a":NaN}', '{"a":1,"a":2}']:
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_json(path)


class Execution(unittest.TestCase):
    def run_module(self, module, **kwargs):
        return evaluate(module, ROOT/'eval/fixtures/manifest.json', ROOT/'eval/fixtures',
                        timeout=kwargs.pop('timeout', 5), **kwargs)

    def test_synthetic(self):
        result = self.run_module('eval.smoke_adapter', warmup=1, repeats=2)
        self.assertTrue(result['pass'])
        self.assertEqual(result['summary']['measured_samples'], 2)
        self.assertFalse(result['summary']['comparable'])
        self.assertTrue(result['cold_start']['process_to_first_result_ms'] > 0)

    def test_wrong_earlier_repeat_not_masked_by_last(self):
        result = self.run_module('eval.tests.adapters.intermittent', warmup=0, repeats=2)
        self.assertFalse(result['pass'])
        self.assertIsNone(result['summary']['p50_ms'])
        self.assertEqual(result['summary']['measured_samples'], 0)

    def test_warmup_failure_is_reported(self):
        result = self.run_module('eval.tests.adapters.intermittent', warmup=1, repeats=2)
        self.assertFalse(result['pass'])
        self.assertIn('warmup 1', result['cases'][0]['failures'][0])

    def test_exception_is_reported(self):
        result = self.run_module('eval.tests.adapters.raises', warmup=1, repeats=1)
        self.assertFalse(result['pass'])
        self.assertIn('RuntimeError', str(result['cases'][0]['failures']))

    def test_import_error_is_reported(self):
        result = self.run_module('no_such_module', repeats=1)
        self.assertFalse(result['pass'])

    def test_timeout_is_reported(self):
        result = self.run_module('eval.tests.adapters.hangs', timeout=.5, repeats=1)
        self.assertFalse(result['pass'])
        self.assertIn('TimeoutError', str(result['cases'][0]['failures']))

    def test_worker_crash_is_reported(self):
        result = self.run_module('eval.tests.adapters.crashes', repeats=1)
        self.assertFalse(result['pass'])
        self.assertIn('Worker exited', str(result['cases'][0]['failures']))

    def test_no_warmup_not_comparable(self):
        result = self.run_module('eval.smoke_adapter', warmup=0, repeats=1)
        self.assertTrue(result['pass'])
        self.assertFalse(result['summary']['comparable'])

    def test_invalid_counts(self):
        for options in [{'warmup':-1}, {'repeats':0}, {'timeout':float('nan')}]:
            with self.assertRaises(ValueError):
                self.run_module('eval.smoke_adapter', **options)


if __name__ == '__main__':
    unittest.main()
