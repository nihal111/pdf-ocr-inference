import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.capture_reference import capture

GOOD = {'ocr_text': 'Hi\n', 'chunks': [{'id': 'c0', 'embedding': [1., 0.]}]}


class CaptureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / 'doc.pdf').write_bytes(b'%PDF-1.4\n')
        self.manifest = self.root / 'manifest.json'
        self.case = {'id': 'one', 'pdf': 'doc.pdf', 'reference': 'refs/one.json', 'pages': 1}
        self.save([self.case])

    def save(self, cases):
        self.manifest.write_text(json.dumps({'cases': cases}))

    def run_capture(self):
        return capture('my.baseline', self.manifest, self.root)

    @patch('eval.capture_reference.Worker')
    def test_captures_valid_baseline(self, worker):
        worker.return_value.call.return_value = {'output': GOOD}
        self.assertEqual(self.run_capture(), 1)
        self.assertEqual(json.loads((self.root / 'refs/one.json').read_text()), GOOD)
        worker.return_value.close.assert_called_once()

    @patch('eval.capture_reference.Worker')
    def test_refuses_existing_output_before_import(self, worker):
        target = self.root / 'refs/one.json'
        target.parent.mkdir()
        target.write_text('existing')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_capture()
        self.assertEqual(target.read_text(), 'existing')
        worker.assert_not_called()

    @patch('eval.capture_reference.Worker')
    def test_rejects_escaping_output_before_import(self, worker):
        self.save([{**self.case, 'reference': '../outside.json'}])
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.run_capture()
        worker.assert_not_called()

    @patch('eval.capture_reference.Worker')
    def test_no_outputs_written_when_later_call_fails(self, worker):
        self.save([self.case, {**self.case, 'id': 'two', 'reference': 'refs/two.json'}])
        worker.return_value.call.side_effect = [{'output': GOOD}, {'error': 'OCR failed'}]
        with self.assertRaisesRegex(ValueError, 'baseline failed'):
            self.run_capture()
        self.assertFalse((self.root / 'refs').exists())
        worker.return_value.close.assert_called_once()

    @patch('eval.capture_reference.Worker')
    def test_rejects_invalid_baseline_output(self, worker):
        worker.return_value.call.return_value = {'output': {**GOOD, 'chunks': []}}
        with self.assertRaisesRegex(ValueError, 'invalid baseline output'):
            self.run_capture()
        self.assertFalse((self.root / 'refs').exists())

    def test_rejects_smoke_adapter(self):
        with self.assertRaisesRegex(ValueError, 'not a baseline'):
            capture('eval.smoke_adapter', self.manifest, self.root)


if __name__ == '__main__':
    unittest.main()
