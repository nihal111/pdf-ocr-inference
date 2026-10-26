#!/usr/bin/env python3
"""Check an embed_pdf module against reference outputs and time each call."""
import argparse
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_json(path):
    def reject_constant(value):
        raise ValueError(f'Non-finite JSON number: {value}')
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=reject_constant, object_pairs_hook=unique_keys)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', delete=False) as out:
        temp = Path(out.name)
        try:
            json.dump(value, out, indent=2, allow_nan=False, ensure_ascii=True)
            out.write('\n')
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_output(value):
    errors = []
    if not isinstance(value, dict):
        return ['Result must be a dict']
    text = value.get('ocr_text')
    if not isinstance(text, str):
        errors.append('ocr_text must be a string')
    else:
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            errors.append('ocr_text must be valid UTF-8')
    chunks = value.get('chunks')
    if not isinstance(chunks, list) or not chunks:
        return errors + ['chunks must be a non-empty list']
    ids = set()
    dimension = None
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            errors.append(f'Chunk {i} must be a dict')
            continue
        identifier = chunk.get('id')
        if not isinstance(identifier, str) or not identifier:
            errors.append(f'Chunk {i} ID must be a non-empty string')
        elif identifier in ids:
            errors.append(f'Chunk {i} duplicate ID')
        else:
            ids.add(identifier)
        vector = chunk.get('embedding')
        if not isinstance(vector, list) or not vector:
            errors.append(f'Chunk {i} embedding must be a non-empty list')
            continue
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            errors.append(f'Chunk {i} inconsistent vector dimension')
        try:
            valid = all(type(x) in (int, float) and math.isfinite(x) for x in vector)
        except (OverflowError, ValueError):
            valid = False
        if not valid:
            errors.append(f'Chunk {i} vector must contain finite numbers, excluding booleans')
        elif not any(vector):
            errors.append(f'Chunk {i} zero vector')
    return errors


def cos(a, b):
    # Scale by the largest element first so very large or small values don't overflow.
    if len(a) != len(b) or not a:
        return float('nan')
    aa, bb = max(abs(x) for x in a), max(abs(x) for x in b)
    if not aa or not bb:
        return float('nan')
    x, y = [v / aa for v in a], [v / bb for v in b]
    result = math.fsum(v * w for v, w in zip(x, y)) / (
        math.sqrt(math.fsum(v*v for v in x)) * math.sqrt(math.fsum(w*w for w in y)))
    return max(-1.0, min(1.0, result))


def validate_minimum(minimum):
    if type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 < minimum <= 1:
        raise ValueError('minimum cosine must be finite and in (0, 1]')


def check(got, ref, minimum=0.99):
    validate_minimum(minimum)
    ref_errors = validate_output(ref)
    if ref_errors:
        raise ValueError('Invalid reference: ' + '; '.join(ref_errors))
    failures = validate_output(got)
    sims = []
    if failures:
        return {'pass': False, 'failures': failures, 'similarities': sims}
    if got['ocr_text'].encode('utf-8') != ref['ocr_text'].encode('utf-8'):
        failures.append('OCR bytes differ')
    if len(got['chunks']) != len(ref['chunks']):
        failures.append('Chunk count mismatch')
    else:
        for i, (chunk, target) in enumerate(zip(got['chunks'], ref['chunks'])):
            if chunk['id'] != target['id']:
                failures.append(f'Chunk {i} ID/order mismatch')
            if len(chunk['embedding']) != len(target['embedding']):
                failures.append(f'Chunk {i} dimension mismatch')
                continue
            similarity = cos(chunk['embedding'], target['embedding'])
            sims.append(similarity)
            if similarity < minimum:
                failures.append(f'Chunk {i} cosine below {minimum}: {similarity}')
    return {'pass': not failures, 'failures': failures, 'similarities': sims}


def confined_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Dataset paths must be non-empty relative paths')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing file or path escapes dataset root: {relative}')
    return path


def load_cases(manifest, root):
    root = Path(root).resolve()
    data = load_json(manifest)
    if not isinstance(data, dict) or not isinstance(data.get('cases'), list) or not data['cases']:
        raise ValueError('Manifest must contain a non-empty cases list')
    cases, seen = [], set()
    for raw in data['cases']:
        if not isinstance(raw, dict):
            raise ValueError('Each manifest case must be an object')
        identifier = raw.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Case IDs must be non-empty and unique')
        seen.add(identifier)
        if type(raw.get('pages')) is not int or raw['pages'] < 1:
            raise ValueError(f'{identifier}: pages must be a positive integer')
        pdf = confined_file(root, raw.get('pdf'))
        reference_path = confined_file(root, raw.get('reference'))
        with pdf.open('rb') as stream:
            if not stream.read(1024).lstrip().startswith(b'%PDF-'):
                raise ValueError(f'{identifier}: missing PDF header')
        reference = load_json(reference_path)
        errors = validate_output(reference)
        if errors:
            raise ValueError(f'{identifier}: invalid reference: ' + '; '.join(errors))
        cases.append({**raw, 'pdf_path': str(pdf), 'reference_data': reference,
                      'pdf_sha256': sha256(pdf), 'reference_sha256': sha256(reference_path)})
    return cases


def percentile(values, q):
    if not values:
        return None
    arr = sorted(values)
    index = (len(arr)-1) * q
    lo = int(index)
    hi = min(lo+1, len(arr)-1)
    return arr[lo] + (arr[hi]-arr[lo]) * (index-lo)


def _worker(pipe, module):
    try:
        fn = getattr(importlib.import_module(module), 'embed_pdf')
        if not callable(fn):
            raise TypeError('embed_pdf must be callable')
        pipe.send({'ready': True})
        while True:
            path = pipe.recv()
            if path is None:
                return
            try:
                started = time.perf_counter()
                value = fn(path)
                elapsed = (time.perf_counter() - started) * 1000
                # Validate before serialising: json.dumps would turn tuples into lists.
                errors = validate_output(value)
                if errors:
                    raise ValueError('Invalid output: ' + '; '.join(errors))
                payload = json.dumps(value, allow_nan=False)
                pipe.send({'output_json': payload, 'latency_ms': elapsed})
            except Exception as exc:
                pipe.send({'error': f'{type(exc).__name__}: {exc}'})
    except BaseException as exc:
        try:
            pipe.send({'error': f'{type(exc).__name__}: {exc}'})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        pipe.close()


class Worker:
    def __init__(self, module, timeout):
        context = mp.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, module))
        self.timeout = timeout
        self.started = time.perf_counter()
        self.process.start()
        child.close()
        try:
            ready = self.receive()
            if not ready.get('ready'):
                raise RuntimeError(ready.get('error', 'Worker failed to initialize'))
        except BaseException:
            self.close()
            raise
        self.startup_ms = (time.perf_counter() - self.started) * 1000
        self.first_result_ms = None

    def receive(self):
        if not self.pipe.poll(self.timeout):
            raise TimeoutError(f'Worker exceeded {self.timeout:g}s timeout')
        try:
            return self.pipe.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError('Worker exited without a result') from exc

    def call(self, path):
        self.pipe.send(path)
        result = self.receive()
        if 'error' in result:
            return result
        if self.first_result_ms is None:
            self.first_result_ms = (time.perf_counter() - self.started) * 1000
        result['output'] = json.loads(result.pop('output_json'))
        return result

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(2)
        self.pipe.close()


def evaluate(module, manifest, root, warmup=1, repeats=5, minimum=.99, timeout=600,
             environment=None):
    validate_minimum(minimum)
    if type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be >=1 and warmup >=0')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    # Validate all inputs before importing the module under test.
    cases = load_cases(manifest, root)
    purpose = load_json(manifest).get('purpose', 'benchmark')
    synthetic = purpose == 'synthetic-harness-test' or module == 'eval.smoke_adapter'
    identity = [{k: c[k] for k in ('id', 'pages', 'pdf_sha256', 'reference_sha256')} for c in cases]
    dataset_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    results, latencies = [], []
    worker, fatal = None, None
    cold = {}
    try:
        try:
            worker = Worker(module, timeout)
        except Exception as exc:
            fatal = f'{type(exc).__name__}: {exc}'
        for case in cases:
            runs, failures, similarities = [], [], []
            if fatal:
                failures.append(f'Not run: {fatal}')
            else:
                for phase, count in (('warmup', warmup), ('measured', repeats)):
                    for repeat in range(count):
                        record = {'phase': phase, 'repeat': repeat + 1}
                        try:
                            response = worker.call(case['pdf_path'])
                            if 'error' in response:
                                record.update({'pass': False, 'failures': [response['error']]})
                            else:
                                checked = check(response['output'], case['reference_data'], minimum)
                                record.update({**checked, 'latency_ms': response['latency_ms']})
                                similarities.extend(checked['similarities'])
                        except Exception as exc:
                            fatal = f'{type(exc).__name__}: {exc}'
                            record.update({'pass': False, 'failures': [fatal]})
                        runs.append(record)
                        if not record['pass']:
                            failures.extend(f'{phase} {repeat+1}: {error}' for error in record['failures'])
                        if fatal:
                            break
                    if fatal:
                        break
            measured = [r['latency_ms'] for r in runs if r['phase'] == 'measured' and r.get('pass')]
            passed = not failures and len(measured) == repeats
            if passed:
                latencies.extend(measured)
            results.append({'id': case['id'], 'pass': passed, 'failures': failures,
                            'min_cosine': min(similarities, default=None), 'runs': runs,
                            'latency_ms': measured, 'p50_ms': percentile(measured, .5) if passed else None,
                            'p95_ms': percentile(measured, .95) if passed else None})
        if worker:
            cold = {'worker_startup_ms': worker.startup_ms,
                    'process_to_first_result_ms': worker.first_result_ms}
    finally:
        if worker:
            worker.close()

    # Aggregate numbers are only reported when every call passed.
    passed = all(c['pass'] for c in results)
    elapsed_s = sum(latencies) / 1000
    pages = sum(c['pages'] for c in cases) * repeats
    summary = {
        'p50_ms': percentile(latencies, .5) if passed else None,
        'p95_ms': percentile(latencies, .95) if passed else None,
        'pages_per_minute_sequential': 60 * pages / elapsed_s if passed and elapsed_s else None,
        'passing_cases': sum(c['pass'] for c in results),
        'total_cases': len(cases),
        'measured_samples': len(latencies) if passed else 0,
        # Whether compare.py will accept this report.
        'comparable': (passed and warmup >= 1 and repeats >= 5 and minimum >= .99
                       and not synthetic),
    }
    return {'schema_version': 1, 'pass': passed,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'purpose': 'synthetic-harness-test' if synthetic else purpose,
            'dataset_sha256': dataset_digest, 'assets': identity,
            'cases': results, 'cold_start': cold, 'summary': summary,
            'configuration': {'module': module, 'warmup_per_case': warmup,
                              'repeats_per_case': repeats, 'minimum_cosine': minimum,
                              'timeout_seconds': timeout},
            'environment': environment or {},
            'runtime': {'python': platform.python_version(), 'platform': platform.platform()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True, help='module that defines embed_pdf')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', required=True, help='directory the manifest paths are relative to')
    parser.add_argument('--out', required=True, help='where to write the JSON report')
    parser.add_argument('--warmup', type=int, default=1, help='warm-up calls per PDF')
    parser.add_argument('--repeats', type=int, default=5, help='measured calls per PDF')
    parser.add_argument('--min-cosine', type=float, default=.99)
    parser.add_argument('--timeout', type=float, default=600, help='seconds, for import and each call')
    parser.add_argument('--environment', help='JSON file describing the hardware and software for this run')
    args = parser.parse_args()
    try:
        result = evaluate(args.module, args.manifest, args.root, args.warmup, args.repeats,
                          args.min_cosine, args.timeout,
                          load_json(args.environment) if args.environment else None)
    except Exception as exc:
        result = {'schema_version': 1, 'pass': False, 'error': f'{type(exc).__name__}: {exc}'}
        write_json(args.out, result)
        print(result['error'], file=sys.stderr)
        return 2
    write_json(args.out, result)
    print(json.dumps(result['summary'], indent=2))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
