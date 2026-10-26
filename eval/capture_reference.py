#!/usr/bin/env python3
"""Run your baseline on every PDF in the manifest and save its outputs as the reference."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.evaluate import Worker, confined_file, load_json, validate_output, write_json


def capture(module, manifest, root, timeout=600):
    root = Path(root).resolve()
    if module == 'eval.smoke_adapter':
        raise ValueError('eval.smoke_adapter is a test fixture, not a baseline')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    data = load_json(manifest)
    if not isinstance(data, dict) or not isinstance(data.get('cases'), list) or not data['cases']:
        raise ValueError('Manifest must contain a non-empty cases list')
    if data.get('purpose') == 'synthetic-harness-test':
        raise ValueError('This is the test fixture manifest; use your own PDFs')
    cases, ids, destinations = [], set(), set()
    for case in data['cases']:
        if not isinstance(case, dict):
            raise ValueError('Each case must be an object')
        identifier = case.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError('Case IDs must be non-empty and unique')
        ids.add(identifier)
        if type(case.get('pages')) is not int or case['pages'] < 1:
            raise ValueError(f'{identifier}: pages must be a positive integer')
        pdf = confined_file(root, case.get('pdf'))
        with pdf.open('rb') as stream:
            if not stream.read(1024).lstrip().startswith(b'%PDF-'):
                raise ValueError(f'{identifier}: missing PDF header')
        relative = case.get('reference')
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f'{identifier}: reference must be a non-empty relative path')
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError(f'{identifier}: reference path escapes dataset root')
        if target.exists() or target in destinations:
            raise ValueError(f'{identifier}: reference already exists or is duplicated: {relative}')
        destinations.add(target)
        cases.append((identifier, pdf, target))
    for a in destinations:
        if any(a != b and (a in b.parents or b in a.parents) for b in destinations):
            raise ValueError('Reference output paths must not nest inside one another')
        if any(parent.exists() and not parent.is_dir() for parent in a.parents):
            raise ValueError('Reference output parent must be a directory')

    # Run everything first so a failure part-way through writes nothing.
    captured = []
    worker = Worker(module, timeout)
    try:
        for identifier, pdf, target in cases:
            response = worker.call(str(pdf))
            if 'error' in response:
                raise ValueError(f"{identifier}: baseline failed: {response['error']}")
            value = response['output']
            errors = validate_output(value)
            if errors:
                raise ValueError(f'{identifier}: invalid baseline output: ' + '; '.join(errors))
            captured.append((target, value))
    finally:
        worker.close()

    for target, _ in captured:
        if target.exists():
            raise ValueError(f'{target} was created during capture; not overwriting it')
    for target, value in captured:
        write_json(target, value)
    return len(captured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    try:
        count = capture(args.module, args.manifest, args.root, args.timeout)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(f'Wrote {count} reference outputs.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
