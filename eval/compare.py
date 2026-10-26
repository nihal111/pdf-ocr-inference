#!/usr/bin/env python3
"""Compare a baseline and an optimized evaluator report run on the same workload."""
import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.evaluate import load_json, write_json

# Environment fields that must be filled in and identical in both reports.
MATCHING_ENVIRONMENT = ('hardware_id', 'gpu_model', 'gpu_count', 'driver_version', 'concurrency')


def compare(baseline, optimized):
    for name, report in [('baseline', baseline), ('optimized', optimized)]:
        if report.get('schema_version') != 1:
            raise ValueError(f'{name}: not an evaluate.py report')
        if not report.get('pass'):
            raise ValueError(f'{name}: report did not pass')
        config = report['configuration']
        if config.get('warmup_per_case', 0) < 1 or config.get('repeats_per_case', 0) < 5:
            raise ValueError(f'{name}: need at least one warm-up and five measured repeats per PDF')
        if config.get('minimum_cosine', 0) < .99:
            raise ValueError(f'{name}: minimum cosine must be at least 0.99')
        if not report['summary'].get('comparable'):
            raise ValueError(f'{name}: report is not comparable (synthetic data?)')
        for key in MATCHING_ENVIRONMENT:
            if not report.get('environment', {}).get(key):
                raise ValueError(f'{name}: environment.{key} is missing')

    if baseline['dataset_sha256'] != optimized['dataset_sha256']:
        raise ValueError('The reports used different PDFs or reference outputs')
    for key in ('warmup_per_case', 'repeats_per_case', 'minimum_cosine'):
        if baseline['configuration'][key] != optimized['configuration'][key]:
            raise ValueError(f'Settings differ: {key}')
    for key in MATCHING_ENVIRONMENT:
        if baseline['environment'][key] != optimized['environment'][key]:
            raise ValueError(f'Environments differ: {key}')
    if baseline['environment']['concurrency'] != 1:
        raise ValueError('Only sequential runs (concurrency 1) can be compared')

    # Speedup > 1 means the optimized version is faster.
    output = {'pass': True, 'dataset_sha256': baseline['dataset_sha256'], 'metrics': {}}
    for metric in ('p50_ms', 'p95_ms', 'pages_per_minute_sequential'):
        a, b = baseline['summary'][metric], optimized['summary'][metric]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (a, b)):
            raise ValueError(f'Invalid metric: {metric}')
        speedup = b / a if metric.startswith('pages') else a / b
        output['metrics'][metric] = {'baseline': a, 'optimized': b, 'speedup': speedup}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--optimized', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        result = compare(load_json(args.baseline), load_json(args.optimized))
    except (ValueError, KeyError, OSError) as exc:
        result = {'pass': False, 'error': str(exc)}
    write_json(args.out, result)
    print(result)
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
