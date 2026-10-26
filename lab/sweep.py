"""Run an experiment file: every variant in its own subprocess, results under results/<name>/<timestamp>/.

    python -m lab.sweep experiments/01_knobs.yaml
    python -m lab.sweep experiments/01_knobs.yaml --only baseline eager --repeats 1
    python -m lab.sweep experiments/01_knobs.yaml --mock          # dry run on a laptop
    python -m lab.sweep experiments/01_knobs.yaml --list          # show expanded variants

Experiment file keys: name, model, engine (nanovllm|mock|vllm), repeats, workloads,
base {engine: {...}, env: {...}, model, flags: [--breakdown, ...]}, variants {name: overrides},
grid {"engine.max_num_seqs": [64, 128]} (cartesian product, each point becomes a variant).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _set_dotted(d: dict, key: str, value):
    *path, last = key.split(".")
    for k in path:
        d = d.setdefault(k, {})
    d[last] = value


def expand(exp: dict) -> dict[str, dict]:
    base = exp.get("base", {})
    variants = {name: deep_merge(base, v or {}) for name, v in (exp.get("variants") or {}).items()}
    grid = exp.get("grid") or {}
    if grid:
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            over: dict = {}
            for k, v in zip(keys, values):
                _set_dotted(over, k, v)
            name = ",".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, values))
            variants[name] = deep_merge(base, over)
    if not variants:
        variants["baseline"] = copy.deepcopy(base)
    return variants


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("experiment")
    p.add_argument("--only", nargs="*", help="run only these variants")
    p.add_argument("--repeats", type=int)
    p.add_argument("--workloads", nargs="*", help="override the workload list")
    p.add_argument("--mock", action="store_true", help="use the mock engine (no GPU)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--out-root", default=os.path.join(ROOT, "results"))
    args = p.parse_args(argv)

    with open(args.experiment) as f:
        exp = yaml.safe_load(f)
    variants = expand(exp)
    if args.only:
        missing = set(args.only) - set(variants)
        if missing:
            sys.exit(f"unknown variants: {sorted(missing)}; have {list(variants)}")
        variants = {k: v for k, v in variants.items() if k in args.only}
    if args.list:
        for name, v in variants.items():
            print(f"{name}: {json.dumps(v)}")
        return

    name = exp.get("name") or os.path.splitext(os.path.basename(args.experiment))[0]
    out_dir = os.path.join(args.out_root, name + ("-mock" if args.mock else ""), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.experiment, os.path.join(out_dir, "experiment.yaml"))
    # Provenance: which engine code produced these numbers.
    for fname, cmd in [("git_rev.txt", ["git", "rev-parse", "HEAD"]), ("engine.diff", ["git", "diff", "HEAD", "--", "engine"])]:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            open(os.path.join(out_dir, fname), "w").write(r.stdout)

    results = os.path.join(out_dir, "results.jsonl")
    workloads = args.workloads or exp.get("workloads") or ["smoke"]
    repeats = args.repeats or exp.get("repeats", 1)
    failed = []
    for vname, v in variants.items():
        kind = "mock" if args.mock else v.get("engine_type", exp.get("engine", "nanovllm"))
        cmd = [sys.executable, "-m", "lab.run", "--engine", kind,
               "--model", v.get("model", exp.get("model", "~/huggingface/Qwen3-0.6B")),
               "--engine-args", json.dumps(v.get("engine", {})),
               "--workloads", *[w if isinstance(w, str) else json.dumps(w) for w in workloads],
               "--repeats", str(repeats), "--experiment", name, "--variant", vname,
               "--out", results, "--steps-dir", os.path.join(out_dir, "steps"),
               *v.get("flags", [])]
        env = {**os.environ, **{k: str(x) for k, x in (v.get("env") or {}).items()}}
        print(f"\n=== {vname} ===\n{' '.join(cmd)}", file=sys.stderr)
        log = open(os.path.join(out_dir, f"{vname}.log"), "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            if line.startswith("[lab]") or "Error" in line or "Traceback" in line:
                sys.stderr.write(line)
        if proc.wait() != 0:
            failed.append(vname)
            print(f"!!! {vname} failed, see {log.name}", file=sys.stderr)
        log.close()

    print(f"\nresults: {results}", file=sys.stderr)
    if failed:
        print(f"failed variants: {failed}", file=sys.stderr)
    if os.path.exists(results):
        from lab.compare import main as compare
        compare([results])


if __name__ == "__main__":
    main()
