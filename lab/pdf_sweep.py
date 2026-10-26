"""Run pipeline variants through the official evaluator and compare each one to the baseline.

    python -m lab.pdf_sweep experiments/pdf/01_embed.yaml --capture   # once: baseline -> references
    python -m lab.pdf_sweep experiments/pdf/01_embed.yaml             # evaluate every variant
    python -m lab.pdf_sweep experiments/pdf/01_embed.yaml --only baseline bf16 --repeats 5

Per variant this runs `eval/evaluate.py --module document_pipeline.adapter` in a fresh process with
DOCUMENT_PIPELINE_CONFIG set, writes <variant>.json (the evaluator report), <variant>.env.json and
<variant>.trace.jsonl (per-call stage timings), then runs eval.compare.compare(baseline, variant).
Official speedups come only from eval.compare. When it refuses (e.g. no GPU, too few repeats),
the table shows the raw ratio marked `unofficial` along with the reason.

YAML keys: name, manifest, root, module (default document_pipeline.adapter), warmup, repeats, min_cosine,
timeout, baseline (variant name, default "baseline"), base {pipeline: {...}, env: {...}},
variants {name: overrides}, grid {"pipeline.embed.batch_size": [...]}.
An optional baseline_report reuses a qualified evaluator report as the comparison
anchor. It is copied byte for byte with a digest; only the listed variants are run.
This keeps the original baseline fixed while allowing a fresh best-configuration control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from lab.sweep import expand

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(p: str) -> str:
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _quiet_run(cmd, env, log_path):
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _env_for(variant: dict, extra: dict) -> dict:
    env = {**os.environ, **{k: str(v) for k, v in (variant.get("env") or {}).items()}, **extra}
    env["DOCUMENT_PIPELINE_CONFIG"] = json.dumps(variant.get("pipeline") or {})
    return env


def _status(out_dir: str, status: str, phase: str, **details):
    path = os.path.join(out_dir, "run-status.json")
    with open(path + ".tmp", "w") as stream:
        json.dump({"status": status, "phase": phase,
                   "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"), **details}, stream, indent=2)
    os.replace(path + ".tmp", path)


def capture(exp: dict, variants: dict, out_dir: str) -> int:
    name = exp.get("baseline", "baseline")
    v = variants[name]
    cmd = [sys.executable, os.path.join(ROOT, "eval", "capture_reference.py"), "--module", exp.get("module", "document_pipeline.adapter"),
           "--manifest", _path(exp["manifest"]), "--root", _path(exp["root"]), "--timeout", str(exp.get("timeout", 900))]
    print(f"capturing references with `{name}` ...", file=sys.stderr)
    rc = _quiet_run(cmd, _env_for(v, {"DOCUMENT_PIPELINE_TRACE": os.path.join(out_dir, "capture.trace.jsonl"),
                                    "DOCUMENT_PIPELINE_PAGE_REVIEW": os.path.join(out_dir, "page-review"),
                                    "DOCUMENT_PIPELINE_OCR_DIAGNOSTICS": os.path.join(out_dir, "ocr-diagnostics")}),
                    os.path.join(out_dir, "capture.log"))
    if rc == 0:
        from lab.pdf_env import collect
        prov = {"captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "variant": name,
                "pipeline": v.get("pipeline") or {}, "env": v.get("env") or {},
                "environment": collect(v.get("pipeline"), _env_for(v, {}))}
        provenance_path = _path(exp.get("reference_provenance") or
                                os.path.join(exp["root"], "reference_provenance.json"))
        Path(provenance_path).parent.mkdir(parents=True, exist_ok=True)
        with open(provenance_path, "w") as f:
            json.dump(prov, f, indent=2)
        print(f"references written; provenance in {provenance_path}", file=sys.stderr)
    else:
        print(open(os.path.join(out_dir, "capture.log")).read()[-3000:], file=sys.stderr)
    return rc


def stage_summary(trace_path: str, warmup: int) -> dict:
    """Mean seconds per measured call for each stage (warm-up calls per PDF skipped)."""
    if not os.path.exists(trace_path):
        return {}
    seen: dict[str, int] = {}
    sums: dict[str, float] = {}
    n = 0
    for line in open(trace_path):
        row = json.loads(line)
        seen[row["pdf"]] = seen.get(row["pdf"], 0) + 1
        if seen[row["pdf"]] <= warmup:
            continue
        n += 1
        for k, v in {**row["stages_s"], "total": row["total_s"]}.items():
            sums[k] = sums.get(k, 0.0) + v
        for k, v in (row.get("ocr_stats") or {}).items():
            if isinstance(v, (int, float)):
                sums[f"ocr.{k}"] = sums.get(f"ocr.{k}", 0.0) + v
    return {k: v / n for k, v in sums.items()} if n else {}


def _summary_row(name, report, cmp, base_report, stages):
    s = report.get("summary") or {}
    cases = report.get("cases") or []
    cos = [c["min_cosine"] for c in cases if c.get("min_cosine") is not None]
    fails = [f for c in cases for f in c.get("failures", [])][:2] or ([report["error"]] if "error" in report else [])
    row = {"variant": name, "pass": report.get("pass", False), "p50_ms": s.get("p50_ms"), "p95_ms": s.get("p95_ms"),
           "pages_min": s.get("pages_per_minute_sequential"), "min_cos": min(cos) if cos else None,
           "speedup": None, "official": False, "note": "; ".join(fails)[:160]}
    if cmp is not None:
        if cmp.get("pass"):
            row["speedup"] = cmp["metrics"]["p50_ms"]["speedup"]
            row["official"] = True
        else:
            bs = (base_report or {}).get("summary") or {}
            if s.get("p50_ms") and bs.get("p50_ms"):
                row["speedup"] = bs["p50_ms"] / s["p50_ms"]
            row["note"] = (row["note"] + " | " if row["note"] else "") + f"compare: {cmp.get('error')}"
    total = stages.get("total")
    if total:
        top = sorted(((k, v) for k, v in stages.items() if k in ("render", "ocr", "chunk", "embed")),
                     key=lambda kv: -kv[1])
        row["stages"] = " ".join(f"{k} {v / total:.0%}" for k, v in top)
        gen, ocr_s = stages.get("ocr.gen_tokens"), stages.get("ocr")
        if gen and ocr_s:
            row["stages"] += f" ({gen / ocr_s:,.0f} gen tok/s)"
    return row


def print_table(rows):
    def f(x, nd=1):
        return "-" if x is None else (f"{x:,.{nd}f}" if isinstance(x, float) else str(x))
    print("\n| variant | pass | p50 ms | p95 ms | pages/min | p50 speedup | min cos | time split | notes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sp = "-" if r["speedup"] is None else f"{r['speedup']:.2f}×" + ("" if r["official"] else " (unofficial)")
        print(f"| {r['variant']} | {'✓' if r['pass'] else '✗'} | {f(r['p50_ms'])} | {f(r['p95_ms'])} | "
              f"{f(r['pages_min'])} | {sp} | {f(r['min_cos'], 4)} | {r.get('stages', '-')} | {r['note'] or ''} |")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("experiment")
    p.add_argument("--capture", action="store_true", help="capture references from the baseline variant first")
    p.add_argument("--capture-only", action="store_true",
                   help="capture references and page-review artifacts, then stop before evaluation")
    p.add_argument("--resume-run", help="evaluate an existing capture-only run after its quality review")
    p.add_argument("--only", nargs="*")
    p.add_argument("--repeats", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--list", action="store_true")
    p.add_argument("--out-root", default=os.path.join(ROOT, "results"))
    args = p.parse_args(argv)

    with open(args.experiment) as fh:
        exp = yaml.safe_load(fh)
    variants = expand(exp)
    base_name = exp.get("baseline", "baseline")
    external_baseline = exp.get("baseline_report")
    if external_baseline and (args.capture or args.capture_only or args.resume_run):
        p.error("A saved comparison baseline cannot be used for reference capture or capture resumption")
    if external_baseline and base_name in variants:
        p.error("The saved baseline name must be separate from the variants being measured")
    if base_name not in variants and not external_baseline:
        sys.exit(f"baseline variant {base_name!r} not in {list(variants)}")
    if args.list:
        for k, v in variants.items():
            print(f"{k}: {json.dumps(v)}")
        return 0
    selected = args.only or list(variants)
    unknown = set(selected) - set(variants)
    if unknown:
        sys.exit(f"unknown variants {sorted(unknown)}")
    if external_baseline:
        from eval.compare import compare
        baseline_bytes = Path(_path(external_baseline)).read_bytes()
        baseline_report = json.loads(baseline_bytes)
        compare(baseline_report, baseline_report)  # Require a qualified evaluator report.
    elif base_name not in selected:
        selected = [base_name, *selected]  # every comparison needs a fresh baseline from the same session
    else:
        selected = [base_name, *[s for s in selected if s != base_name]]

    saved_spec = {**exp, "_expanded": {k: variants[k] for k in selected}}
    if external_baseline:
        saved_spec["_baseline_source"] = {
            "path": external_baseline,
            "sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "note": "Previously measured baseline, copied byte for byte; not rerun in this sweep.",
        }
    if args.resume_run:
        if args.capture or args.capture_only or args.warmup is not None or args.repeats is not None:
            p.error("--resume-run uses the captured configuration and cannot override capture or call counts")
        directory = Path(args.resume_run).resolve()
        original = yaml.safe_load((directory / "experiment.yaml").read_text())
        if original != saved_spec:
            p.error("The experiment differs from the captured run; resume requires its frozen configuration")
        lifecycle = json.loads((directory / "run-status.json").read_text())
        if lifecycle.get("status") != "awaiting_review":
            p.error("Only a capture-only run awaiting review can be resumed")
        review = directory / "quality-review.json"
        if not review.exists() or json.loads(review.read_text()).get("status") != "complete":
            p.error("Complete the captured pages' quality-review.json before evaluation")
        if any((directory / f"{name}.json").exists() or
               (directory / f"{name}.trace.jsonl").exists() for name in selected):
            p.error("Evaluation artifacts already exist; refusing to overwrite or mix measurements")
        out_dir = str(directory)
    else:
        name = exp.get("name") or os.path.splitext(os.path.basename(args.experiment))[0]
        out_dir = os.path.join(args.out_root, name, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(out_dir, exist_ok=False)
        with open(os.path.join(out_dir, "experiment.yaml"), "w") as fh:
            yaml.safe_dump(saved_spec, fh, sort_keys=False)
        if external_baseline:
            Path(out_dir, "baseline-report.json").write_bytes(baseline_bytes)

    if args.capture or args.capture_only:
        _status(out_dir, "running", "reference capture")
        if capture(exp, variants, out_dir) != 0:
            _status(out_dir, "failed", "reference capture", reason="Reference capture failed; see capture.log.")
            return 1
        if args.capture_only:
            _status(out_dir, "awaiting_review", "reference capture finished")
            print(f"Reference review artifacts: {out_dir}")
            return 0

    from eval.compare import compare
    from eval.evaluate import load_json
    from lab.pdf_env import collect
    warmup = args.warmup if args.warmup is not None else exp.get("warmup", 1)
    repeats = args.repeats or exp.get("repeats", 5)
    reports, rows = ({base_name: baseline_report} if external_baseline else {}), []
    for vname in selected:
        _status(out_dir, "running", "evaluation", variant=vname)
        v = variants[vname]
        report_path = os.path.join(out_dir, f"{vname}.json")
        env_path = os.path.join(out_dir, f"{vname}.env.json")
        trace = os.path.join(out_dir, f"{vname}.trace.jsonl")
        child_env = _env_for(v, {"DOCUMENT_PIPELINE_TRACE": trace,
                                 "DOCUMENT_PIPELINE_PAGE_REVIEW": "",
                                 "DOCUMENT_PIPELINE_OCR_DIAGNOSTICS": ""})
        environment = collect(v.get("pipeline"), child_env)
        with open(os.path.join(out_dir, f"{vname}.pipeline.json"), "w") as fh:
            json.dump(v.get("pipeline") or {}, fh, indent=2)
        with open(env_path, "w") as fh:
            json.dump(environment, fh, indent=2)
        cmd = [sys.executable, os.path.join(ROOT, "eval", "evaluate.py"), "--module", exp.get("module", "document_pipeline.adapter"),
               "--manifest", _path(exp["manifest"]), "--root", _path(exp["root"]), "--out", report_path,
               "--warmup", str(warmup), "--repeats", str(repeats), "--min-cosine", str(exp.get("min_cosine", 0.99)),
               "--timeout", str(exp.get("timeout", 900)), "--environment", env_path]
        print(f"=== {vname}: {json.dumps(v.get('pipeline') or {})}", file=sys.stderr)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t = time.perf_counter()
        rc = _quiet_run(cmd, child_env, os.path.join(out_dir, f"{vname}.log"))
        report = load_json(report_path) if os.path.exists(report_path) else {"pass": False, "error": f"exit {rc}, no report"}
        reports[vname] = report
        cmp = None
        if vname != base_name:
            try:
                cmp = compare(reports[base_name], report)
            except (ValueError, KeyError, TypeError) as exc:
                cmp = {"pass": False, "error": str(exc)}
            with open(os.path.join(out_dir, f"{vname}.compare.json"), "w") as fh:
                json.dump(cmp, fh, indent=2)
        row = _summary_row(vname, report, cmp, reports.get(base_name), stage_summary(trace, warmup))
        elapsed = time.perf_counter() - t
        runtime = {"seconds": elapsed, "started_at": started_at,
                   "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "source": "monotonic_clock",
                   "scope": "Evaluator process, worker setup, warm-ups, measured calls, validation and comparison. "
                            "Excludes inference-server startup, environment inventory, reference capture and profiling."}
        with open(os.path.join(out_dir, f"{vname}.suite-runtime.json"), "w") as fh:
            json.dump(runtime, fh, indent=2)
        row["suite_runtime_s"] = elapsed
        rows.append(row)
        print(f"    {'pass' if row['pass'] else 'FAIL'}  p50 {row['p50_ms'] or float('nan'):.0f} ms  "
              f"({elapsed:.0f}s wall)  {row['note'] or ''}", file=sys.stderr)

    print(f"\nresults: {out_dir}")
    print_table(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    _status(out_dir, "passed" if all(r["pass"] for r in rows) else "failed", "finished")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
