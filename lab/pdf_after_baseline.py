"""Wait for a completed evaluator baseline, validate it, then run diagnostic profiles."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.compare import compare
from eval.evaluate import load_cases
from lab.site_data import Catalog

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked_baseline(directory):
    lifecycle = read(directory / "run-status.json")
    if lifecycle.get("phase") != "finished" or lifecycle.get("status") != "passed":
        raise ValueError("The baseline must finish and pass before profiling starts.")
    report = read(directory / "baseline.json")
    spec = yaml.safe_load((directory / "experiment.yaml").read_text())
    expected = read(ROOT / spec["manifest"])["cases"]
    configuration = report["configuration"]
    warmup, repeats = spec["warmup"], spec["repeats"]
    if configuration["warmup_per_case"] != warmup or configuration["repeats_per_case"] != repeats:
        raise ValueError("Call counts differ from the frozen experiment.")
    if not report["pass"] or configuration["minimum_cosine"] < .99:
        raise ValueError("The evaluator's correctness checks must pass.")
    if repeats >= 5:
        # Exercise comparison preconditions without creating a fake comparison.
        compare(report, report)
    if ([case["id"] for case in report["cases"]] != [case["id"] for case in expected]
            or report["summary"]["measured_samples"] != len(expected) * repeats):
        raise ValueError("The report differs from the selected documents or measured-call count.")
    for case in report["cases"]:
        warm = [r for r in case["runs"] if r["phase"] == "warmup"]
        measured = [r for r in case["runs"] if r["phase"] == "measured"]
        if len(warm) != warmup or len(measured) != repeats or not all(r["pass"] for r in case["runs"]):
            raise ValueError(f"Invalid completed call pattern: {case['id']}")
    return report


def publish_baseline(report, directory, note=None):
    warmup = report["configuration"]["warmup_per_case"]
    timing, warnings = Catalog(ROOT).traces(directory / "baseline.trace.jsonl", warmup=warmup)
    if warnings or timing["calls"] != report["summary"]["measured_samples"]:
        raise ValueError("Measured stage records differ from the evaluator report.")
    summary = report["summary"]
    note = note or ROOT / "docs/experiments/001-baseline.md"
    text = note.read_text().split("## Current status")[0].replace("status: running", "status: passed")
    text += "## Results\n\n"
    count = len(report["cases"])
    text += f"All **{count} documents passed**, including all {count * warmup} warm-ups and **{timing['calls']} measured calls**. "
    text += f"Completed at {report['created_at']}.\n\n"
    if not summary["comparable"]:
        text += "This initial measurement has one timed sample per document. It does not qualify for an official speedup comparison, which requires at least five measured repeats per PDF.\n\n"
    text += "| Metric | Result |\n| --- | ---: |\n"
    text += f"| Pooled document latency p50 | {summary['p50_ms'] / 1000:.3f} s |\n"
    text += f"| Pooled document latency p95 | {summary['p95_ms'] / 1000:.3f} s |\n"
    text += f"| Sequential throughput | {summary['pages_per_minute_sequential']:.3f} pages/min |\n\n"
    text += "| Document | p50 (s) | p95 (s) | Minimum cosine |\n| --- | ---: | ---: | ---: |\n"
    for case in report["cases"]:
        text += f"| {case['id']} | {case['p50_ms']/1000:.3f} | {case['p95_ms']/1000:.3f} | {case['min_cosine']:.8f} |\n"
    text += "\n## Stage evidence\n\nStage shares use mean stage time divided by mean total time, excluding warm-ups.\n\n"
    text += "| Stage | Mean duration per document (s) | Share |\n| --- | ---: | ---: |\n"
    for stage in timing["stages"]:
        text += f"| {stage['name']} | {stage['mean_ms']/1000:.3f} | {stage['share']:.2%} |\n"
    text += f"\nThe largest stage is **{timing['dominant_stage']}**. "
    text += "These timings identify where wall time is spent; operator traces are needed to explain the operations underneath.\n\n"
    text += "## Next step\n\nProfile the unchanged four-GPU pipeline on the three-page NACA scan, collecting an OCR trace from each active page worker. "
    text += "Diagnostic timings remain separate from evaluator latency. No optimization or speedup is claimed.\n"
    note.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--baseline-pid", type=int, required=True)
    parser.add_argument("--journal", default="docs/experiments/001-baseline.md")
    args = parser.parse_args()
    directory = (ROOT / "results" / args.baseline_run).resolve()
    if not directory.is_relative_to(ROOT / "results") or not (directory / "experiment.yaml").is_file():
        parser.error("A saved baseline run under results/ is required.")
    state_path = directory / "profiling-workflow.json"
    if state_path.exists():
        raise SystemExit("A profiling workflow already exists for this run.")
    state = {"status": "waiting", "phase": "baseline", "baseline_run": args.baseline_run,
             "required_work": ["Validate completed baseline", "Profile unchanged pipeline",
                               "Publish stage timings, CPU/GPU traces and bottleneck analysis"],
             "profiles": []}
    write(state_path, state)
    try:
        while True:
            lifecycle = read(directory / "run-status.json")
            if lifecycle["status"] in {"failed", "cancelled"}:
                raise RuntimeError("Baseline did not pass; inspect its report before profiling.")
            if lifecycle.get("phase") == "finished":
                break
            process_command = Path(f"/proc/{args.baseline_pid}/cmdline")
            if not process_command.exists() or b"lab.pdf_sweep" not in process_command.read_bytes():
                raise RuntimeError("Baseline process exited before publishing a completed report.")
            time.sleep(10)
        report = checked_baseline(directory)
        spec = yaml.safe_load((directory / "experiment.yaml").read_text())
        cases = load_cases(ROOT / spec["manifest"], ROOT / spec["root"])
        identity = [{key: case[key] for key in ("id", "pages", "pdf_sha256", "reference_sha256")}
                    for case in cases]
        if identity != report["assets"]:
            raise ValueError("Dataset or references changed after baseline evaluation.")
        for relative, digest in report["environment"]["source_files_sha256"].items():
            if relative.startswith("document_pipeline/"):
                import hashlib
                if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Pipeline source changed since baseline: {relative}")
        quality_path = directory / "quality-review.json"
        if quality_path.exists() and read(quality_path).get("status") != "complete":
            raise ValueError("Complete the OCR quality review before profiling.")
        note = (ROOT / args.journal).resolve()
        if not note.is_relative_to(ROOT / "docs/experiments"):
            raise ValueError("The journal must be under docs/experiments.")
        publish_baseline(report, directory, note)
        variant = spec["_expanded"]["baseline"]
        environment = {**os.environ, **variant.get("env", {}), "PYTHONUNBUFFERED": "1"}
        expected_gpus = report["environment"]["gpu_uuids"]
        environment["CUDA_VISIBLE_DEVICES"] = expected_gpus
        pipeline = json.dumps(variant["pipeline"])
        # Stage evidence covers every measured call. Trace the complete historical
        # scan on the same four-GPU group used by the evaluator.
        selected = ["naca-seaplanes"]
        for identifier in selected:
            case = next(case for case in cases if case["id"] == identifier)
            state.update(status="running", phase="profiling", current_document=identifier)
            write(state_path, state)
            command = [sys.executable, "-m", "lab.pdf_prof", str(case["pdf_path"]),
                       "--config", pipeline, "--warmup", str(spec["warmup"]), "--calls", str(spec["repeats"]),
                       "--torch-ocr-forwards", "32",
                       "--label", "baseline", "--baseline-run", args.baseline_run]
            before = set((ROOT / "results/profiles").glob("*/profile.json"))
            with (directory / f"profile-{identifier}.log").open("x") as log:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
            added = set((ROOT / "results/profiles").glob("*/profile.json")) - before
            matching = [path for path in added if read(path).get("baseline_run") == args.baseline_run]
            if completed.returncode != 0 or len(matching) != 1 or read(matching[0])["status"] != "complete":
                raise RuntimeError(f"Profile failed for {identifier}; inspect its log.")
            record = read(matching[0])
            if (record["warmup"] != spec["warmup"] or record["calls"] != spec["repeats"]
                    or record["summary"]["calls"] != spec["repeats"]):
                raise ValueError("Profile measurement pattern differs from the frozen experiment.")
            state["profiles"].append(str(matching[0].parent.relative_to(ROOT / "results")))
            write(state_path, state)
        state.update(status="profiles_complete", phase="interpretation",
                     finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(state_path, state)


if __name__ == "__main__":
    main()
