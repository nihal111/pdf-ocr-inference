"""Profile one PDF; preserve per-call timings, configuration, operators and traces.

    python -m lab.pdf_prof data/real/pdfs/example.pdf --label baseline
    python -m lab.pdf_prof example.pdf --no-torch-profile
    python -m lab.pdf_prof example.pdf --cprofile

Each session gets its own directory under results/profiles/. View it in python -m lab.site.
Diagnostic timings do not count as evaluator speedups.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import re
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from eval.evaluate import sha256, write_json
from lab.site_data import stage_metrics

ROOT = Path(__file__).resolve().parents[1]


def paddle_operator_summary(events):
    """Keep GPU execution separate from CPU kernel-launch ranges."""
    operators, categories = {}, {}
    for event in events:
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
            continue
        category = event.get("cat", "")
        categories[category] = categories.get(category, 0) + 1
        gpu = category in {"Kernel", "Memcpy", "Memset"}
        prefix = "GPU kernel" if category == "Kernel" else "GPU memory" if gpu else category
        # Paddle appends each event's elapsed time to its display name. Remove
        # that suffix so repeated calls to the same operation aggregate together.
        operation = re.sub(r"\[\d+(?:\.\d+)?\s*(?:ns|us|ms|s)\]$", "", event.get("name", ""))
        name = f"{prefix} / {operation}"
        row = operators.setdefault(name, {"name": name, "calls": 0, "cpu_total_us": 0,
                                         "self_cpu_us": None, "device_total_us": 0,
                                         "self_device_us": 0})
        row["calls"] += 1
        row["device_total_us" if gpu else "cpu_total_us"] += event["dur"]
        if gpu:
            row["self_device_us"] += event["dur"]
    return sorted(operators.values(), key=lambda row: -row["device_total_us"]), categories


def paddle_profile(pipe, pdf, directory, max_ocr_forwards=32):
    """Trace layout and the first N native recognition forwards in a Paddle worker."""
    import paddle
    from paddle.profiler import Profiler, ProfilerTarget, RecordEvent

    model = pipe.ocr.model
    forward = model.forward
    vision = getattr(model, "visual", None)
    vision_forward = vision.forward if vision is not None else None
    capture = {"framework": "paddle", "scope": "layout and early region recognition",
               "maximum_ocr_forwards": max_ocr_forwards, "observed_ocr_forwards": 0,
               "captured_phases": {}, "captured_components": {},
               "record_shapes": False, "profile_memory": False,
               "notes": ["A recognition forward belongs to a detected region, not an entire page.",
                         "The first N forwards can span several short regions.",
                         "CPU ranges overlap; GPU rows are kernel durations, not wall time."]}
    prof = Profiler(targets=[ProfilerTarget.CPU, ProfilerTarget.GPU],
                    record_shapes=False, profile_memory=False,
                    on_trace_ready=lambda _: None)
    recording = False

    def stop():
        nonlocal recording
        if recording:
            paddle.device.cuda.synchronize()
            prof.stop()
            recording = False

    def observed_forward(*args, **kwargs):
        if not recording:
            return forward(*args, **kwargs)
        tokens = kwargs.get("input_ids")
        if tokens is None and args:
            tokens = args[0]
        length = tokens.shape[-1] if tokens is not None else None
        phase = "prefill" if length and length > 1 else "decode" if length == 1 else "forward"
        capture["captured_phases"][phase] = capture["captured_phases"].get(phase, 0) + 1
        with RecordEvent(f"ocr::{phase}"):
            output = forward(*args, **kwargs)
        capture["observed_ocr_forwards"] += 1
        if max_ocr_forwards and capture["observed_ocr_forwards"] >= max_ocr_forwards:
            stop()
        return output

    def observed_vision(*args, **kwargs):
        if not recording:
            return vision_forward(*args, **kwargs)
        capture["captured_components"]["vision"] = capture["captured_components"].get("vision", 0) + 1
        with RecordEvent("ocr::vision"):
            return vision_forward(*args, **kwargs)

    try:
        # PaddleX's compatibility setter wraps plain functions and injects self.
        # These closures already hold bound forwards; bypass that setter so the
        # diagnostic preserves the model's arguments and return behavior.
        object.__setattr__(model, "forward", observed_forward)
        if vision is not None:
            object.__setattr__(vision, "forward", observed_vision)
        prof.start()
        recording = True
        pipe(str(pdf))
    finally:
        stop()
        object.__setattr__(model, "forward", forward)
        if vision is not None:
            object.__setattr__(vision, "forward", vision_forward)
    prof.export(str(directory / "trace.json"), format="json")
    events = json.loads((directory / "trace.json").read_text()).get("traceEvents", [])
    operators, categories = paddle_operator_summary(events)
    capture["trace_categories"] = categories
    capture["generation_batches"] = pipe.ocr.stats.get("batches", [])
    write_json(directory / "operators.json", operators)
    return capture


def torch_profile(pipe, pdf, directory, max_ocr_forwards=32, reference=None):
    """Run the whole PDF, recording only the first N OCR forwards when bounded."""
    cfg = getattr(pipe, "cfg", None)
    if cfg and cfg.ocr.backend == "paddleocr" and cfg.ocr.paddle_rec_backend == "vllm-server":
        from lab.vllm_profile import profile
        return profile(pipe, pdf, directory, max_ocr_forwards, reference=reference)
    if hasattr(pipe.ocr, "profile_next_call"):
        pipe.ocr.profile_next_call(directory, max_ocr_forwards)
        pipe(str(pdf))
        workers, operators = [], []
        active = pipe.ocr.stats["active_workers"]
        for worker in active:
            worker_directory = directory / f"worker-{worker}"
            capture = json.loads((worker_directory / "capture.json").read_text())
            workers.append(capture)
            for event in json.loads((worker_directory / "operators.json").read_text()):
                operators.append({**event, "name": f"GPU {worker} / {event['name']}"})
        write_json(directory / "operators.json", sorted(operators, key=lambda event: -event["device_total_us"]))
        return {"scope": "first OCR forwards on each active GPU page worker",
                "maximum_ocr_forwards": max_ocr_forwards, "gpu_workers": pipe.cfg.ocr.gpu_workers,
                "active_workers": active,
                "record_shapes": False, "profile_memory": False, "workers": workers,
                "notes": ["Each active worker has its own CPU/GPU trace under worker-N/.",
                          "Workers without assigned pages are idle and have no OCR trace.",
                          "Operator durations overlap across workers and must not be summed as wall time.",
                          "The trace sample covers OCR; ordinary stage measurements cover the full pipeline."]}
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    options = {"activities": acts, "record_shapes": False, "profile_memory": False}
    capture = {"scope": "first OCR forwards" if max_ocr_forwards else "full PDF",
               "maximum_ocr_forwards": max_ocr_forwards,
               "record_shapes": False, "profile_memory": False,
               "observed_ocr_forwards": 0, "captured_phases": {}, "captured_components": {},
               "generation_batches": []}
    handles = []
    active_range = None
    vision_range = None
    model = getattr(pipe.ocr, "model", None)
    original_generate = getattr(model, "generate", None)
    if max_ocr_forwards:
        if model is None:
            raise ValueError("Bounded OCR tracing requires the Hugging Face backend; use --torch-ocr-forwards 0 for a full trace.")
        # Begin at the first forward to retain prefill and vision work. The pipeline
        # was warmed separately; profiler initialization overhead remains diagnostic.
        options["schedule"] = schedule(wait=0, warmup=0, active=max_ocr_forwards, repeat=1)
    prof = profile(**options)

    def observed_generate(*args, **kwargs):
        sequences = original_generate(*args, **kwargs)
        inputs = kwargs["input_ids"]
        generated = sequences.shape[-1] - inputs.shape[-1]
        limit = kwargs.get("max_new_tokens")
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        last_tokens = sequences[:, -1].tolist()
        capture["generation_batches"].append({
            "batch_size": sequences.shape[0], "input_tokens": inputs.shape[-1],
            "generated_sequence_length": generated, "max_new_tokens": limit,
            "ended_with_eos": [token in eos for token in last_tokens],
            "reached_length_limit": limit is not None and generated >= limit,
        })
        return sequences

    def before_forward(module, args, kwargs):
        nonlocal active_range
        if capture["observed_ocr_forwards"] >= max_ocr_forwards:
            return
        tokens = kwargs.get("input_ids")
        if tokens is None and args:
            tokens = args[0]
        length = tokens.shape[-1] if tokens is not None else None
        phase = "prefill" if length and length > 1 else "decode" if length == 1 else "forward"
        capture["captured_phases"][phase] = capture["captured_phases"].get(phase, 0) + 1
        active_range = record_function(f"ocr::{phase}")
        active_range.__enter__()

    def after_forward(module, args, output):
        nonlocal active_range
        capture["observed_ocr_forwards"] += 1
        if active_range is not None:
            active_range.__exit__(None, None, None)
            active_range = None
        if capture["observed_ocr_forwards"] <= max_ocr_forwards:
            prof.step()

    def before_vision(module, args):
        nonlocal vision_range
        if capture["observed_ocr_forwards"] < max_ocr_forwards:
            capture["captured_components"]["vision"] = capture["captured_components"].get("vision", 0) + 1
            vision_range = record_function("ocr::vision")
            vision_range.__enter__()

    def after_vision(module, args, output):
        nonlocal vision_range
        if vision_range is not None:
            vision_range.__exit__(None, None, None)
            vision_range = None

    try:
        if original_generate is not None:
            model.generate = observed_generate
        if max_ocr_forwards:
            handles.append(model.register_forward_pre_hook(before_forward, with_kwargs=True))
            handles.append(model.register_forward_hook(after_forward))
            # Current Transformers can compute image features before the top-level
            # prefill forward. Label the actual vision module independently.
            vision = getattr(getattr(model, "model", None), "visual", None)
            if vision is not None:
                handles.append(vision.register_forward_pre_hook(before_vision))
                handles.append(vision.register_forward_hook(after_vision))
        else:
            pipe._record_function = record_function
        with prof:
            pipe(str(pdf))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    finally:
        if original_generate is not None:
            model.generate = original_generate
        for handle in handles:
            handle.remove()
        if active_range is not None:
            active_range.__exit__(None, None, None)
        if vision_range is not None:
            vision_range.__exit__(None, None, None)
        pipe._record_function = None
    prof.export_chrome_trace(str(directory / "trace.json"))
    operators = [{"name": event.key, "calls": event.count,
                  "cpu_total_us": event.cpu_time_total, "self_cpu_us": event.self_cpu_time_total,
                  "device_total_us": getattr(event, "device_time_total", 0),
                  "self_device_us": getattr(event, "self_device_time_total", 0)}
                 for event in prof.key_averages()]
    sort_key = "device_total_us" if torch.cuda.is_available() else "cpu_total_us"
    write_json(directory / "operators.json", sorted(operators, key=lambda x: -x[sort_key]))
    return capture


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf")
    p.add_argument("--config", default=None, help="pipeline overrides (JSON or file)")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--calls", type=int, default=5, help="calls for the unprofiled stage table")
    p.add_argument("--label", default="baseline")
    p.add_argument("--baseline-run", default=None, help="related sweep directory, relative to results/")
    p.add_argument("--variant", default=None, help="variant within the related sweep")
    p.add_argument("--reference", help="reference JSON for the full vLLM profile output")
    p.add_argument("--cprofile", action="store_true", help="Python call profile instead of torch profiler")
    p.add_argument("--no-torch-profile", action="store_true", help="stage timings only")
    p.add_argument("--torch-ocr-forwards", type=int, default=32,
                   help="record the first N OCR model forwards; 0 records the full PDF")
    p.add_argument("--out", default=str(ROOT / "results/profiles"))
    args = p.parse_args(argv)
    if args.warmup < 0 or args.calls < 1 or args.torch_ocr_forwards < 0:
        p.error("--warmup and --torch-ocr-forwards must be >= 0; --calls must be >= 1")
    pdf = Path(args.pdf).expanduser().resolve()
    if not pdf.is_file():
        p.error(f"PDF does not exist: {pdf}")

    from document_pipeline.config import load_config
    from document_pipeline.pipeline import Pipeline
    from lab.pdf_env import collect
    from lab.gpu_telemetry import GpuTelemetry

    cfg = load_config(args.config)
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"{pdf.stem}-{args.label}")
    directory = Path(args.out) / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}-{uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    mode = ("cprofile" if args.cprofile else "stages" if args.no_torch_profile
            else "vllm" if cfg.ocr.backend == "paddleocr" and cfg.ocr.paddle_rec_backend == "vllm-server"
            else "paddle" if cfg.ocr.backend == "paddleocr" else "torch")
    if args.reference and (mode != "vllm" or not Path(args.reference).is_file()):
        p.error("--reference requires vLLM tracing and an existing reference JSON")
    record = {
        "schema_version": 1, "kind": "document-profile", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), "label": args.label,
        "pdf": {"name": pdf.name, "sha256": sha256(pdf)},
        "config": cfg.to_dict(), "environment": collect(cfg.to_dict()),
        "warmup": args.warmup, "calls": args.calls, "mode": mode,
        "baseline_run": args.baseline_run, "variant": args.variant, "summary": stage_metrics([]),
        "command": shlex.join([sys.executable, "-m", "lab.pdf_prof", *(argv if argv is not None else sys.argv[1:])]),
        "notes": ["Stage measurements exclude warm-up and the additional profiler call.",
                  "Stage shares use mean stage wall time / mean total wall time.",
                  "Profiling is diagnostic; use eval/compare.py for official speedups."],
    }
    write_json(directory / "profile.json", record)
    print(f"Artifacts: {directory}", flush=True)
    pipe = None
    try:
        pipe = Pipeline(cfg)
        record["init_s"] = pipe.init_s
        pipe.trace_path = None
        for _ in range(args.warmup):
            pipe(str(pdf))
        trace = directory / "stages.jsonl"
        pipe.trace_path = str(trace)
        server_telemetry = nullcontext(None)
        if cfg.ocr.backend == "paddleocr" and cfg.ocr.paddle_rec_backend == "vllm-server":
            from lab.vllm_profile import VllmTelemetry
            server_telemetry = VllmTelemetry(directory, cfg.ocr)
        with GpuTelemetry(directory, "measured") as telemetry, server_telemetry as servers:
            for _ in range(args.calls):
                pipe(str(pdf))
        if servers:
            record["vllm_telemetry"] = servers.summary()
        pipe.trace_path = None
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        record["summary"] = stage_metrics(rows)
        record["summary"]["pages"] = rows[0]["pages"]
        record["summary"]["chunks"] = rows[0]["chunks"]
        record["ocr_stats"] = [r.get("ocr_stats", {}) for r in rows]
        record["embed_stats"] = [r.get("embed_stats", {}) for r in rows]
        ocr_stage = "render + OCR" if cfg.render.placement == "page_workers" else "ocr"
        record["ocr_timing_stage"] = ocr_stage
        record["gpu_telemetry"] = {
            "measured": telemetry.summary(),
            "ocr": telemetry.summary([r["stage_windows"][ocr_stage] for r in rows
                                      if ocr_stage in r.get("stage_windows", {})]),
        }
        tokens = sum(r.get("ocr_stats", {}).get("gen_tokens", 0) for r in rows)
        seconds = sum(r["stages_s"][ocr_stage] for r in rows)
        record["summary"]["generated_tok_s"] = tokens / seconds if seconds else None
        record["summary"]["generated_tok_s_stage"] = ocr_stage
        write_json(directory / "profile.json", record)
        print(f"\n{pdf.name}: median {record['summary']['median_total_ms']:,.1f} ms")
        for stage in record["summary"]["stages"]:
            print(f"  {stage['name']:<10} {stage['median_ms']:10.1f} ms  {stage['share']:6.1%} of mean total")

        if args.cprofile:
            import cProfile
            import pstats
            prof = cProfile.Profile()
            prof.runcall(pipe, str(pdf))
            prof.dump_stats(str(directory / "python.prof"))
            with (directory / "operators.txt").open("w") as stream:
                pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(50)
        elif not args.no_torch_profile:
            with GpuTelemetry(directory, "instrumented") as instrumented:
                record["torch_capture"] = torch_profile(
                    pipe, pdf, directory, args.torch_ocr_forwards, reference=args.reference)
            record["gpu_telemetry"]["instrumented"] = instrumented.summary()
        record["status"] = "complete"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if pipe is not None:
                pipe.close()
        finally:
            write_json(directory / "profile.json", record)
    print(f"Profile saved: {directory / 'profile.json'}")


if __name__ == "__main__":
    main()
