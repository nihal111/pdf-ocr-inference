"""Loopback server telemetry and bounded recognizer traces for Paddle/vLLM."""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.request import Request, urlopen
from uuid import uuid4


def server_request(port, route, body=None):
    payload = None if body is None else json.dumps(body).encode()
    request = Request(f"http://127.0.0.1:{port}/{route}", data=payload,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _sampled_metrics(text):
    names = {
        "vllm:num_requests_running", "vllm:num_requests_waiting",
        "vllm:num_requests_waiting_by_reason", "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total", "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
    }
    metrics = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        key, value = line.rsplit(" ", 1)
        name = key.split("{", 1)[0]
        if name in names or name.startswith(("vllm:iteration_tokens_total",
                                            "vllm:spec_decode_")):
            metrics[key] = float(value)
    return metrics


def _metric_total(metrics, name):
    return sum(value for key, value in metrics.items() if key.split("{", 1)[0] == name)


class VllmTelemetry:
    """Sample scheduler occupancy; this is not a peak FLOPS utilization metric."""
    def __init__(self, directory, cfg):
        self.directory, self.cfg = Path(directory), cfg
        self.stop = threading.Event()
        self.rows, self.errors, self.threads = [], [], []

    def __enter__(self):
        def sample(worker):
            with (self.directory / f"vllm-worker-{worker}-metrics.jsonl").open("x") as stream:
                while not self.stop.is_set():
                    try:
                        text = server_request(self.cfg.paddle_server_base_port + worker, "metrics").decode()
                        row = {"worker": worker, "observed_unix_s": time.time(),
                               "metrics": _sampled_metrics(text)}
                        self.rows.append(row)
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                    except Exception as exc:
                        self.errors.append(f"worker {worker}: {exc}")
                    self.stop.wait(.1)
        for worker in range(self.cfg.gpu_workers):
            thread = threading.Thread(target=sample, args=(worker,), daemon=True)
            thread.start()
            self.threads.append(thread)
        return self

    def __exit__(self, *ignored):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=65)

    def summary(self):
        workers = []
        for worker in range(self.cfg.gpu_workers):
            rows = [r for r in self.rows if r["worker"] == worker]
            running = [_metric_total(r["metrics"], "vllm:num_requests_running") for r in rows]
            waiting = [_metric_total(r["metrics"], "vllm:num_requests_waiting") for r in rows]
            result = {"worker": worker, "samples": len(rows),
                      "maximum_running_requests": max(running, default=0),
                      "maximum_waiting_requests": max(waiting, default=0)}
            if len(rows) >= 2:
                names = {
                    "drafts": "vllm:spec_decode_num_drafts_total",
                    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
                    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
                }
                counts = {key: _metric_total(rows[-1]["metrics"], name) -
                          _metric_total(rows[0]["metrics"], name)
                          for key, name in names.items()}
                if counts["drafts"] > 0 and counts["draft_tokens"] > 0 and \
                        0 <= counts["accepted_tokens"] <= counts["draft_tokens"]:
                    result["speculation"] = {
                        **counts,
                        "draft_acceptance_rate": counts["accepted_tokens"] / counts["draft_tokens"],
                        "mean_acceptance_length": 1 + counts["accepted_tokens"] / counts["drafts"],
                        "scope": "Counter changes between first and last telemetry snapshots "
                                 "of the ordinary diagnostic calls; warm-up and instrumented "
                                 "calls are excluded. Not evaluator samples.",
                    }
            workers.append(result)
        return {"workers": workers, "errors": self.errors,
                "interval_ms": 100, "notes": [
                    "Running requests are scheduler occupancy, not GPU compute utilization.",
                    "Raw Prometheus snapshots include cumulative token counters and iteration histograms."]}


def summarize_events(events):
    grouped = defaultdict(lambda: {"calls": 0, "cpu_total_us": 0.0, "device_total_us": 0.0,
                                   "self_cpu_us": None, "self_device_us": None})
    for event in events:
        if event.get("ph") != "X" or not isinstance(event.get("dur"), (float, int)):
            continue
        category = event.get("cat", "")
        if category not in {"kernel", "gpu_memcpy", "gpu_memset", "cpu_op", "user_annotation"}:
            continue
        key = f"{category} / {event.get('name', 'unknown')}"
        row = grouped[key]
        row["calls"] += 1
        row["device_total_us" if category in {"kernel", "gpu_memcpy", "gpu_memset"}
            else "cpu_total_us"] += event["dur"]
    return sorted(({"name": key, **row} for key, row in grouped.items()),
                  key=lambda row: -row["device_total_us"])


def profile(pipe, pdf, directory, max_iterations, reference=None):
    from eval.evaluate import check
    cfg = pipe.cfg.ocr
    active = pipe.ocr.stats["active_workers"]  # populated by the ordinary measured call
    if getattr(cfg, "paddle_region_dispatch", "page") == "all_gpus":
        active = sorted({gpu for worker in pipe.ocr.stats["workers"]
                         for batch in worker.get("batches", [])
                         for gpu in batch.get("recognizer_gpus", [])})
    prefix = "naca-" + uuid4().hex[:12]
    server_directory = Path(os.environ.get("DOCUMENT_PIPELINE_VLLM_SERVER_ARTIFACTS",
        str(Path(__file__).resolve().parents[1] / ".site/vllm-batching/servers"))).resolve()
    started = []
    try:
        for worker in active:
            server_request(cfg.paddle_server_base_port + worker, "start_profile",
                           {"profile_prefix": prefix, "max_iterations": max_iterations})
            started.append(worker)
        result = pipe(str(pdf))
    finally:
        for worker in started:
            server_request(cfg.paddle_server_base_port + worker, "stop_profile", {})
    captures, operators = [], []
    for worker in active:
        output = directory / f"worker-{worker}"
        output.mkdir()
        traces = list((server_directory / f"worker-{worker}-traces").glob(f"**/*{prefix}*.json.gz"))
        if not traces:
            raise RuntimeError(f"No vLLM trace saved for GPU {worker}.")
        kernel_count = 0
        for trace in traces:
            destination = output / trace.name
            shutil.copy2(trace, destination)
            with gzip.open(destination, "rt") as stream:
                events = json.load(stream)["traceEvents"]
            kernel_count += sum(e.get("cat") == "kernel" for e in events)
            for row in summarize_events(events):
                operators.append({**row, "name": f"GPU {worker} / {row['name']}"})
        captures.append({"worker": worker, "kernel_count": kernel_count,
                         "trace_files": [p.name for p in output.glob("*.json.gz")]})
    (directory / "operators.json").write_text(json.dumps(operators, indent=2) + "\n")
    (directory / "instrumented-output.json").write_text(json.dumps(result) + "\n")
    reference = (Path(reference) if reference else
                 Path(pdf).parent.parent / "refs" / (Path(pdf).stem + ".json"))
    checked = check(result, json.loads(reference.read_text())) if reference.is_file() else None
    return {"framework": "vllm/PyTorch", "scope": "first engine iterations per active recognizer",
            "maximum_engine_iterations": max_iterations, "gpu_workers": cfg.gpu_workers,
            "active_workers": active, "workers": captures, "record_shapes": True,
            "reference_path": str(reference),
            "instrumented_reference_check": checked, "notes": [
                "Each engine iteration may process several recognition regions.",
                "The complete PDF finishes after the bounded trace ends.",
                "Raw traces contain overlapping CPU ranges; self time is not computed.",
                "Profiling timings are separate from evaluator measurements."]}
