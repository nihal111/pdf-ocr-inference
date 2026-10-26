"""Low-rate device observations; these are activity measures, not peak utilization."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import threading
import time

FIELDS = {
    "index": "index", "uuid": "uuid", "utilization.gpu": "gpu_busy_pct",
    "utilization.memory": "memory_busy_pct", "memory.used": "memory_used_mib",
    "memory.total": "memory_total_mib", "power.draw": "power_w",
    "power.limit": "power_limit_w", "clocks.current.sm": "sm_clock_mhz",
    "clocks.current.memory": "memory_clock_mhz", "temperature.gpu": "temperature_c",
    "pstate": "pstate", "clocks_event_reasons.active": "clock_event_mask",
}


class GpuTelemetry:
    def __init__(self, directory, phase):
        self.path = Path(directory) / f"gpu-{phase}.jsonl"
        self.phase = phase
        self.process = None
        self.thread = None
        self.rows = []
        self.errors = []

    def __enter__(self):
        if not shutil.which("nvidia-smi"):
            self.errors.append("nvidia-smi unavailable")
            return self
        command = ["nvidia-smi", "--query-gpu=" + ",".join(FIELDS),
                   "--format=csv,noheader,nounits", "--loop-ms=1000"]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as exc:
            self.errors.append(str(exc))
            return self

        def read():
            with self.path.open("x") as stream:
                for line in self.process.stdout:
                    values = next(csv.reader([line]))
                    if len(values) != len(FIELDS):
                        self.errors.append(line.strip())
                        continue
                    row = {"observed_unix_s": time.time(), "phase": self.phase}
                    for key, value in zip(FIELDS.values(), values):
                        value = value.strip()
                        if key in {"uuid", "pstate", "clock_event_mask"}:
                            row[key] = value
                        else:
                            try:
                                row[key] = float(value)
                            except ValueError:
                                row[key] = None
                    self.rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()

        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *ignored):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.thread.join(timeout=5)
            error = self.process.stderr.read().strip()
            if error:
                self.errors.append(error)
            self.process.stdout.close()
            self.process.stderr.close()

    def summary(self, windows=None):
        rows = self.rows
        if windows is not None:
            rows = [row for row in rows if any(
                w["start_unix_s"] <= row["observed_unix_s"] <= w["end_unix_s"] for w in windows)]
        devices = []
        for uuid in sorted({row["uuid"] for row in rows}):
            selected = [row for row in rows if row["uuid"] == uuid]
            metrics = {}
            for key in FIELDS.values():
                values = [row[key] for row in selected if isinstance(row[key], (int, float))]
                if values and key != "index":
                    metrics[key] = {"mean": mean(values), "min": min(values), "max": max(values)}
            devices.append({"uuid": uuid, "index": int(selected[0]["index"]),
                            "samples": len(selected), "metrics": metrics,
                            "clock_event_masks": sorted({row["clock_event_mask"] for row in selected})})
        return {"phase": self.phase, "interval_ms": 1000, "devices": sorted(devices, key=lambda x: x["index"]),
                "errors": self.errors,
                "notes": ["GPU busy is the percentage of sampled time with one or more kernels executing.",
                          "Memory busy is the percentage of sampled time with device memory reads or writes.",
                          "Neither is percentage of peak compute or peak memory bandwidth.",
                          "One-second sampling is too coarse for individual kernels or short stages."]}
