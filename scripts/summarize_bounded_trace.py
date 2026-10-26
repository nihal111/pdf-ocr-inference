#!/usr/bin/env python3
"""Attribute bounded trace kernels to their CPU launch ranges via correlation IDs."""
import argparse
from collections import Counter
import json
from pathlib import Path


def union_duration(intervals):
    end, total = float("-inf"), 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(path):
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError("Use the streaming trace summarizer for traces larger than 512 MiB.")
    with path.open() as stream:
        events = json.load(stream)["traceEvents"]
    ranges = [e for e in events if e.get("cat") == "user_annotation"
              and e.get("ph") == "X" and e["name"] in {"ocr::vision", "ocr::prefill", "ocr::decode"}]
    launches = {}
    for event in events:
        if event.get("cat") != "cuda_runtime" or "correlation" not in event.get("args", {}):
            continue
        for phase in ranges:
            if (event.get("pid"), event.get("tid")) == (phase.get("pid"), phase.get("tid")) and \
                    phase["ts"] <= event["ts"] < phase["ts"] + phase["dur"]:
                launches[event["args"]["correlation"]] = phase["name"]
                break
    kernels, phases, intervals = {}, {}, []
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        phase = launches.get(event.get("args", {}).get("correlation"), "outside_labeled_ranges")
        interval = (event["ts"], event["ts"] + event["dur"])
        intervals.append(interval)
        record = phases.setdefault(phase, {"kernel_calls": 0, "sum_kernel_us": 0, "intervals": []})
        record["kernel_calls"] += 1
        record["sum_kernel_us"] += event["dur"]
        record["intervals"].append(interval)
        key = (phase, event["name"])
        kernel = kernels.setdefault(key, {"phase": phase, "name": event["name"], "calls": 0, "sum_us": 0})
        kernel["calls"] += 1
        kernel["sum_us"] += event["dur"]
    for phase, record in phases.items():
        selected = record.pop("intervals")
        record["gpu_busy_union_us"] = union_duration(selected)
        record["cpu_range_us"] = sum(e["dur"] for e in ranges if e["name"] == phase)
        record["cpu_range_count"] = sum(e["name"] == phase for e in ranges)
        record["gemv_sum_us"] = sum(k["sum_us"] for k in kernels.values()
                                    if k["phase"] == phase and "gemv" in k["name"].lower())
    span = max(end for _, end in intervals) - min(start for start, _ in intervals) if intervals else 0
    busy = union_duration(intervals)
    return {
        "source": path.name, "source_bytes": path.stat().st_size,
        "method": "GPU kernels linked to CPU launch ranges by CUDA correlation ID.",
        "notes": ["CPU range wall time includes host work, waits and instrumentation.",
                  "Kernel sums may overlap; union time counts overlapping intervals once.",
                  "Busy fraction covers this short instrumented sample, not full-PDF utilization.",
                  "Vision can precede the top-level prefill forward.",
                  "Kernel counters, not these durations, are needed for peak compute/bandwidth utilization."],
        "event_counts": dict(Counter(e.get("cat", "") for e in events)),
        "kernel_span_us": span, "gpu_busy_union_us": busy,
        "gpu_busy_fraction_of_span": busy / span if span else None,
        "phases": phases, "kernels": sorted(kernels.values(), key=lambda k: -k["sum_us"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "kernels"}, indent=2))
