#!/usr/bin/env python3
"""Stream a Chrome trace into kernel totals without loading its event tree."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import ijson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    before = args.trace.stat()
    categories = Counter()
    kernels = {}
    stages = []
    first_kernel, last_kernel = None, None
    trace_seconds = None
    started = time.monotonic()
    with args.trace.open("rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            category = event.get("cat", "")
            categories[category] += 1
            if event.get("ph") != "X":
                continue
            name, duration = event.get("name", ""), event.get("dur", 0)
            if category == "kernel":
                kernel = kernels.setdefault(name, {"name": name, "calls": 0, "total_us": 0})
                kernel["calls"] += 1
                kernel["total_us"] += duration
                start = event["ts"]
                first_kernel = start if first_kernel is None else min(first_kernel, start)
                end = start + duration
                last_kernel = end if last_kernel is None else max(last_kernel, end)
            elif category == "user_annotation" and name.startswith("document_pipeline::"):
                stages.append({"name": name, "elapsed_s": duration / 1e6})
            elif category == "Trace":
                trace_seconds = duration / 1e6
    after = args.trace.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("The trace changed while being read; retry after export completes.")
    result = {
        "source": {"name": args.trace.name, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
        "method": "Streaming sum of complete kernel-event durations grouped by exact kernel name.",
        "notes": ["Kernel durations may overlap; their sum is not wall time.",
                  "These are instrumented timings and must not replace evaluator latency.",
                  "Kernel totals are not attributed to individual pipeline stages."],
        "event_counts": dict(categories), "trace_elapsed_s": trace_seconds,
        "kernel_span_s": (last_kernel - first_kernel) / 1e6 if first_kernel is not None else None,
        "sum_kernel_s": sum(kernel["total_us"] for kernel in kernels.values()) / 1e6,
        "stage_ranges": stages,
        "kernels": sorted(kernels.values(), key=lambda kernel: -kernel["total_us"]),
        "analysis_s": time.monotonic() - started,
    }
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({key: value for key, value in result.items() if key != "kernels"}, indent=2))
    print(json.dumps(result["kernels"][:10], indent=2))


if __name__ == "__main__":
    main()
