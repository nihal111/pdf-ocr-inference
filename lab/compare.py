"""Compare variants: median over repeats, % change vs the baseline variant, per workload.

    python -m lab.compare results/01_knobs/<ts>/results.jsonl
    python -m lab.compare results/01_knobs            # newest run in that dir
    python -m lab.compare a.jsonl b.jsonl --baseline eager --metrics output_tok_s ttft_s.p99
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import median

OFFLINE = ["output_tok_s", "duration_s", "mean_decode_batch", "decode_bw_gbps.p50", "cpu_sched_post_frac",
           "prefill_compute_ratio", "preemptions"]
ONLINE = ["output_tok_s", "ttft_s.p50", "ttft_s.p99", "tpot_s.p50", "tpot_s.p99", "itl_s.p99",
          "slo_attainment", "preemptions"]
# Direction in which a metric improves, for the +/- markers.
LOWER_IS_BETTER = ("duration_s", "ttft_s", "tpot_s", "itl_s", "e2e_s", "queue_s", "preemptions",
                   "cpu_sched_post_frac", "prefill_compute_ratio", "idle_time_s", "peak_mem_gb")


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _files(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "**", "results.jsonl"), recursive=True), key=os.path.getmtime)
            if found:
                out.append(found[-1])
        else:
            out.append(p)
    return out


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x == 0:
            return "0"
        return f"{x:.3g}" if abs(x) < 1 else f"{x:,.1f}"
    return str(x)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--baseline", default=None, help="variant to compare against (default: 'baseline' or first)")
    p.add_argument("--metrics", nargs="*")
    args = p.parse_args(argv)

    rows = [json.loads(line) for f in _files(args.paths) for line in open(f) if line.strip()]
    if not rows:
        print("no results")
        return
    by_wl: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    online = {}
    for r in rows:
        by_wl[r["workload"]][r["variant"]].append(r["metrics"])
        online[r["workload"]] = r["workload_spec"].get("rate") is not None

    for wl, variants in by_wl.items():
        metrics = args.metrics or (ONLINE if online[wl] else OFFLINE)
        names = list(variants)
        base = args.baseline if args.baseline in variants else ("baseline" if "baseline" in variants else names[0])
        med = {v: {m: (median(xs) if (xs := [x for x in (_get(r, m) for r in rs) if x is not None]) else None)
                   for m in metrics} for v, rs in variants.items()}
        print(f"\n### {wl}  (median of n repeats; Δ vs `{base}`)\n")
        print("| variant | n | " + " | ".join(metrics) + " |")
        print("|---|---|" + "---|" * len(metrics))
        for v in names:
            cells = []
            for m in metrics:
                x, b = med[v][m], med[base][m]
                cell = _fmt(x)
                if v != base and isinstance(x, (int, float)) and isinstance(b, (int, float)) and b:
                    d = (x - b) / abs(b) * 100
                    better = (d < 0) if m.split(".")[0] in LOWER_IS_BETTER else (d > 0)
                    cell += f" ({'+' if d >= 0 else ''}{d:.0f}%{' ✓' if better and abs(d) >= 2 else ''})"
                cells.append(cell)
            print(f"| {v} | {len(variants[v])} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
