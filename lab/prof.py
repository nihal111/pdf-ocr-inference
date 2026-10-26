"""torch.profiler over a window of engine steps; writes a Chrome/Perfetto trace + kernel table.

    python -m lab.prof --workload decode_heavy --skip 200 --steps 20
    python -m lab.prof --workload prefill_heavy --skip 0 --steps 5 --engine-args '{"enforce_eager": true}'

Open the trace at https://ui.perfetto.dev. Each step is labelled `prefill n=..` / `decode n=..`.
With CUDA graphs on, decode kernels show up under a single graph launch; profile with
enforce_eager=true to attribute time to individual kernels, then confirm wins with graphs on.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from lab.workloads import resolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--engine-args", default="{}")
    p.add_argument("--workload", default="decode_heavy")
    p.add_argument("--skip", type=int, default=100, help="steps to run before profiling starts")
    p.add_argument("--steps", type=int, default=20, help="steps to profile")
    p.add_argument("--stack", action="store_true", help="record Python stacks (slower, bigger trace)")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "traces"))
    args = p.parse_args(argv)

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from lab import bench
    from lab.engine import LabEngine

    wl = resolve(json.loads(args.workload) if args.workload.lstrip().startswith("{") else args.workload)
    engine = LabEngine(args.model, **json.loads(args.engine_args))
    run_inner = engine._run

    def labelled_run(seqs, is_prefill):
        with record_function(f"{'prefill' if is_prefill else 'decode'} n={len(seqs)}"):
            return run_inner(seqs, is_prefill)
    engine._run = labelled_run

    try:
        bench.warmup(engine, wl)
        engine.reset_stats()
        for r in wl.generate(seed_offset=1):
            engine.add_request(r.prompt_ids, r.max_tokens, wl.temperature)
        for _ in range(args.skip):
            if engine.is_finished():
                break
            engine.step()
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=args.stack)
        with prof:
            for _ in range(args.steps):
                if engine.is_finished():
                    break
                engine.step()
            torch.cuda.synchronize()
    finally:
        engine.exit()

    os.makedirs(args.out, exist_ok=True)
    tag = f"{wl.name}-skip{args.skip}-n{args.steps}-{time.strftime('%Y%m%d-%H%M%S')}"
    trace = os.path.join(args.out, tag + ".json")
    prof.export_chrome_trace(trace)
    ka = prof.key_averages()
    for key in ("device_time_total", "cuda_time_total"):
        try:
            table = ka.table(sort_by=key, row_limit=30)
            break
        except Exception:
            continue
    with open(os.path.join(args.out, tag + ".txt"), "w") as f:
        f.write(table)
    print(table)
    profiled = engine.steps[args.skip:args.skip + args.steps]
    kinds = {"prefill": sum(1 for s in profiled if s[1]), "decode": sum(1 for s in profiled if not s[1])}
    print(f"\nprofiled steps: {kinds}; mean batch {sum(s[2] for s in profiled) / max(1, len(profiled)):.1f}")
    print(f"trace: {trace}  (open in https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
