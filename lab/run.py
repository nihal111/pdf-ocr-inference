"""Run one engine configuration over one or more workloads; append one JSON line per run.

    python -m lab.run --model ~/huggingface/Qwen3-0.6B --workloads offline_mixed decode_heavy
    python -m lab.run --engine mock --workloads smoke            # no GPU needed
    python -m lab.run --engine-args '{"enforce_eager": true}' --variant eager ...

Usually invoked by lab.sweep, one subprocess per variant (fresh CUDA context, env vars
like TORCH_COMPILE_DISABLE apply cleanly, no state leaks between variants).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
import uuid

from lab.workloads import resolve


def make_engine(kind: str, model: str, engine_args: dict, nvtx=False, breakdown=False):
    if kind == "mock":
        from lab.engine import MockEngine
        return MockEngine(model, **engine_args)
    if kind == "nanovllm":
        from lab.engine import LabEngine
        return LabEngine(model, nvtx=nvtx, breakdown=breakdown, **engine_args)
    raise ValueError(kind)


def run_vllm(args, workloads, engine_args, emit):
    """Reference ceiling: same requests through vLLM's offline LLM API (use the separate .venv-vllm)."""
    from vllm import LLM, SamplingParams
    ea = dict(engine_args)
    if "kvcache_block_size" in ea:
        ea["block_size"] = ea.pop("kvcache_block_size")
    llm = LLM(model=os.path.expanduser(args.model), **ea)
    import vllm
    info = {"engine": "vllm", "vllm": vllm.__version__}
    for wl in workloads:
        if wl.online:
            print(f"[lab] skipping {wl.name}: online workloads need `vllm serve` + `vllm bench serve`", file=sys.stderr)
            continue
        llm.generate([{"prompt_token_ids": [1, 2, 3]}], SamplingParams(max_tokens=8), use_tqdm=False)
        for rep in range(args.repeats):
            reqs = wl.generate(seed_offset=rep + 1)
            prompts = [{"prompt_token_ids": r.prompt_ids} for r in reqs]
            sps = [SamplingParams(temperature=wl.temperature, ignore_eos=True, max_tokens=r.max_tokens) for r in reqs]
            t = time.perf_counter()
            outs = llm.generate(prompts, sps, use_tqdm=False)
            dur = time.perf_counter() - t
            out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
            prompt_tok = sum(len(r.prompt_ids) for r in reqs)
            emit(wl, rep, info, {"duration_s": dur, "num_requests": len(reqs), "output_tokens": out_tok,
                                 "prompt_tokens": prompt_tok, "output_tok_s": out_tok / dur,
                                 "total_tok_s": (out_tok + prompt_tok) / dur, "req_s": len(reqs) / dur}, None)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="nanovllm", choices=["nanovllm", "mock", "vllm"])
    p.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    p.add_argument("--engine-args", default="{}", help="JSON kwargs for the engine Config")
    p.add_argument("--workloads", nargs="+", default=["smoke"], help="builtin names or JSON dicts")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--experiment", default="adhoc")
    p.add_argument("--variant", default="default")
    p.add_argument("--out", default=None, help="results .jsonl (default: print to stdout)")
    p.add_argument("--steps-dir", default=None, help="write per-step traces here")
    p.add_argument("--peak-gbps", type=float, default=300.0, help="GPU HBM peak for bandwidth util (L4 ~300)")
    p.add_argument("--nvtx", action="store_true", help="NVTX range per step (for nsys)")
    p.add_argument("--breakdown", action="store_true", help="sync-timed prepare/forward/sample split")
    p.add_argument("--cuda-profiler-range", action="store_true",
                   help="cudaProfilerStart/Stop around the first measured run (nsys --capture-range=cudaProfilerApi)")
    args = p.parse_args(argv)

    engine_args = json.loads(args.engine_args)
    workloads = [resolve(json.loads(w) if w.lstrip().startswith("{") else w) for w in args.workloads]
    out = open(args.out, "a") if args.out else sys.stdout
    env_keys = ("TORCH_COMPILE_DISABLE", "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE")

    def emit(wl, rep, info, metrics, steps):
        run_id = uuid.uuid4().hex[:10]
        row = {"run_id": run_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname(),
               "experiment": args.experiment, "variant": args.variant, "workload": wl.name, "repeat": rep,
               "engine": args.engine, "model": args.model, "engine_args": engine_args,
               "env": {k: os.environ[k] for k in env_keys if k in os.environ},
               "workload_spec": wl.to_dict(), "info": info, "metrics": metrics,
               "python": platform.python_version()}
        out.write(json.dumps(row) + "\n")
        out.flush()
        if steps is not None and args.steps_dir:
            os.makedirs(args.steps_dir, exist_ok=True)
            with open(os.path.join(args.steps_dir, f"{args.variant}__{wl.name}__r{rep}__{run_id}.json"), "w") as f:
                json.dump(steps, f)
        m = metrics
        print(f"[lab] {args.variant:>24} {wl.name:>16} r{rep}  {m['output_tok_s']:9.1f} out tok/s  "
              f"{m['duration_s']:7.2f}s", file=sys.stderr)

    if args.engine == "vllm":
        run_vllm(args, workloads, engine_args, emit)
        return

    from lab import bench
    engine = make_engine(args.engine, args.model, engine_args, args.nvtx, args.breakdown)
    profiled = False
    try:
        for wl in workloads:
            bench.warmup(engine, wl)
            for rep in range(args.repeats):
                prof = args.cuda_profiler_range and not profiled
                if prof:
                    import torch
                    torch.cuda.profiler.start()
                t0, t1 = bench.run(engine, wl, seed_offset=rep + 1)
                if prof:
                    torch.cuda.profiler.stop()
                    profiled = True
                info = engine.info()
                emit(wl, rep, info, bench.summarize(engine, wl, t0, t1, info, args.peak_gbps),
                     bench.steps_as_dicts(engine, t0))
    finally:
        engine.exit()
        if args.out:
            out.close()


if __name__ == "__main__":
    main()
