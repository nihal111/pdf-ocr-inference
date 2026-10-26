"""Drivers (offline / online) and metric summaries."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from lab.engine import STEP_FIELDS
from lab.workloads import Workload


def warmup(engine, wl: Workload, n: int = 32):
    """Short offline run of the same shape (different seed, so no prefix-cache carry-over)."""
    w = replace(wl, num_requests=min(n, wl.num_requests), rate=None, seed=wl.seed + 10_000)
    run_offline(engine, w)
    engine.reset_stats()


def run_offline(engine, wl: Workload, seed_offset: int = 0) -> tuple[float, float]:
    reqs = wl.generate(seed_offset)
    engine.reset_stats()
    t0 = engine.clock()
    for r in reqs:
        engine.add_request(r.prompt_ids, r.max_tokens, wl.temperature, arrival=t0)
    while not engine.is_finished():
        engine.step()
    return t0, engine.clock()


def run_online(engine, wl: Workload, seed_offset: int = 0) -> tuple[float, float]:
    reqs = wl.generate(seed_offset)
    engine.reset_stats()
    t0 = engine.clock()
    i = 0
    while i < len(reqs) or not engine.is_finished():
        now = engine.clock()
        while i < len(reqs) and t0 + reqs[i].arrival <= now:
            engine.add_request(reqs[i].prompt_ids, reqs[i].max_tokens, wl.temperature, arrival=t0 + reqs[i].arrival)
            i += 1
        if not engine.is_finished():
            engine.step()
        elif i < len(reqs):
            engine.wait_until(t0 + reqs[i].arrival)
    return t0, engine.clock()


def run(engine, wl: Workload, seed_offset: int = 0):
    return (run_online if wl.online else run_offline)(engine, wl, seed_offset)


def _pct(xs, qs=(50, 90, 99)) -> dict:
    if len(xs) == 0:
        return {f"p{q}": None for q in qs} | {"mean": None}
    a = np.asarray(xs, dtype=float)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs} | {"mean": float(a.mean())}


def summarize(engine, wl: Workload, t0: float, t1: float, info: dict, peak_gbps: float | None = None) -> dict:
    recs = list(engine.requests.values())
    steps = engine.steps
    dur = t1 - t0
    out_tok = sum(len(r.token_times) for r in recs)
    prompt_tok = sum(r.num_prompt_tokens for r in recs)

    ttft = [r.token_times[0] - r.arrival for r in recs if r.token_times]
    queue = [r.first_scheduled - r.arrival for r in recs if r.first_scheduled is not None]
    tpot, itl = [], []
    for r in recs:
        tt = r.token_times
        if len(tt) > 1:
            tpot.append((tt[-1] - tt[0]) / (len(tt) - 1))
            itl.extend(np.diff(tt).tolist())
    e2e = [r.finish - r.arrival for r in recs if r.finish is not None]

    pre = [s for s in steps if s[1]]
    dec = [s for s in steps if not s[1]]
    step_s = lambda ss: sum(s[5] + s[6] + s[7] for s in ss)  # noqa: E731
    busy = step_s(steps)

    m = {
        "duration_s": dur,
        "num_requests": len(recs),
        "output_tokens": out_tok,
        "prompt_tokens": prompt_tok,
        "output_tok_s": out_tok / dur if dur else None,
        "total_tok_s": (out_tok + prompt_tok) / dur if dur else None,
        "req_s": len(recs) / dur if dur else None,
        "ttft_s": _pct(ttft), "tpot_s": _pct(tpot), "itl_s": _pct(itl), "e2e_s": _pct(e2e), "queue_s": _pct(queue),
        "num_steps": len(steps), "num_prefill_steps": len(pre), "num_decode_steps": len(dec),
        "prefill_time_s": step_s(pre), "decode_time_s": step_s(dec),
        "idle_time_s": max(0.0, dur - busy),
        # Host-side scheduler + postprocess as a share of busy time: pure CPU overhead.
        "cpu_sched_post_frac": sum(s[5] + s[7] for s in steps) / busy if busy else None,
        "mean_decode_batch": float(np.mean([s[2] for s in dec])) if dec else None,
        "mean_prefill_tokens": float(np.mean([s[3] for s in pre])) if pre else None,
        # > prompt_tokens means recompute (preemption); < prompt_tokens means prefix-cache hits.
        "prefill_tokens_computed": sum(s[3] for s in pre),
        "preemptions": engine.num_preemptions,
    }
    m["prefill_compute_ratio"] = m["prefill_tokens_computed"] / prompt_tok if prompt_tok else None
    if wl.slo_ttft_s is not None or wl.slo_tpot_s is not None:
        ok = 0
        for r in recs:
            tt = r.token_times
            if not tt:
                continue
            t_ok = wl.slo_ttft_s is None or tt[0] - r.arrival <= wl.slo_ttft_s
            p_ok = wl.slo_tpot_s is None or len(tt) < 2 or (tt[-1] - tt[0]) / (len(tt) - 1) <= wl.slo_tpot_s
            ok += t_ok and p_ok
        m["slo_attainment"] = ok / len(recs) if recs else None
        m["goodput_req_s"] = ok / dur if dur else None

    # Decode is memory-bound: estimate achieved HBM bandwidth per decode step
    # (weights + KV read) / run time. Compare against your GPU's peak (L4 ~300 GB/s).
    wb, kvb = info.get("weight_bytes_per_rank"), info.get("kv_bytes_per_token_per_rank")
    if dec and wb and kvb:
        bw = [(wb + s[4] * kvb) / s[6] / 1e9 for s in dec if s[6] > 0]
        m["decode_bw_gbps"] = _pct(bw)
        if peak_gbps:
            m["decode_bw_frac_of_peak"] = float(np.median(bw)) / peak_gbps
    if getattr(engine, "breakdown", False):
        m["breakdown_s"] = dict(engine.breakdown_s)
    if "peak_mem_bytes" in info:
        m["peak_mem_gb"] = info["peak_mem_bytes"] / 1e9
    return m


def steps_as_dicts(engine, t0: float) -> list[dict]:
    return [dict(zip(STEP_FIELDS, (s[0] - t0, *s[1:]))) for s in engine.steps]
