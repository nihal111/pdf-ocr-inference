"""Instrumented engines.

`LabEngine` wraps the real nano-vllm engine (needs CUDA + flash-attn).
`MockEngine` drives the *real* nano-vllm Scheduler/BlockManager with a virtual clock
and a toy cost model, so harness and scheduler changes can be developed and tested
on a laptop. Mock numbers say nothing about GPU performance.

Both record, per step: kind, batch size, tokens, context tokens, and the time split
schedule / run (prepare + forward + sample + sync) / postprocess. Per request: arrival,
first-scheduled time and a timestamp for every output token.
"""
from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))

from nanovllm.engine.scheduler import Scheduler  # noqa: E402
from nanovllm.engine.sequence import Sequence  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402

# Field order of each entry in Recorder.steps.
STEP_FIELDS = ("t", "prefill", "num_seqs", "num_tokens", "ctx_tokens", "sched_s", "run_s", "post_s")


@dataclass
class RequestRecord:
    arrival: float
    num_prompt_tokens: int
    max_tokens: int
    first_scheduled: float | None = None
    token_times: list[float] = field(default_factory=list)
    finish: float | None = None


class Recorder:
    """Mixin: replaces LLMEngine.step with an instrumented copy. Requires self.scheduler and self._run."""

    def _lab_init(self, clock):
        self.clock = clock
        self.requests: dict[int, RequestRecord] = {}
        self.steps: list[tuple] = []
        self.num_preemptions = 0
        preempt = self.scheduler.preempt

        def counting_preempt(seq):
            self.num_preemptions += 1
            preempt(seq)
        self.scheduler.preempt = counting_preempt

    def reset_stats(self):
        self.requests.clear()
        self.steps.clear()
        self.num_preemptions = 0

    def add_request(self, prompt_ids: list[int], max_tokens: int, temperature: float = 0.6,
                    arrival: float | None = None) -> int:
        seq = Sequence(prompt_ids, SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=True))
        self.requests[seq.seq_id] = RequestRecord(self.clock() if arrival is None else arrival,
                                                  len(prompt_ids), max_tokens)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        t0 = self.clock()
        seqs, is_prefill = self.scheduler.schedule()
        t1 = self.clock()
        if is_prefill:
            num_tokens = sum(s.num_scheduled_tokens for s in seqs)
            ctx_tokens = sum(s.num_cached_tokens + s.num_scheduled_tokens for s in seqs)
        else:
            num_tokens = len(seqs)
            ctx_tokens = sum(len(s) for s in seqs)
        token_ids = self._run(seqs, is_prefill)
        t2 = self.clock()
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        t3 = self.clock()
        for seq in seqs:
            rec = self.requests[seq.seq_id]
            if rec.first_scheduled is None:
                rec.first_scheduled = t1
            while len(rec.token_times) < seq.num_completion_tokens:
                rec.token_times.append(t3)
            if seq.is_finished:
                rec.finish = t3
        self.steps.append((t0, is_prefill, len(seqs), num_tokens, ctx_tokens, t1 - t0, t2 - t1, t3 - t2))

    def is_finished(self):
        return self.scheduler.is_finished()

    def wait_until(self, t: float):
        """Idle until clock() >= t (online driver, engine has no work)."""
        while self.clock() < t:
            time.sleep(min(1e-3, max(0.0, t - self.clock())))


class LabEngine(Recorder):

    def __init__(self, model: str, nvtx: bool = False, breakdown: bool = False, **engine_kwargs):
        import torch
        from nanovllm import LLM
        self._torch = torch
        self._llm = LLM(os.path.expanduser(model), **engine_kwargs)
        self.scheduler = self._llm.scheduler
        self.model_runner = self._llm.model_runner
        self.nvtx = nvtx
        self.breakdown = breakdown
        self.breakdown_s = {"prepare": 0.0, "forward": 0.0, "sample": 0.0}
        if breakdown:
            self._install_breakdown()
        self._lab_init(time.perf_counter)

    def _run(self, seqs, is_prefill):
        if not self.nvtx:
            return self.model_runner.call("run", seqs, is_prefill)
        nvtx = self._torch.cuda.nvtx
        nvtx.range_push(f"{'prefill' if is_prefill else 'decode'} n={len(seqs)}")
        try:
            return self.model_runner.call("run", seqs, is_prefill)
        finally:
            nvtx.range_pop()

    def _install_breakdown(self):
        """Time ModelRunner phases with explicit syncs. Adds a little overhead; off by default."""
        mr, sync, acc = self.model_runner, self._torch.cuda.synchronize, self.breakdown_s

        def timed(fn, key):
            def wrapper(*args, **kwargs):
                sync()
                t = time.perf_counter()
                out = fn(*args, **kwargs)
                sync()
                acc[key] += time.perf_counter() - t
                return out
            return wrapper
        mr.prepare_prefill = timed(mr.prepare_prefill, "prepare")
        mr.prepare_decode = timed(mr.prepare_decode, "prepare")
        mr.run_model = timed(mr.run_model, "forward")
        mr.sampler = timed(mr.sampler, "sample")

    def reset_stats(self):
        super().reset_stats()
        for k in self.breakdown_s:
            self.breakdown_s[k] = 0.0
        self._torch.cuda.reset_peak_memory_stats()

    def info(self) -> dict:
        torch, cfg, mr = self._torch, self.model_runner.config, self.model_runner
        hf = cfg.hf_config
        tp = cfg.tensor_parallel_size
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        # Bytes a decode step must stream per rank: every weight except the embedding table
        # (only `bs` rows of it are gathered), plus K and V for every context token.
        weight_bytes = sum(p.numel() * p.element_size() for n, p in mr.model.named_parameters()
                           if "embed_tokens" not in n)
        kv_bytes_per_token = 2 * hf.num_hidden_layers * (hf.num_key_value_heads // tp) * head_dim * hf.dtype.itemsize
        try:
            import flash_attn
            fa = flash_attn.__version__
        except Exception:
            fa = None
        return {
            "engine": "nanovllm",
            "gpu": torch.cuda.get_device_name(0),
            "num_gpus_visible": torch.cuda.device_count(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "flash_attn": fa,
            "dtype": str(hf.dtype),
            "num_kvcache_blocks": cfg.num_kvcache_blocks,
            "kvcache_block_size": cfg.kvcache_block_size,
            "kv_capacity_tokens": cfg.num_kvcache_blocks * cfg.kvcache_block_size,
            "weight_bytes_per_rank": weight_bytes,
            "kv_bytes_per_token_per_rank": kv_bytes_per_token,
            "peak_mem_bytes": torch.cuda.max_memory_allocated(),
        }

    def exit(self):
        import atexit
        atexit.unregister(self._llm.exit)
        self._llm.exit()


class MockEngine(Recorder):
    """Real scheduler, fake GPU. Cost model (seconds) is deliberately simple:
    prefill: launch + tokens/prefill_tps + attention ~ ctx_tokens
    decode:  launch + weight-streaming floor + per-seq + KV-read ~ ctx_tokens
    """

    def __init__(self, model: str = "mock", *, max_num_batched_tokens: int = 16384, max_num_seqs: int = 512,
                 max_model_len: int = 4096, kvcache_block_size: int = 256, num_kvcache_blocks: int = -1,
                 gpu_memory_utilization: float = 0.9, kv_tokens_at_90pct: int = 160_000,
                 enable_prefix_caching: bool = True, enforce_eager: bool = False,
                 cost: dict | None = None, **_ignored):
        assert kvcache_block_size % 256 == 0
        if num_kvcache_blocks <= 0:
            # Roughly Qwen3-0.6B on a 24 GB L4: ~3 GB is weights/activations, the rest is KV.
            kv_tokens = kv_tokens_at_90pct * max(0.02, gpu_memory_utilization - 0.12) / 0.78
            num_kvcache_blocks = max(1, int(kv_tokens) // kvcache_block_size)
        cfg = SimpleNamespace(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, eos=-1,
                              kvcache_block_size=kvcache_block_size, num_kvcache_blocks=num_kvcache_blocks,
                              enable_prefix_caching=enable_prefix_caching, max_model_len=max_model_len)
        Sequence.block_size = kvcache_block_size
        self.config = cfg
        self.scheduler = Scheduler(cfg)
        self.cost = {"launch_s": 2e-3 if enforce_eager else 4e-4, "prefill_tps": 40000.0, "decode_floor_s": 4e-3,
                     "per_seq_s": 2e-5, "kv_read_s_per_token": 2e-9, "attn_s_per_token": 5e-9, **(cost or {})}
        self._now = 0.0
        self._rng = random.Random(0)
        self._lab_init(lambda: self._now)

    def _run(self, seqs, is_prefill):
        c = self.cost
        if is_prefill:
            n = sum(s.num_scheduled_tokens for s in seqs)
            ctx = sum(s.num_cached_tokens + s.num_scheduled_tokens for s in seqs)
            dt = c["launch_s"] + n / c["prefill_tps"] + ctx * c["attn_s_per_token"]
        else:
            ctx = sum(len(s) for s in seqs)
            dt = c["launch_s"] + c["decode_floor_s"] + len(seqs) * c["per_seq_s"] + ctx * c["kv_read_s_per_token"]
        self._now += dt
        return [self._rng.randrange(10000) for _ in seqs]

    def wait_until(self, t: float):
        self._now = max(self._now, t)

    def info(self) -> dict:
        cfg = self.config
        return {"engine": "mock", "gpu": None, "num_kvcache_blocks": cfg.num_kvcache_blocks,
                "kvcache_block_size": cfg.kvcache_block_size,
                "kv_capacity_tokens": cfg.num_kvcache_blocks * cfg.kvcache_block_size,
                "weight_bytes_per_rank": 1.2e9, "kv_bytes_per_token_per_rank": 112 * 1024, "cost": self.cost}

    def exit(self):
        pass
