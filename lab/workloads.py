"""Deterministic synthetic workloads.

Prompts are random token ids (like upstream nano-vllm's bench.py), so no dataset
download is needed and every engine/variant sees byte-identical requests.
All requests use ignore_eos so output lengths are fixed by the workload.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict, replace


@dataclass(frozen=True)
class Request:
    prompt_ids: list[int]
    max_tokens: int
    arrival: float = 0.0  # seconds after start; 0 for offline


@dataclass(frozen=True)
class Workload:
    name: str
    num_requests: int
    input_len: tuple[int, int]            # uniform [lo, hi], unique part of the prompt
    output_len: tuple[int, int]           # uniform [lo, hi]
    shared_prefix_len: int = 0            # tokens shared by every prompt (system prompt)
    rate: float | None = None             # req/s; None = offline (everything arrives at t=0)
    burstiness: float = 1.0               # gamma shape for inter-arrivals; 1.0 = Poisson
    temperature: float = 0.6
    seed: int = 0
    vocab: int = 10000
    slo_ttft_s: float | None = None       # optional SLOs for goodput
    slo_tpot_s: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def online(self) -> bool:
        return self.rate is not None

    def generate(self, seed_offset: int = 0) -> list[Request]:
        rng = random.Random(self.seed + seed_offset)
        prefix = [rng.randrange(self.vocab) for _ in range(self.shared_prefix_len)]
        reqs, t = [], 0.0
        for _ in range(self.num_requests):
            n_in = rng.randint(*self.input_len)
            prompt = prefix + [rng.randrange(self.vocab) for _ in range(n_in)]
            if self.online:
                # gamma(shape=k, scale=1/(k*rate)) has mean 1/rate; k<1 is burstier than Poisson
                t += rng.gammavariate(self.burstiness, 1.0 / (self.burstiness * self.rate))
            reqs.append(Request(prompt, rng.randint(*self.output_len), t if self.online else 0.0))
        return reqs

    def to_dict(self) -> dict:
        return asdict(self)


BUILTIN: dict[str, Workload] = {w.name: w for w in [
    # Same shape as upstream bench.py -> directly comparable to the README numbers.
    Workload("offline_mixed", 256, (100, 1024), (100, 1024), tags=("throughput",)),
    # Short prompts, long generations: decode-bound, exposes per-step CPU overhead and batch size.
    Workload("decode_heavy", 256, (64, 128), (512, 1024), tags=("decode",)),
    # Long prompts, tiny outputs: prefill/compute-bound, exposes chunking and attention kernels.
    Workload("prefill_heavy", 256, (1024, 3072), (16, 32), tags=("prefill",)),
    # One 2k-token system prompt shared by every request: measures prefix caching.
    Workload("shared_prefix", 256, (32, 128), (64, 128), shared_prefix_len=2048, tags=("prefix",)),
    # Online serving: Poisson arrivals. Tune `rate` to your GPU so the engine is busy but not drowning.
    Workload("online_poisson", 300, (100, 1024), (100, 512), rate=8.0,
             slo_ttft_s=1.0, slo_tpot_s=0.05, tags=("latency",)),
    Workload("online_bursty", 300, (100, 1024), (100, 512), rate=8.0, burstiness=0.3,
             slo_ttft_s=1.0, slo_tpot_s=0.05, tags=("latency",)),
    # Tiny smoke test.
    Workload("smoke", 8, (16, 64), (8, 16)),
]}


def resolve(spec: str | dict) -> Workload:
    """A workload spec is a builtin name, or a dict with optional `use: <builtin>` plus overrides."""
    if isinstance(spec, str):
        return BUILTIN[spec]
    spec = dict(spec)
    base = spec.pop("use", None)
    for k in ("input_len", "output_len"):
        if k in spec:
            spec[k] = tuple(spec[k])
    if base is None:
        return Workload(**spec)
    w = BUILTIN[base]
    spec.setdefault("name", w.name)
    return replace(w, **spec)
