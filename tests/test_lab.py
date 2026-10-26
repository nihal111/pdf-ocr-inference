"""Harness tests on the mock engine (real nano-vllm scheduler, fake GPU). Run anywhere: `uv run pytest`."""
import json
from dataclasses import replace

from lab import bench
from lab.engine import MockEngine
from lab.sweep import expand
from lab.workloads import BUILTIN, resolve


def _run(wl, **engine_args):
    eng = MockEngine(**engine_args)
    t0, t1 = bench.run(eng, wl, seed_offset=1)
    return eng, bench.summarize(eng, wl, t0, t1, eng.info(), peak_gbps=300)


def test_workloads_deterministic_and_bounded():
    wl = BUILTIN["offline_mixed"]
    a, b = wl.generate(1), wl.generate(1)
    assert [r.prompt_ids for r in a] == [r.prompt_ids for r in b]
    assert a[0].prompt_ids != wl.generate(2)[0].prompt_ids
    assert all(100 <= len(r.prompt_ids) <= 1024 and 100 <= r.max_tokens <= 1024 for r in a)
    online = BUILTIN["online_poisson"].generate(1)
    arrivals = [r.arrival for r in online]
    assert arrivals == sorted(arrivals) and arrivals[-1] > 0


def test_resolve_overrides():
    w = resolve({"use": "shared_prefix", "name": "sp512", "shared_prefix_len": 512, "input_len": [8, 16]})
    assert (w.name, w.shared_prefix_len, w.input_len) == ("sp512", 512, (8, 16))


def test_offline_generates_every_token():
    wl = replace(BUILTIN["offline_mixed"], num_requests=64)
    eng, m = _run(wl)
    assert m["output_tokens"] == sum(r.max_tokens for r in wl.generate(1))
    assert all(r.finish is not None for r in eng.requests.values())
    assert m["num_prefill_steps"] > 0 and m["num_decode_steps"] > 0
    assert m["decode_bw_gbps"]["p50"] > 0


def test_online_latency_metrics():
    wl = replace(BUILTIN["online_poisson"], num_requests=50)
    _, m = _run(wl)
    assert m["num_requests"] == 50
    assert 0 < m["ttft_s"]["p50"] <= m["ttft_s"]["p99"]
    assert m["tpot_s"]["p50"] > 0 and 0 <= m["slo_attainment"] <= 1
    assert m["duration_s"] >= wl.generate(1)[-1].arrival


def test_prefix_caching_toggle():
    wl = replace(BUILTIN["shared_prefix"], num_requests=32)
    _, on = _run(wl, enable_prefix_caching=True)
    _, off = _run(wl, enable_prefix_caching=False)
    assert off["prefill_compute_ratio"] >= 0.99
    assert on["prefill_compute_ratio"] < 0.5        # 2048-token prefix mostly served from cache


def test_small_kv_cache_preempts():
    wl = replace(BUILTIN["decode_heavy"], num_requests=64)
    _, big = _run(wl, num_kvcache_blocks=1000)
    _, small = _run(wl, num_kvcache_blocks=40)
    assert big["preemptions"] == 0 and small["preemptions"] > 0
    assert small["prefill_compute_ratio"] > 1.0     # recompute after preemption


def test_batch_cap_respected():
    wl = replace(BUILTIN["decode_heavy"], num_requests=64)
    eng, m = _run(wl, max_num_seqs=16)
    assert max(s[2] for s in eng.steps) <= 16 and m["mean_decode_batch"] <= 16


def test_chunked_prefill_budget():
    wl = replace(BUILTIN["prefill_heavy"], num_requests=8)
    eng, _ = _run(wl, max_num_batched_tokens=1024)
    assert max(s[3] for s in eng.steps if s[1]) <= 1024


def test_sweep_expand_grid_and_merge():
    exp = {"base": {"engine": {"max_model_len": 4096, "enforce_eager": False}},
           "variants": {"baseline": {}, "eager": {"engine": {"enforce_eager": True}}},
           "grid": {"engine.max_num_seqs": [8, 16]}}
    v = expand(exp)
    assert list(v) == ["baseline", "eager", "max_num_seqs=8", "max_num_seqs=16"]
    assert v["eager"]["engine"] == {"max_model_len": 4096, "enforce_eager": True}
    assert v["max_num_seqs=16"]["engine"]["max_num_seqs"] == 16


def test_run_cli_mock(tmp_path, capsys):
    from lab.run import main
    out = tmp_path / "r.jsonl"
    main(["--engine", "mock", "--workloads", "smoke", '{"use": "online_poisson", "num_requests": 10}',
          "--repeats", "2", "--out", str(out), "--steps-dir", str(tmp_path / "steps")])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 4 and {r["workload"] for r in rows} == {"smoke", "online_poisson"}
    assert len(list((tmp_path / "steps").iterdir())) == 4
    from lab.compare import main as compare
    compare([str(out)])
    assert "| default |" in capsys.readouterr().out
