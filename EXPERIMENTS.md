# Experiments

**Working agreement (2026-10-26):** measure the baseline, then profile it before applying
iterative optimizations. Choose each hypothesis from the observed bottleneck. The ideas below
and the experiment YAMLs are a reference blueprint, not a fixed order or mandatory backlog.
Detailed evidence, artifacts, failures, and decisions belong in `docs/experiments/` and are
visible through the local notebook (`python -m lab.site`).

## Hardware and baseline

| | |
|---|---|
| GPU | Four NVIDIA L4s detected; CUDA driver initialization verified 2026-10-26. Pin one GPU per PDF experiment. |
| Engine | nano-vllm @ bb823b3 + lab hooks |
| torch / CUDA / flash-attn | _fill from results.jsonl `info`_ |

| workload | nano-vllm baseline out tok/s | vLLM out tok/s | gap |
|---|---|---|---|
| offline_mixed (Qwen3-0.6B) | | | |
| decode_heavy | | | |
| prefill_heavy | | | |

## Phase 1: what are the existing knobs worth? (no code changes)

| exp | question | how |
|---|---|---|
| 01 | value of CUDA graphs, torch.compile, prefix caching, KV size, block size | `01_knobs.yaml` |
| 02 | throughput vs decode batch size; where does the bandwidth-bound regime end? | `02_batch_size.yaml`. Plot `mean_decode_batch` vs `decode_bw_gbps` |
| 03 | prefill budget vs tail latency under load | `03_prefill_chunking.yaml`. Watch `itl_s.p99` and `ttft_s.p99` |
| 04 | prefix cache on a shared 2k system prompt | `04_prefix_cache.yaml`. Watch `prefill_compute_ratio` |
| 05 | how the bottleneck moves with model size | `05_models.yaml` |
| 06 | TP=2 over PCIe for 8B: more KV vs all-reduce cost | `06_tensor_parallel.yaml`. Also try 2× data-parallel (two `CUDA_VISIBLE_DEVICES` runs) |

Before running 03, tune `online_poisson.rate` so the baseline sits at roughly 70–90% busy
(`idle_time_s` is small, but TTFT hasn't blown up). Otherwise every variant just looks idle or
drowned.

## Phase 2: code changes (backlog, roughly by expected payoff)

Put each change behind a `Config` flag so the old path can still be ablated.

**Scheduling and batching**
- [ ] **Mixed prefill + decode steps.** `engine/scheduler.py:54` returns prefill-only *or*
      decode-only batches, so every prefill stalls all running decodes (ITL spikes under load).
      Piggyback decodes onto prefill chunks the way vLLM v1 and Sarathi-Serve do. Needs a varlen
      attention path for the mixed batch. Also update `Recorder.step`, which classifies steps as
      prefill or decode.
- [ ] **Chunk more than the first sequence.** `scheduler.py:42` only chunks the head of the
      queue.
- [ ] **Preemption policy.** It always evicts the newest running sequence and recomputes it.
      Try evicting by remaining length, or swap KV to host memory instead of recomputing.
- [ ] **Admission control and `max_num_seqs` tuning** for online SLOs instead of maximum
      throughput.

**CPU overhead (dominant for 0.6B)**
- [ ] `prepare_decode` / `prepare_block_tables` (`engine/model_runner.py:123,172`) rebuild Python
      lists and pad block tables every step. Keep persistent pinned buffers and update them
      incrementally.
- [ ] `.tolist()` at `model_runner.py:218` syncs every step. Overlap scheduling of step N+1
      with GPU execution of step N (async scheduling).
- [ ] `self.running.remove(seq)` (`scheduler.py:92`) is O(n) per finished sequence.
- [ ] `hash_blocks` and xxhash on every postprocess.

**Kernels**
- [ ] Fuse RoPE + KV-cache store (`layers/attention.py:11`, `layers/rotary_embedding.py`).
- [ ] Triton fused add + RMSNorm (`layers/layernorm.py:35`), compared against torch.compile.
- [ ] Sampler (`layers/sampler.py:11`): full softmax plus an exponential over a 151k vocab. Add
      a top-k / greedy fast path, and use a fused Gumbel-max kernel.
- [ ] CUDA graph buckets (`model_runner.py:234`) step by 16, so bs=17 pads to 32. Measure the
      padding waste and try finer buckets. Graphs are disabled above bs 512 (`:197`).

**Memory and precision (L4 specific)**
- [ ] FP8 weights (Ada supports FP8 via `torch._scaled_mm`). Decode is bandwidth-bound, so this
      could be up to ~2× on 8B.
- [ ] FP8 KV cache: twice the KV capacity, so larger batches and fewer preemptions on 8B.
- [ ] Speculative decoding with 0.6B as the draft for 8B. Needs real-text prompts.

## PDF pipeline (scored by `eval/`)

Current pipeline: pypdfium2 → **[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) + [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)** → Markdown chunker → Qwen3-Embedding-0.6B.

The active configuration is `experiments/pdf/00_public_v3_baseline.yaml`: fixed four L4 GPUs, one page per GPU, one region per recognition batch, three PDFs, and one warm-up plus one timed call per PDF. This call count does not qualify for an official speedup comparison.

First establish complete page parsing and reviewed references. Then profile the accepted baseline before selecting changes. Earlier Qwen OCR configurations in `01_ocr.yaml` and related files are historical blueprints, not ready-to-run PaddleOCR comparisons.

### Required baseline work

- [x] Integrate the official native page parser with four persistent GPU workers.
- [x] Isolate Paddle and PyTorch CUDA runtimes.
- [x] Reject unfinished region generation before parser cleanup.
- [x] Complete the three-page NACA setup check without token-limit failures.
- [ ] Review all seven selected pages and freeze new references.
- [ ] Complete the sequential one-warm-up, one-timed-call evaluation.
- [ ] Profile rendering, layout detection, region preparation, recognition, Markdown assembly, chunking and embedding.
- [ ] Collect per-GPU activity, VRAM, power, clocks and bounded hardware-counter evidence before judging saturation.

### Candidate investigations after profiling

| Observed cost | Possible experiment |
| --- | --- |
| Native recognition decode | Kernel/launch overhead, region batching, or a supported VLM execution backend |
| Vision encoding or prefill | Image preparation, attention kernels and padding overhead |
| Layout detection and CPU preparation | Region preprocessing, scheduling or buffer reuse |
| GPU idle gaps | Overlap independent stages while preserving output and page order |
| Embedding | Batch size, precision or packed sequences |

Every optimized variant must preserve exact OCR bytes and chunk structure, and embedding cosine must remain at least 0.99. A backend change needs a correctness probe before timing. Model changes, region-selection policies and rendering resolution establish different reference behavior and require separate quality review.

The [PaddleOCR documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) describes supported recognition backends. A nano-vllm port would require PaddleOCR's architecture and preprocessing; the previous Qwen3-VL port outline does not apply.

## Results log

The current journal is [PaddleOCR-VL-1.6 baseline](docs/experiments/004-paddleocr-baseline.md). Record the motivating profile, hypothesis, change, validation, per-PDF timings, limitations and next decision for each experiment. The experiment list is a reference blueprint; priorities come from the new profile.
