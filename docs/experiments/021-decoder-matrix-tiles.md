---
title: Decoder tile selection · Compilation limitation
description: Smaller tiles reached the vocabulary head; compiled decoder projections retained the original tiles.
order: 21
status: passed
date: 2026-10-26
run: paddleocr_vllm_decoder_tiles/20261026-123223
profile: profiles/20261026-114858-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-4-proposals-Shared-queue-available-GPU-7f647d
---

## Why this experiment

The [latest retained NACA profile](#/profile/a94c9bdbda43f0d45131) puts 85.5% of call time in OCR. Persistent matrix multiplication contributes 68–70% of recorded kernel duration in the bounded trace. That trace includes vision prefill and does not isolate the decoder's share of whole-call latency.

The installed vLLM implementation uses a generic 128-row matrix tile for this PaddleOCR decoder's five projection shapes. Steady decoding often has far fewer rows. vLLM already supports shape-specific, batch-invariant tile selection for other model shapes; its reduction block stays constant across batch sizes.

The [admission-capacity comparison](#/doc/experiments/020-admission-capacity) did not establish an improvement beyond eight slots. Keep that configuration and test the matrix kernel separately.

## Isolated screen

An L4 diagnostic compared the existing kernel with 16-, 32- and 64-row tiles on the decoder's five fused projection shapes and seven row counts. **All 140 cases matched the existing kernel and a larger 128-row batch bit for bit.**

Each measurement replayed a captured kernel after a 256-MiB cache flush outside the timed interval. Twenty repetitions were collected per case. These synthetic matrix timings are separate from PDF evaluation.

At five input rows, corresponding to one request with four speculative proposals:

| Projection | Original kernel | 16-row tile | Kernel speedup |
| --- | ---: | ---: | ---: |
| QKV | 46.02 µs | 36.50 µs | 1.26× |
| Attention output | 51.57 µs | 41.82 µs | 1.23× |
| MLP gate/up | 86.77 µs | 81.02 µs | 1.07× |
| MLP down | 74.54 µs | 61.49 µs | 1.21× |
| Output vocabulary head | 1,081.71 µs | 1,070.75 µs | 1.01× |

At larger row counts, 32-row tiles avoid some regressions seen with 16-row tiles. The candidate therefore uses 16 rows through M=16, 32 rows through M=64, and the original 128-row tile above that. Column tiles remain 128; the reduction block remains 64. Small tiles use four warps.

The optional model hook adds these shapes to vLLM's existing Ada tuning table. It checks the model dimensions, BF16 decoder, tensor parallel size one and batch invariance. It leaves the FP32 vision/projector path unchanged and rejects conflicting upstream tuning. Integration checks also exercised the actual kernel across the 16/17 and 64/65 row boundaries, unchanged defaults, idempotence and conflict handling: four tests passed.

## Full evaluation

Run a fresh default-tile control followed by the smaller-tile candidate. Both retain the five frozen PDFs, four L4 GPUs, eight-slot available-GPU admission, server concurrency eight, four n-gram proposals, dynamic pages, region window 64 and compiled BF16 embeddings.

Each condition receives **one warm-up and five measured calls per PDF**. Require the complete frozen OCR, chunk, embedding and raw-EOS checks. Compare pooled p50, p95 and pages/min with the fresh control; keep the original vLLM batch-1 report as the dashboard anchor.

Both server sessions use fresh compilation-cache directories so the candidate cannot reuse a graph compiled with the default tiles. Model/server startup remains outside PDF latency and suite-runtime measurements. Prefix and image-processor output caches remain disabled.

| Configuration | p50 | p95 | Pages/min | Frozen checks |
| --- | ---: | ---: | ---: | --- |
| Fresh default tiles | 2.875 s | 15.732 s | 36.902 | All 30 calls passed |
| Smaller vocabulary-head tiles | 2.854 s | 16.070 s | 36.474 | All 30 calls passed |

The smaller tiles changed p50 speedup by **+0.74%**, p95 speedup by **−2.10%**, and throughput by **−1.16%** against the fresh default-tile control. This does not establish a useful pipeline improvement: the median difference is smaller than earlier repeat variation, while the tail and throughput regressed.

All 60 calls passed the frozen OCR, chunk and embedding checks; all 5,580 recognition regions ended with EOS. The evaluator, pipeline and plugin source hashes stayed fixed during both conditions. Suite runtime was 272.93 s for the control and 277.97 s for the candidate.

## What the follow-up profile found

The [completed NACA capture](#/profile/983118da65b40c30e2a3) verifies that the change reached **only the vocabulary head**. Each GPU recorded 32 matrix launches with 128-thread blocks during 32 engine iterations. The remaining 3,747–4,080 matrix launches retained 256-thread blocks. The compiled QKV graph still divides the token count into 128-row tiles.

The Python tile lookup runs during compilation with a large token count; its selected tile becomes a constant in the compiled decoder. The vocabulary head runs outside that graph and still uses the smaller buckets. Fresh compilation directories rule out reuse of the earlier default graph. Upstream [vLLM issue #54243](https://github.com/vllm-project/vllm/issues/54243) reports the same mechanism.

All ordinary and instrumented profile outputs passed. NACA's diagnostic median was 2.979 s, including 2.540 s OCR and 0.301 s embedding. Mean stage shares were 85.3% OCR and 10.1% embedding.

**This result does not reject smaller tiles throughout the decoder.** The next experiment will use a fixed 32-row tile for the screened BF16 projections, avoiding token-count-dependent selection during tracing. Verify actual compiled launches, preserve the reduction block and frozen outputs, and measure the prefill cost as part of full-PDF evaluation.

## References

- [vLLM batch-invariance documentation](https://docs.vllm.ai/en/latest/features/batch_invariance/).
- [vLLM invariant matrix-kernel configuration](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/determinism/batch_invariant_configs.py).
- [vLLM invariant matrix implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/determinism/batch_invariant.py).
- [Upstream compilation issue #54243](https://github.com/vllm-project/vllm/issues/54243).
- [vLLM compilation design](https://docs.vllm.ai/en/latest/design/torch_compile/).

Fetched source hashes, the installed-source hashes and raw diagnostic samples are retained with the run artifacts.

[PaddleOCR-VL-1.6 · vLLM · Default decoder tiles · Fresh control](#/run/f541dac8cefb538a422c) · completed; all 30 calls passed.

[PaddleOCR-VL-1.6 · vLLM · Smaller vocabulary-head tiles](#/run/6ad4c84857697f2cd85f) · completed; all 30 calls passed.
