---
title: Optimization sweep · Retained configuration and plateau
description: The validated gains, the configuration to keep, and why the latest trials ended this sweep.
order: 28
status: passed
date: 2026-10-26
run: paddleocr_vllm_six_request_tiles/20261026-150343
profile: profiles/20261026-143036-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-Render-pages-inside-OCR-workers-c911f4
---

## Result

The latest full evaluation of the retained configuration passed at **2.229 s p50**,
**12.402 s p95** and **46.122 pages/min**. Versus the original vLLM batch 1
baseline, these are **6.77× p50 speedup (+576.8%)**, **2.93× p95 speedup** and
**3.78× throughput**. The original baseline remains the comparison anchor.

Identical retained-configuration evaluations measured p50 between 2.229 and
2.271 s. Treat small differences within that range as repeat variation, not a
new optimization. The latest suite took 442.4 s, including a 249.6 s first warm-up;
compiler startup varies and is excluded from pooled latency and throughput.

All measurements use the same five frozen PDFs / 20 pages, four L4 GPUs, and one
warm-up plus five measured calls per PDF. Each report pools 25 full-PDF measured
calls. Frozen OCR and ordered chunks must match; embedding cosine must be at least
0.99, and every recognition region must finish with EOS.

## Configuration to retain

- PaddleOCR-VL-1.6 recognition on vLLM, with batch invariance enabled.
- Four recognizer replicas on the same four L4 GPUs; regions shared across GPUs.
- Eight concurrent requests per recognizer, four GPU n-gram drafts, lookup lengths 2–5.
- Client concurrency four, region window 64, dynamic page assignment, and available-GPU admission with eight slots per GPU.
- Fixed decoder tiles: 32 rows, 128 columns and K=64, with four warps.
- Render at the existing 150 DPI inside OCR page workers, preserving exact pixels.
- BF16 compiled embeddings in 16-chunk batches, replicated across all four GPUs; restore original chunk order after embedding.

The [latest verification](#/run/b75edaea6a9b5182a677)
contains the expanded pipeline, actual server commands, source hashes, raw reports,
per-call stage data and placement/EOS audits. The [retained configuration profile](#/profile/7195172c30ec973339d4)
records ordinary calls separately from the bounded instrumented capture.

## Measured progression

These are saved full-evaluation results, not stage-only estimates. Incremental
tradeoffs and fresh-control comparisons are recorded in each experiment's notes.

| Configuration | Pooled p50 | Pooled p95 | Pages/min |
| --- | ---: | ---: | ---: |
| [vLLM batch 1 · frozen baseline](#/run/d8a8a05ea1b9bc9e9268/vllm_batch_1) | 15.083 s | 36.390 s | 12.193 |
| [Regions shared across GPUs](#/run/d8a8a05ea1b9bc9e9268/vllm_batch_4_all_gpus) | 5.388 s | 30.671 s | 19.886 |
| [BF16 embeddings, batch 16](#/run/2414bf81cf27a32c8c2b/embed_bf16_b16) | 4.189 s | 28.490 s | 22.359 |
| [Eight requests and dynamic pages](#/run/e618414568a8c437d78a/server_8_dynamic) | 3.293 s | 25.767 s | 26.164 |
| [Compiled embeddings](#/run/09dd48268f7b40e92830/server_8_compiled_embeddings) | 3.154 s | 25.299 s | 26.666 |
| [Four-draft speculation](#/run/654d9b217549709275f8/server_8_ngram_gpu_4) | 3.061 s | 15.862 s | 35.597 |
| [Shared available-GPU admission](#/run/7a9c93660221ff8d0318/available_gpu_8) | 2.908 s | 16.053 s | 36.195 |
| [Fixed 32-row decoder tiles](#/run/81ce12032be0a1d87ff4/decoder_tiles_fixed32) | 2.528 s | 12.890 s | 43.697 |
| [Embedding across four GPUs](#/run/901be5b3884c7556a3dd/embedding_four_gpus) | 2.331 s | 12.379 s | 45.627 |
| [Rendering inside OCR workers](#/run/40e811453ad54f72d35a/render_in_page_workers) | 2.271 s | 12.362 s | 45.802 |
| [Latest identical retained-config control](#/run/b75edaea6a9b5182a677/server_8_draft4_control) | 2.229 s | 12.402 s | 46.122 |

## Why this sweep stopped

The latest profiling-led alternatives stopped delivering useful improvement:

| Trial | Verified observation | Decision |
| --- | --- | --- |
| Per-projection decoder columns | Intended kernels ran; p50 speedup −0.14%, p95 +0.87%, throughput +0.53%. | Keep fixed 32-row / 128-column tiles. |
| Three draft tokens | One decoder row tile confirmed; p50 speedup +1.0%, p95 speedup −2.9%, throughput −1.2%. | The median gain is smaller than observed repeat variation; keep four drafts. |
| Six requests with four drafts | One row tile confirmed, but queues returned; p50 latency +5.8%, p95 flat, throughput −0.8%. | Keep eight requests. |

The [three-draft study](#/doc/experiments/026-three-draft-tile-boundary) and
[six-request study](#/doc/experiments/027-six-request-tile-boundary) each include a
fresh control, all 60 paired calls, 5,580 EOS completions and a complete four-GPU
NACA profile. The final capture verifies the intended change, so the negative
result is not explained by the option silently failing to take effect.

This ends the measured tuning sweep; it does not prove a hardware limit.
Rendering/OCR still accounts for about 94% of NACA wall time. One-second GPU busy
samples and bounded kernel traces do not establish peak compute or bandwidth
saturation. A later sweep would need a new concrete hypothesis and the same frozen
checks, rather than promoting these marginal or regressing variants.
