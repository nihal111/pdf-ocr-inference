---
title: Decoder column widths · Per-projection tuning
description: Spread narrow matrix operations across more compute units while accounting for prefill cost.
order: 23
status: passed
date: 2026-10-26
run: paddleocr_vllm_decoder_columns/20261026-130349
profile: profiles/20261026-124729-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-Fixed-32-row-decoder-tiles-fc1057
---

## Evidence

The [retained fixed-row profile](#/profile/24edbbe0670b361c831c) confirms that the smaller decoder kernels execute. NACA OCR remains 82.7% of call time. Some narrow projections launch only 8–16 blocks on the L4's 58 streaming multiprocessors, leaving compute units unused for those operations.

An isolated screen varied column width at the retained 32-row tile over all five decoder matrix shapes and seven decode/prefill sizes. **All 140 cases matched the retained kernel and a 2,048-row batch bit for bit.** The reduction block remained 64.

Kernel measurements favored 64 columns for the attention-output/down projections at the recorded decode sizes. Wider QKV/gate projections benefited from 256 columns during prefill, with small decode penalties. The vocabulary head favored 64 columns at small decode sizes. These tradeoffs require a full-PDF comparison.

## Candidate

| Projection | Matrix N × K | Control columns | Candidate columns |
| --- | ---: | ---: | ---: |
| QKV | 2,560 × 1,024 | 128 | 256 |
| Attention output | 1,024 × 2,048 | 128 | 64 |
| Gate/up | 6,144 × 1,024 | 128 | 256 |
| Down | 1,024 × 3,072 | 128 | 64 |
| Vocabulary head | 103,424 × 1,024 | 128 | 64 |

Both configurations use fixed 32-row tiles, a K=64 reduction and four warps. Column widths depend only on model matrix shapes. FP32 vision kernels remain unchanged.

Six integration tests passed, including dynamically compiled output/down and QKV shapes. They traced large prefill inputs before short decode inputs, checked bitwise outputs, and inspected actual CUDA block/grid sizes for both column choices.

## Full evaluation

Run a fresh 32-row/128-column control and then the candidate, restarting the same four L4 recognizers with fresh compilation directories for each. Keep server concurrency eight, four n-gram proposals, available-GPU admission eight, dynamic pages, region window 64, compiled BF16 embeddings, prompts, crops and decoding fixed.

Each condition runs **one warm-up and five measured calls per frozen PDF**: 30 calls and 25 pooled samples. Require exact OCR/chunks, embedding cosine at least 0.99, raw EOS, the original dataset/reference hashes and complete four-GPU dispatch. Compare p50, p95 and pages/min with the fresh control, retaining the original batch-1 dashboard anchor.

| Configuration | p50 | p95 | Pages/min | Frozen checks |
| --- | ---: | ---: | ---: | --- |
| Fresh 32-row/128-column control | 2.490 s | 12.804 s | 43.887 | All 30 calls passed |
| Columns tuned per projection | 2.493 s | 12.694 s | 44.120 | All 30 calls passed |

The fresh control agrees closely with the retained 2.528 s / 12.890 s / 43.697 pages/min result. Its 2,790 recognition regions all ended with EOS and source hashes remained fixed. Use this fresh measurement for the candidate comparison.

The candidate changes evaluator speedup by **−0.14% p50**, **+0.87% p95** and **+0.53% throughput** against the fresh control. These differences are small compared with the variation between retained-configuration repeats. All 60 calls and 5,580 raw EOS completions passed; both runs preserved their source hashes.

## Decision and execution check

**Retain 32-row/128-column tiles.** The per-projection column change does not establish a useful full-PDF improvement.

The [follow-up NACA trace](#/profile/d4e1d2cbedbb0ce97208) confirms that the candidate executed: each GPU still recorded 2,336 decoder matrix launches, but the narrow projections now commonly launch 32 blocks rather than 16. The QKV/gate grids also changed as predicted. All instrumented outputs passed. This is a measured flat result after the intended kernels ran.

The diagnostic median was 2.496 s, including 2.058 s OCR and 0.300 s embedding. Mean stage shares were 82.7% OCR and 11.9% embedding. These diagnostic calls remain separate from evaluator samples.

The remaining small batching lever is compiled embedding batch size. The earlier batch-size screen preceded compilation; check neighboring batches against the retained compiled batch-16 stage before proposing another full-PDF experiment.

[PaddleOCR-VL-1.6 · vLLM · Fixed 32-row tiles · Fresh control](#/run/10fca35ced9033287977) · completed, all 30 calls passed.

[PaddleOCR-VL-1.6 · vLLM · Decoder columns tuned per projection](#/run/5a11f25bec9f0c33dcf2) · completed, all 30 calls passed.
