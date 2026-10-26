---
title: Compiled embeddings · Batch size and GPU placement
description: Check the remaining embedding cost after decoder tuning reached a plateau.
order: 24
status: passed
date: 2026-10-26
run: paddleocr_vllm_parallel_embeddings/20261026-134054
profile: profiles/20261026-134838-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-Embedding-across-all-four-GPUs-c939a3
---

## Why this stage

The [retained NACA profile](#/profile/24edbbe0670b361c831c) puts 11.3% of call time in embedding. OCR's per-projection column experiment was flat, so retain the 32-row/128-column decoder configuration and examine this remaining cost.

These initial screens isolate embedding using the frozen OCR text and existing chunker. They check the frozen vectors, including order and cosine tolerance, but **are not full-PDF evaluator measurements**.

## Compiled batch-size screen

Each batch size receives one warm-up and five measured calls per PDF. Batch 16 is repeated after the alternatives to check timing drift.

| Batch size | NACA embedding median | Sum of five case medians | Frozen checks |
| --- | ---: | ---: | --- |
| 16 · control | 0.2830 s | 1.6309 s | Passed |
| 8 | 0.2818 s | 1.4898 s | Passed |
| 32 | 0.3875 s | 1.8661 s | Passed |
| 64 | 0.3879 s | 1.9763 s | Passed |
| 16 · repeat | 0.2864 s | 1.6584 s | Passed |

Batch 8 helps English, Attention and W-9 embedding, but NACA—the document setting pooled p50—is almost unchanged. Larger batches are slower. The sum above compares this isolated stage; it is not an evaluator throughput metric.

## Four-GPU prototype

Embedding currently uses GPU 0 while the other recognizers are idle during that stage. Test one unchanged BF16 embedding model replica on each of the same four L4 GPUs. Assign chunks using current input token counts, then restore their original order when gathering vectors.

The prototype includes assignment, dispatch, GPU synchronization and gathering in its stage timing. It compares single-GPU control, four-GPU execution and a repeated single-GPU control, using one warm-up and five measured calls per frozen PDF. Each replica's actual CUDA device is checked, and per-GPU chunk/token assignments are saved.

The test keeps batch size 16, dynamic compilation, model weights, tokenizer, precision and reference vectors fixed. It adds no output cache.

| PDF | GPU 0 control | Four GPUs | GPU 0 repeat |
| --- | ---: | ---: | ---: |
| Earthquake checklist · English | 0.1759 s | 0.0844 s | 0.1774 s |
| NACA seaplanes | 0.2895 s | 0.1224 s | 0.2885 s |
| W-9 | 0.3801 s | 0.1714 s | 0.3787 s |
| Census income | 0.6389 s | 0.2157 s | 0.6340 s |
| Attention paper | 0.1889 s | 0.0763 s | 0.1872 s |

All 90 stage calls passed frozen-vector checks; minimum cosine was 0.99987.
These are embedding-stage medians, not pooled evaluator latencies.
The first four-GPU warm-up took 275.8 s while compilation ran, versus 43.6 s
for the first single-GPU warm-up. Startup cost must remain visible in suite runtime.

## Full-pipeline experiment

The implementation adds `embed.gpu_workers: 4`. It records each call's CUDA
devices and chunk/token distribution, restores output order, and drains all
submitted GPU work before propagating a worker failure. The default stays one worker.

Compare a fresh single-GPU embedding control with four-GPU embeddings through the
unchanged evaluator: five frozen PDFs, one warm-up and five measured calls per PDF.
Both use the same retained OCR servers, decoder tiles and reference vectors.
Both full evaluations passed, including all 60 pipeline calls and 5,580 raw EOS
completions. Every candidate call used all four embedding devices; original chunk
order and frozen-vector checks passed. Source files and frozen inputs stayed fixed.

| Configuration | Pooled p50 | Pooled p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Fresh GPU 0 embedding control | 2.494 s | 12.803 s | 43.918 | 241.7 s |
| Embedding across four GPUs | 2.331 s | 12.379 s | 45.627 | 450.3 s |

Evaluator speedup versus the fresh control improved **7.0% for p50**, **3.4% for
p95**, and **3.9% for throughput**. The original vLLM batch-1 dashboard baseline
remains unchanged; relative to it, the candidate is 6.47× faster at pooled p50.

All five individual PDF medians improved:

| PDF | Control full-PDF median | Four-GPU full-PDF median | Control embedding median | Four-GPU embedding median |
| --- | ---: | ---: | ---: | ---: |
| Earthquake checklist · English | 1.940 s | 1.805 s | 0.171 s | 0.070 s |
| NACA seaplanes | 2.480 s | 2.299 s | 0.282 s | 0.117 s |
| W-9 | 7.632 s | 7.488 s | 0.368 s | 0.162 s |
| Census income | 12.779 s | 12.366 s | 0.616 s | 0.199 s |
| Attention paper | 2.455 s | 2.328 s | 0.181 s | 0.070 s |

**Retain four-GPU embedding for the next experiment.** Startup is the tradeoff:
the first English warm-up grew from 48.1 s to 265.6 s while the replicas compiled.
Warm-ups remain excluded from pooled latency, but included in suite runtime.
Long-lived workers amortize that cost; this result does not claim faster cold starts.

The [fresh NACA profile](#/profile/b87dcac9214c23760dba) is complete: median 2.364 s,
including 2.102 s OCR, 0.119 s embedding and 0.113 s rendering. Mean shares
are 89.2%, 5.1% and 4.9%, respectively. All four embedding replicas were used
and the instrumented output passed frozen checks. The retained decoder kernels
also remained active on all four GPUs.

Rendering costs roughly 0.11 s on both NACA and Attention, which now overlap
around pooled p50. Screen renderer fast paths with exact pixel comparison before
spending a full evaluation on this smaller remaining stage. Keep resolution and
rendering quality fixed. The earlier profile remains linked as the motivation
for the completed embedding experiment.

[Fresh control evaluation](#/run/355fdb74652b8e0f18a0) · five PDFs, 30 calls including warm-ups.

[Four-GPU evaluation](#/run/901be5b3884c7556a3dd) · same frozen PDFs and references.
