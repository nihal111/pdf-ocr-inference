---
title: Qwen3-VL four-GPU baseline — invalidated
description: Sequential PDF evaluation with one page per GPU, one warm-up and one timed call.
order: 1
status: invalidated
date: 2026-10-26
run: pdf_public_v3_4gpu_baseline/20261026-051817
---

## Truncation and repetition

This **Qwen3-VL-2B-Instruct run is invalidated**. Pages 1 and 3 of the seaplane scan reached the 2,048-token limit without EOS. Its saved OCR also repeated each Figure 1 and Figure 2 caption **25 times**. The evaluator matched calls against those incomplete, repetitive references. These measurements remain historical evidence.

A later diagnostic with an 8,192-token ceiling still repeated three page-2 paragraphs 11 times each and failed to emit EOS. [Our saved repetition evidence](#/doc/pages/debugging) records the distinction between the original run and the follow-up.

Related external report: [Qwen issue #1611, “Infinite repetition bug still exists”](https://github.com/QwenLM/Qwen3-VL/issues/1611). It documents similar image-to-Markdown repetition; it does not establish a root cause for our run.

[Setup notes before the first complete baseline](#/doc/experiments/000-setup-notes)

## Question

How long does the unchanged Hugging Face OCR → chunking → embedding pipeline take when four page workers cooperate on each PDF?

## Fixed setup

The library contains ten PDFs capped at five pages, with 42 pages available to preview. This initial measurement uses the three smallest complete documents: the English earthquake checklist (2 pages), Spanish earthquake checklist (2 pages), and NACA seaplane scan (3 pages).

Four NVIDIA L4 GPUs remain allocated throughout. Each holds a Qwen3-VL-2B-Instruct OCR replica, bf16, SDPA, greedy decoding, with a 2,048-token output cap and a batch of one page per GPU. Workers receive pages from the same PDF in round-robin order. Results are assembled in original page order before 512-token chunking and fp32 Qwen3-Embedding-0.6B embedding on GPU 0.

The selected documents activate two, two and three OCR workers respectively. Idle workers retain their model replicas. Four allocated GPUs does not imply four busy GPUs for these short documents.

## Evaluation contract

Capture references once with this configuration, then run the immutable evaluator with **one warm-up and one timed call per PDF**, sequentially. All six evaluation calls must match OCR bytes, chunk IDs/order/dimensions, and embeddings at cosine ≥ 0.99. Startup and reference capture are excluded from measured latency.

One timed sample per PDF is an initial measurement, not a repeat distribution or an official speedup comparison. Official comparisons require at least five timed repeats.

## Saved configuration

Frozen experiment settings, PDF/reference hashes, model revisions, package versions, source hashes and GPU UUIDs are saved with this historical run. The active `00_public_v3_baseline.yaml` now selects PaddleOCR and does not reproduce this Qwen measurement.

## Results

All **3 documents passed**, including all 3 warm-ups and **3 measured calls**. Completed at 2026-10-26T05:25:02.735006+00:00.

This initial measurement has one timed sample per document. It does not qualify for an official speedup comparison, which requires at least five measured repeats per PDF.

| Metric | Result |
| --- | ---: |
| Pooled document latency p50 | 38.204 s |
| Pooled document latency p95 | 57.314 s |
| Sequential throughput | 3.382 pages/min |

| Document | p50 (s) | p95 (s) | Minimum cosine |
| --- | ---: | ---: | ---: |
| earthquake-checklist-en | 26.553 | 26.553 | 1.00000000 |
| earthquake-checklist-es | 38.204 | 38.204 | 1.00000000 |
| naca-seaplanes | 59.438 | 59.438 | 1.00000000 |

## Stage evidence

Stage shares use mean stage time divided by mean total time, excluding warm-ups.

| Stage | Mean duration per document (s) | Share |
| --- | ---: | ---: |
| render | 0.061 | 0.15% |
| ocr | 40.202 | 97.11% |
| chunk | 0.007 | 0.02% |
| embed | 1.128 | 2.72% |

The largest stage is **ocr**. These timings identify where wall time is spent; operator traces are needed to explain the operations underneath.

## Next step

The Qwen diagnostic profile is retained as historical evidence. Current work establishes the [PaddleOCR baseline](#/doc/experiments/004-paddleocr-baseline), then profiles that implementation before choosing an optimization.
