---
title: Batch 4 across GPUs · Embedding and scheduling
description: Profile the current best configuration, test embedding precision, then investigate OCR scheduling.
order: 10
status: passed
date: 2026-10-26
run: paddleocr_vllm_embedding_precision/20261026-085622
---

## Starting point

The best accepted five-PDF experiment uses vLLM recognition batch 4 with regions shared across all four L4 GPUs. Its pooled full-PDF p50 is 5.39 seconds, p95 is 30.67 seconds, and throughput is 19.89 pages/minute. The original vLLM batch 1 report and frozen references remain the comparison baseline.

The five PDFs, page bytes, OCR model and decoding, table resolution, batch invariance, chunks and four GPUs stay fixed. Every full evaluation uses one warm-up and five measured calls per PDF. Diagnostic measurements are separate from evaluator metrics.

## Profiling that motivates this experiment

Open the [fresh NACA study](#/profile/25b86f6693479a2fe6d5) and [fresh Census study](#/profile/379f7cde1f8c400e072d), both linked to the preceding best experiment. NACA spends 35.5% of its 5.30-second mean call in embedding; Census spends 89.2% of its 29.60-second mean call in OCR. Their respective median calls are 5.30 and 29.78 seconds.

NACA motivates lower-cost embeddings. Census motivates scheduling changes: GPUs 1–3 are largely idle while GPU 0 finishes one long region. These are two distinct bottlenecks, so the experiment sequence tests them separately.

## What the existing measurements suggest

| PDF | Median OCR | Median embedding | Reason to investigate |
| --- | ---: | ---: | --- |
| English checklist | 3.00 s | 0.96 s | Several groups of regions wait for each group to finish |
| NACA seaplanes | 3.31 s | 1.94 s | Embedding accounts for about 36% of the call |
| W-9 | 13.10 s | 2.41 s | A long form-table recognition controls OCR latency |
| Census income | 27.21 s | 3.18 s | A 3,252-token region on page 5 dominates the tail |
| Attention paper | 3.74 s | 0.94 s | Short region requests and page scheduling |

These are stage medians from the accepted evaluator run. Fresh profiles of NACA and Census use the current table-resolution setting, GPU telemetry and bounded vLLM traces. Census matters because it dominates p95; the earlier NACA profile alone cannot explain this.

## First lever: embedding precision

The current sentence-transformers embedder uses FP32. Screen FP16 and BF16 with embedding batches of 16, 32 and 64, using the same chunk text and vectors as the frozen references. Every screened vector must meet the existing cosine threshold of 0.99. Screening reuses reference text only to isolate the embedding stage; a speedup is accepted only after running the complete PDF pipeline through `eval/`.

[Sentence Transformers' inference guide](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html) describes lower precision and Flash Attention as GPU inference options. [Qwen's embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) documents the model and its recommended usage. These are leads to test, not evidence of a speedup on this machine.

### Isolated screening results

Each combination used one warm-up and five measured embedding calls per PDF. Every vector passed the frozen cosine check. The duration below is the **sum of the five per-PDF median embedding times**, not pooled full-PDF latency or an official speedup.

| Embedding precision | Chunks per batch | Sum of embedding medians | Minimum cosine |
| --- | ---: | ---: | ---: |
| FP32 | 32 | 9.376 s | 1.000000 |
| BF16 | 16 | 2.807 s | 0.999640 |
| BF16 | 32 | 3.462 s | 0.999640 |
| BF16 | 64 | 3.664 s | 0.999640 |
| FP16 | 16 | 2.883 s | 0.999995 |
| FP16 | 32 | 3.524 s | 0.999995 |
| FP16 | 64 | 3.726 s | 0.999995 |

Select **BF16 with 16 chunks per embedding batch** for full evaluation: it has the lowest measured embedding time and clears the 0.99 cosine threshold. FP16 is close in speed and has a smaller numerical difference, so it remains a useful alternative if later checks expose a problem. The first full comparison runs a fresh FP32/32-chunk control and the BF16/16-chunk candidate. OCR recognition still uses vLLM batch 4 across all GPUs.

The full evaluation passed all 60 calls. Its baseline snapshot remains the original accepted vLLM batch 1 report; the separate control comparison measures the incremental improvement.

## Subsequent levers

Examine region submission barriers, page scheduling and actual vLLM running/waiting requests after embedding improves. A larger client submission window is distinct from increasing the server's simultaneous batch limit. Any change must preserve exact OCR bytes, EOS completion and chunk order.

[vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/) remains enabled. Do not claim that GPU busy percentage establishes peak compute saturation: use measured activity, request queues and traces together.

## Stopping rule

Continue from each accepted improvement. Treat a configuration as an improvement only when the unchanged evaluator accepts its outputs and reports better pooled latency or throughput; check tradeoffs in all three metrics. If gains are small, repeat a control/candidate comparison before drawing a conclusion. Stop when the remaining evidence-backed levers produce no repeatable improvement or violate the frozen output contract, and record those results and limitations.

## Full-PDF evaluation results

| Configuration | p50 | p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Fresh FP32 control | 5.39 s | 30.60 s | 19.90 | 6.45 min |
| BF16 · 16 chunks | 4.19 s | 28.49 s | 22.36 | 5.79 min |

All reference, EOS, page-order and frozen-asset checks passed. Minimum candidate cosine: 0.999640. Relative to the fresh control, eval reports 1.286× p50 speedup, 1.074× p95 speedup and 1.124× throughput.

[Profile of the accepted BF16 configuration](#/profile/4394467834b2cf039d15) confirms OCR is now the dominant stage.
