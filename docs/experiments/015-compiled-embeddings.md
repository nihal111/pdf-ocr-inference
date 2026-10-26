---
title: Concurrency 8 · Compiled embeddings
description: Compile the remaining BF16 embedding forward, with a fresh eager control.
order: 15
status: passed
date: 2026-10-26
run: paddleocr_vllm_compiled_embeddings/20261026-101652
---

## Evidence and change

The [current best NACA profile](#/profile/db72310d9fe71f8fa226) assigns 14.2% of call time to embedding. The [Census profile](#/profile/df9916c0c1fe6277261c) assigns 4.0% to embedding; its remaining tail comes from one long OCR decode.

An isolated embedding screen passed all five frozen references. Summed per-document median embedding times fell from 2.689 to 1.659 seconds (38.3% less embedding time); the first compilation took 113.4 seconds. That is a stage-only diagnostic, not an evaluator speedup. Compiler counters confirm one executed graph and no graph breaks.

Separate NACA embedding traces reduced GPU kernel launches from 3,640 to 936 and summed kernel time from 0.444 to 0.276 seconds. These are kernel-only sums; nested CPU operator timings are not added together. From the previous stage shares, a similar embedding gain predicts roughly 5.5% less NACA call time and only 1.5% less Census call time. The full evaluator must establish the actual result.

Enable `embed.compile_dynamic: true` to call `SentenceTransformer.compile(dynamic=True)`. Keep BF16, 16-chunk batches, tokenization, normalization, OCR settings, region window 64, dynamic pages and server concurrency eight unchanged. The implementation defaults to eager.

This full evaluation reruns the eager control and the compiled candidate: five frozen PDFs, one warm-up and five measured calls per PDF, four L4 GPUs. Compilation and model warm-up are excluded from evaluator latency samples but included in suite runtime. Only compiled kernels may be reused; document outputs, prefixes and image processing are not cached. Every call must match OCR/chunks exactly and embedding cosine must remain at least 0.99.

The dashboard comparison stays anchored to the original vLLM batch-1 baseline. A separate paired comparison isolates compilation against this fresh eager control.

The implementation uses the official [SentenceTransformers compile API](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/base/model.py). Its installed `encode()` path invokes the compiled module forward. The screen saved compiler counters and operator traces separately from its ordinary measurements.


## Full evaluator result

| Configuration | p50 | p95 | Pages/min | Suite runtime | Frozen checks |
| --- | ---: | ---: | ---: | ---: | --- |
| Fresh eager control | 3.312 s | 25.702 s | 26.110 | 310.6 s | 30/30 passed |
| Compiled embeddings | 3.154 s | 25.299 s | 26.666 | 330.4 s | 30/30 passed |

The unchanged evaluator reports a **1.0503× p50 speedup** over the fresh control, 1.0159× for p95 and 1.0213× for pages/minute. Against the original vLLM batch-1 baseline, the corresponding ratios are 4.7824×, 1.4384× and 2.1871×. Original baseline and frozen dataset/reference hashes were verified unchanged.

All 5,580 recognition-region completions across both configurations ended with raw EOS. Every call matched OCR/chunks exactly; minimum embedding cosine was 0.999870 for compilation and 0.999640 for the eager control.

The first compiled full-PDF warm-up took 39.9 s versus 6.4 s in the eager control. Other warm-up costs differed too; total suite time increased by 19.8 s despite faster measured calls. This is a steady-state latency improvement with a startup cost. The isolated screen's first uncached compilation cost remains recorded separately at 113.4 s.

NACA's per-PDF median fell from 3.312 to 3.139 s and embedding stage time fell from 0.468 to 0.285 s. Attention's median is now 3.154 s, so it determines pooled p50. Retain compilation for the warm evaluator workload, profile the new median-limiting PDF, and test GPU n-gram speculation against the unchanged OCR references to address the long Census decode.


## Profile after compilation

The [compiled NACA profile](#/profile/d557cf386179e64081eb) passed its full instrumented reference check. Median call time is 3.11 s, OCR 2.688 s and embedding 0.286 s; mean stage shares are 86.5% OCR and 9.2% embedding. The remaining work is primarily OCR. An Attention profile follows because the pooled median now comes from that PDF.

The [Attention profile](#/profile/ab2b2140b0109f696ed0) also passed. Median call time is 3.13 s, with 2.828 s OCR and 0.184 s embedding. OCR accounts for 90.7% of mean call time. GPU n-gram speculation is the next test; acceptance counters and unchanged frozen checks will determine whether it helps.
