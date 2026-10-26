---
title: BF16 configuration · OCR scheduling
description: Test region submission barriers and waiting-page assignment separately.
order: 11
status: passed
date: 2026-10-26
run: paddleocr_vllm_region_scheduling/20261026-090936
---

## Why these changes

The [accepted BF16 profile](#/profile/4394467834b2cf039d15) shows OCR is now the dominant NACA stage. The [Census study](#/profile/379f7cde1f8c400e072d) shows one late page-5 region controlling the tail while three recognizers are mostly idle.

| Configuration | Change from the BF16 control | Hypothesis |
| --- | --- | --- |
| BF16 control | None; fresh measurement | Check drift against the previous accepted result |
| Region window 64 | Submit 64 regions instead of 16 before waiting | Keep recognition requests available when some regions finish early |
| Dynamic pages | Give the next page to the first free layout worker | Start page 5 earlier instead of waiting behind a slow page on its assigned worker |

The server still runs at most four recognition regions per GPU. The wider client window changes queued work, not that engine limit. Dynamic assignment restores output to original page order. Each candidate changes one scheduling setting; combinations follow only if individual results justify them.

All use the five frozen PDFs, four L4 GPUs, BF16/16-chunk embeddings, identical OCR settings and one warm-up plus five measured calls per PDF. Original vLLM batch 1 remains the dashboard comparison baseline; separate comparisons use the fresh BF16 control. References and `eval/` are unchanged.

## Results

All 90 calls passed the unchanged evaluator and frozen output checks. Every region ended with EOS; all pages appeared exactly once and all four GPUs were used.

| Configuration | p50 | p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Fresh BF16 control | 4.17 s | 28.31 s | 22.41 | 5.78 min |
| Region window 64 | 3.79 s | 28.26 s | 23.42 | 5.55 min |
| Dynamic pages | 4.15 s | 26.68 s | 24.06 | 5.42 min |

The wider region window improved p50 by 1.100× against the fresh control, with little p95 change. Dynamic pages improved p95 by 1.061× and throughput by 1.073×; its p50 was effectively unchanged. These target different delays, so the next experiment combines them against a fresh window-64 control.

The largest page has 52 recognition regions. Window 64 already submits the whole page; increasing it to 128 would not change submission grouping on this dataset. Server concurrency is a separate lever.

## Profile of the window-64 result

The [fresh window-64 NACA profile](#/profile/8af5225b782fa4321fe2) passed the frozen output check. Median full call: 3.77 s; OCR: 3.17 s (84.0% of mean call time); embedding: 0.46 s. All servers reached four running requests, with up to eight waiting. Queues were nonempty in 40–50% of measured OCR-window samples.

This supports trying server concurrency eight after the combined scheduling experiment. The limit controls concurrent recognition requests per GPU; it is independent of the 64-region client submission window. GPU activity alone does not prove peak compute saturation. [vLLM’s tuning guide](https://docs.vllm.ai/en/latest/configuration/optimization/) describes concurrency and token-budget tradeoffs; the frozen evaluator decides whether the change helps this workload.
