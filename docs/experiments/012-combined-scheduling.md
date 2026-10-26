---
title: Window 64 · Dynamic page assignment
description: Combine the two separately successful OCR scheduling changes.
order: 12
status: passed
date: 2026-10-26
run: paddleocr_vllm_combined_scheduling/20261026-092829
---

## Evidence and change

The previous scheduling evaluation found complementary improvements: window 64 reduced p50 from 4.17 to 3.79 s, while dynamic pages reduced p95 from 28.31 to 26.68 s. Every call passed the frozen reference checks.

The [window-64 NACA profile](#/profile/8af5225b782fa4321fe2) confirms OCR still dominates and requests queue at the four-request server limit. The [earlier Census profile](#/profile/379f7cde1f8c400e072d) identified the late page-5 region that dynamic page assignment targets.

| Configuration | Client region window | Page assignment | Concurrent recognition requests per GPU |
| --- | ---: | --- | ---: |
| Fresh window-64 control | 64 | Static round robin | 4 |
| Combined scheduling | 64 | Next free layout worker | 4 |

BF16 embedding with 16 chunks per batch, four L4 GPUs, the five frozen PDFs and all OCR semantics stay fixed. Each configuration runs one warm-up plus five measured calls per PDF. The original vLLM batch-1 report remains the dashboard baseline; a separate comparison measures improvement over the fresh window-64 control.

Accept the combination only if full output checks pass and the measured gains survive together. Then profile the best configuration and test server concurrency eight if queued work remains.

## Results and decision

All 60 full-PDF calls passed the frozen checks; all 5,580 recognition completions ended with EOS.

| Configuration | p50 | p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Fresh window-64 control | 3.80 s | 28.31 s | 23.28 | 5.57 min |
| Window 64 + dynamic pages | 3.96 s | 26.82 s | 24.75 | 5.28 min |

The combined policy improved throughput by 1.063× and p95 by 1.055× against the fresh control, but p50 speedup was 0.958× (latency increased 4.4%). It is a useful tail/throughput tradeoff, not a new p50 winner. Window 64 with static pages remains the best p50 configuration.

Attention's measured OCR median increased from 3.37 to 3.55 s; embedding stayed near 0.30 s. The next concurrency experiment will evaluate static and dynamic pages separately rather than assuming the combination is universally better.

Change only the server's `max_num_seqs` from four to eight. Keep each page worker's client concurrency at four per recognizer and the submission window at 64, so a server-capacity gain is not confused with a client-concurrency change.
