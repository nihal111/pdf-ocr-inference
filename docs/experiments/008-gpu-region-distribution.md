---
title: vLLM · GPU region distribution · Earlier profiling
description: Earlier profiling evidence that motivated distributing regions across all four GPUs.
order: 8
status: superseded
date: 2026-10-26
---

## Comparison

| Variant | Role | Region assignment |
| --- | --- | --- |
| vLLM batch 1 | Baseline | One region at a time on the page worker's GPU |
| vLLM batch 4 · Page-assigned GPUs | Region-batching experiment | Up to four regions on the page worker's GPU |
| vLLM batch 4 · All GPUs | Distribution experiment | Region requests are spread across all four recognizers |

**Batch invariance is enabled for every variant.** The four L4 GPUs, models, precision policy, PDFs, rendering, chunking, embeddings and frozen reference set are fixed. Each variant runs one warm-up and one timed call per PDF through the evaluator. The native PaddlePaddle baseline is superseded and preserved.

## What the all-GPU experiment changes

Layout remains on the assigned page worker. Recognition requests use all four existing localhost servers. A page worker can submit up to 16 regions, distributed round-robin as up to four requests to each GPU. The recognizer servers retain their four-request execution limit. This allows the fourth GPU to help even when the PDF has only two or three pages, and lets a remaining long page use all four GPUs.

Responses are collected in original region order before Paddle assembles Markdown. The adapter records the target GPU of every region, so the experiment can verify actual distribution rather than infer it from configuration. Profiling records all four recognizers when they receive requests.

The per-GPU recognition limit remains four; the distribution variant increases the number of regions a page worker can submit concurrently to use the combined capacity. It is round-robin region distribution, not a cost-aware scheduler.

## Evaluation replacement

The preliminary three-PDF measurements have been replaced by the [five-PDF evaluation](#/doc/experiments/009-five-pdf-evaluation), which runs one warm-up and five measured calls per PDF. The pilot rows were removed from the active runs page; raw evidence remains archived locally. The profiles below are retained as earlier diagnostics and predate the table-image resolution adjustment used in the formal evaluation.

## Profiling

All profiles are complete, and their full instrumented outputs passed the shared reference checks. Each profile uses one warm-up, one ordinary timed NACA call and one separate bounded instrumented call. The page-assigned batch-4 profile was captured during the preceding repeatability check on the same engine setup.

| NACA profile | Batch 1 baseline | Batch 4 · Page-assigned GPUs | Batch 4 · All GPUs |
| --- | ---: | ---: | ---: |
| Complete pipeline | 15.23 s | 9.60 s | 5.39 s |
| OCR | 13.18 s · 86.5% | 7.55 s · 78.7% | 3.34 s · 62.0% |
| Embedding | 1.91 s · 12.5% | 1.91 s · 19.9% | 1.91 s · 35.4% |

The all-GPU recognizers reached four running requests each. Mean GPU busy during OCR was 97.3%, 96.7%, 93.0% and 97.3%; mean power was 67.3, 66.2, 70.3 and 65.6 W. All four GPUs now do OCR work. These are only three one-second samples per GPU, so they describe this short call and do not establish percentage of peak compute or bandwidth. Hardware throughput and occupancy counters were not collected.

**Next hypothesis:** test bfloat16 embedding on GPU 0 in the all-GPU candidate. Embedding remains at about 1.91 s and now occupies 35.4% of the call. Keep the batch-1 baseline, OCR settings and references fixed; require exact OCR/chunk structure and cosine at least 0.99 before accepting a change.

- [Batch 1 baseline profile](#/profile/48c1fb9c3f91abd76b84)
- [Batch 4 · Page-assigned GPUs profile](#/profile/5b3bf972d60a02287485)
- [Batch 4 · All GPUs profile](#/profile/c0465003a3bb83034e18)
