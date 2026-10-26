---
title: Available GPUs · Admission capacity
description: Compare assigning fewer requests in advance with buffering work behind the server.
order: 20
status: passed
date: 2026-10-26
run: paddleocr_vllm_admission_capacity/20261026-115305
profile: profiles/20261026-114858-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-4-proposals-Shared-queue-available-GPU-7f647d
---

## Evidence and question

The [current NACA profile](#/profile/a94c9bdbda43f0d45131) places 85.5% of call time in OCR. Servers briefly reach eight running requests but average about three per GPU within OCR windows; they have no waiting requests or preemptions. Startup and draining contribute to those averages, so occupancy alone does not establish the cause.

The [previous comparison](#/doc/experiments/019-shared-region-admission) improved pooled p50 from 3.03 to 2.91 s and slightly improved throughput, with a small p95 regression. NACA now determines the pooled median. Test capacity before assuming that eight is the right admission limit for available-GPU assignment.

## Controlled comparison

| Variant | Global admission/GPU | Maximum running in vLLM/GPU | Per-worker/GPU limit |
| --- | ---: | ---: | ---: |
| Fresh available-GPU control | 8 | 8 | 4 |
| Assign less work in advance | 4 | 8 | 4 |
| Allow a small server queue | 12 | 8 | 4 |

Four permits fewer active requests and delays GPU assignment until more prior work has finished. Twelve can queue requests behind the eight-request server, potentially reducing gaps between requests. These are competing hypotheses, not expected improvements.

Only the global admission limit changes. Keep the same four L4 GPUs, model precision, four n-gram proposals, lookup lengths 2–5, server configuration, page scheduling, region window, embeddings, input crops, prompts and frozen outputs.

Each configuration runs the five frozen PDFs with **one warm-up and five measured calls per PDF**: 30 complete calls and 25 pooled samples per configuration. Compare every candidate with the fresh eight-slot control using unchanged evaluator p50, p95 and pages/min; the dashboard retains the original batch-1 anchor. Inspect Census separately because the previous gain carried a p95 tradeoff.

## Evaluation results

| Admission/GPU | p50 | p95 | Pages/min | Frozen checks |
| --- | ---: | ---: | ---: | --- |
| 8 · fresh control | 2.977 s | 15.950 s | 35.162 | All 30 calls passed |
| 4 | 3.528 s | 16.153 s | 34.843 | All 30 calls passed |
| 12 | 2.931 s | 15.922 s | 36.315 | All 30 calls passed |

The fresh control's p50 is 2.4% slower than the preceding 2.908 s result; throughput also varies between these repeats. Judge the candidates against this sweep's fresh control and avoid treating changes of this scale as established improvements.

## Validation and decision

Require exact frozen OCR/chunk checks, embedding cosine at least 0.99, raw EOS completion, complete page dispatch, the configured global/per-worker limits, correct routing counters and no outstanding admission leases after a PDF. Source hashes and live server commands are captured before the sweep.

All 90 calls passed, including 8,370 raw EOS completions, frozen references, pooled metrics, admission limits, routing counts and zero outstanding leases. Captured source hashes remained unchanged.

Four slots worsened all three pooled metrics. Twelve improved p50 by 1.016×, p95 by 1.002× and throughput by 1.033× against this sweep's fresh control. Its 2.931 s p50 and 36.315 pages/min are close to the preceding eight-slot result of 2.908 s and 36.195 pages/min. These results do not establish a useful improvement beyond the variation between eight-slot runs.

**Retain eight admission slots.** Capacity tuning has reached a plateau in the tested neighborhood. The next lever is the decoder matrix kernel: the saved NACA trace and an isolated bitwise screen motivate smaller row tiles while retaining batch invariance. Any resulting variant must pass the complete frozen PDF evaluator.
