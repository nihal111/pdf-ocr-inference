---
title: Four proposals · Shared region admission
description: Separate queue-capacity effects from assigning work to an available GPU.
order: 19
status: passed
date: 2026-10-26
run: paddleocr_vllm_region_admission/20261026-113511
profile: profiles/20261026-114858-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-4-proposals-Shared-queue-available-GPU-7f647d
---

## Profiling that motivates this experiment

The [retained Attention profile](#/profile/4f7e7341f92057e60c66) found 16–17 regions per GPU but roughly 3.3× unequal generated-token work. OCR consumed 90.2% of the call. Eager round-robin routing assigns each request before the GPUs finish their queued work.

The [Census profile](#/profile/eea77e8e41feda8616e7) still has a long single-region decode. Admission cannot divide that autoregressive sequence among GPUs; check p95 and throughput alongside the pooled p50.

The [two- and eight-proposal trials](#/doc/experiments/018-ngram-proposals-8) did not establish an improvement beyond four. Keep four proposals for this scheduling comparison.

## Controlled comparison

| Variant | Assignment | Admission across page workers |
| --- | --- | --- |
| Current routing control | Eager per-worker round-robin | Existing four-request client limit per worker/GPU |
| Shared queue · round-robin | Preserve fixed destinations | Eight requests per GPU; four per worker/GPU |
| Shared queue · available GPU | Select a GPU when a slot opens | The same eight/four limits |

The round-robin admission control isolates the effect of reducing queued HTTP requests. Comparing available-GPU admission with that control isolates the assignment policy at the same capacity.

Keep four L4 GPUs, vLLM server concurrency eight, four n-gram proposals, lookup lengths 2–5, standard rejection sampling, region window 64, dynamic pages, compiled BF16 embeddings and all frozen OCR settings fixed. Input regions, crop pixels, prompts and output reading order are unchanged.

Each variant runs **one warm-up and five measured calls per PDF** on the five frozen PDFs: 30 complete calls and 25 pooled measurements per configuration. The original vLLM batch-1 report remains the dashboard anchor. This sweep also measures a fresh retained control, so incremental comparisons use the same code and server session.

## Evaluation results

| Configuration | p50 | p95 | Pages/min | Frozen checks |
| --- | ---: | ---: | ---: | --- |
| Current routing control | 3.028 s | 15.769 s | 35.826 | All 30 calls passed |
| Shared queue · round-robin | 3.058 s | 15.803 s | 35.642 | All 30 calls passed |
| Shared queue · available GPU | 2.908 s | 16.053 s | 36.195 | All 30 calls passed |

The fresh control reproduces the retained four-proposal configuration. Shared admission with fixed destinations was slightly slower on all three pooled metrics; limiting queued requests alone did not improve this benchmark.

Available-GPU assignment improved p50 by **1.041×** and throughput by **1.010×** against the fresh control. Its p95 was slower: **0.982×**. Against admission with fixed destinations at the same capacity, the corresponding ratios are **1.052×**, **1.016×**, and **0.984×**. These are evaluator speedup ratios; the website continues to compare against the original batch-1 baseline.

This is a median/throughput improvement with a tail-latency tradeoff. It does not establish an improvement on every PDF or a statistically significant throughput gain.

## What changed in the measured calls

| PDF | Current routing median | Available GPU median | Median max/min GPU token work, before → after |
| --- | ---: | ---: | ---: |
| Earthquake checklist | 2.071 s | 2.079 s | 1.25× → 1.36× |
| NACA | 2.926 s | 2.906 s | 1.46× → 1.70× |
| W-9 | 9.715 s | 9.294 s | 1.59× → 1.25× |
| Census | 15.769 s | 15.952 s | 17.89× → 8.26× |
| Attention | 3.028 s | 2.798 s | 3.26× → 1.60× |

Token-work ratios are the median of each measured call's largest/smallest per-GPU generated-token totals. They describe output work, not compute utilization. The raw per-call counts and stage durations are in `routing-analysis.json`.

Attention became better balanced and faster, consistent with the motivating profile. W-9 also improved. Census demonstrates the limitation of this measure: its token-work ratio improved, but its long individual decode still dominated and latency increased. NACA now supplies the middle samples of the pooled latency distribution; its slight timing change does not establish a useful gain on that PDF.

The [new NACA profile](#/profile/a94c9bdbda43f0d45131) captures available-GPU admission after this evaluation. Keep the Census p95 tradeoff visible, and confirm small aggregate gains with a fresh control.

All 90 complete calls passed, with 8,370 raw region completions ending normally. Frozen dataset/reference hashes, the original baseline snapshot, four-GPU dispatch, pooled metrics, admission limits and zero outstanding leases were verified. All captured pipeline, lab and evaluator Python files still matched their sweep-start hashes. The evaluator and original baseline were not changed.

## Profiling the new median bottleneck

NACA's diagnostic median was **2.987 s**, including **2.550 s OCR** and **0.297 s embedding**. Mean shares were 85.5% and 9.9%. One warm-up, three ordinary calls and a separate complete instrumented call passed the frozen checks.

Within OCR windows, the servers averaged 2.95–3.43 running requests per GPU, briefly reaching eight. They had no waiting requests or preemptions, and peak KV-cache occupancy stayed below 0.64%. Startup and draining contribute to those averages; they do not prove that request submission is the bottleneck.

N-gram acceptance was 2.01–2.89%, giving 1.08–1.12 tokens per verification step. Per-GPU generated work still varied between calls. Seven one-second GPU samples per device are too sparse to claim peak compute or bandwidth saturation.

The next controlled test compares **admission limits 4, 8 and 12 per GPU**, while keeping vLLM server concurrency at eight and the per-worker/GPU limit at four. Four assigns less work in advance; twelve can buffer requests behind the server. The experiment will determine whether either helps. Every variant must run the full frozen evaluator contract, and the eight-slot control must reproduce the current result.

## Implementation and verification

A broker in the pipeline process grants recognition slots to its page workers. Fixed-destination requests can wait without blocking requests eligible for another GPU. Available-GPU routing considers current admitted work; it does not use frozen output lengths or cached OCR.

The asynchronous client records the selected GPU before publishing its response future. Paddle's existing collector restores region order. Completed PDF calls must leave zero active or pending broker leases; traces record per-call GPU request counts and lifetime capacity peaks.

Linux abstract Unix sockets follow process lifetime, including eval's direct worker termination. Both sides verify the kernel-provided peer UID before exchanging admission messages. This is local process communication; no network port is added. [Linux Unix-socket documentation](https://man7.org/linux/man-pages/man7/unix.7.html).

Preflight checks passed for output ordering, actual GPU labels, global/per-worker limits, fixed-destination queue behavior, backend failure, disconnect, cancellation, worker termination and cleanup. Existing pipeline and website-data checks also passed: 66 passed, one skipped. These checks do not establish GPU performance; every full evaluator call must still pass the frozen text, chunk, embedding and raw-EOS requirements.
