---
title: Concurrency 8 · GPU n-gram speculation
description: Shorten serial OCR decoding with proposals verified by the same model.
order: 16
status: passed
date: 2026-10-26
run: paddleocr_vllm_ngram_gpu_4/20261026-103409
---

## Evidence and hypothesis

The [Census profile](#/profile/df9916c0c1fe6277261c) found a 3,252-token page-5 region controlling the tail. GPU 3 averaged 98.4% busy during ordinary OCR calls; the other GPUs averaged 4.9–9.5%. OCR accounted for 95.6% of the call. The [latest Attention Paper profile](#/profile/ab2b2140b0109f696ed0) records the current control before this change.

If the transcription reuses short token sequences within a region, n-gram proposals may let the OCR model verify several tokens in one decoder step. This is a hypothesis; GPU busy percentage does not establish memory-bandwidth saturation.

## Controlled change

Use vLLM `ngram_gpu`, four speculative tokens, lookup lengths 2–5 and standard rejection sampling. The GPU proposer preserves asynchronous scheduling in our installed vLLM version. It searches only the current request's token history; no draft model or cross-request output cache is added.

Keep server concurrency eight, four L4 GPUs, region window 64, dynamic pages, the selected embedding configuration, batch invariance, prompts, pixels and generation limits fixed. Every full-PDF call must pass the existing frozen OCR, chunk and embedding checks; raw EOS is required.

Run one warm-up and five measured calls for each of the five frozen PDFs. The original vLLM batch-1 baseline remains the dashboard anchor. The immediately preceding accepted control is copied with its SHA-256 for an incremental comparison; a close result requires another control/candidate check.

Capture draft and accepted-token counters before and after the complete evaluator session. Those counts include warm-ups and are separate from evaluator latency samples. A subsequent ordinary-call profile will capture acceptance rates alongside GPU activity and scheduler occupancy.

## References

- [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [Official GPU n-gram proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/ngram_proposer_gpu.py)
- [vLLM acceptance metrics](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/)


## Full evaluator result

| Configuration | p50 | p95 | Pages/min | Suite runtime | Frozen checks |
| --- | ---: | ---: | ---: | ---: | --- |
| Preceding compiled control | 3.154 s | 25.299 s | 26.666 | 330.4 s | 30/30 passed |
| GPU n-gram, four proposals | 3.061 s | 15.862 s | 35.597 | 288.6 s | 30/30 passed |

The unchanged evaluator gives incremental ratios of **1.0303× p50**, **1.5949× p95** and **1.3349× pages/minute**. Against the original vLLM batch-1 baseline, these are 4.9271×, 2.2942× and 2.9195×. The original baseline, five frozen PDFs, references and four GPU UUIDs remain unchanged.

All 2,790 region completions ended with raw EOS, and every call matched OCR and chunks exactly. Minimum embedding cosine remained 0.999870. This accepts the correctness of speculative execution; references were not recaptured.

The tail and throughput improvements are large. The 3% p50 ratio is small enough to require another control/candidate comparison before treating the median gain as repeatable. Profile the speculative Census workload to check draft acceptance and the remaining tail, then repeat the comparison.

Attention's preceding profile found almost equal region counts per GPU (17, 16, 17, 17), but generated-token totals of 913, 454, 854 and 1,510. If this unequal work remains after speculation, test image-derived work estimates for region assignment while restoring the original output order.

## Draft acceptance across the evaluator session

These counter differences include all warm-ups and measured calls. Tokens per verification step includes the bonus token and does not directly measure speedup.

| GPU | Proposed tokens | Accepted tokens | Acceptance | Tokens per step |
| --- | ---: | ---: | ---: | ---: |
| 0 | 116984 | 1812 | 1.5% | 1.06 |
| 1 | 102940 | 1738 | 1.7% | 1.07 |
| 2 | 97608 | 1914 | 2.0% | 1.08 |
| 3 | 191604 | 13802 | 7.2% | 1.29 |


## Profile after speculation

The [speculative Census profile](#/profile/eea77e8e41feda8616e7) passed its full instrumented reference check. Median OCR time is 15.027 s versus 24.597 s in the preceding non-speculative Census study. GPU 3 accepts 15.0% of proposed tokens and emits 1.60 tokens per verification step across the ordinary diagnostic calls; it remains 96.5% busy in the OCR windows while the other GPUs average 6.1–16.6%. The 3,252-token region still ends with EOS.

## Independent confirmation

The candidate-first, fresh-control-second repeat passed all 60 calls and all 5,580 region EOS checks. Each configuration again used one warm-up and five measured calls per frozen PDF. These samples remain separate from the original evaluator report.

| Configuration | p50 | p95 | Pages/min |
| --- | ---: | ---: | ---: |
| GPU n-gram, four proposals · repeat | 3.508 s | 16.199 s | 34.575 |
| Compiled control · fresh servers | 3.127 s | 25.632 s | 26.637 |

Tail latency and throughput improved again: evaluator ratios are **1.5823× p95** and **1.2980× pages/minute**. The p50 ratio is **0.8914×**; the initial small median gain was **not confirmed**.

The slower candidate repeat came from NACA's OCR stage: five measured OCR calls ranged from 2.595–3.739 s, versus 2.497–2.539 s initially, with identical generated-token totals. Rendering also slowed. This evidence does not establish the cause. That repeat followed a profiling session on the same servers.

A further candidate repeat on freshly started servers, with one-second CPU/GPU observations, passed all 30 calls and 2,790 raw EOS checks:

| Fresh-server candidate | p50 | p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| GPU n-gram, four proposals | 3.042 s | 15.669 s | 35.703 | 280.6 s |

Against the fresh compiled control above, evaluator ratios are **1.0279× p50**, **1.6359× p95** and **1.3404× pages/minute**. NACA returned to 2.930 s median; measured OCR calls ranged from 2.506–2.520 s. Its aggregate host CPU busy observation averaged 8.4% across 96 logical CPUs, with near-zero CPU pressure in this repeat. Those observations cannot explain the earlier unobserved slowdown or exclude a single-thread bottleneck.

Tail and throughput gains held across all three candidate evaluations. The median gain is small and variable; retain the slower repeat as evidence. Do not pool these confirmation samples into the original report or replace the original baseline.

Complete reports and comparisons are downloadable as `confirmation-ngram4_repeat.json`, `confirmation-compiled_control_repeat.json`, `confirmation-summary.json`, `confirmation-ngram4_fresh_repeat.json` and `fresh-confirmation-summary.json`. `fresh-confirmation-activity.json` summarizes observations and points to raw CPU/GPU samples.

## Next experiment: proposal depth

The [speculative Attention profile](#/profile/4f7e7341f92057e60c66) passed its instrumented reference check. Median diagnostic latency is 3.036 s, with 2.731 s in OCR; mean shares are 90.2% OCR and 6.2% embedding. Its GPUs emit only 1.01–1.11 tokens per verification step, versus 1.60 on Census's busiest GPU.

Attention's region counts remain balanced at 16–17 per GPU, but generated-token work differs by roughly 3.3× between GPUs 3 and 1. GPU activity samples are sparse on this short PDF, so the saved per-region token totals are the stronger evidence for unequal work.

Saved position counters show that the first two of four proposal positions supplied **76–83% of accepted tokens per GPU** over the initial evaluator session. On Census GPU 3, the corresponding diagnostic fraction is 76.6%. The remaining positions still contribute useful tokens; acceptance fractions alone do not predict speedup.

Test **two proposals** against the fresh four-proposal control. This tests whether less verification work offsets the lost later proposals. Keep all other settings fixed and measure p50, p95 and pages/minute through the complete evaluator. `proposal-position-analysis.json` contains the counter differences and their scopes. After this, test assigning pending regions as GPU capacity becomes available, preserving output order and using no reference-derived length estimates.

The installed GPU proposer requires its requested depth to equal its configured depth. Use a fixed-depth trial; do not assume batch-dependent depth is supported by this proposer.

Follow-up: the [two-proposal trial](#/doc/experiments/017-ngram-proposals-2) regressed all three metrics. The [eight-proposal trial](#/doc/experiments/018-ngram-proposals-8) was effectively flat; four remains the retained control. Next, test shared region admission with a control at the same capacity.
