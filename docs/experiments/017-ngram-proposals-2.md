---
title: GPU n-gram speculation · 2 proposals
description: Tune proposal depth from measured acceptance and OCR activity.
order: 17
status: passed
date: 2026-10-26
run: paddleocr_vllm_ngram_gpu_2/20261026-110828
---

## Evidence and hypothesis

The [four-proposal Census profile](#/profile/eea77e8e41feda8616e7) found 1.60 emitted tokens per verification step on its busiest GPU. Across the initial full evaluator session, the first two proposal positions supplied 76–83% of accepted tokens on each GPU. The [four-proposal Attention profile](#/profile/4f7e7341f92057e60c66) records acceptance and GPU work on the shorter document.

Proposal acceptance alone does not establish speedup. Fewer proposals could reduce verification work while sacrificing some accepted tokens; more proposals could reduce serial steps on long transcriptions while adding verification overhead. Measure all three evaluator metrics to determine the tradeoff.

## Controlled change

Change `ngram_gpu` proposal depth from four to **2**. Keep concurrency eight, lookup lengths 2–5, standard rejection sampling, compiled BF16 embeddings, region window 64, dynamic pages, prompts, image sizes, generation limits and batch invariance fixed. Four original L4 GPUs serve the same five frozen PDFs and references.

Run one warm-up and five measured calls per PDF. Every call must match OCR and chunks exactly, meet the frozen embedding threshold and end every region with raw EOS. The original vLLM batch-1 baseline remains the dashboard anchor. Copy the fresh four-proposal report with its digest for a separate incremental comparison.

This is a fixed server-wide depth experiment. The installed GPU proposer asserts that the requested depth equals its configured depth; batch-dependent speculative depth is therefore not assumed to work here.

## References

- [vLLM speculative configuration](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [GPU n-gram proposer source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/ngram_proposer_gpu.py)
- [Acceptance metrics](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/)

Fetched source URLs and SHA-256 digests are saved in `research-sources.json`. Reports remain separate from profiling and earlier confirmation samples.

## Full evaluator result

Reducing proposal depth did not improve this workload. **Retain four proposals.** Correctness passed for all 30 calls, including all 2,790 raw EOS checks; minimum embedding cosine was 0.999870.

| Configuration | p50 | p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Four proposals · fresh control | 3.042 s | 15.669 s | 35.703 | 280.6 s |
| Two proposals | 3.108 s | 17.175 s | 33.998 | 292.0 s |

The unchanged evaluator's incremental ratios are **0.9789× p50**, **0.9123× p95** and **0.9522× pages/minute**. All three regressed. The original batch-1 comparison remains the dashboard anchor; `server_8_ngram_gpu_2.vs-previous.json` records this incremental comparison.

Census median latency increased from 15.667 to 17.133 s. Attention increased from 3.042 to 3.108 s, while NACA changed little (2.931 to 2.948 s). Fewer proposed tokens did not produce the hoped-for shorter calls, and removing later accepted proposals cost time on the long Census decode.

The whole-session counters include warm-ups and mixed PDFs. On GPU 3 they show 1.247 tokens per verification step with two proposals, compared with 1.288 in the initial four-proposal session. Work placement varies slightly with dynamic pages, so these aggregate counters are supporting evidence rather than an isolated per-region comparison.

## Next experiment

Test **eight proposals** against the same fresh four-proposal control to measure the opposite verification-depth tradeoff. Keep the first four-proposal result, both confirmations, and this unsuccessful optimization as separate evidence. Then address the unequal GPU token work found in the retained four-proposal Attention profile with a shared region-admission experiment.
