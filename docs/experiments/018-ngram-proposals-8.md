---
title: GPU n-gram speculation · 8 proposals
description: Tune proposal depth from measured acceptance and OCR activity.
order: 18
status: passed
date: 2026-10-26
run: paddleocr_vllm_ngram_gpu_8/20261026-111535
---

## Evidence and hypothesis

The [four-proposal Census profile](#/profile/eea77e8e41feda8616e7) found 1.60 emitted tokens per verification step on its busiest GPU. Across the initial full evaluator session, the first two proposal positions supplied 76–83% of accepted tokens on each GPU. The [four-proposal Attention profile](#/profile/4f7e7341f92057e60c66) records acceptance and GPU work on the shorter document.

Proposal acceptance alone does not establish speedup. Fewer proposals could reduce verification work while sacrificing some accepted tokens; more proposals could reduce serial steps on long transcriptions while adding verification overhead. Measure all three evaluator metrics to determine the tradeoff.

The [two-proposal trial](#/doc/experiments/017-ngram-proposals-2) passed correctness but worsened all three evaluator metrics. Its Census median increased from 15.667 to 17.133 s. Four remains the retained configuration; this trial tests the opposite direction, where more accepted later tokens might shorten the serial decode.

## Controlled change

Change `ngram_gpu` proposal depth from four to **8**. Keep concurrency eight, lookup lengths 2–5, standard rejection sampling, compiled BF16 embeddings, region window 64, dynamic pages, prompts, image sizes, generation limits and batch invariance fixed. Four original L4 GPUs serve the same five frozen PDFs and references.

Run one warm-up and five measured calls per PDF. Every call must match OCR and chunks exactly, meet the frozen embedding threshold and end every region with raw EOS. The original vLLM batch-1 baseline remains the dashboard anchor. Copy the fresh four-proposal report with its digest for a separate incremental comparison.

This is a fixed server-wide depth experiment. The installed GPU proposer asserts that the requested depth equals its configured depth; batch-dependent speculative depth is therefore not assumed to work here.

## References

- [vLLM speculative configuration](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [GPU n-gram proposer source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/ngram_proposer_gpu.py)
- [Acceptance metrics](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/)

Fetched source URLs and SHA-256 digests are saved in `research-sources.json`. Reports remain separate from profiling and earlier confirmation samples.

## Full evaluator result

Eight proposals passed all 30 calls and all 2,790 raw EOS checks. Minimum embedding cosine was 0.999870. **Retain four proposals:** this trial did not establish a meaningful improvement.

| Configuration | p50 | p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Four proposals · fresh control | 3.042 s | 15.669 s | 35.703 | 280.6 s |
| Eight proposals | 3.037 s | 15.767 s | 35.757 | 280.1 s |

The unchanged evaluator's incremental ratios are **1.0015× p50**, **0.9938× p95** and **1.0015× pages/minute**. The p50 and throughput differences are approximately 0.15%, much smaller than the variation observed in the earlier four-proposal confirmations. p95 was slightly slower in this trial. These results are retained separately; no samples or reference outputs are replaced.

Two proposals had clearly slowed the Census tail; eight delivered no meaningful overall gain over four. This completes the immediate proposal-depth comparison at two, four and eight. The original vLLM batch-1 report remains the dashboard anchor.

## Next experiment: shared region admission

Return to the four-proposal configuration. Its Attention profile found 16–17 regions per GPU but roughly 3.3× unequal generated-token work. Existing round-robin routing binds requests to GPUs before the queued work finishes.

Test a shared admission queue that assigns pending requests when a GPU has capacity. Include a round-robin admission control at the same capacity so a gain can be distinguished from simply reducing queued HTTP requests. Keep server concurrency eight, client concurrency four per page worker/GPU, region window 64, dynamic pages, compiled embeddings and all frozen OCR settings fixed. Restore reading order before comparing results.

A CPU-only prototype has passed ordering, per-GPU/per-worker limits, backend failure, disconnect, cancellation and shutdown checks using Paddle's real response collector. It is not yet integrated into the measured pipeline and is not evidence of GPU performance improvement.
