---
title: GPU speculation · Three drafts at the decoder tile boundary
description: Test whether three draft tokens avoid extra decoder padding at eight concurrent requests.
order: 26
status: passed
date: 2026-10-26
run: paddleocr_vllm_three_draft_tiles/20261026-145126
profile: profiles/20261026-145728-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-Three-draft-tokens-with-32-row-tiles-17b251
---

## Evidence and hypothesis

The [latest worker-rendering profile](#/profile/7195172c30ec973339d4) puts 93.9%
of wall time in overlapping rendering/OCR. The fixed 32-row decoder kernels
remain active, and narrow output/down projections use two row tiles in 99 of
128 sampled engine iterations across the four GPUs.

That count comes from the actual launch grids: each narrow projection has eight
column tiles, so a 16-block grid covers two row tiles. The capture includes the
first 32 iterations per GPU; it does not establish a whole-PDF time share.

| Draft tokens | Requests/GPU limit | Maximum verification rows | 32-row tiles needed |
| --- | ---: | ---: | ---: |
| 4 · retained control | 8 | 8 × (1 + 4) = 40 | 2 |
| 3 · candidate | 8 | 8 × (1 + 3) = 32 | 1 |

**Hypothesis:** three drafts can reduce padded decoder work during fuller batches.
It also removes one speculative position, which may hurt the long Census decode.
The full evaluation must measure that tradeoff.

Earlier two/eight-draft experiments preceded the fixed 32-row kernel change.
They do not test this tile boundary.

## Implementation and measurement

Use the installed vLLM 0.32.0 GPU n-gram implementation with a static three-token
count. The [versioned configuration source](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/speculative.py)
and [GPU proposer source](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/spec_decode/ngram_proposer_gpu.py)
were retrieved and checked. Actual engine startup and output validation remain
required.

Compare a fresh four-draft control with three drafts. Keep lookup lengths 2–5,
server concurrency eight, fixed 32-row decoder tiles, worker-side rendering,
compiled four-GPU embeddings and all frozen input/reference settings unchanged.
Restart the same four recognizers with isolated compilation directories for each
condition.

Run one warm-up and five measured full-PDF calls per frozen PDF. Require exact OCR,
ordered chunks, cosine at least 0.99 and raw EOS. Compare pooled p50, p95,
pages/min and suite runtime. Profile actual kernel grids before attributing any
gain to reduced padding.

## Full evaluation result

Both conditions passed all 60 full-PDF calls and 5,580 raw EOS completions.
Frozen inputs, references and the original comparison baseline remain unchanged.
Every call used all four embedding GPUs and rendered every page once inside its
OCR worker; source hashes remained unchanged throughout both evaluations.

| Configuration | Pooled p50 | Pooled p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Four drafts · fresh control | 2.271 s | 12.386 s | 45.987 | 386.0 s |
| Three drafts | 2.248 s | 12.749 s | 45.426 | 355.3 s |

Three drafts improve p50 speedup by **1.0%**, but p95 speedup falls **2.9%**
and throughput falls **1.2%**. The fresh four-draft control reproduces the previous
2.271 s p50. Keep four drafts as the retained configuration while investigating
this small median-versus-tail tradeoff.

| PDF | Four-draft median | Three-draft median |
| --- | ---: | ---: |
| earthquake-checklist-en | 1.824 s | 1.811 s |
| naca-seaplanes | 2.255 s | 2.203 s |
| irs-w9 | 7.381 s | 7.404 s |
| census-income | 12.351 s | 12.742 s |
| attention-paper | 2.271 s | 2.247 s |

## Profile and decision

The [fresh three-draft NACA profile](#/profile/4be00774e183411338e9) completed with instrumented
frozen-output validation. Ordinary-call median is 2.242 s, with 93.5% of mean wall
time in rendering/OCR and 5.6% in embedding.

The four bounded 32-iteration GPU captures show 102 narrow-projection iterations
using one row tile and none using two. The previous four-draft capture showed 99
using two. The expected tile reduction happened, but its end-to-end gain is small.
GPU busy averages 68–82% in seven one-second samples per GPU; this does not
establish peak compute or memory-bandwidth saturation.

**Retain four drafts.** The next targeted comparison tests six concurrent requests
per GPU with four drafts: 6 × 5 = 30 verification rows. This could preserve the fourth
speculative position while fitting one 32-row tile. Reduced concurrency can also
cost throughput, so the full evaluation and actual launch grids must decide.

[Next experiment: six requests with four drafts](#/doc/experiments/027-six-request-tile-boundary).

[Fresh four-draft control](#/run/8f40169809b2d6e9a264) · [Three-draft evaluation](#/run/dff8ebe0a0f2172f80ce)

## Follow-up result

The six-request comparison is complete: p50 latency worsened 5.8% and throughput
fell 0.8%, despite the intended tile reduction. An identical eight-request control
also reproduced a p50 1.9% faster than the preceding control, larger than this
three-draft trial’s 1.0% gain. Retain eight requests and four drafts; the
[completed sweep](#/doc/experiments/028-optimization-conclusion) records the stopping evidence.
