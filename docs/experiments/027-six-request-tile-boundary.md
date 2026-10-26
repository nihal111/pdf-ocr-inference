---
title: GPU concurrency · Six requests with four draft tokens
description: Keep four speculative positions while fitting verification into one 32-row decoder tile.
order: 27
status: passed
date: 2026-10-26
run: paddleocr_vllm_six_request_tiles/20261026-151300
profile: profiles/20261026-151811-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-Four-drafts-six-requests-GPU-32-row-tiles-a6ac7e
---

## Evidence

The [three-draft comparison](#/doc/experiments/026-three-draft-tile-boundary)
passed every frozen-output check, but traded a 1.0% p50 speedup for 2.9% lower
p95 speedup and 1.2% lower throughput. Census became slower.

Its [fresh NACA profile](#/profile/4be00774e183411338e9) confirms that reducing
drafts removed the second row tile in the sampled narrow decoder projections.
The retained [four-draft configuration](#/run/40e811453ad54f72d35a) uses worker
rendering, four-GPU compiled embeddings and fixed 32-row decoder tiles.

Test whether fewer concurrent requests can preserve four-draft decoding while
achieving the same tile reduction.

| Configuration | Requests per GPU | Draft tokens | Maximum verification rows |
| --- | ---: | ---: | ---: |
| Fresh control | 8 | 4 | 40 |
| Candidate | 6 | 4 | 30 |

The [versioned vLLM graph-sizing source](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/vllm.py)
defines uniform decode width as requests × (1 + drafts), and includes the maximum
request count when constructing default capture sizes. The downloaded source
matches the installed file byte for byte. Actual launch grids remain the check
that the intended shape is used.

## Measurement

Only server `max_num_seqs` changes. Keep four drafts, lookup lengths 2–5,
client concurrency four, admission eight per GPU, region window 64, dynamic pages,
worker rendering, fixed 32-row tiles and four-GPU compiled embeddings.
Use the same four L4 GPUs, frozen PDFs, references and original evaluator baseline.

Restart servers with isolated compilation directories for each condition.
Run one warm-up and five measured calls per PDF; pool all 25 measured calls
for p50, p95 and pages/min. Record suite runtime separately.
Require exact OCR, ordered chunks, embedding cosine at least 0.99 and raw EOS.

Lower concurrency can reduce useful overlap and throughput. A row-count
calculation alone is insufficient to retain the candidate. Inspect all three
evaluation metrics and profile actual kernel grids before deciding.

## Full evaluation result

Both conditions passed all 60 calls and 5,580 raw EOS completions. Source hashes,
frozen references, page order and all four embedding devices were verified.

| Configuration | Pooled p50 | Pooled p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Eight requests · fresh control | 2.229 s | 12.402 s | 46.122 | 442.4 s |
| Six requests | 2.358 s | 12.396 s | 45.745 | 298.5 s |

Six requests increase p50 latency by **5.8%** (p50 speedup falls **5.5%**).
The **0.05%** p95 speedup is effectively flat, and throughput falls **0.8%**.
Keep eight concurrent requests and four drafts.

The new identical eight-request control is 1.9% faster at p50 than the previous
2.271 s control. This repeat variation exceeds the three-draft experiment's 1.0%
paired p50 gain. Neither tile-boundary change establishes a useful improvement.

The suite-runtime difference comes mostly from the first embedding-compilation
warm-up: 249.6 s for the control versus 105.5 s for the candidate. It is not evidence
that six requests make PDF processing faster. Warm-ups remain excluded from p50,
p95 and pages/min.

| PDF | Eight-request median | Six-request median |
| --- | ---: | ---: |
| earthquake-checklist-en | 1.799 s | 1.817 s |
| naca-seaplanes | 2.214 s | 2.289 s |
| irs-w9 | 7.461 s | 7.383 s |
| census-income | 12.393 s | 12.349 s |
| attention-paper | 2.182 s | 2.312 s |

## Final profile and stopping decision

The [six-request NACA profile](#/profile/270b233aaaaeb8a50c39) completed with frozen-output
validation. Its ordinary-call median is 2.351 s; rendering/OCR takes 93.9% of mean
wall time and embedding 5.2%.

Actual narrow decoder projections used one row tile in 95 of 128 sampled engine
iterations and two in none. The remaining iterations include prefill. Every GPU
peaked at six running and two waiting requests: reducing server concurrency
reintroduced queues. These bounded traces establish execution settings, not
whole-call time shares or peak GPU saturation.

**Stop this tuning sweep and retain eight requests with four drafts.** Three
drafts provided a gain smaller than observed repeat variation while worsening
tail latency and throughput. Six requests preserved four-draft decoding but
regressed p50. The latest measured alternatives have stopped improving the
retained configuration.

[Retained configuration and sweep results](#/doc/experiments/028-optimization-conclusion).

[Fresh eight-request control](#/run/b75edaea6a9b5182a677) · [Six-request evaluation](#/run/e6282c8eaeb358f6414e)
