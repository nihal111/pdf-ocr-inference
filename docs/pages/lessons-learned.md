---
title: Lessons learned
description: Baseline quality problems, profiling findings, useful changes and diminishing returns.
order: 0
---

We first had to make OCR complete and repeatable. The largest performance gains
then came from sharing recognition work across GPUs and reducing decoder cost.

## Results and scope

| Eval metric | vLLM batch-1 baseline | Retained configuration | Speedup |
| --- | ---: | ---: | ---: |
| Full-PDF p50 | 15.08 s | **2.23 s** | **6.77×** |
| Full-PDF p95 | 36.39 s | **12.40 s** | **2.93×** |
| Pages/minute | 12.19 | **46.12** | **3.78×** |

**Five frozen PDFs / 20 pages; four L4 GPUs; one warm-up plus five measured calls
per PDF.** Eval pools 25 measured calls. OCR and ordered chunks must match frozen
references; embedding cosine must be at least 0.99.
[Complete results and configurations](#/doc/experiments/028-optimization-conclusion).

## Quality problems before optimization

**Qwen3-VL-2B-Instruct produced reproducible but incomplete output.** At 2,048 tokens, seaplane
pages were truncated. Matching passed because the references were also truncated.
An 8,192-token retry confirmed repeated paragraphs; larger limits and tested
penalties did not fix the setup. We switched to PaddleOCR-VL-1.6.

**Paddle also repeated a W-9 table.** Penalties of 1.0–1.2 exhausted 8,192 tokens;
1.3 stopped but omitted most of the form. Raising the table-image limit from
1.0 to 2.8 million pixels resolved the tested crop; all 182 regions in the
five-page W-9 reached the end-of-sequence marker (EOS). White padding worked in isolation but
was not retained.
[Debugging evidence and external reports](#/doc/pages/debugging).

**Batching required an engine change.** Native PaddlePaddle silently capped
recognition batches at one. vLLM enabled batching, but matching images, prompts
and precision did not reproduce PaddlePaddle exactly. We accepted engine-related
differences, enabled batch invariance for repeatability, and froze a separate
vLLM batch-1 baseline.
[Engine investigation](#/doc/experiments/006-vllm-region-batching).

EOS means generation finished, not that it was accurate. References still contain
errors in small table digits and merged cells. Speedups preserve that reviewed
baseline, not character-perfect transcription.

## What GPU and system profiles showed

- **Idle GPUs and uneven pages:** native Paddle's NACA call spent 95.3% of wall
  time in OCR. Active GPUs averaged only 11.0–28.8% busy; the fourth had no page.
  A bounded trace found 901 kernels per decode forward, dominated by matrix–vector work.
  [Native profile findings](#/doc/experiments/004-paddleocr-baseline).
- **A different tail bottleneck:** Census's 3,252-token table region kept one GPU
  about 98% busy while the others averaged 5–10%. Region batching could not divide
  that sequence's serial decoding.
  [Census study](#/doc/experiments/016-ngram-speculation).
- **Saturation remains unproven:** retained NACA GPU activity was 68.7–82.3%, with
  only seven samples per GPU. Busy time is not achieved compute or bandwidth.
  A separate confirmation saw 8.4% aggregate CPU busy; that cannot rule out a
  single-thread bottleneck.
  [Profile](#/profile/7195172c30ec973339d4) · [CPU observation](#/doc/experiments/016-ngram-speculation).

The machine had four L4s with 22.5 GiB reported memory and a 72 W limit each,
an AMD EPYC 7R13 CPU, 96 logical CPUs and 363 GiB RAM.

## Changes that paid off

Region batching, a 64-region window, dynamic pages and available-GPU dispatch
reduced idle time. Eight concurrent requests per recognizer beat four. Four-token
n-gram speculation improved tail latency; fixed 32-row decoder tiles reduced
wasted matrix work.

BF16, compilation and four-GPU embedding reduced a stage that had reached about
36% of NACA latency. Rendering inside OCR workers avoided image transfers through
the parent, adding a modest 3.4% p50 speedup against its fresh control.
[Embedding study](#/doc/experiments/010-embedding-and-scheduling) ·
[Rendering study](#/doc/experiments/025-render-placement).

## Hurdles and diminishing returns

[Single-GPU runs were too slow](#/doc/experiments/000-setup-notes), so accepted
comparisons kept four GPUs fixed. Separate runtime environments avoided
CUDA-library conflicts; eager module loading resolved a startup stall.
Profiler repairs kept instrumented calls separate from benchmark samples.

[Token-dependent tile selection](#/doc/experiments/021-decoder-matrix-tiles) initially
changed only the vocabulary head: compilation froze the decoder's larger tiles.
Fixed 32-row tiles reached the intended operations. Compilation also made cold
starts expensive: the latest 442-second suite included a 250-second first warm-up.
The gains describe warmed-up workers, not faster cold starts.

| Trial | What we learned |
| --- | --- |
| [Sixteen requests per GPU](#/doc/experiments/014-server-concurrency-16) | Slower p50, p95 and throughput than eight. |
| [Draft depth and admission limits](#/doc/experiments/028-optimization-conclusion) | Two drafts regressed; three/eight drafts and nearby admission limits did not establish a useful overall gain. |
| [Per-projection decoder columns](#/doc/experiments/023-decoder-column-widths) | Intended kernels ran, but full-PDF changes were under 1%. |
| [Rendering formats and separate render pools](#/doc/experiments/025-render-placement) | Compatible formats were near repeat variation; forced alpha changed pixels. Separate pools had mixed results and image-transfer costs. |

Rendering/OCR now accounts for **97.1% of measured time** across the dataset.
Eliminating embedding entirely would improve throughput by at most **2.5%**;
eliminating preparation and chunking yields at most **0.5%**. These are conditional
bounds, not measured speedups.

The next useful investigation is a complete Census/W-9 profile: those PDFs
account for 76.2% of processing time. Measure queue delay, prefill, tokens advanced
per verification step and achieved bandwidth before another tuning sweep.
[Remaining headroom and calculation details](#/doc/pages/theoretical-limits).
