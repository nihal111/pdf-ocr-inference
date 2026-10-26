---
title: Summary
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

Each row compares a retained change with its recorded control, in experiment
order. **Positive percentages mean faster**, using eval's `(speedup − 1) × 100`
for each metric. These incremental gains are not additive.

| Change | p50 gain | p95 gain | Pages/min gain |
| --- | ---: | ---: | ---: |
| [OCR regions per GPU: 1 → 4](#/run/d8a8a05ea1b9bc9e9268/vllm_batch_4) | **+57.9%** | +13.0% | +29.1% |
| [Share regions across all GPUs instead of page-assigned GPUs](#/run/d8a8a05ea1b9bc9e9268/vllm_batch_4_all_gpus) | **+77.3%** | +5.0% | +26.3% |
| [Embedding: FP32 → BF16; chunk batch 32 → 16](#/run/2414bf81cf27a32c8c2b/embed_bf16_b16) | +28.6% | +7.4% | +12.4% |
| [Region submission window: 16 → 64](#/run/1c2845682cf96081e45c/region_window_64) | +10.0% | +0.2% | +4.5% |
| [Assign each page to the next free worker](#/run/9333355f0563a600b5d8/combined_scheduling) | −4.2% | +5.5% | +6.3% |
| [Server admission: 4 → 8 requests per GPU](#/run/e618414568a8c437d78a/server_8_dynamic) | +20.4% | +4.1% | +5.7% |
| [Compile embedding inference](#/run/09dd48268f7b40e92830/server_8_compiled_embeddings) | +5.0% | +1.6% | +2.1% |
| [Verify four n-gram draft tokens per decoder step](#/run/654d9b217549709275f8/server_8_ngram_gpu_4) | +3.0% | **+59.5%** | **+33.5%** |
| [Dispatch regions when a GPU has capacity](#/run/7a9c93660221ff8d0318/available_gpu_8) | +4.1% | −1.8% | +1.0% |
| [Use fixed 32-row decoder matrix tiles](#/run/81ce12032be0a1d87ff4/decoder_tiles_fixed32) | +13.7% | +22.0% | +18.4% |
| [Embedding replicas: 1 → 4 GPUs](#/run/901be5b3884c7556a3dd/embedding_four_gpus) | +7.0% | +3.4% | +3.9% |
| [Render inside OCR workers, avoiding parent image transfers](#/run/40e811453ad54f72d35a/render_in_page_workers) | +3.4% | +0.2% | +0.9% |

Region batching and sharing produced the largest p50 gains. Speculative decoding
produced the largest tail-latency and throughput gains. Dynamic page assignment
initially traded a slower p50 for better p95 and throughput; raising admission to
eight then improved all three metrics.

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
[Calculation details](#/doc/pages/theoretical-limits).

The final configuration passed the frozen-reference checks and delivered a
**6.77× p50 speedup** on the five-PDF benchmark, using the same four L4 GPUs.
