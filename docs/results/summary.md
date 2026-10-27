---
title: Summary
order: 0
---

**The problem:** Turn PDFs into complete, ordered text and searchable embeddings
faster, while preserving baseline OCR text and chunk structure and keeping
embedding cosine similarity at or above 0.99.

**Benchmark:** Five frozen public PDFs, totaling 20 selected pages: [*Earthquake Preparedness Checklist — English*](#/dataset/public-v3/earthquake-checklist-en/1) (2), [*The Estimation of Maximum Load Capacity of Seaplanes and Flying Boats*](#/dataset/public-v3/naca-seaplanes/1) (3), IRS [*Form W-9*](#/dataset/public-v3/irs-w9/1) (5), [*Income in the United States: 2023*](#/dataset/public-v3/census-income/1) (5), and [*Attention Is All You Need*](#/dataset/public-v3/attention-paper/1) (5).

**System:** Four NVIDIA L4 GPUs (22.5 GiB reported memory and a 72 W limit each); AMD EPYC 7R13, 96 logical CPUs, 363 GiB RAM.

**Evaluation:** Sequential PDF calls with one warm-up and five measured repeats per PDF (25 measured calls), checked against frozen baseline references.

## Document pipeline

| Stage | Transformation | Technology |
| --- | --- | --- |
| Render | PDF pages → images at 150 DPI | pypdfium2 |
| OCR | Page images → text, tables and formulas in reading order | PP-DocLayoutV3 on PaddlePaddle for layout; PaddleOCR-VL-1.6 on vLLM for recognition |
| Chunk | Ordered Markdown → text chunks with heading context and a 512-token target | Markdown chunker |
| Embed | Text chunks → vectors for search by meaning | Qwen3-Embedding-0.6B through sentence-transformers on PyTorch |

## Results and scope

| Eval metric | vLLM batch-1 baseline | Best optimized configuration | Speedup |
| --- | ---: | ---: | ---: |
| Full-PDF p50 | 15.10 s | **2.34 s** | **6.44×** |
| Full-PDF p95 | 36.11 s | **12.41 s** | **2.91×** |
| Pages/minute | 12.27 | **45.23** | **3.69×** |

Both variants were rerun on 2026-10-27 from clean revision `12f9cf6` and
passed all 30 calls each against the existing references; no recapture was needed.
Time-to-first-result was **41.40 s baseline / 131.34 s optimized**. This includes
client startup and the first warm-up, but excludes recognizer-server startup.
[Fresh paired evaluation](#/run/a6e67f74fcdf9cbbf963) ·
[Complete results and configurations](#/doc/experiments/028-optimization-conclusion).

## Quality problems before optimization

**Qwen3-VL-2B-Instruct produced reproducible but incomplete output.** With a 2,048-token
output limit, page transcriptions from the three-page report
[*The Estimation of Maximum Load Capacity of Seaplanes and Flying Boats*](#/dataset/public-v3/naca-seaplanes/1)
were truncated. Matching passed because the references were also truncated.
An 8,192-token retry confirmed repeated paragraphs; larger limits and tested
penalties did not fix the setup. We switched to PaddleOCR-VL-1.6.

**PaddleOCR-VL-1.6 repeated a table in IRS Form W-9.** On the
[*Form W-9* benchmark](#/dataset/public-v3/irs-w9/1), repetition penalties of
1.0–1.2 exhausted 8,192 tokens; 1.3 stopped but omitted most of the form.
Raising the table-image limit from 1.0 to 2.8 million pixels resolved the tested
crop; all 182 regions across the five selected pages reached the end-of-sequence
marker (EOS). This setting is shared by the working baseline and all optimizations.
[Debugging evidence and external reports](#/doc/pages/debugging).

**Repeatable vLLM output required batch invariance.** Enabling it made repeated
calls reproduce the same OCR text. The working baseline uses PaddleOCR-VL-1.6 on
vLLM with one recognition region per GPU at a time; every optimization is checked
against its frozen references.
[Baseline validation](#/doc/experiments/007-vllm-batch4-baseline).

EOS means generation finished, not that it was accurate. References still contain
errors in small table digits and merged cells. For example, on
[page 3 of *The Estimation of Maximum Load Capacity of Seaplanes and Flying Boats*](#/output/d8a8a05ea1b9bc9e9268/naca-seaplanes/3/rendered),
Table V's first maximum-load value is **28,750** in the PDF, but the frozen OCR
reference records **23,750**. Speedups preserve that reviewed baseline, not
character-perfect transcription.

## What GPU and system profiles showed

- **OCR was the first optimization target:** In the initial five-PDF evaluation,
  OCR consumed **89.9% of measured pipeline time** in the vLLM batch-1 baseline,
  excluding warm-ups. Each GPU read one region at a time, and regions were tied
  to the GPU handling their page. Increasing recognition batches to four reduced
  pooled full-PDF p50 from **15.08 to 9.55 seconds**; sharing regions across all
  four GPUs reduced it further to **5.39 seconds**, with embedding unchanged.
  [OCR batching and distribution results](#/doc/experiments/009-five-pdf-evaluation).
- **Long table decoding controlled tail latency:** Before speculative decoding,
  a table region in the fifth selected page of
  [*Income in the United States: 2023*](#/dataset/public-v3/census-income/5)
  generated 3,252 tokens. One GPU stayed about 98% busy while the others averaged
  5–10%. Region batching could not divide that sequence's serial decoding,
  motivating the four-token speculative decoding experiment.
  [Speculative decoding study](#/doc/experiments/016-ngram-speculation).

  **How speculative decoding works:** Normally, the OCR model generates one
  token (a small piece of text) per step. If the phrase “Median household income”
  already appeared in the current table's transcription, our version can reuse
  what followed it as a guess. The OCR model checks up to four guessed tokens
  together against the image.

  | Decoding example | Text so far | Next token pieces |
  | --- | --- | --- |
  | Earlier text in this region | Median household income | was · 80 · , · 000 |
  | Current image matches | Median household income | was · 80 · , · 000 |
  | Current image differs | Median household income | was · 90 · , · 000 |

  Correct guesses let one model pass advance several tokens. At the first
  mismatch, the model supplies the correct token and discards the remaining
  guesses. Every evaluated output still had to match the same frozen OCR reference.

- **Embedding became significant after the initial OCR gains:** With four-region
  batches shared across GPUs,
  [*The Estimation of Maximum Load Capacity of Seaplanes and Flying Boats*](#/dataset/public-v3/naca-seaplanes/1)
  spent 1.89 seconds in embedding, or 35.5% of mean call time. This prompted the
  subsequent tests of lower embedding precision and smaller batches.
  [Embedding profile](#/profile/25b86f6693479a2fe6d5).
- **Best optimized configuration:** When processing
  [*The Estimation of Maximum Load Capacity of Seaplanes and Flying Boats*](#/dataset/public-v3/naca-seaplanes/1),
  GPU activity was 68.7–82.3%, with only seven samples per GPU.
  Busy time is not achieved compute or bandwidth, so saturation remains unproven.
  [GPU profile](#/profile/7195172c30ec973339d4).

## Changes that paid off

Each row compares a retained change with its recorded control, in experiment
order. Expand a change for its explanation and diagram.

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

[Token-dependent tile selection](#/doc/experiments/021-decoder-matrix-tiles) initially
changed only the vocabulary head: compilation froze the decoder's larger tiles.
Fixed 32-row tiles reached the intended operations. Compilation also made cold
starts expensive: an earlier 442-second suite included a 250-second first warm-up.
The fresh time-to-first-result observations are reported above; the latency and
throughput gains describe warmed-up workers.

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
**6.44× p50 speedup** on the five-PDF benchmark, using the same four L4 GPUs.
