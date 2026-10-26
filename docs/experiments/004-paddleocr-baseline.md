---
title: PaddleOCR-VL-1.6 · PaddlePaddle · Four-GPU baseline
description: Establish complete page parsing before timing and profiling on four L4 GPUs.
order: 4
status: superseded
date: 2026-10-26
run: pdf_public_v3_paddleocr_vl_16_baseline/20261026-061753
profile: profiles/20261026-062415-naca-seaplanes-baseline-ff54f4
---

## Pipeline

The OCR stage now uses the official [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) page parser and [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3). Four persistent workers each hold both models, taking one page at a time from the current PDF. Recognition regions are processed with a batch size of one.

The original experiment is `experiments/pdf/00_public_v3_baseline.yaml`. It retains the three smallest PDFs, 150 DPI rendering, a 512-token chunk target, and fp32 Qwen3-Embedding-0.6B on GPU 0. Each recognition region must emit EOS within 8,192 tokens. Page outputs return to original order before chunking.

The native parser supplies task-specific prompts for text, tables, formulas and other regions. It retains headers and footers in Markdown. Page coverage, reading order and numerical recognition still require visual review.

OCR uses PaddleOCR's **native PaddlePaddle inference backend**, while embeddings use **sentence-transformers/PyTorch**. [Inference-engine details](#/doc/pages/pipeline).

## Runtime and model revisions

Paddle workers run in `.venv-paddle`: PaddleOCR 3.7.0, PaddleX 3.7.2 and PaddlePaddle 3.2.1 with CUDA 12.6. The parent and embedder retain the original `.venv` PyTorch/CUDA 13.0 environment. Separating processes and environments avoids overlapping CUDA-library files.

| Model | Revision |
| --- | --- |
| PaddleOCR-VL-1.6 | `c5630abae1d940eafe0697512a0325494b02ab42` |
| PP-DocLayoutV3 | `241f8bdfc77a7c7bee915a5057aaee58c235a8d3` |

## First setup check

The complete three-page NACA scan returned 25 embedding chunks. All 71 recognition regions emitted EOS: 18 on page 1, 23 on page 2 and 30 on page 3. The check took 59.23 seconds including model startup.

This confirms execution and normal generation termination. It is not a steady-state latency measurement or a claim of perfect OCR accuracy. Page transcriptions were subsequently reviewed during reference capture.

## Evaluation and profiling

Capture and review all seven selected pages, then measure each PDF sequentially with one warm-up and one timed call. Exact OCR bytes and chunk structure must match the new references, and embedding cosine must be at least 0.99. One timed sample per PDF does not qualify for an official speedup comparison.

After the baseline passes, collect stage timings, native Paddle CPU/GPU traces, per-GPU telemetry, and bounded hardware-counter evidence where available. Native detailed tracing has completed; its output was checked against the frozen NACA reference. Choose optimization experiments only from the new evidence.

## Reference review

All seven pages returned. **All 268 recognition regions emitted EOS**, with at most 280 generated tokens in any region. The three NACA pages match the reviewed setup output exactly. No long repeated passages were observed; checklist sections, report body and tables I–V are present.

These are execution-baseline references, not character-perfect ground truth. Decorative icons sometimes become spurious characters or emoji, Spanish text has some recognition errors, and the tiny scanned NACA tables contain wrong digits and merged-cell errors. For example, Table V's first maximum load is transcribed as 23,750 rather than 28,750. Future speed experiments must match these frozen outputs; a quality improvement would require a separately identified baseline.

The first capture stalled in CUDA module loading before completing its first PDF. The retry with `CUDA_MODULE_LOADING=EAGER` completed all three PDFs. This runtime setting is fixed for evaluation and profiling. The stalled attempt is retained only as local setup evidence, not a measured experiment.

## Results

All **3 documents passed**, including all 3 warm-ups and **3 measured calls**. Completed at 2026-10-26T06:24:13.813712+00:00.

This initial measurement has one timed sample per document. It does not qualify for an official speedup comparison, which requires at least five measured repeats per PDF.

| Metric | Result |
| --- | ---: |
| Pooled document latency p50 | 30.905 s |
| Pooled document latency p95 | 41.546 s |
| Sequential throughput | 4.400 pages/min |

| Document | p50 (s) | p95 (s) | Minimum cosine |
| --- | ---: | ---: | ---: |
| earthquake-checklist-en | 21.813 | 21.813 | 1.00000000 |
| earthquake-checklist-es | 30.905 | 30.905 | 1.00000000 |
| naca-seaplanes | 42.728 | 42.728 | 1.00000000 |

## Stage evidence

Stage shares use mean stage time divided by mean total time, excluding warm-ups.

| Stage | Mean duration per document (s) | Share |
| --- | ---: | ---: |
| render | 0.067 | 0.21% |
| ocr | 30.315 | 95.29% |
| chunk | 0.015 | 0.05% |
| embed | 1.418 | 4.46% |

The largest stage is **ocr**. These timings identify where wall time is spent; operator traces are needed to explain the operations underneath.

## Detailed profiling findings

The ordinary NACA profile call took **43.27 s**, including **41.23 s of OCR (95.3%)** and **1.90 s of embedding (4.4%)**. It follows one warm-up and excludes the additional instrumented call.

| GPU | Assigned page | Worker duration | Mean GPU busy during OCR | Mean power | Peak device memory used |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 13.00 s | 11.0% | 35.8 W | 13.92 GiB |
| 1 | 2 | 22.69 s | 16.4% | 36.4 W | 9.46 GiB |
| 2 | 3 | 41.20 s | 28.8% | 43.4 W | 9.45 GiB |
| 3 | Idle | 0 s | 0% | 27.8 W | 8.94 GiB |

The four-worker assignment budget is only about 46.6% occupied, calculated from the sum of worker durations divided by four times OCR wall time. GPU 2 determines completion time; the other workers finish earlier or receive no page. Assigned-worker time and GPU kernel activity are different measurements.

Each active worker's native trace covers **3 prefills and 29 decode forwards** across its first few recognition regions. CUDA launch correlation identifies **901 GPU kernels per decode forward**. Matrix–vector kernels account for about **70.6% of sampled decode kernel time**. The numerous small operations and low ordinary GPU activity support testing region batching. CPU trace timings include instrumentation overhead, so they are not substituted for baseline latency.

The first trace attempt failed in a wrapper, which was corrected without changing pipeline execution. The successful retry retained the original ordinary measurement and telemetry, used a fresh warm-up, and checked its full output against the fixed NACA reference. Raw traces, grouped operators and `native-trace-summary.json` are available in the profile artifacts.

Hardware counters for achieved compute throughput, DRAM bandwidth and occupancy have not been collected. These results establish idle time and workload imbalance; they do not quantify percentage of peak arithmetic or memory bandwidth. All 71 recognition regions emitted EOS; known character/table recognition errors remain in the frozen baseline as documented above.

## Next experiment

[The region-batch-4 trial](#/doc/experiments/005-region-batching) revealed that this native PaddleX backend caps batches at 1 and overrides the requested value. It was invalidated after 465 observed batches confirmed size 1. Testing actual region batching requires a separate backend experiment, with the same four GPUs and fixed output checks. No batch-4 speedup is claimed.

[The subsequent vLLM experiment](#/doc/experiments/006-vllm-region-batching) completed actual batch-4 execution and detailed profiling. It reduced observed latency and produced engine-related output differences. Those differences were accepted to establish a [vLLM batch-1 baseline](#/doc/experiments/007-vllm-batch4-baseline); this native run and its references remain the original baseline.
