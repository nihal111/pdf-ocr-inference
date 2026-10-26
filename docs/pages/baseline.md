---
title: Baseline & evaluation
description: What we compare, how we measure it, and what counts as passing.
order: 2
---

The **baseline** is our starting configuration. Each experiment changes part of it, then checks whether the complete PDF pipeline runs faster while preserving its outputs.

## Baseline configuration

| Setting | Value |
| --- | --- |
| OCR model and engine | **PaddleOCR-VL-1.6**, run through **vLLM** |
| Hardware | **Four NVIDIA L4 GPUs**, fixed across comparisons |
| Recognition batch | **One region per GPU at a time**; a region is a detected paragraph, table or other page area |
| Dataset | **Five frozen PDFs · 20 pages**, at most five pages each |

Batch invariance is enabled to reduce output differences caused by changing batch sizes. See [the pipeline](#/doc/pages/pipeline) for how the stages work and [the dataset](#/doc/pages/dataset) for the selected PDFs.

## Measurement method

For **each PDF**, run:

1. **One warm-up:** exercise the pipeline; exclude its time.
2. **Five measured calls:** process the entire PDF each time.

That gives **25 measured calls per configuration**. PDFs run sequentially. Model loading, reference capture and profiling are excluded from these timings; results are never cached between calls.

| Metric | Meaning |
| --- | --- |
| **p50 latency** | The middle value when all 25 measured call times are sorted |
| **p95 latency** | The 95th percentile: an estimated cutoff for the slowest 5% of calls |
| **Pages/minute** | Total measured pages divided by total measured minutes |
| **Suite runtime** | Elapsed time for this configuration's full evaluation: worker setup, all warm-ups and measured calls, validation and comparison |

Suite runtime is shown in minutes and seconds. It excludes inference-server startup, environment inventory, reference capture and profiling; it does not affect eval's speedup calculation. Older runs without recorded wall time show a dash.

p50 and p95 pool the measured calls from **all five PDFs**. The runs page shows latency in seconds and highlights:

```text
p50 improvement (%) = (baseline p50 / experiment p50 − 1) × 100
```

**+100% means 2× speedup.** The small indicators beneath each metric use its corresponding eval speedup: green ↑ means faster, red ↓ means slower. Individual PDF timings remain available in each run’s details.

## Passing and accepted comparisons

**Frozen references** are saved baseline outputs used to check later runs. They stay unchanged throughout a comparison.

Every warm-up and measured call must pass:

- **Text:** exact OCR bytes, including whitespace.
- **Chunks:** same count, IDs, order and vector lengths.
- **Embeddings:** finite, nonzero vectors with cosine similarity ≥ **0.99** to each reference. Cosine similarity compares vector directions.

Reference preparation also requires normal generation completion and page review. Matching a reference does not independently prove OCR accuracy.

A speedup is **accepted** only when `eval/compare.py` validates both passing reports with matching PDFs, references, hardware, driver and measurement settings. The website revalidates saved comparisons before displaying percentages.

## Run the comparison

With the local inference servers running and references prepared:

```bash
.venv/bin/python -m lab.pdf_sweep experiments/pdf/08_vllm_five_pdf_eval.yaml
```

This runs a fresh batch-1 baseline and both batch-4 experiments. [View experiment runs](#/runs).

History and investigations: [setup notes](#/doc/experiments/000-setup-notes) · [switch to vLLM](#/doc/experiments/007-vllm-batch4-baseline) · [table repetition](#/doc/pages/debugging).
