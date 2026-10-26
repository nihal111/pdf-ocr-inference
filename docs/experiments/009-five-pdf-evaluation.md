---
title: vLLM · Five frozen PDFs · Eval comparisons
description: Rerun batch 1 and both batch 4 configurations with one warm-up and five measured calls per PDF.
order: 9
status: passed
date: 2026-10-26
run: paddleocr_vllm_five_pdf_eval/20261026-081822
---

## What the latest profiles motivate

Two fresh studies use this experiment's best configuration, **batch 4 with regions shared across all GPUs**, including the current table-image limit. Both are linked under its run row and passed the frozen-reference checks.

| Study | Finding | Next experiment |
| --- | --- | --- |
| [NACA profile](#/profile/25b86f6693479a2fe6d5) | 5.30 s median call; embedding takes 1.89 s, or 35.5% of mean total time | Lower embedding precision and choose an embedding batch size using measured accuracy and speed |
| [Census profile](#/profile/379f7cde1f8c400e072d) | 29.78 s median call; OCR takes 89.2%. GPU busy averages 93%, 7%, 7%, 15%; one 3,252-token region controls the tail | Give waiting pages to free layout workers sooner; test region submission barriers separately |

Start with embedding because its cost appears on every PDF and the evaluator explicitly checks vector similarity. Then address OCR scheduling to improve the slow-document tail. [Follow the next experiment](#/doc/experiments/010-embedding-and-scheduling).

## Fixed workload

Five PDFs are frozen in `data/public-v3/manifest.json`: the English earthquake checklist (2 pages), NACA seaplanes (3), IRS W-9 (5), Census income (5), and Attention Is All You Need (5). The Spanish checklist was replaced by the attention paper to broaden the sample. Other PDFs were removed from the active dataset. `dataset-freeze.json` records all selected PDF hashes.

The run uses the same four NVIDIA L4 GPUs, batch invariance enabled, one warm-up and five measured calls per PDF. Each configuration contributes **25 measured full-PDF calls**, excluding its five warm-ups. All three configurations together execute 90 calls. Reference capture and profiling are separate.

## Configurations

| Experiment | Change from batch 1 baseline |
| --- | --- |
| vLLM batch 1 · Baseline | One recognition region at a time on its page worker's GPU |
| vLLM batch 4 · Pages assigned to GPUs | Raise recognition batch size to four |
| vLLM batch 4 · Regions shared across GPUs | Batch size four per recognizer; distribute regions across all four GPUs |

The models, precision, rendering, chunking, embeddings, completion checks and references are shared. The experiment is `experiments/pdf/08_vllm_five_pdf_eval.yaml`.

## Reference preparation

The English-checklist and NACA references remain immutable. The three additional references are captured with batch 1 and reviewed before timing.

W-9 initially failed raw EOS checks because its form-table crop repeated dotted filler. Penalties of 1.05, 1.1 and 1.2 did not finish; 1.3 stopped but omitted most content. Increasing the table-image limit from 1,003,520 to 2,822,400 pixels allowed the isolated crop to finish with EOS after 1,167 tokens. This table-only setting is shared by the baseline and both optimization configurations. Full-document review finished: all added pages produced nonempty Markdown, no region hit the token limit, and W-9 retained its later form/certification sections. The standard parser still preserves some charts as image references; this limitation is documented with the references. Repeated reference checks remain required before accepting speedups. [Investigation and upstream references](#/doc/pages/debugging).

## Evaluation metrics

`eval/evaluate.py` pools every measured full-PDF latency for p50 and p95. Sequential pages/minute uses total measured pages divided by total measured duration.

`eval/compare.py` must accept each baseline/candidate pair. Both reports must pass every warm-up and measured output check, share the dataset/reference digest, use matching hardware and evaluation settings, and meet the one-warm-up/five-measurement requirement.

The main visible comparison is **p50 improvement (%)**: `(baseline p50 / candidate p50 − 1) × 100`. The detail view also exposes the p95 and pages/minute speedup multipliers.

## Profiling context

The earlier NACA profiles remain diagnostic evidence for the three configurations. They are not rerun or pooled into these evaluator measurements. Their recorded configuration predates the table-resolution adjustment, and they do not describe performance on the newly added documents. Consult each profile's saved configuration when interpreting it.

## Results

**All 90 calls passed**, including 15 warm-ups and 75 measured calls. Both candidate comparisons were accepted by the unchanged evaluator. Every recognition region reached EOS, OCR and chunk structure matched the frozen references, and embedding checks passed.

| Experiment | p50 | p95 | Pages/min | p50 improvement | Suite runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 baseline | 15.08 s | 36.39 s | 12.19 | Baseline | 10m 16s |
| Batch 4 · Pages assigned to GPUs | 9.55 s | 32.20 s | 15.74 | +57.9% | 8m 03s |
| Batch 4 · Regions shared across GPUs | 5.39 s | 30.67 s | 19.89 | +179.9% | 6m 36s |

Suite runtime is the complete evaluator wall time, including worker setup, warm-ups, measured calls, validation and comparison. It excludes inference-server startup, environment inventory, reference capture and profiling. These runs recorded wall time in their sweep log to the nearest second; future runs save a monotonic timer directly.

The three-PDF pilot rows were replaced by these results on the runs page. Their raw artifacts were archived outside the active results catalog. Earlier profiles remain linked as diagnostics, with their original capture configuration clearly labeled. `dispatch-validation.json` records completed calls, EOS and actual region routing across all four GPUs.
