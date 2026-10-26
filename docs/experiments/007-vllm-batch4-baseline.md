---
title: vLLM · Establishing repeatable outputs
description: Batch invariance fixed same-engine variation; batch 1 is the baseline and batch 4 is an experiment.
order: 7
status: passed
date: 2026-10-26
---

## Why we moved the baseline

The original PaddlePaddle backend processes one recognition region at a time and caps larger requested batches at one. vLLM enables concurrent region recognition. The engine comparison completed both batch 1 and batch 4; batch 4 had lower observed times on all three PDFs.

Changing engines produced differences in OCR output, including checklist icons and LaTeX formatting. Those differences were explicitly accepted as part of switching the baseline. This is a separate baseline with a shared frozen vLLM output set. The original **PaddleOCR-VL-1.6 · PaddlePaddle · Four-GPU baseline**, its measurements and its references remain intact.

Future optimizations must match the new vLLM reference OCR bytes and chunk structure, with embedding cosine at least 0.99. This transition does not establish an official speedup against the original baseline because the references differ.

## Fixed setup

The repeatability check used four L4 GPUs and batch 4 with batch invariance enabled. The subsequent comparison uses **batch 1 as the baseline**, with batch 4 on page-assigned GPUs and batch 4 across all GPUs as experiments. PaddleOCR-VL-1.6, PP-DocLayoutV3, 150 DPI rendering, the 8,192-token region ceiling with EOS checks, 512-token chunk target and fp32 embeddings are unchanged.

The same three PDFs contain seven pages: English and Spanish earthquake checklists with two pages each, and the three-page NACA scan. Capture saves a separate reference set; evaluation then runs one warm-up and one timed call per PDF against those references.

- Configuration: `experiments/pdf/06_paddleocr_vllm_batch4_baseline.yaml`
- Manifest: `data/public-v3/manifest-vllm-batch4-invariant.json`
- New references: `data/public-v3/refs-vllm-batch4-invariant/`
- Original references: `data/public-v3/refs/`

## Verification

The first separate capture completed all seven pages and 268 regions with EOS. All six verification calls also completed, but repeated vLLM calls differed in OCR bytes; one Spanish chunk had cosine 0.98882. This is a same-engine repeatability issue, separate from the accepted PaddlePaddle-to-vLLM transition. Batch-invariant execution (`VLLM_BATCH_INVARIANT=1`) then produced matching repeats on all three PDFs. Its outputs were frozen separately. A fresh evaluator worker then passed all three PDFs, including all three warm-ups and three timed calls. This was a repeatability check, not a separate batch-4 baseline. The earlier repeatability check is retained as setup evidence outside the experiment list.

The earlier batch-invariance-off runs are marked failed. The successful repeatability check is retained as setup evidence, and the [batch 1 versus batch 4 comparison](#/doc/experiments/008-gpu-region-distribution) is the main experiment. See [vLLM batch-invariance guidance](https://docs.vllm.ai/en/latest/features/batch_invariance/).

## Profiling and next experiment

[The completed batch-4 NACA profile](#/profile/28f46803e2448f4d836b) measured 8.83 s total, with OCR at 6.78 s (76.7%) and embedding at 1.92 s (21.7%). Page workers finished after 3.22, 4.33 and 6.76 s; the fourth GPU had no page. The busiest GPU averaged 93.3% busy during OCR. These are activity samples, not measured peak-compute utilization.

The next proposed lever is to distribute detected recognition regions across all four existing recognizers, then restore region and page order. This could let idle GPUs help finish the longest page. Compare that change against the frozen vLLM batch-1 baseline before considering it an improvement.
