---
title: PaddleOCR-VL-1.6 · Region batch 4
description: The native backend overrode batch size 4 with 1; the unsupported trial was archived.
order: 5
status: invalidated
date: 2026-10-26
---

## Evidence and hypothesis

The baseline NACA profile spends 95.3% of wall time in OCR, while its busiest GPU averages 28.8% busy. Early decoding launches 901 GPU kernels per token, and about 70.6% of sampled decode kernel time is matrix–vector work. Processing multiple regions together may amortize execution overhead and improve GPU use.

## Controlled change

Set `ocr.paddle_region_batch_size` to **4**, compared with **1**. Keep four L4 GPUs, one page per GPU, 150 DPI, the same model revisions and native backend, 8,192 tokens per region with EOS required, 512-token chunking and fp32 embeddings.

The sweep runs a fresh batch-1 baseline and batch-4 candidate, each with one warm-up plus one timed call per PDF, against the same frozen references. Exact OCR bytes and chunk structure must match, with embedding cosine ≥ 0.99. One timed call per PDF does not qualify for an official speedup claim.

## Result: the native backend kept batch size 1

**This was not a valid batch-4 experiment.** PaddleX 3.7.2's native `DocVLMLocalPredictor` explicitly caps PaddleOCR-VL recognition at `PADDLEOCR_VL_LOCAL_BATCH_SIZE`, which is **1**. It replaced the requested value 4 during initialization.

The trace confirms **465 generation batches, all containing one region**, across five completed candidate pipeline calls. The trial was stopped after the constraint was confirmed. Its timings cannot establish a batch-4 speedup. The fresh batch-1 control passed all six evaluation calls.

The trial's control and candidate rows have been removed from the runs page because they add no batch-4 comparison. The control passed, but it was only a fresh batch-1 measurement for this trial; the accepted four-GPU baseline remains the reference.

The raw evidence is archived under `.site/archived-experiments/paddleocr_region_batch_4/20261026-063210/`, including `batching-validation.json`, the candidate log and generation statistics. The source check is in the installed PaddleX `inference/models/doc_vlm/predictor.py`, lines 69–78. A configured value alone does not prove the backend executed it.

## Next decision

Region batching remains a useful hypothesis, but this native backend cannot test it through the batch-size setting. A supported serving backend such as PaddleOCR's vLLM-server integration would be a separate engine experiment: first compare its batch-1 output and latency with the native baseline, then test region batching if output checks pass. Keep the same four GPUs and frozen references. No engine change has been made.
