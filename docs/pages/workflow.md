---
title: How we choose experiments
description: Measure the baseline, find the slow work, and test one useful change.
order: 3
---

**Measure → profile → predict → change → compare → record.** Measurements determine the next experiment. The backlog is a set of ideas, not a fixed sequence.

## The cycle

| Step | What to do |
| --- | --- |
| **1. Measure** | Establish a passing baseline on the frozen dataset and hardware. |
| **2. Profile** | Record where time goes and what the CPUs and GPUs are doing. Find the bottleneck: the work that limits overall speed. |
| **3. Predict** | State a testable hypothesis: what should improve, why, and what result would reject the idea. |
| **4. Change** | Make one focused change where practical. Check outputs with a short trial first. |
| **5. Compare** | Run the candidate and a fresh control of the previous best configuration with one warm-up and five measured calls per PDF. Compare against the original accepted baseline and the control; inspect p50, p95, pages/minute and individual PDF results. |
| **6. Record** | Save the evidence, change, correctness results, timings and decision in the journal. |

A **candidate** is the modified pipeline being tested. It must preserve the baseline’s outputs under the [evaluation rules](#/doc/pages/baseline).

The original vLLM batch 1 report remains the dashboard's comparison baseline. Later sweeps can copy this qualified report byte for byte, recording its source and SHA-256, and run the unchanged `eval/compare.py` against new reports. The copied report is explicitly marked as an earlier measurement and does not create a duplicate experiment row. A fresh control checks for performance drift; it does not replace the baseline or references.

Repeat controls appear inside the related experiment's expanded details and full page. Their original reports, per-PDF measurements and profiles remain linked; the main runs table lists configuration changes and historical baselines.

The runs table shows p50 improvement against both the original baseline and the previous best configuration each experiment built on. The latter uses the recorded fresh control where available and links to the comparison measurement. Expanded details also show p95 and throughput changes. These percentages use `(eval speedup − 1) × 100`; repeat verifications are labeled separately.

## Example hypothesis

> If some GPUs are idle while another is still reading a page, sharing its regions across GPUs could shorten the complete PDF call. Keep the change only if outputs pass and measured performance improves.

Here, a **region** is a detected paragraph, table or other page area. The profile must establish that the idle-GPU pattern actually occurs.

## Decide the next step

- **Faster and correct:** record the gain and check whether a different stage now limits speed.
- **Slower or incorrect:** investigate the cause; preserve useful failure evidence.
- **Unclear:** collect more measurements before drawing a conclusion.

Changing the OCR model or how page images are prepared requires a separate quality review and baseline decision. Never replace frozen references merely to make an optimization pass.

[Read a profile](#/doc/pages/profiling) · [Experiment journal](#/journal) · [Experiment runs](#/runs)
