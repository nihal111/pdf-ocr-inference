---
title: Reading a profile
description: Find where time goes, understand GPU activity, and choose the next experiment.
order: 4
---

**Profiling records where time goes and what the hardware is doing.** Use it to explain a slow PDF call and choose an optimization. The evaluator establishes the speedup.

## Read the evidence in this order

| Evidence | Question it answers |
| --- | --- |
| **Stage timings** | How much time goes to rendering, OCR, chunking and embedding? |
| **Per-GPU activity** | Which GPUs have work, and which spend time idle? |
| **Engine trace** | Which CPU/GPU operations, transfers or waits account for the time? A trace is a timeline of recorded operations. |
| **Output checks** | Did the complete call finish and match the reference? |

Stage timings measure **elapsed time**, including CPU work, transfers and waiting inside each stage. Bars show **average stage time ÷ average full-call time**; the durations beside them are **medians**, the middle measured values.

## Are the GPUs fully used?

Inspect each GPU’s activity, memory use, power and assigned work together. For vLLM, also check how many recognition requests are running or waiting.

- **High activity** means the GPU was busy during much of the sample. It does not establish how close it was to its maximum processing rate.
- **Unused GPU memory** shows storage space, not necessarily spare processing capacity.
- **Hardware counters** are specialized measurements needed to assess achieved compute or memory-transfer rates.

GPU readings are sampled once per second; vLLM request counts every 100 milliseconds. Short calls provide few samples, so treat their averages cautiously.

## Keep timings separate

- Ordinary timed calls produce the stage summary, with warm-ups excluded.
- A separate call records the detailed trace. This extra recording can affect execution time.
- Traces may cover only part of generation. Check the saved recording window before generalizing.
- Operations can overlap. Adding their durations does not give the full PDF’s elapsed time.

Neither trace timings nor profile samples contribute to the evaluator’s speedup comparison.

## Open the supporting details

Expand an experiment on [the runs page](#/runs), then open its profile for GPU measurements, operation tables and downloadable traces. Each profile records its PDF, configuration, hardware and reproduction command.

Earlier profiles remain evidence for their **saved configuration and document**. They do not automatically describe a newer setup or every PDF in the dataset.

[How we choose experiments](#/doc/pages/workflow) · [Evaluation metrics](#/doc/pages/baseline)
