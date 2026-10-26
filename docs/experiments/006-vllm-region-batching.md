---
title: PaddleOCR-VL-1.6 · vLLM region batching
description: Batch invariance was off; these runs failed to establish repeatable vLLM output.
order: 6
status: failed
date: 2026-10-26
run: paddleocr_vllm_region_batching/20261026-071555
---

## Question and fixed settings

The native Paddle backend caps region recognition at one, so the earlier batch-size-only trial could not exercise batching. This experiment uses vLLM 0.32.0 for recognition while keeping Paddle's layout parser and the same PaddleOCR-VL-1.6 checkpoint.

Keep the four L4 GPUs, one page per GPU, three smallest PDFs, 150 DPI rendering, 8,192 generated tokens per region with raw EOS checks, 512-token chunking and fp32 embeddings fixed. Each variant runs one warm-up and one timed call per PDF through the evaluator. The existing reference files remain fixed. One repeat does not qualify for an official speedup.

The two variants are `vllm_batch_1` and `vllm_batch_4`. Four independent localhost recognizers serve the four page workers. Region images use lossless PNG instead of the stock client's JPEG transport, preserving crop pixels.

## Result

**These runs had batch invariance disabled and are marked failed.** Actual batch-4 execution completed, but later checks confirmed variation between repeated vLLM calls. Engine-related differences from PaddlePaddle were accepted; same-engine repeatability required enabling batch invariance. Both variants completed one warm-up and one timed call on all three PDFs. Every recognition region finished with EOS. The batch-4 variant recorded **128 four-region batches**, four batches of three and six batches of two across its six PDF calls: 536 region completions in total. The batch-1 control recorded 536 singleton batches.

The table shows observed timed-call durations, including calls whose output checks failed. These are **not accepted speedups**.

| PDF | Accepted native baseline | vLLM batch 1 | vLLM batch 4 | vLLM reference checks |
| --- | ---: | ---: | ---: | --- |
| English earthquake checklist · 2 pages | 21.81 s | 9.53 s | 7.70 s | Both fail OCR bytes and some embedding similarities |
| Spanish earthquake checklist · 2 pages | 30.90 s | 11.54 s | 8.73 s | Both fail OCR bytes and some embedding similarities |
| NACA seaplanes · 3 pages | 42.73 s | 12.59 s | 8.80 s | Both fail OCR bytes; embedding similarities pass |

The original references and evaluator are unchanged. PaddlePaddle remains the original baseline. The [repeatable vLLM references](#/doc/experiments/007-vllm-batch4-baseline) freezes separate outputs for subsequent optimizations. NACA profiling for both vLLM variants is complete.

## Profiling findings

The busiest GPU now averages **91.8% busy at batch 1 and 93.3% at batch 4**, compared with 28.8% in the native baseline. Its mean power is about 70 W against a 72 W limit. This shows much less idle time on the worker that determines completion, but does not establish percentage of peak arithmetic or memory bandwidth.

Each profile includes one warm-up, one ordinary timed NACA call, and a separate full instrumented call. The timings below exclude instrumentation.

| NACA metric | vLLM batch 1 | vLLM batch 4 |
| --- | ---: | ---: |
| Complete pipeline | 12.59 s | 8.83 s |
| OCR | 10.54 s · 83.7% | 6.78 s · 76.7% |
| Embedding | 1.91 s · 15.2% | 1.92 s · 21.7% |
| Rendering | 0.121 s | 0.118 s |
| Chunking | 0.020 s | 0.023 s |
| Maximum running regions on each active GPU | 1 | 4 |

Shares are stage duration divided by complete-call duration. With one ordinary timed call, its mean and median coincide.

| GPU / page | Batch-1 worker duration | Batch-4 worker duration | Mean GPU busy · batch 1 / 4 | Mean power · batch 1 / 4 |
| --- | ---: | ---: | ---: | ---: |
| 0 / page 1 | 4.17 s | 3.22 s | 36.9% / 46.8% | 49.1 / 50.7 W |
| 1 / page 2 | 6.55 s | 4.33 s | 54.7% / 61.3% | 54.8 / 58.7 W |
| 2 / page 3 | 10.53 s | 6.76 s | 91.8% / 93.3% | 70.8 / 69.7 W |
| 3 / no page | — | — | 0% / 0% | 27.5 / 27.7 W |

GPU activity and power means cover the OCR window, including time after shorter pages finish. Sampling every second produced only ten readings per GPU for batch 1 and six for batch 4. The fourth GPU retains its recognizer but has no assigned page in this three-page PDF. Redistributing regions across workers could address that imbalance in a future experiment; embedding is also becoming a larger fraction of total latency.

Scheduler samples every 100 ms confirm up to four running regions on each active GPU at batch 4. Bounded PyTorch traces cover the first 32 engine iterations on each active recognizer, including vision, prefill and decoding. An iteration can contain multiple regions, so kernel counts in these traces are not directly comparable to the native profile's kernels per decode forward. Raw traces, grouped operators, physical device memory, clocks, power and server metrics are available in the profile artifacts. Hardware counters for achieved compute throughput, DRAM bandwidth and occupancy were not collected.

Both full instrumented outputs completed all 71 regions with EOS, but failed exact OCR matching. NACA batch 1 differs at two LaTeX spacing sites; batch 4 differs at four. All NACA chunk embeddings remain above 0.99 cosine. English and Spanish evaluator calls also fail some embedding comparisons, with minimum cosines of 0.82582 and 0.87893 respectively.

**Decision:** mark batch-invariance-off runs as failed. Preserve PaddlePaddle as superseded history. Use vLLM batch 1 with batch invariance enabled as the baseline; batch 4 is an optimization experiment. The next proposed experiment distributes recognition regions across all four GPUs to reduce page imbalance, while requiring output matching against the new vLLM baseline.

- [Batch-1 detailed profile](#/profile/7870dda6ed09d4d0bdea)
- [Batch-4 detailed profile](#/profile/28f46803e2448f4d836b)

## Integration and correctness investigation

The four localhost servers started successfully after CUDA toolkit fixes and selecting vLLM's PyTorch sampler path for this greedy workload. The client also requires `--trust-request-mm-kwargs` to preserve Paddle's per-region image-size settings; these servers accept requests only on localhost.

The first complete English PDF call finished all 99 recognition regions with EOS, but differed from the native reference in five decorative-icon outputs and bullet/line-break formatting. Two chunk embeddings fell below 0.99 cosine. A representative crop produced **identical image tensors and identical prompt token IDs** in both stacks.

Inspection found a precision mismatch: native Paddle declares `_keep_in_fp32_modules = ["visual", "mlp_AR"]`, while vLLM initially loaded those components in bf16. The native model holds 2,901,046,528 parameter bytes; the all-bf16 checkpoint contains 958,588,736 parameters. The local vLLM extension keeps the vision encoder and visual projector in fp32, preserves the bf16 language decoder, and casts image embeddings at the same boundary as native Paddle. **Matching that policy did not restore exact output matching.** The timed table above uses the aligned policy for both variants.

The adapter checks raw generated token IDs and finish reasons before Markdown assembly, records submitted region batches, and rejects silently overridden batch settings. Diagnostic profiling captured scheduler occupancy, GPU activity and bounded engine traces. Timed measurements exclude the extra instrumented call.

These findings rule out image transport, the checked crop's preprocessing/tokenization, and the identified component-precision mismatch as sufficient explanations for the remaining output differences. They do not prove numerical equivalence of Paddle and vLLM kernels. No output cleanup or reference replacement was used to make a failed comparison pass.

## Reproduction setup

Install the isolated recognizer environment with `requirements-vllm.txt`, then install the local model extension using `uv pip install --python .venv-vllm/bin/python --no-deps -e engine/paddle_vllm_plugin`. The server launcher is `lab.paddle_servers`; it records each GPU UUID and complete launch command. Both batch sizes use the same server configuration and precision policy.

## References

- [PaddleOCR-VL-1.6 model and server integration](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [vLLM PaddleOCR recipe](https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html)
- [Native backend batching constraint](#/doc/experiments/005-region-batching)
