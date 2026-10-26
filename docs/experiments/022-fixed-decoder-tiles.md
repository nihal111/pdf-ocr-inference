---
title: Compiled decoder · Fixed 32-row tiles
description: Apply the smaller matrix tile throughout the compiled BF16 decoder.
order: 22
status: passed
date: 2026-10-26
run: paddleocr_vllm_fixed_decoder_tiles/20261026-124322
profile: profiles/20261026-123707-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-Smaller-decoder-matrix-tiles-16-32-rows-b40ec8
---

## Evidence and change

The [preceding profile](#/profile/983118da65b40c30e2a3) found that token-count-dependent tile selection was fixed during compilation: only the vocabulary head used the smaller kernels. The rest of the decoder kept its 128-row tile. That experiment did not test the intended change throughout the decoder.

Use one **32-row tile** for all five screened BF16 projection shapes, including the vocabulary head. Keep the column tile at 128, reduction block at 64 and launch at four warps. A single choice survives tracing with a large prefill input and subsequent execution with small decode inputs. FP32 vision kernels remain unchanged.

This also changes large prefill matrices, so a small-kernel speedup alone is insufficient. The full-PDF measurement must include any prefill penalty.

## Verification

Five integration tests passed, including an actual dynamically compiled matrix operation traced at 2,048 rows and then executed at 5, 65 and 128 rows. Outputs matched the original kernel bit for bit; GPU trace inspection confirmed 128-thread launches for the compiled small inputs. Earlier matrix screening covered all five decoder shapes.

The full experiment keeps four L4 GPUs, server concurrency eight, four n-gram proposals, available-GPU admission eight, region window 64, dynamic page assignment and compiled BF16 embeddings. Use the five frozen PDFs with one warm-up and five measured calls each, retaining all OCR, chunk, embedding and raw-EOS checks.

Compare with the [fresh default-tile control](#/run/f541dac8cefb538a422c), and preserve the original batch-1 baseline as the dashboard anchor. The default control's p50 is 2.875 s, p95 15.732 s and throughput 36.902 pages/min.

## Full-PDF result

| Configuration | p50 | p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Fresh default tiles | 2.875 s | 15.732 s | 36.902 | 272.93 s |
| Fixed 32-row tiles | 2.528 s | 12.890 s | 43.697 | 241.66 s |

Against the fresh control, evaluator speedup ratios are **1.1371× p50**, **1.2205× p95** and **1.1841× throughput**. All five per-PDF medians improved. Against the original batch-1 baseline, the ratios are 5.9654×, 2.8231× and 3.5839× respectively.

All 30 calls passed frozen OCR and chunk checks, embedding cosine remained at least 0.999870, and all 2,790 recognition regions ended with raw EOS. The 53 recorded source hashes stayed fixed during evaluation. The original baseline and references remain unchanged.

## Follow-up profile and next lever

The [completed NACA profile](#/profile/24edbbe0670b361c831c) confirms **2,336 smaller-tile decoder launches per GPU**: 32 iterations × (18 layers × 4 projections + the vocabulary head). The previous attempt changed only 32 head launches. The generated decoder graph also now divides the token count into 32-row tiles.

All ordinary and instrumented outputs passed. Diagnostic median is 2.561 s, including 2.115 s OCR and 0.286 s embedding. Mean stage shares are 82.7% OCR and 11.3% embedding.

The trace shows another concrete limit: narrow decoder projections frequently launch only 16 blocks; GPU 3 also records 180 launches with eight blocks. A block executes on one streaming multiprocessor (SM), so these kernels can occupy at most 16 or eight of the L4's 58 SMs at once. This identifies unused compute units for those operations; it does not measure whole-GPU or memory-bandwidth saturation.

The 128-column tile produces small grids for the 1,024-wide attention-output and down projections. **Test narrower column tiles** to create more independent blocks, preserving the K=64 reduction order.

A separate fixed-row screen passed all 120 bitwise cases. Fixed 16-row tiles gave small mixed decode gains with larger prefill penalties; 64-row tiles often slowed decode while improving some prefill cases. Keep 32 rows for the next column-width screen. Raw samples and grid counts are attached to the profile.

## References

- [Preceding tile experiment](#/doc/experiments/021-decoder-matrix-tiles).
- [vLLM issue #54243: tile buckets collapse during compilation](https://github.com/vllm-project/vllm/issues/54243).
- [vLLM compilation design](https://docs.vllm.ai/en/latest/design/torch_compile/).

[Full experiment](#/run/81ce12032be0a1d87ff4)
