---
title: Theoretical limits
description: Measured cost, hardware bounds and remaining headroom.
order: 7
---

**Rendering and OCR account for 97.1% of measured time. Eliminating embedding would improve throughput by at most 2.5%. Hardware saturation remains unproven.**

Scope: **five frozen PDFs, 20 pages, four NVIDIA L4 GPUs**, one warm-up and five measured calls per PDF. [Evaluation](#/run/b75edaea6a9b5182a677/server_8_draft4_control) · [Calculation data](/artifact?path=analysis/theoretical-limits-20261026-retained.json).

## Measured cost

| Eval metric | Result | Speedup vs baseline |
| --- | ---: | ---: |
| Pooled p50 | **2.229 s** | **6.77× / +576.8%** |
| Pooled p95 | **12.402 s** | **2.93×** |
| Pages/minute | **46.122** | **3.78×** |

Configuration: eight requests per recognizer, four n-gram drafts, 32-row decoder tiles and shared region scheduling. Embeddings use compiled BF16, batch 16, across all four GPUs. Pages render inside OCR workers.

The following totals sum **one mean call per PDF**, using five measured calls each:

| Stage | Time | Share of full-call time |
| --- | ---: | ---: |
| Rendering + OCR | **25.267 s** | **97.1%** |
| Embedding | 0.625 s | 2.4% |
| Preparation + chunking | 0.121 s | 0.5% |

Rendering overlaps OCR, so their combined wall time is measured once. **Census and W-9 account for 76.2% of total processing time**; they dominate throughput and tail latency. NACA and Attention overlap around pooled p50.

## Remaining speedup

For each measured call, replace `stage time` with `stage time / speedup`, then recompute eval's pooled metrics. These are **conditional estimates**, with all other work fixed.

| Hypothetical change | p50 | p95 | Pages/min | Throughput gain |
| --- | ---: | ---: | ---: | ---: |
| Embedding 2× faster | 2.193 s | 12.302 s | 46.682 | +1.2% |
| Embedding takes zero time | 2.157 s | 12.202 s | 47.256 | **+2.5% ceiling** |
| Rendering + OCR 1.10× faster | 2.034 s | 11.297 s | 50.588 | +9.7% |
| Rendering + OCR 2× faster | 1.177 s | 6.324 s | 89.656 | +94.4% |

Removing embedding entirely caps **p50 speedup at +3.3%**. Removing preparation and chunking caps throughput gain at **+0.5%**. Further substantial gains must come from rendering/OCR.

## Hardware bounds

The machine has **22.5 GiB reported memory per GPU**, 96 logical CPUs and 363 GiB system RAM.

| Resource | One L4 | Four L4s, independent work |
| --- | ---: | ---: |
| Device-memory bandwidth | 300 GB/s | 1,200 GB/s |
| FP32 compute | 30.3 TFLOP/s | 121.2 TFLOP/s |
| Dense BF16 Tensor Core compute | 121 TFLOP/s | 484 TFLOP/s |
| Power limit | 72 W | 288 W |

These are advertised peaks. NVIDIA's 242 BF16 TFLOP/s figure includes sparsity; **121 TFLOP/s is the dense limit** used here. [Specifications][l4]

Each GPU holds approximately **2.90 GB of OCR weights**—FP32 vision and BF16 decoding—and **1.19 GB of BF16 embedding weights**. These figures exclude layout detection, caches and activations.

Four replicas process independent regions. Their bandwidth does not automatically combine for a single long sequence.

## OCR: the long-sequence bound

Census's page-5 table produces **3,252 tokens with EOS**. Its six-region submission batch takes **10.536–10.648 s**. This includes work beyond the long sequence's decoding.

The decoder's linear weights contain **360.7 million parameters**, or **721.4 MB in BF16**. Streaming them once at 300 GB/s takes **2.40 ms**. With speculative decoding:

```text
Ideal weight-stream time ≈ 7.82 seconds / a
a = average output tokens advanced per verification pass
```

For example, `a = 2` gives **3.91 s of ideal weight traffic**. This assumes one complete weight stream per pass, peak bandwidth and no persistent weight-cache reuse. It excludes KV traffic, vision, prefill, proposal/rejection work and launch gaps.

**We need the current sequence's acceptance rate and achieved memory bandwidth to tighten this estimate.** Dividing this model by the measured batch time would not establish hardware efficiency. [Model configuration][ocr-config] · [Performance model][roofline]

## Embedding: the compute bound

The workload contains **42,220 tokens across 136 chunks**. Its linear layers require approximately **37.19 TFLOP** of useful work: `2 × 440.4 million parameters × tokens`. [Model configuration][embed-config]

| Four-GPU embedding | Time |
| --- | ---: |
| Ideal dense-BF16 compute, perfectly balanced tokens | **0.077 s** |
| Ideal compute, recorded token placement | **0.078 s** |
| Measured stage total | **0.625 s** |

The placement estimate uses each PDF's busiest GPU. Its **0.0016 s** gap from perfect balance suggests little opportunity in redistributing tokens alone.

These compute floors exclude padding, attention, normalization, memory traffic, dispatch and gathering. The gap to measured time is not a GPU-efficiency percentage.

## What remains unknown

The [NACA profile](#/profile/7195172c30ec973339d4) records **68.7–82.3% GPU busy**, with only seven samples per device. Busy means time executing kernels, not percentage of peak compute or bandwidth. Operator traces cover only **32 recognition iterations per GPU**. [Metric definitions][smi]

The most useful next measurement is a **complete Census and W-9 capture** on this configuration:

- **Queue delay and vision/prefill time:** how much time precedes decoding?
- **Output tokens per verification pass:** how much does speculation advance each sequence?
- **DRAM bytes/s, matrix shapes, clocks and launch gaps:** what limits the GPU work?

Those measurements would distinguish unavoidable model work from scheduling delays and wasted computation. Preserve the four GPUs, frozen outputs and evaluation protocol.

## Evidence

[Calculation data](/artifact?path=analysis/theoretical-limits-20261026-retained.json) includes unrounded values, input hashes, checkpoint counts and GPU token placement. Estimates use all 25 measured full-PDF calls and eval's percentile function.

CPU-only reproduction:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m lab.theoretical_limits \
  --output results/analysis/theoretical-limits-20261026-retained.json
```

[l4]: https://www.nvidia.com/en-us/data-center/l4/
[roofline]: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
[smi]: https://docs.nvidia.com/deploy/nvidia-smi/index.html
[ocr-config]: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/main/config.json
[embed-config]: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config.json
