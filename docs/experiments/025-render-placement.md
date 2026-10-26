---
title: Rendering · Formats and page-worker placement
description: Preserve page pixels while reducing the remaining rendering and image-transfer cost.
order: 25
status: passed
date: 2026-10-26
run: paddleocr_vllm_worker_render/20261026-141545
profile: profiles/20261026-143036-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-Render-pages-inside-OCR-workers-c911f4
---

## Evidence

The [four-GPU embedding profile](#/profile/b87dcac9214c23760dba) leaves rendering
at 4.9% of NACA wall time, about 0.113 s. The full evaluation also spends about
0.115 s rendering Attention. These two PDFs now overlap around pooled p50.
The retained configuration uses four-GPU embeddings and fixed 32-row decoder tiles.

These initial screens measure rendering alone. They keep 150 DPI and compare
the RGB bytes, dimensions and order of all 20 pages with the existing renderer.
Each configuration receives one warm-up and five measured calls per PDF.
The numbers below are stage medians, not full-PDF evaluator metrics.

## Bitmap formats

The installed pypdfium2 version is 5.14.0 in both pipeline and Paddle worker
environments. Its [versioned rendering API source](https://github.com/pypdfium2-team/pypdfium2/blob/5.14.0/src/pypdfium2/_helpers/page.py)
documents reverse byte order, four-byte pixel storage and optional alpha formats.
The matching source was retrieved and archived with this study.

| Format | Sum of five rendering medians | Exact page pixels |
| --- | ---: | --- |
| Existing control | 0.3914 s | Passed |
| Reverse byte order | 0.3885 s | Passed |
| BGRx | 0.4079 s | Passed |
| RGBx | 0.3927 s | Passed |
| Alpha when needed | 0.3846 s | Passed |
| RGBA when needed | 0.3848 s | Passed |
| Force BGRA | 0.4228 s | Failed |
| Force RGBA | 0.3941 s | Failed |
| Existing control repeated | 0.3857 s | Passed |

**Keep the existing format.** The compatible alternatives do not establish a useful
gain beyond the repeated control. Forcing alpha changes pixels on both English
checklist pages, W-9 page 1 and Census pages 1–4. Those candidates are excluded
before any full-pipeline evaluation.

## Separate rendering processes

The project's [threading guidance](https://github.com/pypdfium2-team/pypdfium2/blob/5.14.0/docs/source/python_api.rst)
requires separate processes for concurrent PDFium calls. The prototype therefore
uses persistent processes started with `spawn`.

Timing includes reading the current PDF's page count, opening documents in the
workers, rendering, transferring full images back to the parent and restoring page
order. It uses the existing bitmap format. All 120 calls preserved exact pixels.

| PDF | Single process | Two processes | Four processes | Single process repeated |
| --- | ---: | ---: | ---: | ---: |
| English checklist | 0.0417 s | 0.0485 s | 0.0399 s | 0.0293 s |
| NACA seaplanes | 0.1329 s | 0.1193 s | 0.0696 s | 0.1105 s |
| W-9 | 0.0751 s | 0.1015 s | 0.0596 s | 0.0514 s |
| Census income | 0.0651 s | 0.1046 s | 0.0637 s | 0.0479 s |
| Attention paper | 0.0901 s | 0.1280 s | 0.0672 s | 0.0892 s |

Four processes help NACA and Attention, but the repeated single-process control
is faster on simpler PDFs. The experiment includes additional document opens and
large image transfers; it does not isolate which overhead causes those regressions.

## Next full-pipeline candidate

Render pages inside the existing OCR page workers. This permits independent page
rendering while avoiding transfers of rendered images through the parent. The
current pixel-preserving screen motivates the approach, but does not prove its
end-to-end benefit.

Keep a fresh parent-render control with the retained four-GPU embeddings. Before
evaluation, verify exact pixels, source page order, worker cleanup and the default
path. Record overlapping rendering and OCR as a combined wall-time stage.
Then run the unchanged frozen evaluation and compare pooled p50, p95, pages/min
and suite runtime. The implementation passed 44 tests and actual-worker checks for exact pixels on
all 20 pages and frozen outputs on all five PDFs. That preflight used eager
embeddings to isolate renderer/OCR integration; both full evaluation conditions
retain compiled four-GPU embeddings and the required one warm-up plus five
measured calls per PDF.

## Full evaluation result

Both conditions passed all 60 calls and 5,580 raw EOS completions. Every page in
the candidate rendered inside its assigned OCR worker, output order was preserved,
and every embedding call still used all four GPUs. Pixel hashing was disabled
during evaluation; its cost belongs only to the separate preflight.

| Configuration | Pooled p50 | Pooled p95 | Pages/min | Suite runtime |
| --- | ---: | ---: | ---: | ---: |
| Fresh parent-render control | 2.349 s | 12.382 s | 45.414 | 550.3 s |
| Rendering inside OCR workers | 2.271 s | 12.362 s | 45.802 | 343.0 s |

The candidate improves evaluator speedup by **3.4% for p50**, **0.2% for p95** and
**0.9% for throughput** versus the fresh control. This is a modest p50 gain;
the other metrics are nearly flat.

| PDF | Parent-render median | Worker-render median |
| --- | ---: | ---: |
| English checklist | 1.843 s | 1.851 s |
| NACA seaplanes | 2.320 s | 2.253 s |
| W-9 | 7.511 s | 7.535 s |
| Census income | 12.379 s | 12.348 s |
| Attention paper | 2.348 s | 2.271 s |

NACA and Attention benefit, while the simpler documents change little. The
combined rendering/OCR median fell from 2.181 to 2.112 s on NACA, and from
2.256 to 2.183 s on Attention.

Suite runtime is dominated by variable compilation startup: the first English
warm-up took 364.9 s in the control and 158.9 s in the candidate. That difference
does not measure rendering speedup. Warm-ups remain excluded from pooled latency.

**Retain page-worker rendering for the next comparison**, while keeping the
small size of the gain visible. The [fresh NACA profile](#/profile/7195172c30ec973339d4)
is complete: 2.237 s median, with 93.9% of mean wall time in combined rendering/OCR
and 5.2% in embedding. Instrumented outputs passed.

The trace found narrow decoder projections using two 32-row tiles in 99 of 128
sampled engine iterations. At eight requests, four draft tokens plus one target
token can produce 40 rows; three drafts produce 32. This motivates a new
three-draft comparison after the fixed 32-row kernel change. Check the Census
tail and actual kernel grids before drawing a performance conclusion.

The collector initially rejected the combined stage. Its aggregation fix passed
29 tests; the corrected full capture completed. Earlier raw ordinary-call data
was preserved with the collector-repair artifacts.

[Fresh parent-render control](#/run/48bcf1f74ad36b6dc4b0) · full evaluation.

[Candidate evaluation](#/run/40e811453ad54f72d35a) · rendering inside existing OCR page workers.
