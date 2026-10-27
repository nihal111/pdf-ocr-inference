---
title: Theoretical limits
description: How the best-case model compares with our measured five-PDF workload.
order: 7
---

**Our optimized reference takes 26.53 seconds to convert all five PDFs, one after another: 20 pages into 136 chunk embeddings.** That is the total for the dataset, with warm models on four NVIDIA L4 GPUs.

The best-case calculation gives **4.4–5.4 seconds for the same serial workload**, but requires perfect speculative acceptance, peak GPU performance and negligible CPU/transfer overhead. It is an optimistic limit, not a promised runtime.

## Where the time goes

**All times below cover all five PDFs processed serially, in seconds.** The measured column is the retained optimized configuration, revalidated on 2026-10-27.

| Stage | Measured reference | Optimistic minimum | Stage estimate |
| --- | ---: | ---: | ---: |
| Rendering + OCR | **25.764** | **4.289** | **5.344** |
| Embedding | **0.633** | **0.081** | **0.081** |
| Preparation, chunking and other overhead | 0.131 | 0, assumed | 0, assumed |
| **All five PDFs, serially** | **26.528** | **4.371** | **5.425** |

**Optimistic minimum** assumes independent GPU work overlaps perfectly. **Stage estimate** adds each stage's ideal time. Both assume perfect four-draft acceptance; neither is a worst case or a guaranteed achievable result.

Rendering/OCR accounts for nearly all of the gap. The saved traces time rendering, layout detection, vision, prompt processing and decoding together, so we compare that combined stage rather than invent separate measured timings.

## What limits the speed?

- **Model size:** OCR uses about **906 million parameters**, and the embedding model **596 million**. The decoder repeatedly reads **361 million** of the OCR parameters—about **721 MB** in BF16 per pass.
- **Sequential decoding:** ideal weight bandwidth allows about **416 tokens/second per sequence**, before KV-cache overhead. Four GPUs can handle different regions in parallel, but cannot generate one region's dependent tokens simultaneously. The longest Census table produces **3,252 tokens**.
- **Speculation:** accepting four draft tokens allows up to five tokens of progress per pass. Full GPU utilization does not guarantee that acceptance. The serial-workload estimate rises to **7.7–9.3 seconds** at two tokens per pass, or **14–16 seconds** with ordinary decoding.

The OCR workload is **24,890 generated tokens across 465 regions**. The final Markdown has **47,081 tokens with the OCR tokenizer** because post-processing adds text such as HTML table-cell styling. This difference is not caused by speculation: accepted draft tokens are already included in the generated count. Embedding processes **42,220 tokens across 136 chunks**.

## How the five PDFs add up

The ideal column uses the stage estimate with **perfect four-draft acceptance**. The measured column averages five warm calls per PDF; summing those means gives the **26.53-second dataset total**.

| PDF | Ideal, perfect drafts | Measured mean |
| --- | ---: | ---: |
| Earthquake checklist | 0.58 s | 1.84 s |
| NACA seaplanes | 0.53 s | 2.36 s |
| IRS W-9 | 1.68 s | 7.68 s |
| Census income | 2.13 s | 12.36 s |
| Attention paper | 0.51 s | 2.29 s |
| **All five PDFs, serially** | **5.42 s** | **26.53 s** |

Startup and warm-up calls are excluded. The reference uses source revision `12f9cf6`.

[Measured reference](#/run/a6e67f74fcdf9cbbf963/optimized) · [Detailed calculations, assumptions and input hashes](#/doc/pages/theoretical-limits-calculations)
