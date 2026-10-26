---
title: Setup notes before the first complete baseline
description: Why the initial measurements were scrapped and what changed.
order: 0
status: superseded
date: 2026-10-26
---

These were setup attempts, not accepted baseline measurements.

- **One GPU was too slow for practical iteration.** We moved to a fixed pool of four L4 GPUs, each handling one page at a time from the same PDF. To keep experiments manageable, we retained the three smallest PDFs and one warm-up plus one timed call **per PDF**.
- **The 2,048-token OCR cap cut off output.** The seaplane report's pages 1 and 3 reached the limit without EOS. Reference-fidelity checks passed because repeated calls reproduced those truncated references. We needed to fix output completion, inspect the page transcriptions, and establish new references before optimizing.
- **A larger allowance alone did not address the underlying failure.** Inspection found page 1 repeating figure captions after its body text, while page 3 cut off mid-table. The transcription instruction was made explicit about including every table row, transcribing each text element once, and stopping after the page was complete.
- **Qwen completion diagnostics were stopped in favor of PaddleOCR-VL-1.6.** The exact 8,192-token retry confirmed repeated paragraphs on NACA page 2, which also failed to finish. The current baseline uses the [PaddleOCR-VL-1.6 page parser](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) with PP-DocLayoutV3 and region-level EOS checks.

The old single-GPU run data was removed. The initial four-GPU profile documents how the truncation was discovered; its timings do not establish an accepted baseline or an optimization speedup.
