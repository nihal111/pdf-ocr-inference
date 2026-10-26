# Handoff: inference-lab on four L4 GPUs

The current OCR baseline is **[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)** with [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3), using the official native page parser. Read `docs/experiments/004-paddleocr-baseline.md` and the active configuration in `experiments/pdf/00_public_v3_baseline.yaml` before continuing.

## User constraints

- Keep `eval/` unchanged. OCR bytes and chunk structure must match fixed references; embedding cosine must be at least 0.99.
- Keep the same four L4 GPUs. Each persistent worker handles one page at a time from the same PDF. Restore original page order and evaluate PDFs sequentially.
- The library contains ten public PDFs under `data/public-v3/`, each capped at five pages. All 42 pages are previewable on the localhost site.
- The initial measured baseline uses the three smallest PDFs: two earthquake checklists and the three-page NACA scan. Run one warm-up plus one timed call per PDF.
- Establish complete, reviewed OCR before freezing references. Profile the accepted baseline before applying optimizations. Experiment ideas are a blueprint, not a required sequence.
- Show meaningful results, per-PDF timings in seconds, detailed profiling findings and documentation on the site at `127.0.0.1:8766`.

## Runtime

The parent and Qwen3-Embedding-0.6B use the original `.venv` PyTorch/CUDA 13.0 environment. Paddle workers use `.venv-paddle` with PaddleOCR 3.7.0, PaddleX 3.7.2, PaddlePaddle 3.2.1 and CUDA 12.6. Do not combine these CUDA-library packages in one environment. Workers communicate over private inherited pipes.

The parser uses one region per recognition batch. Every region must emit EOS within 8,192 tokens, checked before parser cleanup. The active render resolution is 150 DPI, and Markdown chunks target 512 tokens. The embedding stage is fp32 on GPU 0.

## Verified and outstanding

The first complete NACA setup check finished all three pages and 71 regions normally. That check included startup and is not a measured baseline. Capture and review all seven selected pages, then complete the evaluator run. Native Paddle detailed tracing and hardware-counter collection remain required after the baseline.

Qwen OCR diagnostics are superseded. Do not use their references, timings or profiler findings as PaddleOCR results. Keep their short setup notes as historical context.

## Continue

1. Inspect the live run status and process handles before restarting work. `.site/baseline.run` points to the latest intended baseline; verify it against saved lifecycle files.
2. Capture with `lab.pdf_sweep --capture-only`, inspect page transcripts and region diagnostics, and record a quality review.
3. Run the frozen baseline through `eval/evaluate.py` via the sweep. Keep reference files fixed.
4. Profile the unchanged pipeline, including layout, recognition, CPU/GPU work, per-GPU activity and capacity evidence. Report what the measurements cannot establish.
5. Update the site and journal with the measured per-PDF results and evidence-based next experiments.

Do not cache PDF results between calls, loosen evaluator thresholds, or recapture references to make an optimized variant pass. Official speedups require `eval.compare` acceptance and at least five measured repeats per PDF. Commit at logical checkpoints on a branch; do not push unless asked.
