# OCR models

The current OCR stage uses **[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)**, a document-focused 0.9B recognizer, with **[PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)** and the official native PaddleOCR page parser.

The active experiment is `experiments/pdf/00_public_v3_baseline.yaml`. Four persistent GPU workers each hold both models. The parser detects page regions and reading order, recognizes text, tables and formulas with task-specific prompts, and returns page Markdown. The standalone Transformers example in the model card handles individual regions; our baseline uses the full page parser.

| Installed model | Pinned revision |
| --- | --- |
| PaddlePaddle/PaddleOCR-VL-1.6 | `c5630abae1d940eafe0697512a0325494b02ab42` |
| PaddlePaddle/PP-DocLayoutV3 | `241f8bdfc77a7c7bee915a5057aaee58c235a8d3` |

Paddle workers use `.venv-paddle`; the parent and Qwen3-Embedding-0.6B use `.venv`. The original Qwen OCR attempt was superseded after repetition and completion failures. No cross-model speedup is claimed.

The [model-options handbook](docs/pages/ocr-models.md) holds the research shortlist and source links. Use each publisher's documented preprocessing and evaluate coverage, reading order, tables, formulas and failure modes before adopting another model. Model changes require separately reviewed references; byte equality against another model is not an OCR-accuracy ranking.

Sources: [PaddleOCR-VL-1.6 model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6), [PP-DocLayoutV3 model card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3), [official page-parser guide](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html), [PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR).
