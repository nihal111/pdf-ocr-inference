---
title: OCR model options
description: The current model, alternatives to investigate, and how to compare them.
order: 5
---

We use **PaddleOCR-VL-1.6 through vLLM**. PaddleOCR first finds page regions, such as paragraphs and tables; the model reads those cropped images, and PaddleOCR assembles the text in reading order.

A **model** supplies the learned ability to read images. An **inference engine**, such as vLLM, runs that model. Changing either can affect the output.

## Models in scope

This is a research shortlist. Only our saved experiments establish results on this dataset.

| Model | Role here |
| --- | --- |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | **Current model.** Used with the complete PaddleOCR page parser and PP-DocLayoutV3 layout detector. |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | **Previous setup.** Superseded after incomplete and repetitive transcription. |
| [olmOCR-2-7B-1025](https://huggingface.co/allenai/olmOCR-2-7B-1025) | Alternative document-extraction model to investigate. |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | Alternative OCR and structured-document model to investigate. |
| [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | Alternative document-reading model to investigate. |

The linked publisher pages describe each model’s supported software, input preparation and usage instructions.

## How to evaluate a replacement

1. **Read the same PDFs.** Use each model’s documented image preparation and instructions.
2. **Review quality.** Check missing or invented text, reading order, table values and formulas.
3. **Measure the complete pipeline.** Compare time, memory use and failures on the same hardware, including layout and text assembly.
4. **Choose a baseline explicitly.** An adopted model gets its own reviewed references; preserve the previous results.

Our current evaluator checks exact agreement with saved outputs. Different models can produce different formatting, so those checks alone cannot rank their OCR accuracy.

[Current pipeline](#/doc/pages/pipeline) · [Evaluation rules](#/doc/pages/baseline) · [Debugging findings](#/doc/pages/debugging)
