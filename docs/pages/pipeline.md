---
title: Document pipeline
description: How a PDF becomes readable text and vectors for search by meaning.
order: 1
---

This pipeline turns a PDF into **extracted text and embeddings**: lists of numbers representing text meaning, useful for search by meaning. It processes every page and preserves page order.

## The four stages

| Stage | What happens | Technology |
| --- | --- | --- |
| **Render** | Convert each PDF page into an image | pypdfium2 |
| **OCR** | Read text, tables and formulas from the images | [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6), run through **vLLM** |
| **Chunk** | Divide the extracted text into smaller pieces, carrying relevant headings into each piece | Markdown chunker |
| **Embed** | Convert each piece into a vector: a list of numbers representing its meaning | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), run through sentence-transformers on PyTorch |

Chunks target **512 tokens**. A token is a unit of text used by a model, such as a word or part of a word. Chunk lengths vary because the pipeline also respects text boundaries and heading context.

## How OCR reads a page

**OCR means optical character recognition: reading text from an image.** Here it has three steps:

1. **Find regions.** Layout detection locates areas such as paragraphs, tables and formulas, and determines their reading order. It uses PP-DocLayoutV3 on PaddlePaddle.
2. **Read each region.** Recognition means converting a cropped image into text. PaddleOCR-VL-1.6 reads those regions, using **vLLM to run the model on the GPU**.
3. **Assemble the page.** PaddleOCR combines the results into Markdown: text with formatting for headings, lists and tables.

An **inference engine** is the software that runs a trained model. vLLM is the engine for the reading step; PaddlePaddle runs layout detection.

## How the baseline runs

The **baseline** is the starting configuration used to measure later optimizations.

- **Four NVIDIA L4 GPUs cooperate on one PDF.** Each GPU handles one page at a time. Short PDFs may leave some GPUs idle.
- **Recognition batch size is one region per GPU.** A batch is a group of inputs processed together; it is separate from the number of pages in the PDF.
- After OCR, pages are combined in their original order, then chunked and embedded. Embedding runs on GPU 0. The next PDF starts after the current PDF finishes.

## Current optimized configuration

The [retained optimized configuration](#/doc/experiments/028-optimization-conclusion)
renders pages inside the existing OCR workers, allowing rendering and recognition
to overlap. Their combined wall time is reported as **render + OCR**.

Recognition regions are shared across all four GPUs, with eight concurrent requests
per GPU and four speculative draft tokens. The completed tuning sweep records why
these settings were retained.

It also distributes embedding across all four GPUs. Each GPU holds the same embedding
model; chunks are assigned using their input token lengths, then vectors are
restored to the original chunk order. This builds on shared OCR region scheduling
and decoder optimizations. Compilation adds startup time, which is included in
the reported suite runtime.

## What a valid result means

- **Finish reading.** Every region must generate an end-of-sequence (**EOS**) marker, indicating the model has finished.
- **Preserve the reference.** Optimizations must match the saved baseline text exactly, keep the same chunks in order, and preserve their IDs and the number of values in each vector.
- **Preserve embedding similarity.** Each vector needs cosine similarity of at least **0.99** to its saved reference. This score compares the directions of two vectors.

These checks test agreement with the baseline. Reviewing the PDF alongside its transcription is still necessary to assess OCR accuracy.

## Python interface

The evaluator imports `embed_pdf` from `document_pipeline.adapter` and calls it
with a PDF path:

```python
def embed_pdf(path: str) -> dict:
    ...

# Example result; actual vectors have the embedding model's dimension.
{
    "ocr_text": "Hello world\n",
    "chunks": [
        {"id": "chunk-000", "embedding": [0.12, -0.03, 0.98]}
    ],
}
```

`ocr_text` contains the OCR text of every page in document order. `chunks` is
non-empty and follows that same order. Each chunk has a unique, non-empty string
ID and a non-empty plain Python list of finite floats. Every vector has the same
length. A blank document still produces one chunk.

Missing files raise `FileNotFoundError`. Malformed PDFs, including truncated files
without an EOF marker and documents with no pages, raise `ValueError`. Callers can
catch these exceptions and process another PDF using the same pipeline. Both
rendering placements use the same input checks; page workers report input errors
and drain outstanding requests before the next document.

Incomplete OCR or invalid embedding output raises `RuntimeError` instead of
returning a partial or invalid result.

## Further details

- [Baseline settings and evaluation rules](#/doc/pages/baseline)
- [Experiment runs and optimization results](#/runs)
- [Theoretical limits and remaining headroom](#/doc/pages/theoretical-limits)
- [OCR model options](#/doc/pages/ocr-models)
- [Table-repetition investigation](#/doc/pages/debugging)
