"""The module eval/evaluate.py loads:  python -m eval.evaluate --module document_pipeline.adapter ...

Models load at import time, so they count toward the evaluator's worker start-up and not
toward per-PDF latency. Select a variant with DOCUMENT_PIPELINE_CONFIG (see document_pipeline/config.py).
"""
import atexit

from document_pipeline.pipeline import Pipeline

_pipeline = Pipeline()
atexit.register(_pipeline.close)


def embed_pdf(path: str) -> dict:
    """Return full OCR text and ordered chunks with finite Python float vectors.

    Missing paths raise FileNotFoundError; malformed PDFs raise ValueError.
    Input errors are recoverable: callers may catch them and process another PDF.
    """
    return _pipeline(path)
