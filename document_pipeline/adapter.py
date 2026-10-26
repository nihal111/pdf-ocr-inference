"""The module eval/evaluate.py loads:  python -m eval.evaluate --module document_pipeline.adapter ...

Models load at import time, so they count toward the evaluator's worker start-up and not
toward per-PDF latency. Select a variant with DOCUMENT_PIPELINE_CONFIG (see document_pipeline/config.py).
"""
import atexit

from document_pipeline.pipeline import Pipeline

_pipeline = Pipeline()
atexit.register(_pipeline.close)


def embed_pdf(path: str) -> dict:
    return _pipeline(path)
