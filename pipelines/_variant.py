"""Shared set-up for the pinned variants. Must run before document_pipeline.adapter is imported."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Client-side environment shared by both measured runs. CUDA_VISIBLE_DEVICES is host-specific:
# export the four assigned GPU UUIDs before running the evaluator.
ENVIRONMENT = {
    "CUDA_MODULE_LOADING": "EAGER",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_BATCH_INVARIANT": "1",
}


def pin(pipeline: dict, environment: dict | None = None) -> None:
    """Fix DOCUMENT_PIPELINE_CONFIG and the client environment for this process."""
    if "document_pipeline.adapter" in sys.modules:
        raise RuntimeError("Import only one pinned pipeline per process.")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the four assigned GPU UUIDs.")
    pipeline = json.loads(json.dumps(pipeline))
    ocr = pipeline["ocr"]
    # The frozen settings named the GPU host's checkout; resolve the same venvs from this checkout.
    ocr.setdefault("paddle_python", str(ROOT / ".venv-paddle/bin/python"))
    ocr.setdefault("paddle_server_python", str(ROOT / ".venv-vllm/bin/python"))
    os.environ.update({**ENVIRONMENT, **(environment or {})})
    os.environ["DOCUMENT_PIPELINE_CONFIG"] = json.dumps(pipeline)
