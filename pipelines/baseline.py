"""Frozen baseline: vLLM batch 1 (experiments/pdf/08_vllm_five_pdf_eval.yaml, variant vllm_batch_1).

Start the recognizers first, with the launcher defaults (four requests per GPU, no speculation,
no decoder tiles):

    python -m lab.paddle_servers --directory results/servers/baseline

References are captured from this module only.
"""
from pipelines._variant import pin

PIPELINE = {
    "render": {"dpi": 150},
    "ocr": {
        "backend": "paddleocr",
        "model": "~/huggingface/PaddleOCR-VL-1.6",
        "max_new_tokens": 8192,
        "require_eos": True,
        "device": "auto",
        "dtype": "bfloat16",
        "page_batch_size": 1,
        "attn_implementation": "sdpa",
        "gpu_workers": 4,
        "min_pixels": None,
        "max_pixels": None,
        "gpu_memory_utilization": 0.4,
        "max_model_len": 16384,
        "enforce_eager": False,
        "paddle_pipeline_version": "v1.6",
        "paddle_layout_model": "~/huggingface/PP-DocLayoutV3",
        "paddle_region_batch_size": 1,
        "paddle_rec_backend": "vllm-server",
        "paddle_region_dispatch": "page",
        "paddle_table_max_pixels": 2822400,
    },
    "chunk": {"tokenizer": "~/huggingface/Qwen3-Embedding-0.6B", "max_tokens": 512},
    "embed": {
        "backend": "sentence_transformers",
        "model": "~/huggingface/Qwen3-Embedding-0.6B",
        "device": "cuda:0",
        "dtype": "float32",
        "batch_size": 32,
        "max_batch_tokens": 16384,
        "max_length": 8192,
    },
}

pin(PIPELINE)

from document_pipeline.adapter import embed_pdf  # noqa: E402

__all__ = ["embed_pdf"]
