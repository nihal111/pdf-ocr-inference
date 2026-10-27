"""Retained configuration (experiments/pdf/26_six_request_tiles.yaml, variant server_8_draft4_control).

Start the recognizers first: eight requests per GPU, four GPU n-gram drafts,
fixed 32-row tiles and an isolated server compilation cache.

    export CUDA_MODULE_LOADING=EAGER VLLM_BATCH_INVARIANT=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True VLLM_USE_FLASHINFER_SAMPLER=0
    python -m lab.paddle_servers --directory results/servers/optimized \
        --max-num-seqs 8 --ngram-tokens 4 --ngram-method ngram_gpu --decode-matmul-fixed32 \
        --isolate-compile-cache

Export the four assigned CUDA_VISIBLE_DEVICES UUIDs before launching.
Wait for ready.json, indicating all four workers are healthy.
"""
from pipelines._variant import pin

PIPELINE = {
    "render": {"dpi": 150, "placement": "page_workers"},
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
        "paddle_region_batch_size": 4,
        "paddle_rec_backend": "vllm-server",
        "paddle_region_dispatch": "all_gpus",
        "paddle_table_max_pixels": 2822400,
        "paddle_region_submission_size": 64,
        "page_schedule": "dynamic",
        "paddle_region_admission": "available",
        "paddle_region_admission_slots_per_gpu": 8,
    },
    "chunk": {"tokenizer": "~/huggingface/Qwen3-Embedding-0.6B", "max_tokens": 512},
    "embed": {
        "backend": "sentence_transformers",
        "model": "~/huggingface/Qwen3-Embedding-0.6B",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "batch_size": 16,
        "max_batch_tokens": 16384,
        "max_length": 8192,
        "compile_dynamic": True,
        "gpu_workers": 4,
    },
}

pin(PIPELINE, {"DOCUMENT_PIPELINE_PADDLE_DECODE_TILES": "32"})

from document_pipeline.adapter import embed_pdf  # noqa: E402

__all__ = ["embed_pdf"]
