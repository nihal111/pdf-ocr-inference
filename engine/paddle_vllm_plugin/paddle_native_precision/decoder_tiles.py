"""Optional Paddle decoder tiles for the pinned vLLM batch-invariant kernel."""
from __future__ import annotations

import logging
import os

ENVIRONMENT_KEY = "DOCUMENT_PIPELINE_PADDLE_DECODE_TILES"


def decoder_shapes(config):
    """Fused projection sizes (N, K) for one tensor-parallel rank."""
    config = getattr(config, "text_config", config)
    hidden = config.hidden_size
    head = getattr(config, "head_dim", None) or hidden // config.num_attention_heads
    query = config.num_attention_heads * head
    kv = config.num_key_value_heads * head
    return (
        (query + 2 * kv, hidden),
        (hidden, query),
        (2 * config.intermediate_size, hidden),
        (hidden, config.intermediate_size),
        (config.vocab_size, hidden),
    )


def install_decoder_tiles(vllm_config):
    selected = os.environ.get(ENVIRONMENT_KEY, "0")
    if selected == "0":
        return ()
    if selected not in {"1", "32", "shape_columns"}:
        raise ValueError(
            f"{ENVIRONMENT_KEY} must be 0, 1 (bucketed), 32, or shape_columns")
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise ValueError("Decoder tile tuning requires batch invariance")
    if vllm_config.parallel_config.tensor_parallel_size != 1:
        raise ValueError("Decoder tile tuning requires tensor parallel size one")

    import torch
    from vllm.model_executor.determinism import batch_invariant_configs as configs
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or (capability.major, capability.minor) != (8, 9):
        raise ValueError("Decoder tile tuning was screened only on Ada GPUs")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise ValueError("Decoder tile tuning requires the BF16 language model")
    shapes = decoder_shapes(vllm_config.model_config.hf_config)
    screened = {(2560, 1024), (1024, 2048), (6144, 1024),
                (1024, 3072), (103424, 1024)}
    if set(shapes) != screened:
        raise ValueError("Decoder matrix shapes differ from the screened PaddleOCR model")

    # Upstream keeps BLOCK_K constant across M to preserve batch invariance.
    # All five shapes originally fall back to K=64, N=128, M=128. Retain K;
    # row/column choices and four warps were checked bitwise against that kernel.
    # A single bucket survives Dynamo tracing. M-dependent Python selection
    # otherwise fixes the prefill tile into the compiled decoder graph.
    buckets = (
        ((2**31 - 1, configs._MatmulMConfig(32, 128, 4, 3)),)
        if selected in {"32", "shape_columns"} else (
            (16, configs._MatmulMConfig(16, 128, 4, 3)),
            (64, configs._MatmulMConfig(32, 128, 4, 3)),
            (2**31 - 1, configs._MatmulMConfig(128, 128, 8, 3)),
        )
    )
    proposed = {}
    for shape in shapes:
        selected_buckets = buckets
        if selected == "shape_columns":
            # Screened model shapes: wide QKV/gate projections benefit during
            # prefill; narrower output/down/head tiles improve small decode.
            columns = 256 if shape[0] in {2560, 6144} else 64
            selected_buckets = (
                (2**31 - 1, configs._MatmulMConfig(32, columns, 4, 3)),)
        proposed[shape] = configs._MatmulShapeConfig(
            block_k=64, m_buckets=selected_buckets)
    table = configs._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS["ada"]
    # Check every shape before changing the table. A future upstream tuning
    # change needs review; identical entries permit repeated model construction.
    for shape in shapes:
        existing = table.get(shape)
        if existing is not None and existing != proposed[shape]:
            raise ValueError(f"vLLM already tunes {shape}; review before overriding it")
    for shape in shapes:
        table[shape] = proposed[shape]
    description = (
        "fixed 32 rows; columns 256 for QKV/gate-up and 64 for output/down/head; "
        "four warps; K=64 fixed"
        if selected == "shape_columns" else
        "fixed 32 rows for all M; four warps; N=128/K=64 fixed"
        if selected == "32" else
        "16 rows through M=16, 32 rows through M=64; "
        "N=128/K=64 fixed; original tiles above M=64"
    )
    logging.getLogger("vllm.paddle_native_precision").info(
        "Paddle decoder tiles enabled: %s. Shapes: %s", description, shapes)
    return shapes
