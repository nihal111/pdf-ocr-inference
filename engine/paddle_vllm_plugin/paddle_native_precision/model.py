"""vLLM recognition with Paddle's fp32 visual path and bf16 language model.

PaddleX 3.7.2's PaddleOCRVLForConditionalGeneration declares
_keep_in_fp32_modules = ["visual", "mlp_AR"]. The checkpoint stores those
weights in bf16; converting them before loading reproduces the native policy.
"""
from collections import Counter
import logging

import torch
from vllm.model_executor.models.paddleocr_vl import (
    PaddleOCRVLForConditionalGeneration,
    PaddleOCRVLDummyInputsBuilder,
    PaddleOCRVLMultiModalProcessor,
    PaddleOCRVLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .decoder_tiles import install_decoder_tiles


@MULTIMODAL_REGISTRY.register_processor(
    PaddleOCRVLMultiModalProcessor,
    info=PaddleOCRVLProcessingInfo,
    dummy_inputs=PaddleOCRVLDummyInputsBuilder,
)
class PaddleOCRNativePrecision(PaddleOCRVLForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        install_decoder_tiles(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.visual.float()
        self.mlp_AR.float()
        self.decoder_dtype = vllm_config.model_config.dtype
        counts = Counter()
        for name, parameter in self.named_parameters():
            counts[(name.split(".")[0], str(parameter.dtype))] += parameter.numel()
        logging.getLogger(__name__).info("Native precision parameter counts: %s", dict(counts))

    def embed_multimodal(self, **kwargs):
        # Native Paddle casts image embeddings only at the language-model boundary.
        return tuple(value.to(self.decoder_dtype)
                     for value in super().embed_multimodal(**kwargs))
