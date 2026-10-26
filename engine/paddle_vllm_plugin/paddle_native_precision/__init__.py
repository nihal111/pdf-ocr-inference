"""Match the native Paddle model's explicit fp32 vision/projector policy."""


def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "PaddleOCRVLForConditionalGeneration",
        "paddle_native_precision.model:PaddleOCRNativePrecision")
