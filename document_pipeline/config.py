"""Pipeline configuration.

`eval/evaluate.py` imports a module and calls `embed_pdf(path)` with no other arguments, so
variants are selected through the environment:

    DOCUMENT_PIPELINE_CONFIG='{"ocr": {"backend": "vllm"}}'   # JSON overrides, or a path to a .json/.yaml file
    DOCUMENT_PIPELINE_TRACE=results/x/trace.jsonl             # optional: append per-call stage timings here

The defaults use pypdfium2, the native PaddleOCR-VL-1.6 page parser on four GPU workers,
our markdown chunker, and Qwen3-Embedding-0.6B through sentence-transformers in fp32.
The active experiment freezes these settings in experiments/pdf/00_public_v3_baseline.yaml.
References are captured from the
baseline, and every variant must reproduce the OCR text byte for byte and each chunk embedding
to cosine >= 0.99.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass

DEFAULT_VLM = "~/huggingface/PaddleOCR-VL-1.6"
DEFAULT_EMBED_MODEL = "~/huggingface/Qwen3-Embedding-0.6B"
DEFAULT_PROMPT = None  # Native PaddleOCR selects a prompt for each region type.


@dataclass
class RenderConfig:
    dpi: int = 150                       # page -> RGB image; changes the visual tokens and so the OCR output
    placement: str = "parent"             # parent | page_workers; page_workers overlaps rendering with OCR


@dataclass
class OcrConfig:
    backend: str = "paddleocr"           # paddleocr | hf | vllm | nanovllm
    model: str = DEFAULT_VLM
    prompt: str | None = DEFAULT_PROMPT
    max_new_tokens: int = 8192
    require_eos: bool = True             # reject truncated OCR before it can become a reference
    repetition_penalty: float = 1.0      # fixed across backends; candidate values need coverage review
    paddle_python: str = "/local/home/nhsng/inference-lab/.venv-paddle/bin/python"
    paddle_pipeline_version: str = "v1.6"
    paddle_layout_model: str = "~/huggingface/PP-DocLayoutV3"
    paddle_region_batch_size: int = 1
    paddle_region_dispatch: str = "page"  # page | all_gpus; up to batch_size regions per GPU
    paddle_region_submission_size: int | None = None  # client window; server batch limit is separate
    paddle_region_admission: str = "none"  # none | round_robin | available; shared page-worker admission
    paddle_region_admission_slots_per_gpu: int = 8  # used only when shared admission is enabled
    paddle_table_max_pixels: int | None = None  # optional table-only recognition resolution limit
    paddle_rec_backend: str = "native"   # native | vllm-server; layout remains Paddle
    paddle_server_base_port: int = 8800  # one loopback recognizer per GPU
    paddle_server_python: str = "/local/home/nhsng/inference-lab/.venv-vllm/bin/python"
    device: str = "auto"                 # hf only: auto | cpu | cuda | cuda:N | mps
    dtype: str = "bfloat16"
    # Pages per forward. hf: pages per generate() call (padded). vllm/nanovllm: max pages in flight.
    # Batching changes the floating-point results and can flip a greedy token. Measure it.
    page_batch_size: int = 1
    attn_implementation: str = "sdpa"    # hf: sdpa | eager | flash_attention_2
    gpu_workers: int = 4                 # independent page workers, one parser/model replica per GPU
    page_schedule: str = "round_robin"    # round_robin | dynamic (next page goes to a free worker)
    min_pixels: int | None = None        # processor image-size bounds (None = model default)
    max_pixels: int | None = None
    # vllm
    gpu_memory_utilization: float = 0.55  # leave room for the embedder in the same process
    max_model_len: int = 8192
    enforce_eager: bool = False


@dataclass
class ChunkConfig:
    tokenizer: str = DEFAULT_EMBED_MODEL  # token counts use the embedding model's tokenizer
    max_tokens: int = 512


@dataclass
class EmbedConfig:
    backend: str = "sentence_transformers"  # sentence_transformers | nanovllm
    model: str = DEFAULT_EMBED_MODEL
    device: str = "auto"
    dtype: str = "float32"                  # float32 | bfloat16 | float16
    batch_size: int = 32                    # sentence_transformers: sequences per forward
    compile_dynamic: bool = False           # sentence_transformers: compile forward; first call pays setup cost
    gpu_workers: int = 1                    # sentence_transformers: replicate across consecutive CUDA devices
    max_batch_tokens: int = 16384           # nanovllm: packed tokens per forward (varlen, no padding)
    max_length: int = 8192


@dataclass
class PipelineConfig:
    render: RenderConfig = field(default_factory=RenderConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _build(cls, data: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        kwargs[name] = _build(type(default), value) if is_dataclass(default) and isinstance(value, dict) else value
    return cls(**kwargs)


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(overrides: dict | str | None = None) -> PipelineConfig:
    """Defaults, then DOCUMENT_PIPELINE_CONFIG (JSON string or file path), then explicit overrides."""
    data = PipelineConfig().to_dict()
    for source in (os.environ.get("DOCUMENT_PIPELINE_CONFIG"), overrides):
        if not source:
            continue
        if isinstance(source, str):
            if source.lstrip().startswith("{"):
                source = json.loads(source)
            else:
                with open(os.path.expanduser(source)) as f:
                    if source.endswith((".yaml", ".yml")):
                        import yaml
                        source = yaml.safe_load(f) or {}
                    else:
                        source = json.load(f)
        data = _merge(data, source)
    return _build(PipelineConfig, data)
