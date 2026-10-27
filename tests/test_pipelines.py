"""The pinned evaluator modules reproduce the measured sweep variants exactly."""
import ast
from pathlib import Path

import pytest
import yaml

from document_pipeline.config import load_config
from lab.sweep import expand

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "baseline": ("experiments/pdf/08_vllm_five_pdf_eval.yaml", "vllm_batch_1"),
    "optimized": ("experiments/pdf/26_six_request_tiles.yaml", "server_8_draft4_control"),
}
HOST_PATHS = {"paddle_python", "paddle_server_python"}


def pinned(name):
    """Read PIPELINE without importing the module, which would start the OCR workers."""
    tree = ast.parse((ROOT / "pipelines" / f"{name}.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets] == ["PIPELINE"]:
            return ast.literal_eval(node.value)
    raise AssertionError(f"pipelines/{name}.py has no PIPELINE literal")


@pytest.mark.parametrize("name", VARIANTS)
def test_pinned_pipeline_matches_measured_variant(name):
    path, variant = VARIANTS[name]
    measured = expand(yaml.safe_load((ROOT / path).read_text()))[variant]["pipeline"]
    for key in HOST_PATHS:
        measured["ocr"].pop(key, None)
    assert load_config(pinned(name)).to_dict() == load_config(measured).to_dict()


def test_optimized_environment_selects_fixed_decoder_tiles():
    source = (ROOT / "pipelines/optimized.py").read_text()
    assert '"DOCUMENT_PIPELINE_PADDLE_DECODE_TILES": "32"' in source
    assert "--max-num-seqs 8 --ngram-tokens 4 --ngram-method ngram_gpu --decode-matmul-fixed32" in source
