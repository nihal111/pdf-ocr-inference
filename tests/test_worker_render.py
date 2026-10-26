"""PDF metadata dispatch, ownership and truthful timing for worker-side rendering."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from document_pipeline.config import load_config
from document_pipeline.pipeline import Pipeline
from document_pipeline.worker_render import PdfPageSource, page_request, request_images, request_image_scope

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "eval/fixtures/synthetic.pdf"


def test_metadata_request_has_no_image_payload_and_keeps_order():
    source = PdfPageSource("/example.pdf", 150, 5)
    assert len(source) == 5
    request = page_request(source, [4, 1])
    assert request == {"pdf": {"path": "/example.pdf", "dpi": 150},
                       "pages_1_based": [5, 2], "profile": None}
    assert page_request(["a", "b"], [1])["images"] == ["b"]


def test_image_input_stays_owned_by_caller():
    image = Image.new("RGB", (2, 2), "red")
    try:
        with request_image_scope({"images": [image]}) as images:
            assert images[0] is image
        assert image.getpixel((0, 0)) == (255, 0, 0)
    finally:
        image.close()


def test_pdf_images_close_when_ocr_raises():
    source = PdfPageSource.open(PDF, 150)
    with pytest.raises(RuntimeError, match="OCR failed"):
        with request_image_scope(page_request(source, [0])) as images:
            image = images[0]
            assert image.mode == "RGB"
            raise RuntimeError("OCR failed")
    with pytest.raises(ValueError, match="closed image"):
        image.getpixel((0, 0))


@pytest.mark.parametrize("indices", [[0], [-1], [True], [1, 1], [999]])
def test_invalid_page_requests(indices):
    with pytest.raises(ValueError):
        request_images({"pdf": {"path": str(PDF), "dpi": 150}, "pages_1_based": indices})


def test_pdf_and_images_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either"):
        request_images({"pdf": {}, "images": []})


@pytest.mark.parametrize("placement,expected_stages", [
    ("parent", {"render", "ocr", "chunk", "embed"}),
    ("page_workers", {"prepare PDF", "render + OCR", "chunk", "embed"}),
])
def test_pipeline_records_real_combined_wall_stage(placement, expected_stages, monkeypatch, tmp_path):
    from document_pipeline import pipeline as module
    pipe = Pipeline.__new__(Pipeline)
    pipe.cfg = load_config({"render": {"placement": placement}})
    pipe.trace_path = str(tmp_path / "trace.jsonl")
    pipe._nvtx = pipe._record_function = None
    seen = []
    pipe.ocr = SimpleNamespace(ocr=lambda pages: seen.append(pages) or ["page text"], stats={})
    pipe.chunker = SimpleNamespace(chunk=lambda text: [text])
    pipe.embedder = SimpleNamespace(embed=lambda texts: [[1.0]])
    monkeypatch.setattr(module, "render_pages", lambda path, dpi: ["rendered page"])
    monkeypatch.setattr(PdfPageSource, "open", lambda path, dpi: PdfPageSource(str(path), dpi, 1))
    output = pipe("input.pdf")
    assert output == {"ocr_text": "page text\n", "chunks": [{"id": "chunk-000", "embedding": [1.0]}]}
    trace = json.loads(Path(pipe.trace_path).read_text())
    assert set(trace["stages_s"]) == expected_stages
    assert set(trace["stage_windows"]) == expected_stages
    assert trace["pages"] == 1
    assert isinstance(seen[0], PdfPageSource) == (placement == "page_workers")


def test_unsupported_placement_is_rejected_before_model_loading():
    with pytest.raises(ValueError, match="Unsupported"):
        Pipeline(load_config({"render": {"placement": "typo"}}))
    with pytest.raises(ValueError, match="persistent OCR"):
        Pipeline(load_config({"render": {"placement": "page_workers"},
                              "ocr": {"backend": "vllm", "gpu_workers": 1}}))
