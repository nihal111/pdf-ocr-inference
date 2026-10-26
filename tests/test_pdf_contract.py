"""Public adapter contract, real PDF parsing, and recovery without GPU models."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from document_pipeline import pipeline as module
from document_pipeline.config import load_config
from document_pipeline.pipeline import Pipeline, result_dict
from document_pipeline.worker_render import PdfPageSource, page_request, request_image_scope
from eval.evaluate import validate_output
from scripts.make_sample_pdfs import text_pdf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pdfs(tmp_path):
    good = tmp_path / "three-pages.pdf"
    text_pdf(good, [["First page"], ["Second page"], ["Third page"]])
    zero = tmp_path / "zero-pages.pdf"
    text_pdf(zero, [])
    data = good.read_bytes()
    bad = [(tmp_path / "missing.pdf", FileNotFoundError), (tmp_path, ValueError),
           (zero, ValueError)]
    for name, payload in {
        "empty": b"", "text": b"This is not a PDF",
        "invalid-objects": b"%PDF-1.4\nbroken document\n%%EOF\n",
        "truncated": data[:len(data) // 2],
        "missing-eof": data[:data.rfind(b"%%EOF")],
    }.items():
        path = tmp_path / f"{name}.pdf"
        path.write_bytes(payload)
        bad.append((path, ValueError))
    dangling = tmp_path / "dangling.pdf"
    dangling.symlink_to(tmp_path / "absent.pdf")
    bad.append((dangling, FileNotFoundError))
    return good, bad


def cpu_pipeline(monkeypatch, placement, page_texts=None):
    seen = []
    ocr = SimpleNamespace(stats={})

    def recognize(source):
        seen.append(source)
        if isinstance(source, PdfPageSource):
            with request_image_scope(page_request(source, list(range(len(source))))):
                pass
        return page_texts if page_texts is not None else [
            f"Page {i + 1} complete text" for i in range(len(source))]

    ocr.ocr = recognize
    monkeypatch.setattr(module, "make_ocr", lambda cfg: ocr)
    monkeypatch.setattr(module, "MarkdownChunker", lambda cfg: SimpleNamespace(
        chunk=lambda text: [part for part in text.split("\n\n") if part.strip()] or [text]))
    monkeypatch.setattr(module, "make_embedder", lambda cfg: SimpleNamespace(
        embed=lambda texts: [[float(i + 1), -0.25] for i in range(len(texts))]))
    return Pipeline(load_config({"render": {"placement": placement, "dpi": 72}})), seen


@pytest.mark.parametrize("placement", ["parent", "page_workers"])
def test_embed_pdf_contract_and_recovery_after_every_bad_input(placement, pdfs, monkeypatch):
    good, bad = pdfs
    pipe, seen = cpu_pipeline(monkeypatch, placement)
    # Load the actual public entry point with CPU stand-ins for model inference.
    monkeypatch.setattr(module, "Pipeline", lambda: pipe)
    spec = importlib.util.spec_from_file_location("contract_adapter", ROOT / "document_pipeline/adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    expected = {
        "ocr_text": "Page 1 complete text\n\nPage 2 complete text\n\nPage 3 complete text\n",
        "chunks": [{"id": f"chunk-{i:03d}", "embedding": [float(i + 1), -0.25]}
                   for i in range(3)],
    }
    assert adapter.embed_pdf(str(good)) == expected
    assert validate_output(expected) == []
    for path, error_type in bad:
        calls_before = len(seen)
        with pytest.raises(error_type):
            adapter.embed_pdf(str(path))
        assert len(seen) == calls_before, "Bad inputs must not reach OCR"
        assert adapter.embed_pdf(str(good)) == expected


@pytest.mark.parametrize("vectors", [
    None, [], [[1.0]], [[1.0], [2.0], [3.0]], [[1.0], []],
    [[1.0], [2.0, 3.0]], ((1.0,), (2.0,)), [(1.0,), [2.0]],
    [[True], [2.0]], [[1], [2.0]], [["1.0"], [2.0]],
    [[float("nan")], [2.0]], [[float("inf")], [2.0]],
    [[float("-inf")], [2.0]],
])
def test_invalid_embeddings_are_never_returned(vectors):
    with pytest.raises(RuntimeError, match="vector|embedding"):
        result_dict("first\n\nsecond\n", ["first", "second"], vectors)


def test_chunk_ids_remain_unique_past_three_digits():
    result = result_dict("text\n", ["text"] * 1001, [[1.0, -0.5] for _ in range(1001)])
    ids = [chunk["id"] for chunk in result["chunks"]]
    assert ids == [f"chunk-{index:03d}" for index in range(1001)]
    assert len(set(ids)) == len(ids)
    assert validate_output(result) == []


@pytest.mark.parametrize("pages", [[], ["only one page"], ["first", "second", None],
                                  ["first", "second", "\ud800"]])
def test_missing_pages_or_invalid_ocr_text_are_not_returned(pages, pdfs, monkeypatch):
    pipe, _ = cpu_pipeline(monkeypatch, "parent", page_texts=pages)
    with pytest.raises(RuntimeError, match="every page|UTF-8"):
        pipe(str(pdfs[0]))


def test_blank_document_still_has_a_chunk(tmp_path, monkeypatch):
    blank = tmp_path / "blank.pdf"
    text_pdf(blank, [[]])
    pipe, _ = cpu_pipeline(monkeypatch, "parent", page_texts=[""])
    output = pipe(str(blank))
    assert output["ocr_text"] == "\n"
    assert len(output["chunks"]) == 1
    assert validate_output(output) == []


def test_parent_images_are_released_after_ocr_failure(pdfs, monkeypatch):
    pipe, _ = cpu_pipeline(monkeypatch, "parent")
    images = []

    def fail(pages):
        images.extend(pages)
        raise RuntimeError("OCR failed")

    pipe.ocr.ocr = fail
    with pytest.raises(RuntimeError, match="OCR failed"):
        pipe(str(pdfs[0]))
    assert len(images) == 3
    for image in images:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))


def test_native_pdf_errors_do_not_terminate_the_calling_process(pdfs):
    good, bad = pdfs
    # Keep the native parser in a child test process: an abort or a segfault fails
    # this assertion instead of taking the entire test runner down with it.
    script = r'''
import json, sys
from document_pipeline.ocr import render_pages
from document_pipeline.worker_render import PdfPageSource
good, bad = sys.argv[1], json.loads(sys.argv[2])
for path, kind in bad:
    for reader in (lambda p: render_pages(p, 72), lambda p: PdfPageSource.open(p, 72)):
        try:
            reader(path)
        except (FileNotFoundError, ValueError) as exc:
            assert type(exc).__name__ == kind, (path, type(exc).__name__, kind)
        else:
            raise AssertionError("Bad PDF was accepted: " + path)
        images = render_pages(good, 72)
        assert len(images) == 3
        for image in images:
            image.close()
print("survived and processed a valid PDF after every bad input")
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(good),
         json.dumps([(str(path), error.__name__) for path, error in bad])],
        capture_output=True, text=True, timeout=30, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "survived" in result.stdout
