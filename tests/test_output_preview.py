"""Preview correctness and inert rendering of model-generated Markdown."""
import hashlib
import json

import pytest
import yaml

from lab.output_preview import prepare, render_markdown


def test_preview_preserves_tables_but_has_no_executable_content_or_fetches():
    rendered = render_markdown(
        '<table style="color:red" onclick="alert(1)"><tr><td colspan="2">Total &amp; cost</td>'
        '</tr></table>\n\n<script>alert(1)</script><iframe src="/api/overview"></iframe>'
        '<svg onload="alert(1)"></svg><img src="/api/overview" onerror="alert(1)" alt="Chart">\n\n'
        '[link](https://example.com) ![Diagram](https://example.com/image.png)'
    )
    assert '<td colspan="2">Total &amp; cost</td>' in rendered
    assert "[Chart]" in rendered and "[Diagram]" in rendered
    for forbidden in ("<script", "<iframe", "<svg", "<img", "onclick", "onerror", "href=", "src=", "style="):
        assert forbidden not in rendered


def test_preview_requires_exact_page_text_and_run_asset_identity(tmp_path):
    dataset = tmp_path / "data/example"
    dataset.mkdir(parents=True)
    (dataset / "x.pdf").write_bytes(b"%PDF example")
    reference = {"ocr_text": "First page\n\nSecond page\n", "chunks": []}
    (dataset / "x.json").write_text(json.dumps(reference))
    run = tmp_path / "results/example/20261026-120000"
    run.mkdir(parents=True)
    (run / "experiment.yaml").write_text(yaml.safe_dump({"root": "data/example"}))
    (run / "manifest.json").write_text(json.dumps({"cases": [
        {"id": "x", "pdf": "x.pdf", "reference": "x.json", "pages": 2}]}))
    (run / "baseline.json").write_text(json.dumps({
        "pass": True, "dataset_sha256": "dataset",
        "assets": [{"id": "x", "pages": 2,
                    "pdf_sha256": hashlib.sha256((dataset / "x.pdf").read_bytes()).hexdigest(),
                    "reference_sha256": hashlib.sha256((dataset / "x.json").read_bytes()).hexdigest()}]}))
    review = tmp_path / "review/x"
    review.mkdir(parents=True)
    (review / "page-1.md").write_text("First page\n")
    (review / "page-2.md").write_text("Wrong output\n")
    with pytest.raises(ValueError, match="No saved page set matches"):
        prepare(tmp_path, run, [review.parent])
    assert not (run / "output-preview.json").exists()
    (review / "page-2.md").write_text("Second page\n")
    path = prepare(tmp_path, run, [review.parent])
    saved = json.loads(path.read_text())
    assert saved["documents"][0]["markdown"] == reference["ocr_text"]
    assert saved["documents"][0]["pages"][1]["html"] == "<p>Second page</p>\n"
    (dataset / "x.pdf").write_bytes(b"changed PDF")
    with pytest.raises(ValueError, match="Saved run identity differs"):
        prepare(tmp_path, run, [review.parent])
