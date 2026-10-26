"""The exported site is self-contained, bounded in size, and preserves evidence."""
import builtins
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from lab.export_site import MAX_FILE_BYTES, check_site, export_site


def workspace(root):
    shutil.copytree(Path(__file__).resolve().parents[1] / "web", root / "web")
    docs = root / "docs/pages"
    docs.mkdir(parents=True)
    (docs / "example.md").write_text(
        "---\ntitle: Example\n---\n<script>bad()</script>\n\n"
        "[Numbers](/artifact?path=analysis/example.json)\n")
    analysis = root / "results/analysis"
    analysis.mkdir(parents=True)
    (analysis / "example.json").write_text('{"value":42}')
    profile = root / "results/profiles/example"
    profile.mkdir(parents=True)
    (profile / "profile.json").write_text(json.dumps(
        {"kind": "document-profile", "status": "complete", "label": "Example"}))
    (profile / "operators.json").write_text('[{"name":"matmul","calls":1000}]')
    (profile / "trace.json").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    (profile / "notes.log").write_text("Local diagnostic")
    data = root / "data/example"
    (data / "pdfs").mkdir(parents=True)
    pdf = b"%PDF-1.4\nexample"
    (data / "pdfs/example.pdf").write_bytes(pdf)
    (data / "manifest.json").write_text(json.dumps({
        "cases": [{"id": "example", "pdf": "pdfs/example.pdf", "pages": 1}]}))
    (data / "previews").mkdir()
    (data / "previews/example.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (data / "previews/index.json").write_text(json.dumps({"example": {
        "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
        "pages": {"1": "previews/example.png"}}}))


def test_export_is_standalone_and_excludes_raw_traces(tmp_path, monkeypatch):
    workspace(tmp_path)
    original_import = builtins.__import__

    def no_gpu(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "paddle", "vllm", "transformers"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_gpu)
    manifest = export_site(tmp_path)
    output = tmp_path / "site"
    assert check_site(output) == manifest
    assert manifest["omitted_downloads"] == 2
    assert not list(output.rglob("trace.json"))
    assert not list(output.rglob("*.log"))
    assert all(entry["bytes"] <= MAX_FILE_BYTES for entry in manifest["files"].values())
    assert (output / "artifacts/analysis/example.json").read_bytes() == b'{"value":42}'
    doc = json.loads((output / "site-data/docs/pages/example.json").read_text())
    assert 'href="artifacts/analysis/example.json"' in doc["html"]
    assert "<script>" not in doc["html"] and "&lt;script&gt;" in doc["html"]
    overview = json.loads((output / "site-data/overview.json").read_text())
    pdf = overview["datasets"][0]["documents"][0]
    assert (output / pdf["pdf_url"]).read_bytes().startswith(b"%PDF")
    assert (output / pdf["preview_urls"]["1"]).is_file()
    assert "/static/" not in (output / "index.html").read_text().replace("./static/", "")
    # Local raw experiment inputs can be removed; the snapshot remains complete.
    shutil.rmtree(tmp_path / "results")
    shutil.rmtree(tmp_path / "data")
    assert check_site(output)["profiles"] == 1


def test_oversized_required_asset_fails_without_replacing_previous_export(tmp_path):
    workspace(tmp_path)
    export_site(tmp_path)
    before = (tmp_path / "site/export-manifest.json").read_bytes()
    (tmp_path / "results/analysis/example.json").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds 1 MB"):
        export_site(tmp_path)
    assert (tmp_path / "site/export-manifest.json").read_bytes() == before
    assert check_site(tmp_path / "site")
    assert not list(tmp_path.glob(".site-export-*"))


def test_export_replaces_only_owned_snapshot_and_detects_changed_files(tmp_path):
    workspace(tmp_path)
    output = tmp_path / "site"
    output.mkdir()
    (output / "important.txt").write_text("Keep this")
    with pytest.raises(ValueError, match="Refusing to replace"):
        export_site(tmp_path)
    shutil.rmtree(output)
    export_site(tmp_path)
    (tmp_path / "results/analysis/example.json").write_text('{"value":43}')
    export_site(tmp_path)
    assert json.loads((output / "artifacts/analysis/example.json").read_text())["value"] == 43
    (output / "static/app.js").write_text("changed")
    with pytest.raises(ValueError, match="Export file changed"):
        check_site(output)


def test_export_rejects_link_escape_and_input_overlap(tmp_path):
    workspace(tmp_path)
    with pytest.raises(ValueError, match="overlaps source"):
        export_site(tmp_path, tmp_path / "results/site")
    (tmp_path / "docs/pages/example.md").write_text("[bad](/artifact?path=../outside.json)")
    with pytest.raises(ValueError, match="allowed directory"):
        export_site(tmp_path)
