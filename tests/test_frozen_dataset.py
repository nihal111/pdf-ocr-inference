"""The shipped benchmark is complete, portable, frozen and smaller than 1 MB per file."""
import hashlib
import json
from pathlib import Path
import shutil

import pypdfium2 as pdfium

from eval.evaluate import load_cases

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/public-v3"


def test_frozen_dataset_files_sizes_pages_and_hashes(tmp_path):
    manifest = json.loads((DATA / "manifest.json").read_text())
    freeze = json.loads((DATA / "dataset-freeze.json").read_text())
    sources = json.loads((DATA / "sources.json").read_text())
    assert len(manifest["cases"]) == freeze["pdf_count"] == sources["selected_documents"] == 5
    assert sum(case["pages"] for case in manifest["cases"]) == freeze["page_count"] == sources["selected_pages"] == 20
    assert hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest() == freeze["manifest_sha256"]
    paths = {"manifest.json", "dataset-freeze.json", "sources.json"}
    for case in manifest["cases"]:
        paths.update([case["pdf"], case["reference"]])
        with pdfium.PdfDocument(DATA / case["pdf"]) as pdf:
            assert len(pdf) == case["pages"] <= 5
        frozen = next(item for item in freeze["pdfs"] if item["id"] == case["id"])
        reference = next(item for item in freeze["references"] if item["id"] == case["id"])
        assert hashlib.sha256((DATA / case["pdf"]).read_bytes()).hexdigest() == frozen["sha256"]
        assert hashlib.sha256((DATA / case["reference"]).read_bytes()).hexdigest() == reference["sha256"]
    for relative in paths:
        path = Path(relative)
        assert not path.is_absolute() and ".." not in path.parts
        assert (DATA / path).stat().st_size < 1_000_000
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / path, target)
    # Resolve every PDF/reference using only the shipped files, under a different root.
    cases = load_cases(tmp_path / "manifest.json", tmp_path)
    assert len(cases) == 5
    assert all(Path(case["pdf_path"]).is_relative_to(tmp_path) for case in cases)
