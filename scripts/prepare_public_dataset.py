#!/usr/bin/env python3
"""Freeze ten public PDFs with at most five selected pages each."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "experiments/pdf/public_sources.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/public-v3")
    args = parser.parse_args()
    target = args.out.resolve()
    if (target / "manifest.json").exists() or (target / "sources.json").exists():
        raise SystemExit("Dataset already frozen; refusing to replace inputs or provenance.")
    inventory = json.loads(INVENTORY.read_text())["sources"]
    (target / "pdfs").mkdir(parents=True, exist_ok=True)
    sources = []
    for original in inventory:
        source = {key: original[key] for key in ("id", "title", "source_page", "url", "category")}
        path = target / "pdfs" / f"{source['id']}.pdf"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
        with urlopen(Request(source["url"], headers={"User-Agent": "Mozilla/5.0"}),
                     timeout=120) as response:
            data = response.read(100 * 1024 * 1024 + 1)
            if len(data) > 100 * 1024 * 1024 or not data.lstrip().startswith(b"%PDF-"):
                raise ValueError(f"Invalid or oversized PDF: {source['id']}")
            if hashlib.sha256(data).hexdigest() != original["expected_sha256"]:
                raise ValueError(f"Publisher PDF changed: {source['id']}; refusing to alter the corpus")
            source["resolved_url"] = response.url
        source["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        selection = original["selected_pages_1_based"]
        if not 1 <= len(selection) <= 5 or len(set(selection)) != len(selection):
            raise ValueError(f"Expected one to five distinct source pages: {source['id']}")
        with pdfium.PdfDocument(data) as document:
            pages = len(document)
            if pages != original["expected_pages"]:
                raise ValueError(f"Page count differs from frozen inventory: {path}")
            if min(selection) < 1 or max(selection) > pages:
                raise ValueError(f"Invalid source-page selection: {source['id']}")
            complete = selection == list(range(1, pages + 1))
            if complete:
                path.write_bytes(data)
            else:
                with pdfium.PdfDocument.new() as selected:
                    selected.import_pages(document, [number - 1 for number in selection])
                    selected.save(path)
        with pdfium.PdfDocument(path) as document:
            assert len(document) == len(selection)
            for number in range(len(document)):
                page = document[number]
                assert all(dimension > 0 for dimension in page.get_size())
                page.close()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source.update({
            "pdf": f"pdfs/{path.name}",
            "pdf_sha256": digest, "original_sha256": original["expected_sha256"],
            "bytes": path.stat().st_size,
            "original_pages": pages, "selected_pages_1_based": selection,
            "full_document": complete,
            "selection_notes": ("Complete source PDF." if complete else
                                "Selected source pages preserve text, figures and table layouts; no rasterization or rotation."),
        })
        sources.append(source)
        print(source["id"], len(selection), "pages", digest, flush=True)
    assert len(sources) == 10 and len(list((target / "pdfs").glob("*.pdf"))) == 10
    provenance = {
        "dataset": target.name, "sources": sources, "selected_documents": len(sources),
        "selected_pages": sum(len(source["selected_pages_1_based"]) for source in sources),
        "maximum_pages_per_document": 5,
        "selection_method": "At most five source pages per document, copied with pypdfium2 without rasterization, scaling or rotation.",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest = {
        "purpose": "public-web-five-page-benchmark-v3",
        "cases": [{"id": source["id"], "pdf": source["pdf"],
                   "reference": f"refs/{source['id']}.json", "pages": len(source["selected_pages_1_based"])}
                  for source in sources],
    }
    (target / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    smallest = {"purpose": "public-web-three-smallest-initial-baseline",
                "cases": sorted(manifest["cases"], key=lambda case: (case["pages"], case["id"]))[:3]}
    (target / "manifest-smallest.json").write_text(json.dumps(smallest, indent=2) + "\n")
    print(f"Frozen {len(sources)} PDFs, {provenance['selected_pages']} pages; at most five pages each.")


if __name__ == "__main__":
    main()
