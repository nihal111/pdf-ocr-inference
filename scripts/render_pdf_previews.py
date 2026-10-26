#!/usr/bin/env python3
"""Render manifest-selected PDFs for the local notebook; never changes benchmark inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    output = root / "previews"
    output.mkdir(exist_ok=True)
    index = {}
    for case in manifest["cases"]:
        path = (root / case["pdf"]).resolve()
        if not path.is_relative_to((root / "pdfs").resolve()):
            raise ValueError("PDF is outside the selected input directory")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        doc = pdfium.PdfDocument(path)
        pages = {}
        try:
            if len(doc) != case["pages"]:
                raise ValueError("Manifest page count differs from the PDF")
            for number in range(len(doc)):
                target = output / f"{digest[:16]}-page-{number + 1}.png"
                page = doc[number]
                bitmap = page.render(scale=1.5)
                try:
                    bitmap.to_pil().convert("RGB").save(target)
                finally:
                    bitmap.close()
                    page.close()
                pages[str(number + 1)] = str(target.relative_to(root))
        finally:
            doc.close()
        index[case["id"]] = {"pdf_sha256": digest, "pages": pages}
        print(case["id"], len(pages), "page previews")
    temporary = output / "index.json.tmp"
    temporary.write_text(json.dumps(index, indent=2) + "\n")
    temporary.replace(output / "index.json")


if __name__ == "__main__":
    main()
