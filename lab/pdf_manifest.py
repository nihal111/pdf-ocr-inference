"""Build an eval manifest from a directory of PDFs (page counts read with pypdfium2).

    python -m lab.pdf_manifest data/real --pdf-subdir pdfs     # writes data/real/manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root", help="dataset root; manifest paths are relative to it")
    p.add_argument("--pdf-subdir", default="pdfs")
    p.add_argument("--refs-subdir", default="refs")
    p.add_argument("--purpose", default="benchmark")
    args = p.parse_args(argv)

    import pypdfium2 as pdfium
    root = Path(args.root)
    pdfs = sorted((root / args.pdf_subdir).rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs under {root / args.pdf_subdir}")
    cases, ids = [], set()
    for pdf in pdfs:
        ident = re.sub(r"[^A-Za-z0-9._-]+", "-", str(pdf.relative_to(root / args.pdf_subdir).with_suffix("")))
        assert ident not in ids, ident
        ids.add(ident)
        doc = pdfium.PdfDocument(str(pdf))
        cases.append({"id": ident, "pdf": str(pdf.relative_to(root)), "reference": f"{args.refs_subdir}/{ident}.json",
                      "pages": len(doc)})
        doc.close()
    out = root / "manifest.json"
    out.write_text(json.dumps({"purpose": args.purpose, "cases": cases}, indent=2) + "\n")
    print(f"{out}: {len(cases)} PDFs, {sum(c['pages'] for c in cases)} pages")


if __name__ == "__main__":
    main()
