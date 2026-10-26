#!/usr/bin/env python3
"""Generate a small dev dataset: digital-text, scanned (image-only, needs OCR) and mixed PDFs,
plus an eval manifest. Real benchmarks should use your own PDFs (see lab/pdf_manifest.py).

    python scripts/make_sample_pdfs.py data/sample
"""
import argparse
import json
import random

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WORDS = ("latency throughput kernel batch scheduler memory bandwidth cache token prefill decode "
         "embedding chunk document layout table figure section result method baseline model "
         "inference profile optimize queue request page parse").split()


def paragraphs(rng, n):
    out = []
    for _ in range(n):
        sentences = []
        for _ in range(rng.randint(3, 6)):
            words = [rng.choice(WORDS) for _ in range(rng.randint(6, 14))]
            sentences.append(" ".join(words).capitalize() + ".")
        out.append(" ".join(sentences))
    return out


def wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def page_lines(rng, title):
    lines = [title, ""]
    for p in paragraphs(rng, 5):
        lines += wrap(p, 80) + [""]
    return lines


def text_pdf(path, pages):
    """Minimal PDF with a real text layer (Helvetica), no dependencies."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        ops = ["BT", "/F1 11 Tf", "14 TL", "56 760 Td"]
        for i, line in enumerate(lines):
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(("/F1 16 Tf " if i == 0 else "/F1 11 Tf ") + f"({esc}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        content = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(len(objs))
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        body = obj if isinstance(obj, bytes) else obj.encode()
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    Path(path).write_bytes(bytes(out))


def scanned_page(rng, lines, dpi=150):
    w, h = int(8.5 * dpi), int(11 * dpi)
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    try:
        body = ImageFont.truetype("DejaVuSans.ttf", 22)
        head = ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        body = head = ImageFont.load_default(size=22)
    y = 110
    for i, line in enumerate(lines):
        draw.text((110, y), line, fill=0, font=head if i == 0 else body)
        y += 48 if i == 0 else 30
    # Mild scan artefacts: slight rotation, blur and noise.
    img = img.rotate(rng.uniform(-0.6, 0.6), fillcolor=255, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    px = img.load()
    for _ in range(w * h // 400):
        px[rng.randrange(w), rng.randrange(h)] = rng.randint(120, 220)
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    cases = []

    def add(name, pages):
        cases.append({"id": name, "pdf": f"pdfs/{name}.pdf", "reference": f"refs/{name}.json", "pages": pages})

    text_pdf(out / "pdfs/digital-3p.pdf", [page_lines(rng, f"Section {i + 1}: Results") for i in range(3)])
    add("digital-3p", 3)
    imgs = [scanned_page(rng, page_lines(rng, f"Scanned report page {i + 1}")) for i in range(4)]
    imgs[0].save(out / "pdfs/scanned-4p.pdf", save_all=True, append_images=imgs[1:], resolution=150)
    add("scanned-4p", 4)
    imgs = [scanned_page(rng, page_lines(rng, "Appendix scan"))]
    imgs[0].save(out / "pdfs/scanned-1p.pdf", resolution=150)
    add("scanned-1p", 1)

    (out / "manifest.json").write_text(json.dumps({"purpose": "dev-sample", "cases": cases}, indent=2) + "\n")
    print(f"wrote {len(cases)} PDFs and {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
