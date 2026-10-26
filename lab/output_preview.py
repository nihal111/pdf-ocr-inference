"""Prepare a PDF/Markdown preview from saved pages, without running inference."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from html.parser import HTMLParser
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

from eval.evaluate import write_json
from lab.site_data import identifier


class PreviewHTML(HTMLParser):
    """Keep document formatting, including OCR tables, with no active content."""

    tags = frozenset("p div span h1 h2 h3 h4 h5 h6 strong em b i s del sub sup "
                     "ul ol li blockquote pre code br hr table thead tbody tfoot tr th td caption".split())
    void = {"br", "hr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            label = dict(attrs).get("alt") or "Image placeholder"
            self.parts.append(f'<span class="markdown-image-placeholder">[{html.escape(label)}]</span>')
        elif tag in self.tags:
            spans = "".join(f' {key}="{value}"' for key, value in attrs
                            if tag in {"td", "th"} and key in {"rowspan", "colspan"}
                            and value and value.isascii() and value.isdigit()
                            and 1 <= int(value) <= 100)
            self.parts.append(f"<{tag}{spans}>")

    def handle_endtag(self, tag):
        if tag in self.tags and tag not in self.void:
            self.parts.append(f"</{tag}>")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data):
        self.parts.append(html.escape(data))


def render_markdown(text: str) -> str:
    # PaddleOCR emits HTML tables inside Markdown. Preserve table structure but
    # remove links, image fetches, styles, event handlers and executable elements.
    parser = PreviewHTML()
    parser.feed(MarkdownIt("commonmark", {"html": True}).enable("table").render(text))
    parser.close()
    return "".join(parser.parts)


def prepare(root: Path, run: Path, page_reviews: list[Path]) -> Path:
    root, run = root.resolve(), run.resolve()
    relative = str(run.relative_to(root / "results"))
    spec = yaml.safe_load((run / "experiment.yaml").read_text())
    baseline = spec.get("baseline", "baseline")
    report = json.loads((run / f"{baseline}.json").read_text())
    if report.get("pass") is not True:
        raise ValueError("A passing baseline is required to identify the preview output")
    dataset = (root / spec["root"]).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    sources_path = dataset / "sources.json"
    sources = json.loads(sources_path.read_text()) if sources_path.exists() else {}
    titles = {source["id"]: source.get("title", source["id"]) for source in sources.get("sources", [])}
    assets = {asset["id"]: asset for asset in report["assets"]}
    documents = []
    for case in manifest["cases"]:
        asset = assets[case["id"]]
        reference_bytes = (dataset / case["reference"]).read_bytes()
        if (hashlib.sha256(reference_bytes).hexdigest() != asset["reference_sha256"]
                or hashlib.sha256((dataset / case["pdf"]).read_bytes()).hexdigest() != asset["pdf_sha256"]
                or case["pages"] != asset["pages"]):
            raise ValueError(f"Saved run identity differs for {case['id']}")
        markdown = json.loads(reference_bytes)["ocr_text"]
        page_texts = None
        for review in page_reviews:
            files = [review / case["id"] / f"page-{page}.md" for page in range(1, case["pages"] + 1)]
            if all(path.is_file() for path in files):
                texts = [path.read_text().strip() for path in files]
                if "\n\n".join(texts) + "\n" == markdown:
                    page_texts = texts
                    break
        if page_texts is None:
            raise ValueError(f"No saved page set matches the reference for {case['id']}")
        documents.append({
            "id": case["id"], "title": titles.get(case["id"], case["id"]),
            "pdf_sha256": asset["pdf_sha256"], "reference_sha256": asset["reference_sha256"],
            "markdown": markdown,
            "pages": [{"number": number, "markdown": text, "html": render_markdown(text)}
                      for number, text in enumerate(page_texts, 1)],
        })
    output = run / "output-preview.json"
    write_json(output, {
        "schema_version": 1, "kind": "document-output-preview",
        "run_id": identifier(relative), "run_path": relative,
        "dataset_id": dataset.name, "dataset_sha256": report["dataset_sha256"],
        "reference_variant": baseline, "documents": documents,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--page-review", action="append", required=True, type=Path)
    args = parser.parse_args()
    print(prepare(Path(__file__).resolve().parents[1], args.run, args.page_review))


if __name__ == "__main__":
    main()
