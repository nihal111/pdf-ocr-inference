"""Export the PDF OCR dashboard as plain files for GitHub Pages.

Export needs only PyYAML and markdown-it-py. Validation and serving use the
Python standard library. No inference, model loading, or GPU access occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import parse_qs, quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 1_000_000
MARKER = "export-manifest.json"
KIND = "pdf-ocr-static-site"
RAW_TRACE = re.compile(r"(?:^|/)(?:trace\.json(?:\.gz)?|[^/]*\.trace\.json(?:\.gz)?|[^/]*\.(?:nsys-rep|ncu-rep))$")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves the export directory: {relative}")
    return path


def check_site(output: Path) -> dict:
    """Check every exported byte; this also runs on a clean GitHub runner."""
    output = output.resolve()
    manifest = json.loads((output / MARKER).read_text())
    if manifest.get("kind") != KIND:
        raise ValueError("Not a PDF OCR static export")
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {MARKER}:
        raise ValueError("The export contains missing or unrecorded files")
    for relative, expected in manifest["files"].items():
        path = safe_path(output, relative)
        if (output / relative).is_symlink() or RAW_TRACE.search(relative):
            raise ValueError(f"Raw trace or symlink in export: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds 1 MB: {relative}")
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Export file changed: {relative}")
    if (output / MARKER).stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Export manifest exceeds 1 MB")
    overview = json.loads((output / "site-data/overview.json").read_text())
    required = {"index.html", ".nojekyll", "static/app.js", "static/style.css", "static/favicon.svg"}
    required.update(f"site-data/runs/{r['id']}.json" for r in overview["runs"])
    required.update(f"site-data/profiles/{p['id']}.json" for p in overview["profiles"])
    required.update(f"site-data/docs/{d['id']}.json" for d in overview["documents"])
    urls = list(overview["asset_urls"].values())
    for dataset in overview["datasets"]:
        for document in dataset["documents"]:
            urls.append(document["pdf_url"])
            urls.extend(document["preview_urls"].values())
    from urllib.parse import unquote
    required.update(unquote(url) for url in urls)
    if not required.issubset(actual):
        raise ValueError(f"Broken export links: {sorted(required - actual)}")
    return manifest


def export_site(root: Path = ROOT, output: Path | None = None) -> dict:
    from markdown_it import MarkdownIt
    from lab.site_data import Catalog

    root = root.resolve()
    output = (output or root / "site").resolve()
    if root.is_relative_to(output) or any(
            output.is_relative_to(root / name)
            for name in ("results", "data", "docs", "web", "lab", ".git")):
        raise ValueError("Export destination overlaps source data")
    if output.exists() and (
            not (output / MARKER).is_file()
            or json.loads((output / MARKER).read_text()).get("kind") != KIND):
        raise ValueError("Refusing to replace a directory that is not an owned export")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".site-export-", dir=output.parent))
    catalog = Catalog(root)
    overview = catalog.overview()
    runs, profiles = catalog.runs(True), catalog.profiles(True)
    urls = {}
    omitted = []

    def write(relative, content):
        path = safe_path(temporary, relative)
        payload = content if isinstance(content, bytes) else content.encode()
        if len(payload) > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds 1 MB: {relative} ({len(payload)} bytes)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def write_json(relative, content):
        write(relative, json.dumps(content, ensure_ascii=False, allow_nan=False,
                                   separators=(",", ":")) + "\n")

    def export_artifact(relative):
        if relative in urls:
            return urls[relative]
        path = catalog.artifact(relative)
        if RAW_TRACE.search(relative) or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Required artifact is a raw trace or exceeds 1 MB: {relative}")
        target = f"artifacts/{relative}"
        write(target, path.read_bytes())
        urls[relative] = quote(target, safe="/")
        return urls[relative]

    def keep_artifacts(record, names):
        kept = []
        for artifact in record["artifacts"]:
            if artifact["name"] in names and artifact["bytes"] <= MAX_FILE_BYTES:
                export_artifact(artifact["path"])
                kept.append(artifact)
            else:
                omitted.append(artifact["path"])
        record["artifacts"] = kept

    try:
        for run in runs:
            names = {"output-preview.json", "embedding-preview.json"}
            for variant in run["variants"]:
                names.update({f"{variant['name']}.json", f"{variant['name']}.compare.json"})
                previous = variant.get("previous_comparison", {}).get("source")
                if previous:
                    export_artifact(previous["report_path"])
            keep_artifacts(run, names)
            write_json(f"site-data/runs/{run['id']}.json", run)
        for profile in profiles:
            keep_artifacts(profile, {"profile.json", "operators.json"})
            write_json(f"site-data/profiles/{profile['id']}.json", profile)

        # Preview images and the five frozen PDFs are already individually < 1 MB.
        for dataset in overview["datasets"]:
            for document in dataset["documents"]:
                prefix = f"datasets/{dataset['id']}/{document['id']}"
                source = catalog.dataset_asset(dataset["id"], document["id"])
                write(f"{prefix}/document.pdf", source.read_bytes())
                document["pdf_url"] = quote(f"{prefix}/document.pdf", safe="/")
                document["preview_urls"] = {}
                for page in document["preview_pages"]:
                    image = catalog.dataset_asset(dataset["id"], document["id"], page)
                    relative = f"{prefix}/{page}.png"
                    write(relative, image.read_bytes())
                    document["preview_urls"][str(page)] = quote(relative, safe="/")

        renderer = MarkdownIt("commonmark", {"html": False}).enable("table")

        def rewrite_url(url):
            parsed = urlsplit(url)
            if not parsed.scheme and parsed.path == "/artifact":
                return export_artifact(parse_qs(parsed.query)["path"][0])
            if parsed.scheme in {"http", "https"} and parsed.hostname in {"localhost", "127.0.0.1"}:
                if parsed.fragment.startswith("/"):
                    return "#" + parsed.fragment
            if not parsed.scheme and parsed.path.startswith("/"):
                raise ValueError(f"Root-relative link is not portable to GitHub Pages: {url}")
            return url

        def rewrite_tokens(tokens):
            for token in tokens:
                for attribute in ("href", "src"):
                    value = token.attrGet(attribute)
                    if value:
                        token.attrSet(attribute, rewrite_url(value))
                if token.children:
                    rewrite_tokens(token.children)

        for doc in catalog.documents():
            tokens = renderer.parse(doc["body"])
            rewrite_tokens(tokens)
            doc["html"] = renderer.renderer.render(tokens, renderer.options, {})
            write_json(f"site-data/docs/{doc['id']}.json", doc)
        overview["asset_urls"] = urls
        overview["exported_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json("site-data/overview.json", overview)
        write("index.html", (root / "web/index.html").read_bytes())
        for name in ("app.js", "style.css", "favicon.svg"):
            write(f"static/{name}", (root / "web" / name).read_bytes())
        write(".nojekyll", b"")
        files = {str(p.relative_to(temporary)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(temporary.rglob("*")) if p.is_file()}
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
        manifest = {"kind": KIND, "schema_version": 1, "exported_at": overview["exported_at"],
                    "source_revision": revision, "max_file_bytes": MAX_FILE_BYTES,
                    "total_bytes": sum(f["bytes"] for f in files.values()),
                    "runs": len(runs), "profiles": len(profiles),
                    "pdfs": sum(len(d["documents"]) for d in overview["datasets"]),
                    "omitted_downloads": len(omitted), "files": files}
        write_json(MARKER, manifest)
        check_site(temporary)
        # Only replace an export previously created by this tool.
        old = output.parent / (temporary.name + "-old")
        if output.exists():
            output.rename(old)
        try:
            temporary.rename(output)
        except BaseException:
            if old.exists() and not output.exists():
                old.rename(output)
            raise
        if old.exists():
            shutil.rmtree(old)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path, metavar="SITE_DIRECTORY")
    args = parser.parse_args()
    manifest = check_site(args.check) if args.check else export_site(args.root, args.output)
    print(json.dumps({key: manifest[key] for key in
                      ("runs", "profiles", "pdfs", "total_bytes", "max_file_bytes", "omitted_downloads")}))


if __name__ == "__main__":
    main()
