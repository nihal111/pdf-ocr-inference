"""Prepare a chunk map and cosine similarities from frozen embedding references."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from document_pipeline.chunker import MarkdownChunker
from document_pipeline.config import ChunkConfig
from eval.evaluate import write_json
from lab.output_preview import render_markdown
from lab.site_data import identifier


def geometry(vectors):
    """PCA is only for display; similarities use the full normalized vectors."""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected at least two finite embeddings of equal dimension")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero vectors have no cosine similarity")
    normalized = matrix / norms[:, None]
    similarities = np.clip(normalized @ normalized.T, -1, 1)
    centered = normalized - normalized.mean(axis=0)
    u, singular, axes = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :2] * singular[:2]
    # Resolve SVD's arbitrary axis signs for repeatable presentation.
    signs = np.sign(axes[np.arange(2), np.abs(axes[:2]).argmax(axis=1)])
    coordinates *= signs
    energy = float(np.sum(singular ** 2))
    explained = singular[:2] ** 2 / energy if energy else np.zeros(2)
    return coordinates, similarities, explained


def prepare(root: Path, run: Path) -> Path:
    root, run = root.resolve(), run.resolve()
    relative = str(run.relative_to(root / "results"))
    spec = yaml.safe_load((run / "experiment.yaml").read_text())
    baseline = spec.get("baseline", "baseline")
    report = json.loads((run / f"{baseline}.json").read_text())
    if report.get("pass") is not True:
        raise ValueError("A passing baseline is required to identify the frozen vectors")
    config = json.loads((run / f"{baseline}.pipeline.json").read_text())
    environment = json.loads((run / f"{baseline}.env.json").read_text())
    chunker_hash = environment.get("source_files_sha256", {}).get("document_pipeline/chunker.py")
    if chunker_hash != hashlib.sha256((root / "document_pipeline/chunker.py").read_bytes()).hexdigest():
        raise ValueError("The chunker differs from the saved baseline; chunk text cannot be reconstructed safely")
    dataset = (root / spec["root"]).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    sources_path = dataset / "sources.json"
    sources = json.loads(sources_path.read_text()) if sources_path.exists() else {}
    titles = {source["id"]: source.get("title", source["id"]) for source in sources.get("sources", [])}
    assets = {asset["id"]: asset for asset in report["assets"]}
    chunker = MarkdownChunker(ChunkConfig(**config["chunk"]))
    documents, points, vectors = [], [], []
    for case in manifest["cases"]:
        asset = assets[case["id"]]
        raw = (dataset / case["reference"]).read_bytes()
        if (hashlib.sha256(raw).hexdigest() != asset["reference_sha256"]
                or hashlib.sha256((dataset / case["pdf"]).read_bytes()).hexdigest() != asset["pdf_sha256"]
                or case["pages"] != asset["pages"]):
            raise ValueError(f"The PDF or reference differs from the saved run: {case['id']}")
        reference = json.loads(raw)
        texts = chunker.chunk(reference["ocr_text"])
        chunks = reference["chunks"]
        if len(texts) != len(chunks) or any(c["id"] != f"chunk-{i:03d}" for i, c in enumerate(chunks)):
            raise ValueError(f"Reconstructed chunks differ from saved IDs or count: {case['id']}")
        documents.append({
            "id": case["id"], "title": titles.get(case["id"], case["id"]),
            "chunks": len(chunks), "pages": case["pages"],
            "pdf_sha256": asset["pdf_sha256"], "reference_sha256": asset["reference_sha256"],
        })
        for text, chunk in zip(texts, chunks):
            points.append({
                "index": len(points), "document_id": case["id"], "chunk_id": chunk["id"],
                "text": text, "html": render_markdown(text), "tokens": chunker._n(text),
            })
            vectors.append(chunk["embedding"])
    coordinates, similarities, explained = geometry(vectors)
    for point, position in zip(points, coordinates):
        point["position"] = position.tolist()
    output = run / "embedding-preview.json"
    write_json(output, {
        "schema_version": 1, "kind": "document-embedding-preview",
        "run_id": identifier(relative), "run_path": relative,
        "dataset_id": dataset.name, "dataset_sha256": report["dataset_sha256"],
        "reference_variant": baseline, "model": config["embed"]["model"].split("/")[-1],
        "dimensions": len(vectors[0]), "chunk_config": config["chunk"],
        "projection": {"method": "PCA", "explained_variance_ratio": explained.tolist()},
        "documents": documents, "points": points,
        "similarities": similarities.tolist(),
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    # Only tokenize saved text. No model inference, network access or GPU work.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(prepare(Path(__file__).resolve().parents[1], args.run))


if __name__ == "__main__":
    main()
