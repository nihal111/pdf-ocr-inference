"""PDF -> page images -> VLM OCR -> chunks -> embeddings, timed per stage.

Stages:
  render    pypdfium2: each page -> RGB image at `render.dpi`
  ocr       PaddleOCR parses each page to markdown (greedy); ocr_text = pages joined by blank lines
  chunk     MarkdownChunker (token-limited, heading context)
  embed     embedding backend

No results are cached between calls. The evaluator calls the same PDF repeatedly, and a
cache would turn the benchmark into a cache-hit measurement.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

from document_pipeline.chunker import MarkdownChunker
from document_pipeline.config import PipelineConfig, load_config
from document_pipeline.embedders import make_embedder
from document_pipeline.ocr import make_ocr, render_pages


def join_pages(pages: list[str]) -> str:
    return "\n\n".join(p.strip() for p in pages) + "\n"


class Pipeline:

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or load_config()
        if self.cfg.render.placement not in {"parent", "page_workers"}:
            raise ValueError("Unsupported render placement")
        if self.cfg.render.placement == "page_workers" and not (
                self.cfg.ocr.backend == "paddleocr"
                or self.cfg.ocr.backend == "hf" and self.cfg.ocr.gpu_workers > 1):
            raise ValueError("Page-worker rendering requires the persistent OCR page pool")
        self.trace_path = os.environ.get("DOCUMENT_PIPELINE_TRACE")
        self._nvtx = None
        self._record_function = None  # enabled only during a diagnostic torch.profiler call
        if os.environ.get("DOCUMENT_PIPELINE_NVTX"):
            import torch
            self._nvtx = torch.cuda.nvtx
        t = time.perf_counter()
        self.ocr = make_ocr(self.cfg.ocr)
        self.chunker = MarkdownChunker(self.cfg.chunk)
        self.embedder = make_embedder(self.cfg.embed)
        self.init_s = time.perf_counter() - t

    def close(self):
        try:
            if hasattr(self.embedder, "close"):
                self.embedder.close()
        finally:
            if hasattr(self.ocr, "close"):
                self.ocr.close()

    @contextmanager
    def _stage(self, timings: dict, name: str, windows: dict):
        if self._nvtx:
            self._nvtx.range_push(name)
        t = time.perf_counter()
        started_unix_s = time.time()
        try:
            if self._record_function:
                with self._record_function(f"document_pipeline::{name}"):
                    yield
            else:
                yield
        finally:
            timings[name] = time.perf_counter() - t
            windows[name] = {"start_unix_s": started_unix_s, "end_unix_s": time.time()}
            if self._nvtx:
                self._nvtx.range_pop()

    def __call__(self, path: str) -> dict:
        timings: dict[str, float] = {}
        windows = {}
        t0 = time.perf_counter()
        worker_render = self.cfg.render.placement == "page_workers"
        with self._stage(timings, "prepare PDF" if worker_render else "render", windows):
            if worker_render:
                from document_pipeline.worker_render import PdfPageSource
                images = PdfPageSource.open(path, self.cfg.render.dpi)
            else:
                images = render_pages(path, self.cfg.render.dpi)
        with self._stage(timings, "render + OCR" if worker_render else "ocr", windows):
            page_texts = self.ocr.ocr(images)
            text = join_pages(page_texts)
        # Optional quality-review artifacts; ordinary benchmark calls leave this
        # unset. The text returned to the evaluator has no extra page markers.
        review = os.environ.get("DOCUMENT_PIPELINE_PAGE_REVIEW")
        if review:
            from pathlib import Path
            directory = Path(review) / Path(path).stem
            directory.mkdir(parents=True, exist_ok=True)
            for page, page_text in enumerate(page_texts, 1):
                (directory / f"page-{page}.md").write_text(page_text + "\n")
            (directory / "completion.json").write_text(json.dumps(
                {"pdf": os.path.basename(path), "pages": len(images),
                 "ocr_stats": getattr(self.ocr, "stats", {})}, indent=2) + "\n")
        with self._stage(timings, "chunk", windows):
            texts = self.chunker.chunk(text)
        with self._stage(timings, "embed", windows):
            vectors = self.embedder.embed(texts)
        total = time.perf_counter() - t0
        if self.trace_path:
            with open(self.trace_path, "a") as f:
                f.write(json.dumps({"pdf": os.path.basename(path), "pages": len(images), "chunks": len(texts),
                                    "total_s": total, "stages_s": timings,
                                    "stage_windows": windows,
                                    "ocr_stats": getattr(self.ocr, "stats", {}),
                                    "embed_stats": getattr(self.embedder, "stats", {})}) + "\n")
        return {"ocr_text": text,
                "chunks": [{"id": f"chunk-{i:03d}", "embedding": v} for i, v in enumerate(vectors)]}
