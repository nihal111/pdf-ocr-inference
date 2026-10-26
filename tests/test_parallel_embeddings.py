"""Ordering, device placement and failure draining for replicated embedding."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from document_pipeline import embedders
from document_pipeline.config import EmbedConfig


@pytest.fixture
def fake_workers(monkeypatch):
    calls, synchronized = [], []

    class FakeEmbedder:
        def __init__(self, cfg):
            self.cfg = cfg
            self.model = SimpleNamespace(parameters=lambda: iter([
                SimpleNamespace(device=torch.device(cfg.device))]))
            self.tokenizer = lambda texts, **kwargs: {
                "input_ids": [[0] * len(text) for text in texts]}

        def embed(self, texts):
            calls.append((self.cfg.device, texts))
            return [[float(ord(text[0]))] for text in texts]

    monkeypatch.setattr(embedders, "SentenceTransformersEmbedder", FakeEmbedder)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    return calls, synchronized


def test_balancing_preserves_original_order_and_records_devices(fake_workers):
    calls, synchronized = fake_workers
    pool = embedders.make_embedder(EmbedConfig(device="cuda:0", gpu_workers=4))
    try:
        texts = ["a", "bbbbbbbb", "cc", "ddddddd", "eee", "ffffff"]
        assert pool.embed(texts) == [[float(ord(text[0]))] for text in texts]
        assert {device for device, _ in calls} == {f"cuda:{i}" for i in range(4)}
        assert sorted(synchronized) == [f"cuda:{i}" for i in range(4)]
        assert sum(pool.stats["chunks_per_gpu"]) == len(texts)
        assert sum(pool.stats["tokens_per_gpu"]) == sum(map(len, texts))
        calls.clear()
        assert pool.embed([]) == []
        assert not calls
        assert pool.stats["chunks_per_gpu"] == [0] * 4
        assert pool.embed(["x"]) == [[120.0]]
        assert pool.stats["chunks_per_gpu"] == [1, 0, 0, 0]
    finally:
        pool.close()
        pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.embed(["x"])


def test_failed_worker_drains_other_work_before_returning(fake_workers, monkeypatch):
    _, synchronized = fake_workers
    pool = embedders.make_embedder(EmbedConfig(device="cuda:1", gpu_workers=2))
    started, release, finished = Event(), Event(), Event()

    def fail(texts):
        assert started.wait(5)
        raise ValueError("worker failed")

    def slow(texts):
        started.set()
        assert release.wait(5)
        finished.set()
        return [[98.0]]

    monkeypatch.setattr(pool.workers[0], "embed", fail)
    monkeypatch.setattr(pool.workers[1], "embed", slow)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(pool.embed, ["aaaa", "b"])
            assert started.wait(5)
            assert not future.done()
            release.set()
            with pytest.raises(ValueError, match="worker failed"):
                future.result(timeout=5)
            assert finished.is_set()
            assert sorted(synchronized) == ["cuda:1", "cuda:2"]
            assert pool.stats == {}
    finally:
        release.set()
        pool.close()


def test_bad_vector_count_is_rejected(fake_workers, monkeypatch):
    pool = embedders.make_embedder(EmbedConfig(device="cuda:0", gpu_workers=2))
    monkeypatch.setattr(pool.workers[0], "embed", lambda texts: [])
    try:
        with pytest.raises(RuntimeError, match="vector count"):
            pool.embed(["long", "b"])
    finally:
        pool.close()


def test_single_worker_keeps_existing_backend(monkeypatch):
    marker = object()
    monkeypatch.setitem(embedders.BACKENDS, "sentence_transformers", lambda cfg: marker)
    assert embedders.make_embedder(EmbedConfig()) is marker


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count(workers):
    with pytest.raises(ValueError, match="positive integer"):
        embedders.make_embedder(EmbedConfig(gpu_workers=workers))


def test_unsupported_devices_and_backend(fake_workers):
    for cfg, message in [
        (EmbedConfig(device="cpu", gpu_workers=2), "require CUDA"),
        (EmbedConfig(device="cuda:3", gpu_workers=2), "Not enough"),
        (EmbedConfig(backend="nanovllm", gpu_workers=2), "sentence_transformers"),
    ]:
        with pytest.raises(ValueError, match=message):
            embedders.make_embedder(cfg)


def test_pipeline_close_still_closes_ocr_on_embedder_error():
    from document_pipeline.pipeline import Pipeline
    closed = []
    pipe = Pipeline.__new__(Pipeline)

    def fail():
        raise RuntimeError("close failed")

    pipe.embedder = SimpleNamespace(close=fail)
    pipe.ocr = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="close failed"):
        pipe.close()
    assert closed == [True]
