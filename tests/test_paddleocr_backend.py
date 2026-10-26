"""Completion checks operate on raw generated tokens before parser cleanup."""
from types import SimpleNamespace

import numpy as np
import pytest

from document_pipeline.paddleocr_backend import PaddleOcr


@pytest.mark.parametrize("overrides,expected", [
    ({"paddle_region_batch_size": 4}, 4),
    ({"paddle_rec_backend": "vllm-server", "paddle_region_batch_size": 4,
      "paddle_region_dispatch": "all_gpus", "paddle_region_submission_size": 64}, 64),
])
def test_backend_cannot_silently_override_requested_region_batch(monkeypatch, overrides, expected):
    import sys
    from document_pipeline.config import load_config

    requested, closed = [], []
    parser = SimpleNamespace(
        paddlex_pipeline=SimpleNamespace(vl_rec_model=SimpleNamespace(
            batch_sampler=SimpleNamespace(batch_size=1))),
        close=lambda: closed.append(True))

    def make_parser(**kwargs):
        requested.append(kwargs["paddlex_config"]["SubModules"]["VLRecognition"]["batch_size"])
        return parser

    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(set_device=lambda _: None))
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=make_parser))
    monkeypatch.setitem(sys.modules, "paddlex.inference", SimpleNamespace(
        load_pipeline_config=lambda _: {"SubModules": {"LayoutDetection": {}, "VLRecognition": {}}}))
    cfg = load_config({"ocr": overrides}).ocr
    with pytest.raises(ValueError, match="backend overrode"):
        PaddleOcr(cfg, device="cuda:0")
    assert requested == [expected] and closed == [True]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_region_submission_window_rejects_invalid_sizes(value):
    from document_pipeline.config import load_config
    cfg = load_config({"ocr": {"paddle_rec_backend": "vllm-server",
                              "paddle_region_submission_size": value}}).ocr
    with pytest.raises(ValueError, match="positive integer"):
        PaddleOcr(cfg, device="cuda:0")


def test_shared_admission_requires_broker_before_loading_models(monkeypatch):
    from document_pipeline.config import load_config
    monkeypatch.delenv("DOCUMENT_PIPELINE_REGION_ADMISSION_SOCKET", raising=False)
    cfg = load_config({"ocr": {"paddle_rec_backend": "vllm-server",
                              "paddle_region_dispatch": "all_gpus",
                              "paddle_region_admission": "available"}}).ocr
    with pytest.raises(ValueError, match="page-worker broker"):
        PaddleOcr(cfg, device="cuda:0")


@pytest.mark.parametrize("tokens,finished", [([8, 2, 0], True), ([8, 9, 10], False)])
def test_raw_region_eos_guard(tokens, finished, monkeypatch):
    monkeypatch.delenv("DOCUMENT_PIPELINE_OCR_DIAGNOSTICS", raising=False)
    ocr = PaddleOcr.__new__(PaddleOcr)
    ocr.cfg = SimpleNamespace(require_eos=True)
    ocr.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2),
                               config=SimpleNamespace(image_token_id=7))
    ocr._local_page = 1
    ocr.stats = {"batches": [], "gen_tokens": 0, "visual_tokens": 0}
    output = (SimpleNamespace(numpy=lambda: np.array([tokens])), None)
    ocr._generate = lambda inputs, **kwargs: output
    inputs = {"input_ids": np.array([[7, 7, 4]])}
    if finished:
        assert ocr._checked_generate(inputs, max_new_tokens=3) is output
        assert ocr.stats["gen_tokens"] == 2
    else:
        with pytest.raises(RuntimeError, match="Refusing truncated output"):
            ocr._checked_generate(inputs, max_new_tokens=3)
    assert ocr.stats["batches"][0]["ended_with_eos"] == [finished]
    assert ocr.stats["visual_tokens"] == 2


def test_server_requests_preserve_pixels_and_greedy_decoding():
    from document_pipeline.paddle_server import ServerRecognition
    formats = []
    def build(client, data, image_format, *args):
        formats.append(image_format)
        return [([], {"extra_body": {}})]
    ocr = SimpleNamespace(recognizer=SimpleNamespace(_doc_vlm_genai_build_request_specs=build))
    bridge = ServerRecognition(ocr)
    specs = bridge.prepare(None, [{"image": "unused"}], "JPEG")
    assert formats == ["PNG"]
    assert specs[0][1]["temperature"] == 0
    assert specs[0][1]["extra_body"]["return_token_ids"] is True
    assert specs[0][1]["extra_body"]["repetition_penalty"] == 1


def test_table_resolution_override_preserves_text_and_formula_settings():
    from document_pipeline.paddle_server import ServerRecognition
    def build(client, data, image_format, *args):
        return [([], {"extra_body": {"mm_processor_kwargs": {
            "min_pixels": 112896, "max_pixels": 1003520}}}) for _ in data]
    ocr = SimpleNamespace(
        cfg=SimpleNamespace(paddle_table_max_pixels=2822400),
        recognizer=SimpleNamespace(_doc_vlm_genai_build_request_specs=build))
    bridge = ServerRecognition(ocr)
    items = [{"query": query} for query in ["OCR:", "Table Recognition:", "Formula Recognition:"]]
    specs = bridge.prepare(None, items, "JPEG")
    assert [options["extra_body"]["mm_processor_kwargs"]["max_pixels"]
            for _, options in specs] == [1003520, 2822400, 1003520]
    assert all(options["extra_body"]["mm_processor_kwargs"]["min_pixels"] == 112896
               for _, options in specs)
    assert all(options["extra_body"]["repetition_penalty"] == 1 for _, options in specs)


def test_regions_use_all_gpus_and_keep_order_when_responses_finish_out_of_order(monkeypatch):
    from collections import Counter
    from concurrent.futures import Future
    from document_pipeline.paddle_server import RegionClients, ServerRecognition
    monkeypatch.delenv("DOCUMENT_PIPELINE_OCR_DIAGNOSTICS", raising=False)
    class Client:
        closed = False
        def create_chat_completion(self, messages, **kwargs):
            return Future()
        def close(self):
            self.closed = True
    clients = [Client() for _ in range(4)]
    router = RegionClients(clients, 2)
    futures = [router.create_chat_completion([], return_future=True) for _ in range(16)]
    assert Counter(f.document_pipeline_gpu for f in futures) == {0: 4, 1: 4, 2: 4, 3: 4}
    for index in reversed(range(16)):
        futures[index].set_result(SimpleNamespace(
            choices=[SimpleNamespace(token_ids=[index + 10, 2], finish_reason="stop",
                                     message=SimpleNamespace(content=str(index)))],
            usage=SimpleNamespace(prompt_tokens=10)))
    ocr = SimpleNamespace(recognizer=SimpleNamespace(_doc_vlm_genai_build_request_specs=lambda *a: []),
                          cfg=SimpleNamespace(require_eos=True), _local_page=1,
                          stats={"batches": [], "gen_tokens": 0})
    assert ServerRecognition(ocr).collect(futures) == [str(i) for i in range(16)]
    assert ocr.stats["batches"][0]["recognizer_gpus"] == [2, 3, 0, 1] * 4
    router.close()
    assert all(client.closed for client in clients)


@pytest.mark.parametrize("tokens,reason,valid", [
    ([9, 2], "stop", True), ([9, 2], "length", False),
    ([9], "stop", False), (None, "stop", False),
])
def test_server_requires_raw_eos_and_records_actual_region_count(tokens, reason, valid, monkeypatch):
    from concurrent.futures import Future
    from document_pipeline.paddle_server import ServerRecognition
    monkeypatch.delenv("DOCUMENT_PIPELINE_OCR_DIAGNOSTICS", raising=False)
    ocr = SimpleNamespace(
        recognizer=SimpleNamespace(_doc_vlm_genai_build_request_specs=lambda *a: []),
        cfg=SimpleNamespace(require_eos=True), _local_page=1,
        stats={"batches": [], "gen_tokens": 0})
    bridge = ServerRecognition(ocr)
    future = Future()
    future.set_result(SimpleNamespace(
        choices=[SimpleNamespace(token_ids=tokens, finish_reason=reason,
                                 message=SimpleNamespace(content="sample"))],
        usage=SimpleNamespace(prompt_tokens=10)))
    if valid:
        assert bridge.collect([future, future]) == ["sample", "sample"]
        assert ocr.stats["gen_tokens"] == 4
    else:
        with pytest.raises(RuntimeError, match="Refusing unchecked output"):
            bridge.collect([future, future])
    assert ocr.stats["batches"][0]["regions"] == 2
    assert ocr.stats["batches"][0]["ended_with_eos"] == [valid, valid]
