"""Remote trace capture must finish the full PDF and preserve correctness evidence."""
import gzip
import json
from types import SimpleNamespace

import pytest

from lab.vllm_profile import VllmTelemetry, _sampled_metrics, profile


def test_telemetry_keeps_speculation_and_does_not_double_count_queue_labels(tmp_path):
    text = """# HELP vllm:num_requests_waiting Waiting requests.
vllm:num_requests_running{engine="0"} 8
vllm:num_requests_waiting{engine="0"} 4
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 4
vllm:spec_decode_num_drafts_total{engine="0"} 100
vllm:spec_decode_num_draft_tokens_total{engine="0"} 400
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 120
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 90
vllm:unrelated_metric{engine="0"} 7
"""
    first = _sampled_metrics(text)
    last = dict(first)
    # Set explicit cumulative values so the ratio reflects this capture, not server lifetime.
    last['vllm:spec_decode_num_drafts_total{engine="0"}'] = 120
    last['vllm:spec_decode_num_draft_tokens_total{engine="0"}'] = 480
    last['vllm:spec_decode_num_accepted_tokens_total{engine="0"}'] = 160
    assert not any("unrelated_metric" in key for key in first)
    assert any("per_pos_total" in key for key in first)
    telemetry = VllmTelemetry(tmp_path, SimpleNamespace(gpu_workers=1))
    telemetry.rows = [{"worker": 0, "metrics": first}, {"worker": 0, "metrics": last}]
    worker = telemetry.summary()["workers"][0]
    assert worker["maximum_waiting_requests"] == 4
    assert worker["maximum_running_requests"] == 8
    assert worker["speculation"]["drafts"] == 20
    assert worker["speculation"]["draft_acceptance_rate"] == .5
    assert worker["speculation"]["mean_acceptance_length"] == 3
    # A server counter reset must not be presented as an acceptance rate.
    telemetry.rows.reverse()
    assert "speculation" not in telemetry.summary()["workers"][0]


@pytest.mark.parametrize("reference_dir", ["refs", "refs-vllm"])
def test_remote_profile_collects_trace_and_checks_full_output(tmp_path, monkeypatch, reference_dir):
    servers = tmp_path / "servers"
    trace_dir = servers / "worker-0-traces"
    trace_dir.mkdir(parents=True)
    monkeypatch.setenv("DOCUMENT_PIPELINE_VLLM_SERVER_ARTIFACTS", str(servers))
    result = {"ocr_text": "complete document", "chunks": [
        {"id": "chunk-000", "embedding": [1.0, 0.0]}]}
    refs = tmp_path / "data" / reference_dir
    refs.mkdir(parents=True)
    (refs / "example.json").write_text(json.dumps(result))
    directory = tmp_path / "profile"
    directory.mkdir()
    calls = []

    def request(port, route, body):
        calls.append((port, route, body))
        if route == "start_profile":
            with gzip.open(trace_dir / f"{body['profile_prefix']}.trace.json.gz", "wt") as stream:
                json.dump({"traceEvents": [
                    {"ph": "X", "cat": "kernel", "name": "gemm", "dur": 4},
                    {"ph": "X", "cat": "kernel", "name": "gemm", "dur": 6},
                    {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 20},
                ]}, stream)
        return b""

    monkeypatch.setattr("lab.vllm_profile.server_request", request)

    class Pipe:
        cfg = SimpleNamespace(ocr=SimpleNamespace(paddle_server_base_port=8800, gpu_workers=4))
        ocr = SimpleNamespace(stats={"active_workers": [0]})
        complete_calls = 0

        def __call__(self, pdf):
            self.complete_calls += 1
            return result

    pipe = Pipe()
    capture = profile(pipe, tmp_path / "data/pdfs/example.pdf", directory, 5,
                      reference=refs / "example.json" if reference_dir != "refs" else None)
    assert pipe.complete_calls == 1
    assert capture["instrumented_reference_check"]["pass"]
    assert capture["workers"][0]["kernel_count"] == 2
    assert [(port, route) for port, route, _ in calls] == [
        (8800, "start_profile"), (8800, "stop_profile")]
    assert calls[0][2]["max_iterations"] == 5
    operators = json.loads((directory / "operators.json").read_text())
    assert operators[0]["device_total_us"] == 10
    assert operators[0]["calls"] == 2
    assert operators[0]["self_cpu_us"] is None
