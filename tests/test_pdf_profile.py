"""A bounded trace must keep processing the PDF after recording stops."""
import json
from types import SimpleNamespace

import torch

from lab.pdf_prof import torch_profile


def test_bounded_trace_keeps_full_workload_and_removes_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self.calls = 0
            self.generation_config = SimpleNamespace(eos_token_id=99)
            self.model = SimpleNamespace(visual=torch.nn.Linear(4, 4))

        def forward(self, input_ids):
            self.calls += 1
            return self.linear(torch.ones(1, 4))

        def generate(self, input_ids, max_new_tokens, finish_eos):
            self.model.visual(torch.ones(1, 4))  # image features can precede the prefill forward
            count = max_new_tokens - 1 if finish_eos else max_new_tokens
            for index in range(count):
                self(input_ids=input_ids if index == 0 else torch.ones(1, 1, dtype=torch.long))
            generated = torch.full((1, count), 10)
            if finish_eos:
                generated[0, -1] = 99
            return torch.cat([input_ids, generated], dim=-1)

    model = Model()

    class Pipeline:
        ocr = SimpleNamespace(model=model)
        _record_function = None
        outputs = []

        def __call__(self, pdf):
            for finish_eos in [True, False]:
                self.outputs.append(model.generate(input_ids=torch.ones(1, 8, dtype=torch.long),
                                                   max_new_tokens=5, finish_eos=finish_eos))

    pipe = Pipeline()
    capture = torch_profile(pipe, "test.pdf", tmp_path, max_ocr_forwards=3)
    assert model.calls == capture["observed_ocr_forwards"] == 9
    assert capture["captured_phases"] == {"prefill": 1, "decode": 2}
    operators = {row["name"]: row for row in json.loads((tmp_path / "operators.json").read_text())}
    assert operators["aten::linear"]["calls"] == 4
    assert operators["ocr::vision"]["calls"] == 1
    assert capture["captured_components"] == {"vision": 1}
    assert operators["ocr::prefill"]["calls"] == 1
    assert operators["ocr::decode"]["calls"] == 2
    assert json.loads((tmp_path / "trace.json").read_text())["traceEvents"]
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert model.generate.__func__ is Model.generate
    assert [out.shape[-1] for out in pipe.outputs] == [12, 13]
    first, second = capture["generation_batches"]
    assert first["generated_sequence_length"] == 4 and first["ended_with_eos"] == [True]
    assert not first["reached_length_limit"]
    assert second["generated_sequence_length"] == 5 and second["ended_with_eos"] == [False]
    assert second["reached_length_limit"]


def test_kernel_attribution_uses_launch_not_execution_time(tmp_path):
    from scripts.summarize_bounded_trace import summarize
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"cat": "user_annotation", "ph": "X", "name": "ocr::decode",
         "pid": 10, "tid": 10, "ts": 100, "dur": 10},
        {"cat": "cuda_runtime", "pid": 10, "tid": 10, "ts": 105, "args": {"correlation": 1}},
        {"cat": "cuda_runtime", "pid": 10, "tid": 10, "ts": 108, "args": {"correlation": 2}},
        {"cat": "kernel", "ph": "X", "name": "gemv", "ts": 120, "dur": 6,
         "args": {"correlation": 1}},
        {"cat": "kernel", "ph": "X", "name": "other", "ts": 124, "dur": 5,
         "args": {"correlation": 2}},
    ]}))
    result = summarize(trace)
    phase = result["phases"]["ocr::decode"]
    assert phase["sum_kernel_us"] == 11 and phase["gpu_busy_union_us"] == 9
    assert phase["gemv_sum_us"] == 6 and phase["cpu_range_us"] == 10
    assert result["gpu_busy_fraction_of_span"] == 1
def test_paddle_summary_separates_cpu_launches_and_groups_timed_names():
    from lab.pdf_prof import paddle_operator_summary
    events = [
        {"ph": "X", "cat": "Kernel", "name": "gemv[4.000 us]", "dur": 4},
        {"ph": "X", "cat": "Kernel", "name": "gemv[6.000 us]", "dur": 6},
        {"ph": "X", "cat": "DygraphKernelLaunch", "name": "gemv kernel launch[20.000 us]", "dur": 20},
        {"ph": "X", "cat": "Memcpy", "name": "HtoD[1.000 us]", "dur": 1},
    ]
    rows, categories = paddle_operator_summary(events)
    kernel = next(row for row in rows if row["name"] == "GPU kernel / gemv")
    launch = next(row for row in rows if row["name"].startswith("DygraphKernelLaunch"))
    assert kernel["calls"] == 2 and kernel["device_total_us"] == 10
    assert launch["device_total_us"] == 0 and launch["cpu_total_us"] == 20
    assert launch["self_cpu_us"] is None
    assert categories["Kernel"] == 2
