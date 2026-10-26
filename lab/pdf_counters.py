"""Bounded Nsight Compute diagnostic: first decode forward after a short prefix warm-up.

Run through ncu with --profile-from-start off, --devices 0, --clock-control none,
--cache-control all. Keep kernel filtering and launch counts bounded. Timings
from this instrumented call are not evaluator measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def install_counter_hooks():
    """Executed in spawned GPU processes too; first prefix call is unprofiled."""
    from document_pipeline.ocr import HfOcr

    original = HfOcr.ocr

    def observed(self, images):
        import torch
        self._counter_calls = getattr(self, "_counter_calls", 0) + 1
        if self.device != "cuda:0" or self._counter_calls != 2:
            return original(self, images)
        state = {"active": False, "done": False}

        def before(module, args, kwargs):
            tokens = kwargs.get("input_ids")
            if not state["done"] and tokens is not None and tokens.shape[-1] == 1:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("counter_sample::first_decode")
                torch.cuda.profiler.start()
                state["active"] = True

        def after(module, args, output):
            if state["active"]:
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                torch.cuda.nvtx.range_pop()
                state.update(active=False, done=True)

        handles = [self.model.register_forward_pre_hook(before, with_kwargs=True),
                   self.model.register_forward_hook(after)]
        try:
            output = original(self, images)
            if not state["done"]:
                raise RuntimeError("No decode forward was observed for the counter sample.")
            return output
        finally:
            for handle in handles:
                handle.remove()
            if state["active"]:
                torch.cuda.profiler.stop()
                torch.cuda.nvtx.range_pop()

    HfOcr.ocr = observed


# multiprocessing imports this module as __mp_main__ before invoking the page
# worker target, so the diagnostic hooks are installed in each model process.
if os.environ.get("DOCUMENT_PIPELINE_COUNTER_SAMPLE") == "1":
    install_counter_hooks()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=4)
    args = parser.parse_args()
    if not 2 <= args.prefix_tokens <= 32:
        parser.error("Use 2–32 prefix tokens to keep the counter diagnostic bounded.")
    if os.environ.get("DOCUMENT_PIPELINE_COUNTER_SAMPLE") != "1":
        parser.error("Set DOCUMENT_PIPELINE_COUNTER_SAMPLE=1 under ncu.")
    import yaml
    from document_pipeline.config import load_config
    from document_pipeline.pipeline import Pipeline
    from eval.evaluate import check, load_cases

    directory = (ROOT / "results" / args.baseline_run).resolve()
    assert directory.is_relative_to(ROOT / "results")
    report = json.loads((directory / "baseline.json").read_text())
    assert report["pass"], "A passing baseline is required"
    quality = json.loads((directory / "quality-review.json").read_text())
    assert quality["status"] == "complete", "A reviewed complete baseline is required"
    for path, digest in report["environment"]["source_files_sha256"].items():
        if path.startswith("document_pipeline/"):
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    spec = yaml.safe_load((directory / "experiment.yaml").read_text())
    baseline_configuration = spec["_expanded"]["baseline"]["pipeline"]
    configuration = copy.deepcopy(baseline_configuration)
    # sudo has a different home directory. Resolve the frozen model paths for
    # the workspace owner explicitly rather than loading another model/cache.
    for section, key in (("ocr", "model"), ("chunk", "tokenizer"), ("embed", "model")):
        value = configuration[section][key]
        if value.startswith("~/"):
            configuration[section][key] = str(ROOT.parent / value[2:])
    os.environ["CUDA_VISIBLE_DEVICES"] = report["environment"]["gpu_uuids"]
    cases = load_cases(ROOT / spec["manifest"], ROOT / spec["root"])
    case = next(case for case in cases if case["id"] == "naca-seaplanes")
    cfg = load_config(configuration)
    assert cfg.ocr.gpu_workers == 4 and cfg.ocr.page_batch_size == 1
    # Only the counter diagnostic processes a short prefix. These outputs never
    # become references, baseline timings, or a claim of full-page correctness.
    cfg.ocr.max_new_tokens = args.prefix_tokens
    cfg.ocr.require_eos = False
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(cfg)
    pipe.trace_path = None
    try:
        warm_prefix = pipe(str(case["pdf_path"]))
        started = time.perf_counter()
        output = pipe(str(case["pdf_path"]))
        elapsed = time.perf_counter() - started
        checked = check(output, warm_prefix, .99)
    finally:
        pipe.ocr.close()
    record = {
        "baseline_run": args.baseline_run, "pdf_sha256": case["pdf_sha256"],
        "reference_sha256": case["reference_sha256"],
        "baseline_config": baseline_configuration, "diagnostic_config": cfg.to_dict(),
        "gpu_uuids": report["environment"]["gpu_uuids"],
        "scope": "GPU 0, selected kernels in first decode forward on page 1, after one short-prefix warm-up",
        "prefix_tokens_per_page": args.prefix_tokens,
        "warmup_calls": 1, "instrumented_calls": 1, "diagnostic_elapsed_s": elapsed,
        "prefix_repeat_check": checked, "ocr_stats": pipe.ocr.stats,
        "notes": ["All four page-worker replicas remain loaded; all three full-page images are encoded.",
                  "Generation stops at the diagnostic prefix limit; complete-page references are unchanged.",
                  "Kernel replay alters diagnostic execution; this is not evaluator latency.",
                  "Clock control is disabled; cache-control all uses cold replay caches and can differ from natural decoding.",
                  "A few early kernels cannot establish saturation across every page or decode position."]}
    (args.out / "counter-capture.json").write_text(json.dumps(record, indent=2) + "\n")
    assert checked["pass"], checked
    print(json.dumps({"status": "complete", "prefix_repeat_check": checked,
                      "diagnostic_elapsed_s": elapsed}), flush=True)


if __name__ == "__main__":
    main()
