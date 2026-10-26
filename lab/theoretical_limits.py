"""Reproduce the theoretical-limits note from saved results; no inference or GPU work."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import struct

from eval.evaluate import percentile
from lab.site_data import identifier

BASELINE_RUN = "paddleocr_vllm_five_pdf_eval/20261026-081822"
RUN = "paddleocr_vllm_six_request_tiles/20261026-150343"
PROFILES = (
    "20261026-143036-naca-seaplanes-PaddleOCR-VL-1.6-vLLM-Render-pages-inside-OCR-workers-c911f4",
    "20261026-104101-census-income-PaddleOCR-VL-1.6-vLLM-Concurrency-8-GPU-n-gram-speculation-564e43",
    "20261026-084914-census-income-PaddleOCR-VL-1.6-vLLM-Batch-4-Regions-shared-across-GPUs-03e0ff",
)
VARIANT = "server_8_draft4_control"


def parameter_inventory(directory):
    """Read only safetensors metadata; never materialize weight tensors."""
    tensors, header_hashes = {}, {}
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            raw = file.read(length)
        header_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        tensors.update({name: record for name, record in json.loads(raw).items()
                        if name != "__metadata__"})
    if not tensors:
        raise ValueError(f"No checkpoint headers in {directory}")
    counts = {name: math.prod(record["shape"]) for name, record in tensors.items()}
    linear = {name: count for name, count in counts.items()
              if len(tensors[name]["shape"]) == 2 and "embed_tokens" not in name}
    return counts, linear, header_hashes


def calculate(root, models):
    fingerprints = {}

    def read(path):
        raw = path.read_bytes()
        fingerprints[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    run = root / "results" / RUN
    original = root / "results" / BASELINE_RUN
    report = read(run / f"{VARIANT}.json")
    baseline = read(original / "vllm_batch_1.json")
    initial = read(original / "vllm_batch_4_all_gpus.json")
    comparison = read(run / f"{VARIANT}.compare.json")
    config = read(run / f"{VARIANT}.pipeline.json")
    preview = read(original / "embedding-preview.json")
    assert report["pass"] and baseline["pass"] and comparison["pass"]
    assert preview["dataset_sha256"] == report["dataset_sha256"]
    assert initial["assets"] == baseline["assets"] == report["assets"]
    stage_names = ("prepare PDF", "render + OCR", "chunk", "embed")
    raw = (run / f"{VARIANT}.trace.jsonl").read_bytes()
    fingerprints[str((run / f"{VARIANT}.trace.jsonl").relative_to(root))] = hashlib.sha256(raw).hexdigest()
    grouped = defaultdict(list)
    for line in raw.splitlines():
        row = json.loads(line)
        grouped[row["pdf"]].append(row)

    corpus, measured, full_times = [], [], []
    for case in report["cases"]:
        rows = grouped[f"{case['id']}.pdf"]
        assert [r["phase"] for r in case["runs"]] == ["warmup"] + ["measured"] * 5
        assert len(rows) == len(case["runs"]) == 6
        for row, call in zip(rows, case["runs"]):
            # Evaluator timing surrounds the pipeline call and is slightly larger.
            assert abs(row["total_s"] - call["latency_ms"] / 1000) < .01
        rows = rows[1:]
        measured.extend(rows)
        full_times.extend(value / 1000 for value in case["latency_ms"])
        stats = rows[0]["ocr_stats"]
        lengths = [length for worker in stats["workers"] for batch in worker.get("batches", [])
                   for length in batch["generated_lengths"]]
        assert all(row["ocr_stats"]["gen_tokens"] == sum(lengths) for row in rows)
        assert all(set(row["stages_s"]) == set(stage_names) for row in rows)
        assert all(row["embed_stats"]["devices"] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
                   for row in rows)
        corpus.append({
            "id": case["id"], "pages": rows[0]["pages"], "chunks": rows[0]["chunks"],
            "mean_call_s": mean(case["latency_ms"]) / 1000,
            "mean_stage_s": {stage: mean(row["stages_s"][stage] for row in rows)
                             for stage in stage_names},
            "ocr_output_tokens": sum(lengths), "longest_region_tokens": max(lengths),
            "regions": len(lengths),
            "embedding_tokens_per_gpu": rows[0]["embed_stats"]["tokens_per_gpu"],
        })
    assert len(measured) == 25 and sum(doc["pages"] for doc in corpus) == 20
    total = sum(full_times)
    pages = sum(row["pages"] for row in measured)
    assert math.isclose(60 * pages / total, report["summary"]["pages_per_minute_sequential"])
    stages = {stage: sum(row["stages_s"][stage] for row in measured)
              for stage in stage_names}

    scenarios = []
    for label, changed, factor in [
        ("Embedding 2x faster", ("embed",), 2),
        ("Embedding takes zero time", ("embed",), None),
        ("Render + OCR 10% faster", ("render + OCR",), 1.1),
        ("Render + OCR 2x faster", ("render + OCR",), 2),
        ("Preparation and chunking take zero time", ("prepare PDF", "chunk"), None),
    ]:
        simulated = [time - sum(row["stages_s"][stage] for stage in changed)
                     * (1 - (1 / factor if factor else 0))
                     for time, row in zip(full_times, measured)]
        scenarios.append({
            "scenario": label, "p50_s": percentile(simulated, .5),
            "p95_s": percentile(simulated, .95),
            "pages_per_minute": 60 * pages / sum(simulated),
            "throughput_speedup": total / sum(simulated),
            "p50_speedup": percentile(full_times, .5) / percentile(simulated, .5),
            "seconds_per_20_pages": sum(simulated) / 5,
        })

    ocr, ocr_linear, ocr_headers = parameter_inventory(models / "PaddleOCR-VL-1.6")
    embed, embed_linear, embed_headers = parameter_inventory(models / "Qwen3-Embedding-0.6B")
    decoder_linear = sum(count for name, count in ocr_linear.items()
                         if name.startswith(("model.", "lm_head.")))
    visual = sum(count for name, count in ocr.items() if name.startswith(("visual.", "mlp_AR.")))
    ocr_config = json.loads((models / "PaddleOCR-VL-1.6/config.json").read_text())
    kv_bytes = 2 * ocr_config["num_hidden_layers"] * ocr_config["num_key_value_heads"] * ocr_config["head_dim"] * 2
    model_numbers = {
        "ocr_total_parameters": sum(ocr.values()),
        "ocr_visual_projector_parameters_fp32": visual,
        "ocr_decode_linear_parameters": decoder_linear,
        "ocr_resident_weight_bytes": visual * 4 + (sum(ocr.values()) - visual) * 2,
        "ocr_decode_linear_weight_bytes_bf16": decoder_linear * 2,
        "ocr_kv_bytes_per_context_token": kv_bytes,
        "embedding_total_parameters": sum(embed.values()),
        "embedding_linear_parameters": sum(embed_linear.values()),
        "embedding_linear_flops_per_token": 2 * sum(embed_linear.values()),
        "embedding_resident_weight_bytes_fp32": sum(embed.values()) * 4,
        "embedding_resident_weight_bytes_bf16_per_gpu": sum(embed.values()) * 2,
        "safetensors_header_sha256": {"ocr": ocr_headers, "embedding": embed_headers},
    }
    # Tokenize saved chunk text on CPU. Count useful input tokens, excluding padding.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(models / "Qwen3-Embedding-0.6B", local_files_only=True)
    for doc in corpus:
        texts = [p["text"] for p in preview["points"] if p["document_id"] == doc["id"]]
        assert len(texts) == doc["chunks"]
        tokens = sum(len(ids) for ids in tokenizer(
            texts, truncation=True, max_length=config["embed"]["max_length"])["input_ids"])
        flops = tokens * model_numbers["embedding_linear_flops_per_token"]
        doc["embedding_input_tokens"] = tokens
        doc["embedding_useful_linear_tflop"] = flops / 1e12
        doc["embedding_fp32_linear_floor_s"] = flops / 30.3e12
        doc["embedding_bf16_linear_floor_s"] = flops / 121e12
        assert sum(doc["embedding_tokens_per_gpu"]) == tokens
        doc["embedding_bf16_four_gpu_linear_floor_s"] = flops / (4 * 121e12)
        doc["embedding_bf16_recorded_placement_floor_s"] = (
            max(doc["embedding_tokens_per_gpu"]) * model_numbers["embedding_linear_flops_per_token"] / 121e12)

    profiles = []
    for name in PROFILES:
        profile = read(root / "results/profiles" / name / "profile.json")
        devices = profile["gpu_telemetry"]["ocr"]["devices"]
        profiles.append({
            "id": identifier(f"profiles/{name}"), "path": f"profiles/{name}",
            "pdf": profile["pdf"]["name"], "summary": profile["summary"],
            "gpu_ocr_busy_means": [d["metrics"]["gpu_busy_pct"]["mean"] for d in devices],
            "gpu_ocr_samples": [d["samples"] for d in devices],
            "gpu_ocr_power_w_means": [d["metrics"]["power_w"]["mean"] for d in devices],
            "vllm_telemetry": profile.get("vllm_telemetry"),
            "capture_scope": profile["torch_capture"]["scope"],
            "maximum_engine_iterations": profile["torch_capture"]["maximum_engine_iterations"],
        })
        if "084914" not in name:
            continue
        batches = [batch for worker in profile["ocr_stats"][0]["workers"] for batch in worker.get("batches", [])]
        batch = max(batches, key=lambda b: max(b["generated_lengths"]))
        index = batch["generated_lengths"].index(max(batch["generated_lengths"]))
        output_tokens, input_tokens = batch["generated_lengths"][index], batch["input_tokens"][index]
        # Prefill emits the first output token; then G-1 serial decode steps.
        steps = output_tokens - 1
        weight_seconds = steps * model_numbers["ocr_decode_linear_weight_bytes_bf16"] / 300e9
        kv_seconds = kv_bytes * (steps * input_tokens + steps * (steps - 1) / 2) / 300e9
        decode = {
            "scope": "Historical non-speculative model; not a lower bound for current speculative decoding.",
            "profile_id": identifier(f"profiles/{name}"),
            "output_tokens": output_tokens, "input_tokens": input_tokens, "serial_decode_steps": steps,
            "observed_region_submission_batch_s": batch["elapsed_s"],
            "effective_output_tokens_per_batch_second": output_tokens / batch["elapsed_s"],
            "ideal_weight_stream_seconds": weight_seconds,
            "ideal_weight_plus_kv_stream_seconds": weight_seconds + kv_seconds,
            "illustrative_speculative_tokens_per_pass": 1.6,
            "illustrative_speculative_weight_stream_seconds": weight_seconds / 1.6,
            "weight_only_floor_fraction_of_batch_s": weight_seconds / batch["elapsed_s"],
            "weight_plus_kv_floor_fraction_of_batch_s": (weight_seconds + kv_seconds) / batch["elapsed_s"],
            "assumptions": [
                "Standard BF16 decoding, one token per sequence per iteration.",
                "Each iteration streams linear weights from DRAM once; persistent cache reuse excluded.",
                "KV estimate reads each prior K/V once and excludes KV writes.",
                "300 GB/s ideal per-GPU memory bandwidth; no quantization or speculative decoding.",
                "Excludes prefill, vision, layout, transfers, launch gaps and other work.",
                "Submission-batch wall time is not a separately measured pure-decode duration.",
            ],
        }
    census_batches = []
    for row in grouped["census-income.pdf"][1:]:
        batches = [batch for worker in row["ocr_stats"]["workers"] for batch in worker.get("batches", [])]
        batch = max(batches, key=lambda b: max(b["generated_lengths"]))
        index = batch["generated_lengths"].index(max(batch["generated_lengths"]))
        assert batch["generated_lengths"][index] == decode["output_tokens"] == 3252
        census_batches.append({
            "elapsed_s": batch["elapsed_s"], "region_count": len(batch["generated_lengths"]),
            "longest_region_tokens": batch["generated_lengths"][index],
            "recognizer_gpu": batch["recognizer_gpus"][index],
        })
    progression = []
    for label, relative, variant in [
        ("Batch 1 baseline", BASELINE_RUN, "vllm_batch_1"),
        ("Shared regions, original note", BASELINE_RUN, "vllm_batch_4_all_gpus"),
        ("BF16 embeddings", "paddleocr_vllm_embedding_precision/20261026-085622", "embed_bf16_b16"),
        ("Compiled embeddings", "paddleocr_vllm_compiled_embeddings/20261026-101652", "server_8_compiled_embeddings"),
        ("Four-draft speculation", "paddleocr_vllm_ngram_gpu_4/20261026-103409", "server_8_ngram_gpu_4"),
        ("Fixed 32-row tiles", "paddleocr_vllm_fixed_decoder_tiles/20261026-124322", "decoder_tiles_fixed32"),
        ("Four-GPU embeddings", "paddleocr_vllm_parallel_embeddings/20261026-134054", "embedding_four_gpus"),
        ("Retained configuration", RUN, VARIANT),
    ]:
        entry = read(root / "results" / relative / f"{variant}.json")
        assert entry["pass"] and entry["assets"] == baseline["assets"]
        progression.append({"label": label, "run_id": identifier(relative), "variant": variant,
                            "summary": entry["summary"]})
    sensitivity = []
    for label, control_run, control_variant, candidate_run, candidate_variant in [
        ("Decoder columns", "paddleocr_vllm_decoder_columns/20261026-125752", "decoder_columns_control",
         "paddleocr_vllm_decoder_columns/20261026-130349", "decoder_columns_shaped"),
        ("Three drafts", "paddleocr_vllm_three_draft_tiles/20261026-144303", "draft_4_control",
         "paddleocr_vllm_three_draft_tiles/20261026-145126", "draft_3_tiles32"),
        ("Six requests", RUN, VARIANT,
         "paddleocr_vllm_six_request_tiles/20261026-151300", "server_6_draft4_tiles32"),
    ]:
        control = read(root / "results" / control_run / f"{control_variant}.json")
        candidate = read(root / "results" / candidate_run / f"{candidate_variant}.json")
        assert control["pass"] and candidate["pass"]
        assert candidate["assets"] == control["assets"] == baseline["assets"]
        a, b = control["summary"], candidate["summary"]
        sensitivity.append({
            "trial": label, "control_run_id": identifier(control_run), "candidate_run_id": identifier(candidate_run),
            "p50_speedup_gain_pct": 100 * (a["p50_ms"] / b["p50_ms"] - 1),
            "p95_speedup_gain_pct": 100 * (a["p95_ms"] / b["p95_ms"] - 1),
            "throughput_gain_pct": 100 * (b["pages_per_minute_sequential"] / a["pages_per_minute_sequential"] - 1),
        })
    return {
        "schema_version": 2, "as_of": "2026-10-26", "report_created_at": report["created_at"],
        "run_id": identifier(RUN),
        "run_path": RUN, "variant": VARIANT, "dataset_sha256": report["dataset_sha256"],
        "method": "Exclude warmups; use 25 evaluator latencies and paired stage traces. No new inference.",
        "hardware_peaks_per_l4": {"memory_bytes_s": 300e9, "fp32_flops_s": 30.3e12,
                                "dense_bf16_flops_s": 121e12, "power_limit_w": 72},
        "eval": report["summary"], "baseline_eval": baseline["summary"], "initial_eval": initial["summary"],
        "configuration": config, "official_comparison": comparison,
        "progression": progression, "latest_trials": sensitivity,
        "corpus": corpus, "seconds_per_20_pages": total / 5,
        "mean_stage_sum_s": {key: value / 5 for key, value in stages.items()},
        "stage_fraction": {key: value / total for key, value in stages.items()},
        "what_if": scenarios, "models": model_numbers, "historical_census_decode_model": decode,
        "current_census_long_region_batches": census_batches,
        "profiles": profiles, "input_sha256": fingerprints,
        "references": [
            "https://www.nvidia.com/en-us/data-center/l4/",
            "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html",
            "https://docs.nvidia.com/deploy/nvidia-smi/index.html",
            "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/main/config.json",
            "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config.json",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path.home() / "huggingface")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                      TOKENIZERS_PARALLELISM="false")
    root = Path(__file__).resolve().parents[1]
    result = calculate(root, args.models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
