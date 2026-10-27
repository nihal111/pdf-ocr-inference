"""Model the frozen corpus at ideal L4 compute/bandwidth rates, without inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

from lab.theoretical_limits import parameter_inventory

RUN = "results/revalidation/20261027-020444"
GPU_COUNT = 4
SLOTS = 8
FP32 = 30.3e12
BF16 = 121e12  # Dense; the advertised 242 TFLOP/s includes structured sparsity.
BANDWIDTH = 300e9


@dataclass(frozen=True)
class Decoder:
    linear_parameters: int
    layers: int
    query_width: int
    kv_bytes_per_token: int
    bandwidth: float = BANDWIDTH
    compute: float = BF16

    def step(self, active, verification_width):
        """One weight stream shared by active sequences; one KV read per sequence."""
        traffic = 2 * self.linear_parameters
        flops = 0
        for context, remaining in active:
            q = min(verification_width, remaining)
            traffic += self.kv_bytes_per_token * (context + q)
            flops += 2 * self.linear_parameters * q
            flops += 4 * self.layers * self.query_width * (
                q * context + q * (q + 1) // 2)
        return max(traffic / self.bandwidth, flops / self.compute), flops, traffic

    def schedule(self, regions, advance, verification_width, gpus=GPU_COUNT, slots=SLOTS):
        """An explicit ideal schedule, not an optimal-scheduling proof.

        Place longest outputs first in the least-loaded independent slot. Ties
        spread across GPUs. Within a GPU, slots share each decoder weight stream.
        Prefill has already emitted the first token, so only G-1 advances remain.
        """
        if not 1 <= advance <= verification_width or gpus < 1 or slots < 1:
            raise ValueError("Invalid decoder schedule")
        queues = [deque() for _ in range(gpus * slots)]
        loads = [0] * len(queues)
        for prompt, generated in sorted(regions, key=lambda r: (-r[1], r[0])):
            if prompt < 1 or generated < 1:
                raise ValueError("Prompt and generated lengths must be positive")
            remaining = generated - 1
            if not remaining:
                continue
            slot = min(range(len(queues)), key=loads.__getitem__)
            queues[slot].append((prompt, remaining))
            loads[slot] += math.ceil(remaining / advance)
        devices = []
        for gpu in range(gpus):
            local = [queues[gpu + gpus * slot] for slot in range(slots)]
            active = [q.popleft() if q else None for q in local]
            seconds, flops, traffic, passes = 0.0, 0, 0, 0
            while any(active):
                duration, work, data = self.step(
                    [entry for entry in active if entry], verification_width)
                seconds += duration
                flops += work
                traffic += data
                passes += 1
                for i, entry in enumerate(active):
                    if entry is None:
                        continue
                    context, remaining = entry
                    n = min(advance, remaining)
                    active[i] = ((context + n, remaining - n) if remaining > n
                                 else local[i].popleft() if local[i] else None)
            devices.append({"seconds": seconds, "flops": flops,
                            "bytes": traffic, "passes": passes})
        return {"seconds": max(d["seconds"] for d in devices),
                "flops": sum(d["flops"] for d in devices), "devices": devices}


def layout_inventory(graph):
    """Count static conv/matmul work at batch one; fail on unresolved dimensions."""
    ops = graph["program"]["regions"][0]["blocks"][0]["ops"]
    shapes, stored_floats = {}, 0
    for op in ops:
        outputs = op.get("O", [])
        outputs = [outputs] if isinstance(outputs, dict) else outputs
        for output in outputs:
            typ = output.get("TT", {}).get("D", [])
            if len(typ) > 1 and isinstance(typ[1], list):
                shapes[output["%"]] = typ[1]
                if op["#"] == "p" and typ[0]["#"] == "0.t_f32":
                    stored_floats += math.prod(typ[1])
    flops, counts = Counter(), Counter()
    for op in ops:
        name = op["#"]
        if name not in {"1.conv2d", "1.depthwise_conv2d", "1.matmul", "1.bmm"}:
            continue
        attributes = {a["N"]: a["AT"].get("D") for a in op.get("A", [])}
        x = shapes[op["I"][0]["%"]]
        weight = shapes[op["I"][1]["%"]]
        output = list(shapes[op["O"][0]["%"]])
        if output[0] == -1:
            output[0] = 1
        if name in {"1.conv2d", "1.depthwise_conv2d"}:
            # Filter shape already contains input_channels/groups.
            dimensions = output + weight[1:]
        else:
            reduction = x[-2] if attributes.get("transpose_x") else x[-1]
            dimensions = output + [reduction]
        if not all(n > 0 for n in dimensions):
            raise ValueError(f"Unresolved layout dimensions: {name}, {dimensions}")
        flops[name] += 2 * math.prod(dimensions)
        counts[name] += 1
    return {"stored_fp32_values_including_constants": stored_floats,
            "flops_per_page": sum(flops.values()), "flops_by_operator": dict(flops),
            "operator_counts": dict(counts), "input_shape": [1, 3, 800, 800],
            "excluded": "Normalization, activations, interpolation, sorting and other non-matrix ops."}


def region_lengths(row):
    regions = []
    for worker in row["ocr_stats"]["workers"]:
        for batch in worker.get("batches", []):
            assert all(batch["ended_with_eos"])
            regions.extend(zip(batch["input_tokens"], batch["generated_lengths"], strict=True))
    assert sum(g for _, g in regions) == row["ocr_stats"]["gen_tokens"]
    return sorted(regions)


def calculate(root, models):
    from transformers import AutoTokenizer
    from document_pipeline.chunker import MarkdownChunker
    from document_pipeline.config import ChunkConfig

    fingerprints = {}

    def raw(path, label=None):
        data = path.read_bytes()
        fingerprints[label or str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return data

    def read(path, label=None):
        return json.loads(raw(path, label))

    baseline = read(root / "reports/baseline.json")
    report = read(root / "reports/optimized.json")
    environment = read(root / "reports/optimized-env.json")
    provenance = read(root / "reports/provenance.json")
    assert baseline["pass"] and report["pass"]
    assert baseline["assets"] == report["assets"]
    assert environment["gpu_count"] == GPU_COUNT
    manifest = read(root / "data/public-v3/manifest.json")
    for source in ("lab/model_limits.py", "document_pipeline/chunker.py",
                   "document_pipeline/embedders.py", "document_pipeline/config.py",
                   "document_pipeline/paddle_server.py", "pipelines/optimized.py",
                   "engine/paddle_vllm_plugin/paddle_native_precision/model.py"):
        raw(root / source)

    ocr_dir, embed_dir = models / "PaddleOCR-VL-1.6", models / "Qwen3-Embedding-0.6B"
    for directory in (ocr_dir, embed_dir):
        for path in sorted(directory.glob("*token*")):
            if path.is_file():
                raw(path, f"models/{directory.name}/{path.name}")
    oc = read(ocr_dir / "config.json", "models/PaddleOCR-VL-1.6/config.json")
    ec = read(embed_dir / "config.json", "models/Qwen3-Embedding-0.6B/config.json")
    ocr, _, ocr_hashes = parameter_inventory(ocr_dir)
    emb, _, emb_hashes = parameter_inventory(embed_dir)
    unused = sum(n for k, n in ocr.items() if k.startswith("visual.vision_model.head.")
                 or "packing_position_embedding" in k)
    visual = sum(n for k, n in ocr.items() if k.startswith("visual.")) - unused
    projector = sum(n for k, n in ocr.items() if k.startswith("mlp_AR."))
    visual_linear = sum(n for k, n in ocr.items()
                        if k.startswith("visual.vision_model.encoder.") and k.endswith(".weight")
                        and any(part in k for part in ("self_attn", "mlp")))
    projector_linear = sum(n for k, n in ocr.items()
                           if k.startswith("mlp_AR.linear_") and k.endswith(".weight"))
    decoder_linear = sum(n for k, n in ocr.items()
                         if k.startswith("model.layers.") and k.endswith(".weight")
                         and any(part in k for part in ("self_attn", "mlp")))
    lm_head = ocr["lm_head.weight"]
    embedding_linear = sum(n for k, n in emb.items()
                           if k.startswith("layers.") and k.endswith("_proj.weight"))
    vc = oc["vision_config"]
    patch_linear = vc["hidden_size"] * vc["num_channels"] * vc["patch_size"] ** 2
    query_width = oc["num_attention_heads"] * oc["head_dim"]
    decoder = Decoder(decoder_linear + lm_head, oc["num_hidden_layers"], query_width,
                      2 * oc["num_hidden_layers"] * oc["num_key_value_heads"] * oc["head_dim"] * 2)
    # The saved implementation excludes the pooler/packing embedding at load time
    # and disables TF32. Hash it as evidence; importing it would initialize vLLM.
    runtime_root = root / ".venv-vllm/lib/python3.12/site-packages/vllm"
    for relative in ("model_executor/models/paddleocr_vl.py",
                     "model_executor/determinism/batch_invariant.py"):
        raw(runtime_root / relative, f"runtime/vllm/{relative}")
    layout = layout_inventory(read(models / "PP-DocLayoutV3/inference.json",
                                   "models/PP-DocLayoutV3/inference.json"))

    ot = AutoTokenizer.from_pretrained(ocr_dir, local_files_only=True)
    et = AutoTokenizer.from_pretrained(embed_dir, local_files_only=True)
    chunker = MarkdownChunker(ChunkConfig(tokenizer=str(embed_dir), max_tokens=512))
    prompt_overheads = {}
    for prompt in ("OCR:", "Table Recognition:", "Chart Recognition:",
                   "Formula Recognition:", "Spotting:"):
        messages = [{"role": "user", "content": [{"type": "image"},
                     {"type": "text", "text": prompt}]}]
        ids = ot.apply_chat_template(messages, add_generation_prompt=True,
                                     tokenize=True)["input_ids"]
        prompt_overheads[prompt] = len(ids) - ids.count(oc["image_token_id"])
    assert set(prompt_overheads.values()) == {13}
    overhead = next(iter(prompt_overheads.values()))

    def vision_flops(prompt):
        image_tokens = prompt - overhead
        assert image_tokens > 0
        patches = image_tokens * vc["spatial_merge_size"] ** 2
        return (2 * (visual_linear + patch_linear) * patches
                + 2 * projector_linear * image_tokens
                + 4 * vc["num_hidden_layers"] * vc["hidden_size"] * patches ** 2)

    def prefill_flops(prompt):
        return (2 * decoder_linear * prompt + 2 * lm_head
                + 2 * decoder.layers * query_width * prompt * (prompt + 1))

    traces = {}
    for variant in ("baseline", "optimized"):
        grouped = defaultdict(list)
        for line in raw(root / RUN / f"{variant}-stages.jsonl").splitlines():
            row = json.loads(line)
            grouped[row["pdf"]].append(row)
        traces[variant] = grouped
    docs, scenarios = [], []
    for case, result, asset in zip(manifest["cases"], report["cases"], report["assets"], strict=True):
        assert case["id"] == result["id"] == asset["id"]
        reference_path = root / "data/public-v3" / case["reference"]
        ref = read(reference_path)
        assert fingerprints[str(reference_path.relative_to(root))] == asset["reference_sha256"]
        pdf = root / "data/public-v3" / case["pdf"]
        raw(pdf)
        assert fingerprints[str(pdf.relative_to(root))] == asset["pdf_sha256"]
        rows = traces["optimized"][f"{case['id']}.pdf"]
        assert len(rows) == 6
        assert [r["phase"] for r in result["runs"]] == ["warmup"] + ["measured"] * 5
        stage_names = ("prepare PDF", "render + OCR", "chunk", "embed")
        for row, call in zip(rows, result["runs"], strict=True):
            assert set(row["stages_s"]) == set(stage_names)
            assert abs(row["total_s"] - call["latency_ms"] / 1000) < .01
        mean_stages = {stage: mean(row["stages_s"][stage] for row in rows[1:])
                       for stage in stage_names}
        mean_call = mean(result["latency_ms"]) / 1000
        other_seconds = mean_call - sum(mean_stages.values())
        assert other_seconds >= 0
        regions = region_lengths(rows[1])
        for variant in traces:
            repeats = traces[variant][f"{case['id']}.pdf"]
            assert len(repeats) == 6
            assert all(region_lengths(r) == regions for r in repeats)
        chunks = chunker.chunk(ref["ocr_text"])
        lengths = [len(ids) for ids in et(chunks, truncation=True, max_length=8192)["input_ids"]]
        assert len(chunks) == rows[1]["chunks"] == len(ref["chunks"])
        assert all(sum(r["embed_stats"]["tokens_per_gpu"]) == sum(lengths) for r in rows)
        flops = {
            "layout": case["pages"] * layout["flops_per_page"],
            "vision": sum(vision_flops(p) for p, _ in regions),
            "prefill": sum(prefill_flops(p) for p, _ in regions),
            "embedding": (2 * embedding_linear * sum(lengths)
                          + 2 * ec["num_hidden_layers"] * ec["num_attention_heads"] * ec["head_dim"]
                          * sum(n * (n + 1) for n in lengths)),
        }
        greedy = decoder.schedule(regions, 1, 1)
        flops["greedy_decode"] = greedy["flops"]
        longest_batches = []
        for row in rows[1:]:
            batches = [batch for worker in row["ocr_stats"]["workers"]
                       for batch in worker.get("batches", [])]
            batch = max(batches, key=lambda b: max(b["generated_lengths"]))
            longest_batches.append({
                "elapsed_seconds": batch["elapsed_s"],
                "regions": len(batch["generated_lengths"]),
                "longest_generated_tokens": max(batch["generated_lengths"]),
            })
        docs.append({
            "id": case["id"], "pages": case["pages"], "regions": len(regions),
            "ocr_generated_tokens_including_eos": sum(g for _, g in regions),
            "decode_advances_after_prefill": sum(g - 1 for _, g in regions),
            "longest_region_tokens": max(g for _, g in regions),
            "ocr_prompt_tokens": sum(p for p, _ in regions),
            "visual_tokens": sum(p - overhead for p, _ in regions),
            "vision_patches": sum(p - overhead for p, _ in regions) * vc["spatial_merge_size"] ** 2,
            "markdown_tokens_ocr_tokenizer": len(ot(ref["ocr_text"], add_special_tokens=False)["input_ids"]),
            "markdown_tokens_embedding_tokenizer": len(et(ref["ocr_text"], add_special_tokens=False)["input_ids"]),
            "embedding_tokens": sum(lengths), "chunks": len(chunks), "embedding_chunk_lengths": lengths,
            "region_prompt_and_generated_lengths": regions, "flops": flops,
            "measured_mean_call_seconds": mean_call,
            "measured_mean_stage_seconds": mean_stages,
            "measured_other_seconds": other_seconds,
            "measured_longest_region_submission_batches": longest_batches,
        })

    all_flops = {key: sum(doc["flops"][key] for doc in docs) for key in docs[0]["flops"]}
    for label, advance, width in (("greedy", 1, 1), ("two_tokens_per_verification", 2, 5),
                                  ("perfect_four_draft_speculation", 5, 5)):
        cases = []
        for doc in docs:
            regions, f = doc["region_prompt_and_generated_lengths"], doc["flops"]
            schedule = decoder.schedule(regions, advance, width)
            layout_one = layout["flops_per_page"] / FP32
            critical = max(layout_one + vision_flops(p) / FP32 + prefill_flops(p) / BF16
                           + decoder.schedule([(p, g)], advance, width)["seconds"]
                           for p, g in regions)
            # Relax scheduling and overlap FP32/BF16 resources completely. The
            # largest resource/individual-chain requirement is an optimistic minimum,
            # conditional on our weight-stream and KV-traffic assumptions.
            fp32_floor = (f["layout"] + f["vision"]) / (GPU_COUNT * FP32)
            bf16_floor = (f["prefill"] + schedule["flops"]) / (GPU_COUNT * BF16)
            embedding = f["embedding"] / (GPU_COUNT * BF16)
            optimistic_minimum = max(fp32_floor, bf16_floor, critical) + embedding
            # A different estimate: ideal stage service times added, with page
            # waves for layout and the explicit replicated decoder schedule.
            model_stages = {
                "layout": math.ceil(doc["pages"] / GPU_COUNT) * layout_one,
                "vision": f["vision"] / (GPU_COUNT * FP32),
                "prefill": f["prefill"] / (GPU_COUNT * BF16),
                "decode": schedule["seconds"],
                "embed": embedding,
            }
            phase = sum(model_stages.values())
            phase_stages = {
                "prepare PDF": 0, "render + OCR": phase - embedding,
                "chunk": 0, "embed": embedding, "other": 0,
            }
            minimum_stages = {
                **phase_stages, "render + OCR": optimistic_minimum - embedding,
            }
            cases.append({"id": doc["id"], "optimistic_minimum_seconds": optimistic_minimum,
                          "phase_estimate_seconds": phase, "critical_ocr_chain_seconds": critical,
                          "fp32_resource_seconds": fp32_floor, "bf16_resource_seconds": bf16_floor,
                          "embedding_seconds": embedding, "decode_schedule": schedule,
                          "model_stage_seconds": model_stages,
                          "phase_estimate_stage_seconds": phase_stages,
                          "optimistic_minimum_stage_seconds": minimum_stages})
        scenarios.append({"name": label, "output_advance_per_pass": advance,
                          "verification_positions_per_pass": width, "cases": cases,
                          "optimistic_minimum_seconds": sum(c["optimistic_minimum_seconds"] for c in cases),
                          "phase_estimate_seconds": sum(c["phase_estimate_seconds"] for c in cases),
                          "phase_estimate_stage_seconds": {
                              stage: sum(c["phase_estimate_stage_seconds"][stage] for c in cases)
                              for stage in phase_stages},
                          "optimistic_minimum_stage_seconds": {
                              stage: sum(c["optimistic_minimum_stage_seconds"][stage] for c in cases)
                              for stage in minimum_stages}})

    totals = {key: sum(d[key] for d in docs) for key in (
        "pages", "regions", "ocr_generated_tokens_including_eos", "decode_advances_after_prefill",
        "ocr_prompt_tokens", "visual_tokens", "vision_patches", "markdown_tokens_ocr_tokenizer",
        "markdown_tokens_embedding_tokenizer", "embedding_tokens", "chunks")}
    totals["flops"] = all_flops
    totals["ideal_arithmetic_stage_seconds"] = {
        k: f / (GPU_COUNT * (FP32 if k in ("layout", "vision") else BF16))
        for k, f in totals["flops"].items()}
    totals["ideal_arithmetic_sum_seconds"] = sum(totals["ideal_arithmetic_stage_seconds"].values())
    measured_stages = {
        stage: sum(doc["measured_mean_stage_seconds"][stage] for doc in docs)
        for stage in stage_names}
    measured_stages["other"] = sum(doc["measured_other_seconds"] for doc in docs)
    measured_total = sum(doc["measured_mean_call_seconds"] for doc in docs)
    assert math.isclose(sum(measured_stages.values()), measured_total, rel_tol=1e-12)
    assert math.isclose(measured_total, 60 * totals["pages"]
                        / report["summary"]["pages_per_minute_sequential"], rel_tol=1e-12)
    weight_time = 2 * decoder.linear_parameters / BANDWIDTH
    return {
        "schema_version": 1, "as_of": "2026-10-27",
        "method": "CPU calculation from checkpoint headers, frozen reference text and saved traces; no new inference.",
        "measurement_source_revision": provenance["source_revision"],
        "model_revisions": environment["model_revisions"], "dataset_sha256": report["dataset_sha256"],
        "scope": {"pdfs": len(docs), "gpus": GPU_COUNT, "slots_per_gpu": SLOTS,
                  "pdf_concurrency": 1, "warm_models": True, "passes_over_dataset": 1},
        "hardware_per_gpu": {"fp32_flops_s": FP32, "dense_bf16_flops_s": BF16,
                             "memory_bytes_s": BANDWIDTH},
        "models": {
            "ocr": {"checkpoint_parameters": sum(ocr.values()), "unused_checkpoint_parameters": unused,
                    "active_parameters": sum(ocr.values()) - unused,
                    "active_vision_parameters": visual, "projector_parameters": projector,
                    "resident_weight_bytes": (visual + projector) * 4
                    + (sum(ocr.values()) - unused - visual - projector) * 2,
                    "vision_encoder_linear_parameters": visual_linear,
                    "patch_projection_linear_parameters": patch_linear,
                    "projector_linear_parameters": projector_linear,
                    "decoder_transformer_linear_parameters": decoder_linear,
                    "lm_head_parameters": lm_head, "decode_linear_parameters": decoder.linear_parameters,
                    "decode_weight_bytes": 2 * decoder.linear_parameters,
                    "kv_bytes_per_token": decoder.kv_bytes_per_token, "config": oc,
                    "safetensors_header_sha256": ocr_hashes},
            "embedding": {"parameters": sum(emb.values()), "linear_parameters": embedding_linear,
                          "resident_weight_bytes_bf16": 2 * sum(emb.values()), "config": ec,
                          "safetensors_header_sha256": emb_hashes},
            "layout": layout},
        "prompt_nonimage_tokens": prompt_overheads,
        "decode_rates_ignoring_kv": {"weight_stream_seconds": weight_time,
                                    "greedy_tokens_s_per_sequence": 1 / weight_time,
                                    "greedy_tokens_s_32_busy_sequences": GPU_COUNT * SLOTS / weight_time},
        "totals": totals, "documents": docs, "scenarios": scenarios,
        "measured_optimized_reference": {
            "report": "reports/optimized.json", "trace": f"{RUN}/optimized-stages.jsonl",
            "configuration": "Retained optimized configuration, clean-revision revalidation.",
            "aggregation": "Mean of five measured calls per PDF, summed over all five PDFs processed serially.",
            "warmups_excluded": True, "stage_seconds": measured_stages,
            "total_seconds": measured_total,
            "stage_scope": "Rendering, layout, vision, prefill, decoding and OCR postprocessing share one measured render + OCR interval.",
            "other_scope": "Evaluator call time minus named stage intervals, including result assembly and call-boundary overhead.",
        },
        "measured_seconds_per_20_pages": {
            name: 60 * totals["pages"] / r["summary"]["pages_per_minute_sequential"]
            for name, r in (("baseline", baseline), ("optimized", report))},
        "assumptions": [
            "Four full replicas; no tensor parallelism, quantization, precision changes or sparse weights.",
            "FP32 vision/layout and dense BF16 language/embedding; peaks are sustained without launch gaps.",
            "Free PDF rendering, crop transport, tokenization, Markdown assembly and host orchestration.",
            "No startup, compilation, padding, activation traffic, normalization or sampling overhead.",
            "One full decoder matrix-weight stream per GPU pass; no persistent weight residency in cache.",
            "Read each sequence's KV once and write its verification positions; ideal reuse within the pass.",
            "Attention includes QK and AV; causal useful triangles for language/embedding, full vision attention.",
            "Every region is ready for the decoder; longest-first placement has advance-length foresight.",
            "Embedding follows completed OCR for each PDF; PDFs are processed sequentially.",
            "Advance 2 is an illustration; advance 5 requires every four-token draft to be accepted.",
            "Optimistic minimum and stage-by-stage estimate both assume ideal conditions; neither predicts actual runtime or a worst case.",
        ],
        "input_sha256": fingerprints,
        "references": [
            "https://www.nvidia.com/en-us/data-center/l4/",
            "https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html",
            "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/c5630abae1d940eafe0697512a0325494b02ab42/config.json",
            "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config.json",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path.home() / "huggingface")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                      TOKENIZERS_PARALLELISM="false")
    result = calculate(Path(__file__).resolve().parents[1], args.models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
