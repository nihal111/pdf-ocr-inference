"""Fill eval/environment.example.json automatically for this machine and pipeline config.

    python -m lab.pdf_env --out env.json [--config '{"embed": {"dtype": "bfloat16"}}']

eval/compare.py only compares reports whose hardware_id, gpu_model, gpu_count,
driver_version and concurrency match, so collect these, don't type them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sh(*cmd) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=ROOT)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _gpus() -> list[dict]:
    out = _sh("nvidia-smi", "--query-gpu=name,uuid,driver_version,power.limit,clocks.max.sm,utilization.gpu,memory.used",
              "--format=csv,noheader,nounits")
    if not out:
        return []
    keys = ("name", "uuid", "driver", "power_limit_w", "max_sm_mhz", "util_pct", "mem_used_mib")
    return [dict(zip(keys, (x.strip() for x in line.split(",")))) for line in out.splitlines()]


def _visible(gpus: list[dict], environment=None) -> list[dict]:
    vis = (environment if environment is not None else os.environ).get("CUDA_VISIBLE_DEVICES")
    if vis is None or not gpus:
        return gpus
    picked = []
    for token in filter(None, (t.strip() for t in vis.split(","))):
        if token.isdigit() and int(token) < len(gpus):
            picked.append(gpus[int(token)])
        else:
            picked += [g for g in gpus if g["uuid"].startswith(token)]
    return picked


def _model_revision(path: str) -> str:
    path = os.path.expanduser(path)
    provenance = os.path.join(path, "inference-lab-provenance.json")
    if os.path.isfile(provenance):
        with open(provenance) as stream:
            revision = json.load(stream).get("revision")
        if revision:
            return f"{os.path.basename(path)}@{revision}"
    meta = os.path.join(path, ".cache", "huggingface", "download")
    if os.path.isdir(meta):  # hf download --local-dir keeps the commit hash here
        for name in sorted(os.listdir(meta)):
            if name.endswith(".safetensors.metadata"):
                return f"{os.path.basename(path)}@{open(os.path.join(meta, name)).readline().strip()}"
    return f"{os.path.basename(path)}@unknown"


def collect(pipeline_cfg: dict | None = None, environment: dict | None = None) -> dict:
    from document_pipeline.config import load_config
    cfg = load_config(pipeline_cfg or {})
    gpus = _visible(_gpus(), environment)
    try:
        import torch
        cuda = torch.version.cuda
    except ImportError:
        cuda = None
    rev = _sh("git", "rev-parse", "--short", "HEAD") or "unknown"
    if _sh("git", "status", "--porcelain", "--", "document_pipeline", "engine", "lab"):
        rev += "-dirty"
    versions = {}
    for package in ("torch", "torchvision", "transformers", "sentence-transformers",
                    "pypdfium2", "pillow", "numpy", "accelerate", "flash-attn", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    worker = {}
    if cfg.ocr.backend == "paddleocr":
        # Query the isolated interpreter without importing Paddle into the parent.
        output = _sh(os.path.expanduser(cfg.ocr.paddle_python), "-c",
                     "import importlib.metadata as m,json,platform;"
                     "names=['paddleocr','paddlex','paddlepaddle-gpu','nvidia-cuda-runtime-cu12'];"
                     "print(json.dumps({'python_version':platform.python_version(),"
                     "'package_versions':{n:m.version(n) for n in names}}))")
        if output:
            worker = json.loads(output)
        worker["python"] = os.path.expanduser(cfg.ocr.paddle_python)
        worker["engine"] = ("PaddleOCR/PaddleX native PaddlePaddle" if
                            cfg.ocr.paddle_rec_backend == "native" else
                            "PaddleOCR/PaddleX layout + vLLM server recognition")
        if cfg.ocr.paddle_rec_backend == "vllm-server":
            output = _sh(os.path.expanduser(cfg.ocr.paddle_server_python), "-c",
                         "import importlib.metadata as m,json,platform;"
                         "names=['vllm','torch','transformers','inference-lab-paddle-native-precision'];"
                         "print(json.dumps({'python_version':platform.python_version(),"
                         "'package_versions':{n:m.version(n) for n in names}}))")
            worker["recognizer_server_environment"] = json.loads(output) if output else {}
            worker["recognizer_server_environment"]["python"] = cfg.ocr.paddle_server_python
            worker["recognizer_server_environment"]["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            worker["recognizer_server_environment"]["VLLM_PLUGINS"] = "paddle_native_precision"
            worker["recognizer_server_environment"]["VLLM_BATCH_INVARIANT"] = (
                environment or os.environ).get("VLLM_BATCH_INVARIANT", "0")
            worker["recognizer_server_environment"]["precision_policy"] = {
                "visual": "float32", "mlp_AR": "float32", "language_model": "bfloat16"}
    # This workspace may be a source copy without .git. Record code fingerprints as well.
    from pathlib import Path
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("document_pipeline", "engine", "lab")
        for path in sorted((Path(ROOT) / directory).rglob("*.py"))
    }
    if gpus:
        hw = hashlib.sha256(",".join(sorted(g["uuid"] for g in gpus)).encode()).hexdigest()[:12]
        hardware_id = f"{socket.gethostname()}-{hw}"
    else:
        hardware_id = f"{socket.gethostname()}-{platform.machine()}"
    return {
        "hardware_id": hardware_id,
        "gpu_model": gpus[0]["name"] if gpus else f"none ({platform.processor() or platform.machine()})",
        # 0 GPUs is honest, and eval/compare.py rejects it: official comparisons need the GPU box.
        "gpu_count": len(gpus),
        "gpu_uuids": ",".join(g["uuid"] for g in gpus),
        "driver_version": gpus[0]["driver"] if gpus else "none",
        "cuda_runtime": cuda or "",
        "python_version": platform.python_version(),
        "engine": f"ocr={cfg.ocr.backend}"
                  + (f"/{cfg.ocr.paddle_rec_backend}" if cfg.ocr.backend == "paddleocr" else "")
                  + f":{os.path.basename(cfg.ocr.model)} | embed={cfg.embed.backend}",
        "precision": f"ocr={cfg.ocr.dtype} embed={cfg.embed.dtype}",
        "ocr_gpu_workers": cfg.ocr.gpu_workers,
        "batch_size": f"ocr_pages={cfg.ocr.page_batch_size} embed="
                      + str(cfg.embed.max_batch_tokens if cfg.embed.backend == "nanovllm" else cfg.embed.batch_size),
        "ocr_region_batch_size": cfg.ocr.paddle_region_batch_size if cfg.ocr.backend == "paddleocr" else None,
        "concurrency": 1,
        "source_revision": rev,
        "source_files_sha256": source_hashes,
        "package_versions": versions,
        "ocr_worker_environment": worker,
        "cuda_module_loading": (environment or os.environ).get("CUDA_MODULE_LOADING", "default"),
        "model_revisions": ",".join(
            [_model_revision(cfg.ocr.model), _model_revision(cfg.embed.model)]
            + ([_model_revision(cfg.ocr.paddle_layout_model)] if cfg.ocr.backend == "paddleocr" else [])),
        "power_settings": ";".join(f"{g['power_limit_w']}W/{g['max_sm_mhz']}MHz" for g in gpus),
        "cache_state": "warm models (loaded at import), >=1 warm-up call per PDF, no result cache",
        "other_gpu_activity": ";".join(f"util {g['util_pct']}% mem {g['mem_used_mib']}MiB" for g in gpus)
                              + " (at start)" if gpus else "",
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None, help="pipeline overrides (JSON), as in DOCUMENT_PIPELINE_CONFIG")
    args = p.parse_args(argv)
    env = collect(json.loads(args.config) if args.config else None)
    with open(args.out, "w") as f:
        json.dump(env, f, indent=2)
        f.write("\n")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
