"""Launch four loopback vLLM recognizers; exiting terminates their process groups."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def cuda_toolkit(python):
    purelib = subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True).strip()
    directory = Path(purelib) / "nvidia/cu13"
    if not (directory / "bin/nvcc").is_file():
        raise RuntimeError(f"CUDA compiler missing from {directory}")
    # NVIDIA's Python wheels use lib/ and a versioned runtime library. JIT
    # linkers expect the conventional toolkit lib64/ and unversioned soname.
    for link, target in ((directory / "lib64", "lib"),
                         (directory / "lib/libcudart.so", "libcudart.so.13")):
        if not link.exists():
            link.symlink_to(target)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--python", default=str(ROOT / ".venv-vllm/bin/python"))
    parser.add_argument("--model", default="~/huggingface/PaddleOCR-VL-1.6")
    parser.add_argument("--base-port", type=int, default=8800)
    parser.add_argument("--max-num-seqs", type=int, default=4,
                        help="maximum simultaneously running recognition requests per GPU")
    parser.add_argument("--ngram-tokens", type=int, default=0,
                        help="within-request n-gram speculative tokens; zero disables speculation")
    parser.add_argument("--ngram-method", choices=("ngram", "ngram_gpu"), default="ngram",
                        help="CPU or GPU n-gram proposer; used only when speculation is enabled")
    parser.add_argument("--iteration-details", action="store_true",
                        help="log actual scheduler batch sizes for a diagnostic session")
    tiles = parser.add_mutually_exclusive_group()
    tiles.add_argument("--decode-matmul-tiles", action="store_true",
                        help="use screened 16/32-row BF16 decoder tiles; batch invariance required")
    tiles.add_argument("--decode-matmul-fixed32", action="store_true",
                       help="use fixed 32-row BF16 decoder tiles, including compiled projections")
    tiles.add_argument("--decode-matmul-shape-columns", action="store_true",
                       help="use fixed 32-row decoder tiles with screened per-projection column widths")
    parser.add_argument("--isolate-compile-cache", action="store_true",
                        help="use a fresh compilation-cache directory for this server session")
    args = parser.parse_args()
    tile_mode = ("shape_columns" if args.decode_matmul_shape_columns else
                 "32" if args.decode_matmul_fixed32 else "1" if args.decode_matmul_tiles else "0")
    if not 1 <= args.max_num_seqs <= 64 or not 0 <= args.ngram_tokens <= 16:
        parser.error("--max-num-seqs must be 1–64 and --ngram-tokens must be 0–16.")
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or not all(d.startswith("GPU-") for d in devices):
        parser.error("Expose the four fixed GPUs using four distinct UUIDs.")
    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    processes = []
    cuda_home = cuda_toolkit(args.python)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for worker, gpu in enumerate(devices):
            trace = directory / f"worker-{worker}-traces"
            command = [
                args.python, "-m", "vllm.entrypoints.cli.main", "serve", os.path.expanduser(args.model),
                "--host", "127.0.0.1", "--port", str(args.base_port + worker),
                "--served-model-name", "PaddleOCR-VL-1.6-0.9B", "--trust-remote-code",
                "--trust-request-mm-kwargs",
                "--mm-encoder-attn-backend", "TORCH_SDPA",
                "--dtype", "bfloat16", "--max-model-len", "16384",
                "--max-num-seqs", str(args.max_num_seqs), "--max-num-batched-tokens", "16384",
                "--gpu-memory-utilization", "0.40", "--no-enable-prefix-caching",
                "--mm-processor-cache-gb", "0", "--generation-config", "vllm",
                "--profiler-config", json.dumps({"profiler": "torch",
                    "torch_profiler_dir": str(trace),
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_with_stack": False}),
            ]
            if args.ngram_tokens:
                command += ["--speculative-config", json.dumps({
                    "method": args.ngram_method, "num_speculative_tokens": args.ngram_tokens,
                    "prompt_lookup_min": 2, "prompt_lookup_max": 5,
                    "rejection_sample_method": "standard",
                })]
            if args.iteration_details:
                command += ["--observability-config",
                            '{"enable_logging_iteration_details":true}']
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1",
                           "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                           "VLLM_USE_FLASHINFER_SAMPLER": "0",
                           "VLLM_PLUGINS": "paddle_native_precision",
                           "DOCUMENT_PIPELINE_PADDLE_DECODE_TILES":
                               tile_mode,
                           "CUDA_HOME": str(cuda_home),
                           "PATH": f"{Path(args.python).parent}:{cuda_home / 'bin'}:{os.environ['PATH']}"}
            if args.isolate_compile_cache:
                environment.update(
                    VLLM_CACHE_ROOT=str(directory / "vllm-cache"),
                    TORCHINDUCTOR_CACHE_DIR=str(directory / "inductor-cache"))
            with (directory / f"worker-{worker}.log").open("w") as log:
                process = subprocess.Popen(command, env=environment, cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            (directory / f"worker-{worker}.json").write_text(json.dumps({
                "pid": process.pid, "gpu": gpu, "command": command,
                "decoder_matmul_tiles": tile_mode != "0",
                "decoder_matmul_tile_mode": tile_mode,
                "isolated_compile_cache": args.isolate_compile_cache}, indent=2) + "\n")
        pending = set(range(4))
        deadline = time.monotonic() + 900
        while pending:
            for worker in list(pending):
                if processes[worker].poll() is not None:
                    raise RuntimeError(f"Worker {worker} exited; inspect its server log.")
                try:
                    with urlopen(f"http://127.0.0.1:{args.base_port + worker}/health", timeout=2) as response:
                        if response.status == 200:
                            pending.remove(worker)
                            print(f"Worker {worker} healthy", flush=True)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"Server readiness timeout: {sorted(pending)}")
            time.sleep(2)
        (directory / "ready.json").write_text(json.dumps({"pids": [p.pid for p in processes],
            "gpu_uuids": devices, "base_port": args.base_port, "ready_at": time.time()}) + "\n")
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A recognizer server exited.")
            time.sleep(2)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        (directory / "ready.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
