"""Persistent OCR replicas cooperate on a PDF, one page at a time per GPU."""
from __future__ import annotations

import atexit
import ctypes
from contextlib import ExitStack
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from document_pipeline.worker_render import page_request, request_image_scope


def _parent_lifetime(parent_pid):
    """eval.Worker terminates its pipeline process; GPU children must exit too."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # Linux PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Could not attach GPU worker to parent lifetime")
    if os.getppid() != parent_pid:
        raise RuntimeError("Pipeline process exited while its GPU worker was starting")


def _worker_main(connection, cfg, worker, parent_pid):
    _parent_lifetime(parent_pid)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        if cfg.backend == "paddleocr":
            import paddle
            from document_pipeline.paddleocr_backend import PaddleOcr
            ocr = PaddleOcr(cfg, device=f"cuda:{worker}")
            gpu_name = paddle.device.cuda.get_device_name(worker)
            parameter_bytes = ocr.parameter_bytes
            def reset_peaks():
                paddle.device.cuda.reset_max_memory_allocated(worker)
                paddle.device.cuda.reset_max_memory_reserved(worker)
            peak_allocated = lambda: paddle.device.cuda.max_memory_allocated(worker)
            peak_reserved = lambda: paddle.device.cuda.max_memory_reserved(worker)
        else:
            import torch
            from document_pipeline.ocr import HfOcr
            torch.set_num_threads(1)
            torch.cuda.set_device(worker)
            ocr = HfOcr(cfg, device=f"cuda:{worker}")
            gpu_name = torch.cuda.get_device_name(worker)
            parameter_bytes = sum(p.numel() * p.element_size() for p in ocr.model.parameters())
            reset_peaks = lambda: torch.cuda.reset_peak_memory_stats(worker)
            peak_allocated = lambda: torch.cuda.max_memory_allocated(worker)
            peak_reserved = lambda: torch.cuda.max_memory_reserved(worker)
        identity = {
            "worker": worker, "pid": os.getpid(), "device": worker,
            "gpu_name": gpu_name, "parameter_bytes": parameter_bytes,
            "backend": cfg.backend,
        }
        import importlib.metadata
        identity["renderer_version"] = importlib.metadata.version("pypdfium2")
        connection.send({"ready": True, "identity": identity})
        while True:
            request = connection.recv()
            if request is None:
                break
            started = time.perf_counter()
            reset_peaks()
            render_started = time.time()
            with ExitStack() as resources:
                try:
                    images = resources.enter_context(request_image_scope(request))
                except (FileNotFoundError, ValueError) as exc:
                    connection.send({
                        "input_error": {
                            "type": "FileNotFoundError" if isinstance(exc, FileNotFoundError) else "ValueError",
                            "message": str(exc),
                        },
                        "pages_1_based": request["pages_1_based"],
                    })
                    continue
                render_finished = time.time()
                render_s = time.perf_counter() - started if "pdf" in request else 0
                render_stats = {"render_s": render_s,
                                "rendered_pages_1_based": request["pages_1_based"] if "pdf" in request else [],
                                "render_windows": [{"start_unix_s": render_started,
                                                    "end_unix_s": render_finished,
                                                    "pages_1_based": request["pages_1_based"]}]
                                                  if "pdf" in request else []}
                if "pdf" in request and os.environ.get("DOCUMENT_PIPELINE_RENDER_PIXEL_CHECK") == "1":
                    import hashlib
                    render_stats["render_pixels"] = [
                        {"page": page, "size": list(image.size), "mode": image.mode,
                         "sha256": hashlib.sha256(image.tobytes()).hexdigest()}
                        for page, image in zip(request["pages_1_based"], images)]
                if request.get("profile"):
                    if cfg.backend == "paddleocr":
                        from lab.pdf_prof import paddle_profile as profile_call
                    else:
                        from lab.pdf_prof import torch_profile as profile_call
                    directory = Path(request["profile"]["directory"]) / f"worker-{worker}"
                    directory.mkdir(parents=True, exist_ok=False)

                    class OcrCall:
                        _record_function = None

                        def __init__(self):
                            self.ocr = ocr
                            self.output = None

                        def __call__(self, ignored):
                            self.output = ocr.ocr(images)

                    call = OcrCall()
                    capture = profile_call(call, "in-memory-pages", directory,
                                           request["profile"]["max_ocr_forwards"])
                    texts = call.output
                    (directory / "capture.json").write_text(json.dumps(
                        {"identity": identity, "pages_1_based": request["pages_1_based"], **capture},
                        indent=2) + "\n")
                else:
                    texts = ocr.ocr(images)
            connection.send({"texts": texts, "stats": {**ocr.stats, **render_stats},
                             "pages_1_based": request["pages_1_based"],
                             "peak_allocated_bytes": peak_allocated(),
                             "peak_reserved_bytes": peak_reserved(),
                             "elapsed_s": time.perf_counter() - started})
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send({"error": traceback.format_exc()})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class _PaddleProcess:
    """Popen with the lifecycle interface used by the persistent page pool."""
    def __init__(self, connection, cfg, worker, admission_path=None):
        root = Path(__file__).resolve().parents[1]
        environment = {**os.environ, "PYTHONPATH": str(root),
                       "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"}
        environment.pop("DOCUMENT_PIPELINE_REGION_ADMISSION_SOCKET", None)
        if admission_path:
            environment["DOCUMENT_PIPELINE_REGION_ADMISSION_SOCKET"] = admission_path
        self.process = subprocess.Popen(
            [os.path.expanduser(cfg.paddle_python), "-m", "document_pipeline.paddle_worker",
             "--fd", str(connection.fileno())],
            pass_fds=(connection.fileno(),), cwd=root, env=environment)
        self.pid = self.process.pid

    def is_alive(self):
        return self.process.poll() is None

    def join(self, timeout):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()


class PageParallelOcr:
    """Persistent page workers; results restored to original page order."""

    def __init__(self, cfg):
        from document_pipeline.region_admission import validate_region_admission
        validate_region_admission(cfg)
        import torch
        self.cfg = cfg
        self.stats = {}
        self.identities = []
        self.connections = []
        self.processes = []
        self._profile_next = None
        self._closed = False
        self._region_admission = None
        if not torch.cuda.is_available() or torch.cuda.device_count() != cfg.gpu_workers:
            raise ValueError(f"Expose exactly {cfg.gpu_workers} CUDA GPUs for this page-worker baseline.")
        if cfg.device not in {"auto", "cuda", "cuda:0"}:
            raise ValueError("Page workers choose their CUDA devices; use ocr.device=auto.")
        if cfg.page_batch_size != 1:
            raise ValueError("The page-worker baseline requires page_batch_size=1 per GPU.")
        if cfg.page_schedule not in {"round_robin", "dynamic"}:
            raise ValueError("Unsupported page scheduling policy.")
        context = mp.get_context("spawn")
        try:
            if cfg.paddle_region_admission != "none":
                from document_pipeline.region_admission import RegionAdmission
                self._region_admission = RegionAdmission(
                    gpus=cfg.gpu_workers, workers=cfg.gpu_workers,
                    slots_per_gpu=cfg.paddle_region_admission_slots_per_gpu,
                    slots_per_worker_gpu=cfg.paddle_region_batch_size)
            for worker in range(cfg.gpu_workers):
                parent, child = context.Pipe()
                try:
                    if cfg.backend == "paddleocr":
                        from dataclasses import asdict
                        process = _PaddleProcess(child, cfg, worker,
                            self._region_admission.path if self._region_admission else None)
                    else:
                        process = context.Process(target=_worker_main,
                            args=(child, cfg, worker, os.getpid()), daemon=True)
                        process.start()
                    self.connections.append(parent)
                    self.processes.append(process)
                    if cfg.backend == "paddleocr":
                        parent.send({"config": asdict(cfg), "worker": worker,
                                     "parent_pid": os.getpid()})
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
            for worker in range(cfg.gpu_workers):
                response = self._receive(worker, timeout=300)
                if not response.get("ready"):
                    raise RuntimeError("GPU worker failed to initialize")
                self.identities.append(response["identity"])
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _receive(self, worker, timeout=1800):
        connection = self.connections[worker]
        deadline = time.monotonic() + timeout
        while not connection.poll(min(1, max(0, deadline - time.monotonic()))):
            if any(not process.is_alive() for process in self.processes):
                raise RuntimeError("A GPU page worker exited unexpectedly.")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"GPU page worker {worker} exceeded {timeout} seconds.")
        try:
            result = connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"GPU page worker {worker} exited without a result.") from exc
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def profile_next_call(self, directory, max_ocr_forwards):
        self._profile_next = {"directory": str(directory), "max_ocr_forwards": max_ocr_forwards}

    @staticmethod
    def _input_error(response):
        error = response["input_error"]
        classes = {"FileNotFoundError": FileNotFoundError, "ValueError": ValueError}
        if error.get("type") not in classes or not isinstance(error.get("message"), str):
            raise RuntimeError("GPU page worker returned an invalid input error")
        return classes[error["type"]](error["message"])

    def ocr(self, images):
        admission = getattr(self, "_region_admission", None)
        before = admission.snapshot() if admission else None
        texts = self._ocr_pages(images)
        if admission:
            after = admission.snapshot()
            if any(after["active"]) or after["pending"]:
                self.close()
                raise RuntimeError("Region admission retained work after the PDF completed.")
            self.stats["region_admission"] = {
                "policy": self.cfg.paddle_region_admission,
                "slots_per_gpu": self.cfg.paddle_region_admission_slots_per_gpu,
                "slots_per_worker_gpu": self.cfg.paddle_region_batch_size,
                "requests_per_gpu": [end - start for start, end in
                                     zip(before["grants"], after["grants"])],
                "lifetime_peak_per_gpu": after["peak"],
                "lifetime_peak_per_worker_gpu": after["worker_peak"],
                "active_after_call": after["active"], "pending_after_call": after["pending"],
            }
        return texts

    def _ocr_pages(self, images):
        profile, self._profile_next = self._profile_next, None
        if getattr(getattr(self, "cfg", None), "page_schedule", "round_robin") == "dynamic":
            if profile:
                raise ValueError("Dynamic pages require server-side tracing, not native worker tracing.")
            return self._ocr_dynamic(images)
        assignments = [list(range(worker, len(images), len(self.connections)))
                       for worker in range(len(self.connections))]
        texts = [None] * len(images)
        input_error = None
        stats = {"pages": len(images), "gen_tokens": 0, "visual_tokens": 0,
                 "gpu_workers": len(self.connections),
                 "active_workers": [i for i, pages in enumerate(assignments) if pages],
                 "workers": []}
        try:
            for worker, pages in enumerate(assignments):
                if pages:
                    self.connections[worker].send(page_request(images, pages, profile))
            for worker, pages in enumerate(assignments):
                if not pages:
                    stats["workers"].append({"worker": worker, "pages_1_based": [], "elapsed_s": 0})
                    continue
                response = self._receive(worker)
                if response["pages_1_based"] != [i + 1 for i in pages]:
                    raise RuntimeError(f"GPU page worker {worker} returned inconsistent pages.")
                if "input_error" in response:
                    input_error = input_error or self._input_error(response)
                    continue
                if (len(response["texts"]) != len(pages)
                        or any(not isinstance(text, str) for text in response["texts"])):
                    raise RuntimeError(f"GPU page worker {worker} returned inconsistent pages.")
                for index, text in zip(pages, response["texts"]):
                    texts[index] = text
                for key in ("gen_tokens", "visual_tokens"):
                    stats[key] += response["stats"][key]
                stats["workers"].append({"worker": worker, "pages_1_based": response["pages_1_based"],
                                         "elapsed_s": response["elapsed_s"],
                                         "peak_allocated_bytes": response["peak_allocated_bytes"],
                                         "peak_reserved_bytes": response["peak_reserved_bytes"],
                                         **response["stats"]})
            self.stats = stats
        except BaseException:
            self.close()
            raise
        # Every submitted request has been drained, so a bad PDF leaves the
        # workers and their pipes ready for the next document.
        if input_error:
            raise input_error
        return texts

    def _ready_worker(self, pending, timeout=1800):
        from multiprocessing.connection import wait
        deadline = time.monotonic() + timeout
        while True:
            ready = wait([self.connections[w] for w in pending], timeout=1)
            if ready:
                return next(w for w in pending if self.connections[w] in ready)
            if any(not process.is_alive() for process in self.processes):
                raise RuntimeError("A GPU page worker exited unexpectedly.")
            if time.monotonic() >= deadline:
                raise TimeoutError("Dynamic page scheduling exceeded its worker timeout.")

    def _ocr_dynamic(self, images):
        """Dispatch the next source page to whichever layout worker finishes first."""
        texts, pending, next_page = [None] * len(images), {}, 0
        input_error = None
        workers = [{"worker": w, "pages_1_based": [], "pages": 0, "elapsed_s": 0,
                    "gen_tokens": 0, "visual_tokens": 0, "batches": [], "page_blocks": [],
                    "render_s": 0, "rendered_pages_1_based": [], "render_windows": [],
                    "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
                   for w in range(len(self.connections))]

        def send(worker):
            nonlocal next_page
            pending[worker] = next_page
            self.connections[worker].send(page_request(images, [next_page]))
            next_page += 1

        try:
            for worker in range(min(len(images), len(workers))):
                send(worker)
            while pending:
                worker = self._ready_worker(pending)
                page = pending.pop(worker)
                response = self._receive(worker)
                if response["pages_1_based"] != [page + 1]:
                    raise RuntimeError(f"GPU page worker {worker} returned inconsistent pages.")
                if "input_error" in response:
                    input_error = input_error or self._input_error(response)
                    continue
                if (len(response["texts"]) != 1
                        or not isinstance(response["texts"][0], str)):
                    raise RuntimeError(f"GPU page worker {worker} returned inconsistent pages.")
                texts[page] = response["texts"][0]
                accumulated, local = workers[worker], response["stats"]
                offset = accumulated["pages"]
                accumulated["pages_1_based"].append(page + 1)
                accumulated["pages"] += 1
                accumulated["elapsed_s"] += response["elapsed_s"]
                accumulated["render_s"] += local.get("render_s", 0)
                for key in ("rendered_pages_1_based", "render_windows", "render_pixels"):
                    if key in local:
                        accumulated.setdefault(key, []).extend(local[key])
                for key in ("gen_tokens", "visual_tokens"):
                    accumulated[key] += local[key]
                for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
                    accumulated[key] = max(accumulated[key], response[key])
                accumulated["batches"].extend(
                    {**batch, "first_local_page": batch["first_local_page"] + offset}
                    for batch in local.get("batches", []))
                accumulated["page_blocks"].extend(
                    {**block, "local_page": block["local_page"] + offset}
                    for block in local.get("page_blocks", []))
                for key, value in local.items():
                    if key not in accumulated:
                        accumulated[key] = value
                if next_page < len(images) and input_error is None:
                    send(worker)
            self.stats = {
                "pages": len(images), "gpu_workers": len(workers), "page_schedule": "dynamic",
                "active_workers": [w["worker"] for w in workers if w["pages"]],
                "workers": workers,
                **{key: sum(w[key] for w in workers) for key in ("gen_tokens", "visual_tokens")},
            }
        except BaseException:
            self.close()
            raise
        if input_error:
            raise input_error
        return texts

    def close(self):
        if self._closed:
            return
        self._closed = True
        for connection in self.connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        try:
            if self._region_admission:
                self._region_admission.close()
        finally:
            for process in self.processes:
                process.join(2)
                if process.is_alive():
                    process.terminate()
                    process.join(2)
                if process.is_alive():
                    process.kill()
                    process.join(2)
            for connection in self.connections:
                connection.close()
