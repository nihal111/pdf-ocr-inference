"""Official PaddleOCR page parser, confined to a worker with its own CUDA runtime."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time


class PaddleOcr:
    def __init__(self, cfg, *, device):
        from document_pipeline.region_admission import validate_region_admission
        validate_region_admission(cfg)
        admission_path = None
        if cfg.paddle_region_admission != "none":
            from document_pipeline.region_admission import socket_address
            admission_path = os.environ.get("DOCUMENT_PIPELINE_REGION_ADMISSION_SOCKET")
            if not admission_path:
                raise ValueError("Shared region admission requires its page-worker broker.")
            socket_address(admission_path)
        if cfg.paddle_rec_backend not in {"native", "vllm-server"}:
            raise ValueError("Unsupported Paddle recognition backend.")
        if cfg.paddle_region_dispatch not in {"page", "all_gpus"}:
            raise ValueError("Unsupported recognition-region dispatch policy.")
        if cfg.paddle_region_dispatch == "all_gpus" and cfg.paddle_rec_backend != "vllm-server":
            raise ValueError("Dispatching regions across GPUs requires vLLM recognizers.")
        submission_override = cfg.paddle_region_submission_size
        if submission_override is not None:
            if cfg.paddle_rec_backend != "vllm-server":
                raise ValueError("A separate region submission window requires vLLM recognizers.")
            if type(submission_override) is not int or submission_override < 1:
                raise ValueError("Region submission size must be a positive integer.")
        if not 1024 <= cfg.paddle_server_base_port <= 65535 - cfg.gpu_workers:
            raise ValueError("Invalid loopback recognizer port range.")
        import paddle
        from paddleocr import PaddleOCRVL
        from paddlex.inference import load_pipeline_config

        self.cfg, self.device, self.paddle = cfg, device.replace("cuda:", "gpu:"), paddle
        paddle.set_device(self.device)
        config = load_pipeline_config(f"PaddleOCR-VL-{cfg.paddle_pipeline_version[1:]}")
        config["batch_size"] = 1
        config["use_queues"] = False
        config["SubModules"]["LayoutDetection"]["batch_size"] = 1
        submission_batch_size = submission_override or cfg.paddle_region_batch_size * (
            cfg.gpu_workers if cfg.paddle_region_dispatch == "all_gpus" else 1)
        config["SubModules"]["VLRecognition"]["batch_size"] = submission_batch_size
        server_options = {}
        if cfg.paddle_rec_backend == "vllm-server":
            port = cfg.paddle_server_base_port + int(self.device.split(":")[-1])
            server_options = {
                "vl_rec_server_url": f"http://127.0.0.1:{port}/v1",
                "vl_rec_api_model_name": "PaddleOCR-VL-1.6-0.9B",
                "vl_rec_max_concurrency": cfg.paddle_region_batch_size,
            }
        self.parser = PaddleOCRVL(
            pipeline_version=cfg.paddle_pipeline_version, paddlex_config=config,
            device=self.device, vl_rec_backend=cfg.paddle_rec_backend,
            vl_rec_model_dir=os.path.expanduser(cfg.model),
            layout_detection_model_dir=os.path.expanduser(cfg.paddle_layout_model),
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_layout_detection=True, use_queues=False,
            use_ocr_for_image_block=True, markdown_ignore_labels=[],
            **server_options,
        )
        self.recognizer = self.parser.paddlex_pipeline.vl_rec_model
        actual_batch_size = self.recognizer.batch_sampler.batch_size
        if actual_batch_size != submission_batch_size:
            self.parser.close()
            raise ValueError(
                f"PaddleOCR {cfg.paddle_rec_backend} selected region batch size {actual_batch_size}, "
                f"but {submission_batch_size} was requested. The backend overrode "
                "the setting; refusing an ineffective batch-size experiment.")
        self.stats = {}
        self._call = 0
        self._local_page = 0
        if cfg.paddle_rec_backend == "vllm-server":
            from document_pipeline.paddle_server import RegionClients, ServerRecognition
            if cfg.paddle_region_dispatch == "all_gpus":
                from paddlex.inference.models.common.genai import GenAIClient
                original = self.recognizer.genai_client
                worker = int(self.device.split(":")[-1])
                clients = [original if gpu == worker else GenAIClient(
                    backend="vllm-server",
                    base_url=f"http://127.0.0.1:{cfg.paddle_server_base_port + gpu}/v1",
                    model_name=original._model_name,
                    max_concurrency=cfg.paddle_region_batch_size,
                ) for gpu in range(cfg.gpu_workers)]
                if cfg.paddle_region_admission == "none":
                    self.recognizer._genai_client = RegionClients(clients, worker)
                else:
                    from document_pipeline.region_admission import AdmittedRegionClients
                    from paddlex.inference.models.common.genai import run_async
                    self.recognizer._genai_client = AdmittedRegionClients(
                        clients, worker, admission_path,
                        lambda coroutine: run_async(coroutine, return_future=True),
                        policy=cfg.paddle_region_admission)
            self.server_recognition = ServerRecognition(self)
            self.model = None
            self.parameter_bytes = None  # recognizer lives in a separate server process
            return
        self.model = self.recognizer.infer
        self._generate = self.model.generate
        # Native generation returns generated tokens before its text postprocessor.
        # Check them here so parser cleanup cannot conceal a length-limit failure.
        if not self.model.generation_config.trunc_input:
            raise ValueError("Paddle completion checks require generated-token-only output.")
        self.model.generate = self._checked_generate
        widths = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}
        self.parameter_bytes = sum(
            math.prod(parameter.shape) * widths[str(parameter.dtype).split(".")[-1]]
            for parameter in self.model.parameters())

    def _checked_generate(self, inputs, **kwargs):
        started = time.perf_counter()
        output = self._generate(inputs, **kwargs)
        rows = output[0].numpy().tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        finished = [any(token in eos for token in row) for row in rows]
        lengths = [next((i + 1 for i, token in enumerate(row) if token in eos), len(row))
                   for row in rows]
        event = {
            "first_local_page": self._local_page, "regions": len(rows),
            "generated_lengths": lengths, "ended_with_eos": finished,
            "elapsed_s": time.perf_counter() - started,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
        }
        self.stats["batches"].append(event)
        self.stats["gen_tokens"] += sum(lengths)
        self.stats["visual_tokens"] += int(
            (inputs["input_ids"] == self.model.config.image_token_id).sum().item())
        review = os.environ.get("DOCUMENT_PIPELINE_OCR_DIAGNOSTICS")
        if review:
            directory = Path(review) / self.device.replace(":", "-")
            directory.mkdir(parents=True, exist_ok=True)
            batch = len(self.stats["batches"])
            prefix = directory / f"call-{self._call}-page-{self._local_page}-regions-{batch}"
            texts = self.recognizer.processor.postprocess(output)
            prefix.with_suffix(".json").write_text(json.dumps(
                {**event, "generated_tokens": rows, "texts": texts}, indent=2) + "\n")
        if self.cfg.require_eos and not all(finished):
            raise RuntimeError(
                f"PaddleOCR incomplete on local page {self._local_page}: a region reached "
                f"{kwargs.get('max_new_tokens')} tokens without EOS. Refusing truncated output.")
        return output

    def ocr(self, images):
        import numpy as np
        self._call += 1
        self.stats = {"pages": len(images), "gen_tokens": 0, "visual_tokens": 0,
                      "batches": [], "page_blocks": [], "backend": "paddleocr",
                      "region_batch_size": self.cfg.paddle_region_batch_size,
                      "region_dispatch": self.cfg.paddle_region_dispatch,
                      "region_admission": self.cfg.paddle_region_admission,
                      "submission_batch_size": self.recognizer.batch_sampler.batch_size,
                      "recognition_backend": self.cfg.paddle_rec_backend,
                      "decoding": self.model.generation_config.decode_strategy if self.model else "greedy"}
        texts = []
        for page, image in enumerate(images, 1):
            self._local_page = page
            # The page renderer returns RGB; the official parser expects BGR arrays.
            bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            results = self.parser.predict(bgr, max_new_tokens=self.cfg.max_new_tokens,
                                          use_queues=False, markdown_ignore_labels=[])
            if len(results) != 1:
                raise RuntimeError("PaddleOCR must return exactly one result for each page.")
            result = results[0]
            text = result.markdown["markdown_texts"]
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"PaddleOCR returned empty text on local page {page}.")
            texts.append(text)
            self.stats["page_blocks"].append({
                "local_page": page,
                "blocks": [{"label": block.label, "characters": len(block.content)}
                           for block in result["parsing_res_list"]],
            })
        return texts

    def close(self):
        self.parser.close()
