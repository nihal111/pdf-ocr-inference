"""Observe Paddle's server requests before postprocessing and preserve region pixels."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time


class RegionClients:
    """Distribute region requests across existing recognizers, preserving future order."""
    backend = "vllm-server"

    def __init__(self, clients, start_gpu):
        self.clients, self.start_gpu = clients, start_gpu
        self.reset()

    def reset(self):
        self.next_gpu = self.start_gpu

    def create_chat_completion(self, messages, *, return_future=False, **kwargs):
        if not return_future:
            raise ValueError("Region distribution requires asynchronous requests.")
        gpu = self.next_gpu
        self.next_gpu = (gpu + 1) % len(self.clients)
        future = self.clients[gpu].create_chat_completion(
            messages, return_future=True, **kwargs)
        future.document_pipeline_gpu = gpu
        return future

    def close(self):
        for client in self.clients:
            client.close()


class ServerRecognition:
    def __init__(self, ocr):
        self.ocr = ocr
        recognizer = ocr.recognizer
        self.build_requests = recognizer._doc_vlm_genai_build_request_specs
        recognizer._doc_vlm_genai_build_request_specs = self.prepare
        recognizer._doc_vlm_genai_collect_responses = self.collect
        self.started = 0.0

    def prepare(self, client, data, image_format, *args):
        self.started = time.perf_counter()
        if isinstance(client, RegionClients) and not self.ocr.stats["batches"]:
            client.reset()
        # The stock server client uses JPEG. PNG preserves the native crop pixels.
        specs = self.build_requests(client, data, "PNG", *args)
        for item, (_, kwargs) in zip(data, specs):
            kwargs["temperature"] = 0.0
            kwargs["extra_body"].update(return_token_ids=True, skip_special_tokens=True,
                                        repetition_penalty=1.0)
            table_pixels = getattr(getattr(self.ocr, "cfg", None), "paddle_table_max_pixels", None)
            if table_pixels is not None and item.get("query") == "Table Recognition:":
                if type(table_pixels) is not int or table_pixels <= 0:
                    raise ValueError("Table image limit must be a positive integer.")
                kwargs["extra_body"].setdefault("mm_processor_kwargs", {})["max_pixels"] = table_pixels
        return specs

    def collect(self, futures):
        responses = [future.result() for future in futures]
        choices = [response.choices[0] for response in responses]
        tokens = [getattr(choice, "token_ids", None) for choice in choices]
        # No custom stop strings/tokens are supplied. Require raw EOS plus finish_reason.
        finished = [choice.finish_reason == "stop" and bool(row) and row[-1] == 2
                    for choice, row in zip(choices, tokens)]
        lengths = [len(row) if row is not None else 0 for row in tokens]
        texts = [choice.message.content for choice in choices]
        event = {
            "first_local_page": self.ocr._local_page, "regions": len(responses),
            "generated_lengths": lengths, "ended_with_eos": finished,
            "finish_reasons": [choice.finish_reason for choice in choices],
            "elapsed_s": time.perf_counter() - self.started,
            "input_tokens": [response.usage.prompt_tokens for response in responses],
            "transport": "lossless PNG",
            "recognizer_gpus": [
                getattr(future, "document_pipeline_gpu",
                        int(getattr(self.ocr, "device", "gpu:0").split(":")[-1]))
                for future in futures],
        }
        self.ocr.stats["batches"].append(event)
        self.ocr.stats["gen_tokens"] += sum(lengths)
        review = os.environ.get("DOCUMENT_PIPELINE_OCR_DIAGNOSTICS")
        if review:
            directory = Path(review) / self.ocr.device.replace(":", "-")
            directory.mkdir(parents=True, exist_ok=True)
            batch = len(self.ocr.stats["batches"])
            path = directory / f"call-{self.ocr._call}-page-{self.ocr._local_page}-regions-{batch}.json"
            path.write_text(json.dumps({**event, "generated_tokens": tokens, "texts": texts},
                                       indent=2) + "\n")
        if self.ocr.cfg.require_eos and not all(finished):
            raise RuntimeError(
                f"Paddle/vLLM incomplete on local page {self.ocr._local_page}: "
                f"finish reasons {[choice.finish_reason for choice in choices]}, "
                f"raw EOS checks {finished}. Refusing unchecked output.")
        return texts
