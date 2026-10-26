"""Page rendering and VLM OCR backends (Qwen3-VL). Every backend decodes greedily with the same
chat prompt. The evaluator compares OCR text byte for byte, so a backend only passes if it
reproduces the baseline's tokens exactly."""
from __future__ import annotations

import os
import time

from document_pipeline.embedders import _dtype, resolve_device


def render_pages(path: str, dpi: int):
    from document_pipeline.pdf_input import open_pdf, render_page
    images = []
    try:
        with open_pdf(path) as document:
            for index in range(len(document)):
                images.append(render_page(document, index, dpi))
        return images
    except BaseException:
        for image in images:
            image.close()
        raise


def _processor(cfg):
    from transformers import AutoProcessor
    kwargs = {k: getattr(cfg, k) for k in ("min_pixels", "max_pixels") if getattr(cfg, k) is not None}
    proc = AutoProcessor.from_pretrained(os.path.expanduser(cfg.model), **kwargs)
    proc.tokenizer.padding_side = "left"   # batched generation needs left padding
    return proc


def _chat(proc, prompt: str | None) -> str:
    prompt = prompt or "Transcribe all text on this page as Markdown. Output only the transcription."
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class HfOcr:
    """Baseline: transformers generate(), one padded batch of `page_batch_size` pages at a time."""

    def __init__(self, cfg, *, device=None):
        import torch
        from transformers import AutoModelForImageTextToText
        self.cfg, self.torch = cfg, torch
        self.device = device or resolve_device(cfg.device)
        self.proc = _processor(cfg)
        self.model = AutoModelForImageTextToText.from_pretrained(
            os.path.expanduser(cfg.model), dtype=_dtype(cfg.dtype),
            attn_implementation=cfg.attn_implementation).to(self.device)
        self.model.eval()
        self.text = _chat(self.proc, cfg.prompt)
        self.stats: dict = {}

    def ocr(self, images) -> list[str]:
        out, gen_tokens, vis_tokens, batches = [], 0, 0, []
        bs = max(1, self.cfg.page_batch_size)
        image_token = self.model.config.image_token_id
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        for i in range(0, len(images), bs):
            started = time.perf_counter()
            group = images[i:i + bs]
            inputs = self.proc(text=[self.text] * len(group), images=group, return_tensors="pt",
                               padding=True).to(self.device)
            with self.torch.inference_mode():
                seqs = self.model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                                           repetition_penalty=self.cfg.repetition_penalty,
                                           max_new_tokens=self.cfg.max_new_tokens)
            new = seqs[:, inputs["input_ids"].shape[1]:]
            # Each sequence must contain an EOS token. In padded batches the EOS
            # need not be the final token; padding may follow a finished sequence.
            finished = [any(token in eos for token in row) for row in new.tolist()]
            decoded = self.proc.batch_decode(new, skip_special_tokens=True)
            diagnostic_root = os.environ.get("DOCUMENT_PIPELINE_OCR_DIAGNOSTICS")
            if diagnostic_root:
                import json
                from pathlib import Path
                self._diagnostic_batch = getattr(self, "_diagnostic_batch", 0) + 1
                directory = Path(diagnostic_root) / self.device.replace(":", "-")
                directory.mkdir(parents=True, exist_ok=True)
                for index, text in enumerate(decoded):
                    prefix = directory / f"batch-{self._diagnostic_batch}-page-{i + index + 1}"
                    prefix.with_suffix(".md").write_text(text + "\n")
                    prefix.with_suffix(".json").write_text(json.dumps({
                        "local_page": i + index + 1, "device": self.device,
                        "generated_tokens": new[index].tolist(), "ended_with_eos": finished[index],
                        "max_new_tokens": self.cfg.max_new_tokens,
                        "repetition_penalty": self.cfg.repetition_penalty,
                        "elapsed_s": time.perf_counter() - started,
                    }, indent=2) + "\n")
            if self.cfg.require_eos and not all(finished):
                unfinished = [i + index + 1 for index, done in enumerate(finished) if not done]
                raise RuntimeError(
                    f"OCR incomplete on local page(s) {unfinished}: generation stopped without EOS "
                    f"at max_new_tokens={self.cfg.max_new_tokens}. Refusing truncated output.")
            out += decoded
            gen_tokens += int((new != self.proc.tokenizer.pad_token_id).sum())
            vis_tokens += int((inputs["input_ids"] == image_token).sum())
            batches.append({"first_local_page": i + 1, "pages": len(group),
                            "elapsed_s": time.perf_counter() - started,
                            "input_tokens": inputs["input_ids"].shape[-1],
                            "generated_sequence_length": new.shape[-1], "ended_with_eos": finished})
        self.stats = {"pages": len(images), "gen_tokens": gen_tokens, "visual_tokens": vis_tokens,
                      "batches": batches}
        return out


class VllmOcr:
    """vLLM offline engine: continuous batching across a document's pages, CUDA graphs, paged KV."""

    def __init__(self, cfg):
        from vllm import LLM, SamplingParams
        self.cfg = cfg
        self.proc = _processor(cfg)
        mm_kwargs = {k: getattr(cfg, k) for k in ("min_pixels", "max_pixels") if getattr(cfg, k) is not None}
        self.llm = LLM(model=os.path.expanduser(cfg.model), dtype=cfg.dtype, max_model_len=cfg.max_model_len,
                       gpu_memory_utilization=cfg.gpu_memory_utilization, enforce_eager=cfg.enforce_eager,
                       max_num_seqs=max(1, cfg.page_batch_size), limit_mm_per_prompt={"image": 1},
                       mm_processor_kwargs=mm_kwargs or None, seed=0)
        self.params = SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens,
                                     repetition_penalty=cfg.repetition_penalty)
        self.text = _chat(self.proc, cfg.prompt)
        self.stats: dict = {}

    def ocr(self, images) -> list[str]:
        prompts = [{"prompt": self.text, "multi_modal_data": {"image": img}} for img in images]
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        if self.cfg.require_eos and any(o.outputs[0].finish_reason == "length" for o in outs):
            raise RuntimeError("OCR incomplete: vLLM reached the generation length limit.")
        self.stats = {"pages": len(images), "gen_tokens": sum(len(o.outputs[0].token_ids) for o in outs),
                      "prompt_tokens": sum(len(o.prompt_token_ids) for o in outs)}
        return [o.outputs[0].text for o in outs]


class NanoVllmOcr:
    def __init__(self, cfg):
        raise NotImplementedError(
            "Qwen3-VL on nano-vllm is the porting project; see EXPERIMENTS.md 'Qwen3-VL on nano-vllm'.")


BACKENDS = {"hf": HfOcr, "vllm": VllmOcr, "nanovllm": NanoVllmOcr}


def make_ocr(cfg):
    if type(cfg.gpu_workers) is not int or cfg.gpu_workers < 1:
        raise ValueError("gpu_workers must be a positive integer")
    if cfg.backend == "paddleocr":
        from document_pipeline.page_parallel import PageParallelOcr
        return PageParallelOcr(cfg)
    if cfg.backend not in BACKENDS:
        raise ValueError(f"unknown ocr backend {cfg.backend!r}; have {sorted(BACKENDS)}")
    if cfg.backend == "hf" and cfg.gpu_workers > 1:
        from document_pipeline.page_parallel import PageParallelOcr
        return PageParallelOcr(cfg)
    if cfg.backend != "hf" and cfg.gpu_workers != 1:
        raise ValueError("Multiple GPU page workers require the Hugging Face backend.")
    return BACKENDS[cfg.backend](cfg)
