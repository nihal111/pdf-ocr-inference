"""Embedding backends. Every backend returns L2-normalised last-token embeddings for the same
token ids, so their outputs can be checked against each other by cosine similarity."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from threading import Lock


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(name: str):
    import torch
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


class SentenceTransformersEmbedder:
    """Baseline: off-the-shelf sentence-transformers (padded batches, its own length sorting)."""

    def __init__(self, cfg):
        from sentence_transformers import SentenceTransformer
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.model = SentenceTransformer(os.path.expanduser(cfg.model), device=self.device,
                                         model_kwargs={"dtype": _dtype(cfg.dtype)})
        self.model.max_seq_length = cfg.max_length
        self.tokenizer = self.model.tokenizer
        if cfg.compile_dynamic:
            self.model.compile(dynamic=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=self.cfg.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype("float64").tolist()

    def count_tokens(self, texts: list[str]) -> int:
        return sum(len(ids) for ids in self.tokenizer(texts, truncation=True, max_length=self.cfg.max_length)["input_ids"])


class NanoVllmEmbedder:
    """Qwen3-Embedding on nano-vllm's Qwen3 layers (FlashAttention varlen, fused/compiled ops).

    No KV cache and no padding: chunks are packed into one varlen batch of up to
    `max_batch_tokens` tokens, and the hidden state of each sequence's last token
    (the EOS the tokenizer appends) is the embedding. Needs CUDA + flash-attn.
    """

    def __init__(self, cfg):
        import socket
        import sys

        import torch
        import torch.distributed as dist
        from transformers import AutoConfig, AutoTokenizer

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
        from nanovllm.models.qwen3 import Qwen3ForCausalLM, Qwen3Model

        self.cfg = cfg
        self.torch = torch
        path = os.path.expanduser(cfg.model)
        device = resolve_device(cfg.device)
        assert device.startswith("cuda"), "nanovllm backend needs CUDA"
        assert cfg.dtype in ("bfloat16", "float16"), "flash-attn needs bfloat16 or float16"
        device = "cuda:0" if device == "cuda" else device
        torch.cuda.set_device(torch.device(device))
        if not dist.is_initialized():
            # nano-vllm's layers query the TP world; a private single-rank group on a free port.
            with socket.socket() as s:
                s.bind(("localhost", 0))
                port = s.getsockname()[1]
            dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=1, rank=0)
        hf_config = AutoConfig.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_dtype(cfg.dtype))
        try:
            with torch.device(device):
                self.model = Qwen3Model(hf_config)
        finally:
            torch.set_default_dtype(prev_dtype)
        self._load(path, Qwen3ForCausalLM.packed_modules_mapping)
        self.model.eval()
        self.device = device

    def _load(self, path, packed):
        from glob import glob

        from safetensors import safe_open
        loaded = 0
        for file in glob(os.path.join(path, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for name in f.keys():
                    if name.startswith("lm_head."):
                        continue
                    target = name.removeprefix("model.")
                    for k, (v, shard) in packed.items():
                        if k in target:
                            param = self.model.get_parameter(target.replace(k, v))
                            param.weight_loader(param, f.get_tensor(name), shard)
                            break
                    else:
                        param = self.model.get_parameter(target)
                        loader = getattr(param, "weight_loader", None)
                        if loader:
                            loader(param, f.get_tensor(name))
                        else:
                            param.data.copy_(f.get_tensor(name))
                    loaded += 1
        assert loaded, f"no weights found in {path}"

    def _batches(self, ids: list[list[int]]):
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        batch, tokens = [], 0
        for i in order:
            if batch and tokens + len(ids[i]) > self.cfg.max_batch_tokens:
                yield batch
                batch, tokens = [], 0
            batch.append(i)
            tokens += len(ids[i])
        if batch:
            yield batch

    def embed(self, texts: list[str]) -> list[list[float]]:
        torch = self.torch
        from nanovllm.utils.context import reset_context, set_context
        ids = self.tokenizer(texts, truncation=True, max_length=self.cfg.max_length)["input_ids"]
        out: list = [None] * len(texts)
        with torch.inference_mode():
            for batch in self._batches(ids):
                flat, positions, cu = [], [], [0]
                for i in batch:
                    flat.extend(ids[i])
                    positions.extend(range(len(ids[i])))
                    cu.append(cu[-1] + len(ids[i]))
                max_len = max(len(ids[i]) for i in batch)
                to = dict(device=self.device, non_blocking=True)
                input_ids = torch.tensor(flat, dtype=torch.int64, pin_memory=True).to(**to)
                pos = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(**to)
                cu_t = torch.tensor(cu, dtype=torch.int32, pin_memory=True).to(**to)
                set_context(True, cu_t, cu_t, max_len, max_len)
                try:
                    hidden = self.model(input_ids, pos)
                finally:
                    reset_context()
                last = hidden[cu_t[1:] - 1].float()
                vecs = torch.nn.functional.normalize(last, dim=-1).double().cpu().tolist()
                for i, v in zip(batch, vecs):
                    out[i] = v
        return out

    def count_tokens(self, texts: list[str]) -> int:
        return sum(len(x) for x in self.tokenizer(texts, truncation=True, max_length=self.cfg.max_length)["input_ids"])


BACKENDS = {"sentence_transformers": SentenceTransformersEmbedder, "nanovllm": NanoVllmEmbedder}


class ParallelSentenceTransformersEmbedder:
    """Replicate the same embedder, balance input tokens, and restore chunk order."""

    def __init__(self, cfg):
        import torch
        self.cfg = cfg
        self.torch = torch
        device = torch.device(resolve_device(cfg.device))
        if device.type != "cuda":
            raise ValueError("Multiple embedding workers require CUDA")
        first = device.index if device.index is not None else torch.cuda.current_device()
        if first + cfg.gpu_workers > torch.cuda.device_count():
            raise ValueError("Not enough visible CUDA devices for embedding workers")
        self.devices = [f"cuda:{gpu}" for gpu in range(first, first + cfg.gpu_workers)]
        self.workers = [
            SentenceTransformersEmbedder(replace(cfg, device=target, gpu_workers=1))
            for target in self.devices
        ]
        for target, worker in zip(self.devices, self.workers):
            if next(worker.model.parameters()).device != torch.device(target):
                raise RuntimeError(f"Embedding replica did not load on {target}")
        self.tokenizer = self.workers[0].tokenizer
        self.pool = ThreadPoolExecutor(max_workers=cfg.gpu_workers,
                                       thread_name_prefix="document-embedding")
        self.lock = Lock()
        self.closed = False
        self.stats = {}

    def _embed_worker(self, worker, texts):
        device = self.devices[worker]
        with self.torch.cuda.device(device):
            try:
                return self.workers[worker].embed(texts)
            finally:
                # Drain device work on both success and failure before reusing a replica.
                self.torch.cuda.synchronize(device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self.lock:
            if self.closed:
                raise RuntimeError("Embedding workers are closed")
            self.stats = {}
            lengths = ([len(ids) for ids in self.tokenizer(
                texts, truncation=True, max_length=self.cfg.max_length)["input_ids"]]
                if texts else [])
            if len(lengths) != len(texts):
                raise RuntimeError("Tokenizer returned an inconsistent number of inputs")
            groups, loads = [[] for _ in self.workers], [0] * len(self.workers)
            for index in sorted(range(len(texts)), key=lambda i: -lengths[i]):
                worker = min(range(len(groups)), key=lambda i: (loads[i], len(groups[i]), i))
                groups[worker].append(index)
                loads[worker] += lengths[index]
            pending, output = [], [None] * len(texts)
            try:
                for worker, indices in enumerate(groups):
                    if indices:
                        pending.append((indices, self.pool.submit(
                            self._embed_worker, worker, [texts[i] for i in indices])))
                for indices, future in pending:
                    vectors = future.result()
                    if len(vectors) != len(indices):
                        raise RuntimeError("Embedding worker returned an inconsistent vector count")
                    for index, vector in zip(indices, vectors):
                        output[index] = vector
                if any(vector is None for vector in output):
                    raise RuntimeError("Embedding worker omitted a vector")
            finally:
                # A failed request must not leave another replica running into the next call.
                wait([future for _, future in pending])
            self.stats = {"devices": self.devices, "chunks_per_gpu": list(map(len, groups)),
                          "tokens_per_gpu": loads, "assignment": "current_input_token_balance"}
            return output

    def count_tokens(self, texts: list[str]) -> int:
        return self.workers[0].count_tokens(texts)

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.pool.shutdown(wait=True)
                self.workers.clear()


def make_embedder(cfg):
    if cfg.backend not in BACKENDS:
        raise ValueError(f"unknown embed backend {cfg.backend!r}; have {sorted(BACKENDS)}")
    if type(cfg.gpu_workers) is not int or cfg.gpu_workers < 1:
        raise ValueError("embed.gpu_workers must be a positive integer")
    if cfg.gpu_workers > 1:
        if cfg.backend != "sentence_transformers":
            raise ValueError("Multiple embedding workers require sentence_transformers")
        return ParallelSentenceTransformersEmbedder(cfg)
    return BACKENDS[cfg.backend](cfg)
