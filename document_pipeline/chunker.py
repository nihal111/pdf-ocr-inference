"""Markdown-aware chunking of the OCR text.

Blocks are separated by blank lines. Consecutive blocks are packed until the next one would
push the chunk past `max_tokens` (counted with the embedding tokenizer). A chunk that doesn't
start with a heading is prefixed with the most recent heading, so it embeds with its section
context. A single block longer than the budget is split on token boundaries. The output is a
pure function of the text, so chunk IDs and order are stable across backends.
"""
from __future__ import annotations

import os
import re

_HEADING = re.compile(r"^#{1,6}\s+\S")


class MarkdownChunker:

    def __init__(self, cfg):
        from transformers import AutoTokenizer
        self.max_tokens = cfg.max_tokens
        self.tok = AutoTokenizer.from_pretrained(os.path.expanduser(cfg.tokenizer))

    def _n(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _split_long(self, text: str, budget: int) -> list[str]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return [self.tok.decode(ids[i:i + budget]) for i in range(0, len(ids), budget)]

    def chunk(self, text: str) -> list[str]:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        chunks: list[str] = []
        heading = None
        cur: list[str] = []
        cur_tokens = 0

        def flush():
            nonlocal cur, cur_tokens
            if cur:
                body = "\n\n".join(cur)
                if heading and not _HEADING.match(cur[0]):
                    body = f"{heading}\n\n{body}"
                chunks.append(body)
            cur, cur_tokens = [], 0

        for block in blocks:
            if _HEADING.match(block):
                flush()
                heading = block.splitlines()[0]
            n = self._n(block)
            prefix = self._n(heading) + 2 if heading and not _HEADING.match(block) else 0
            if n + prefix > self.max_tokens:
                flush()
                for piece in self._split_long(block, self.max_tokens - prefix):
                    cur, cur_tokens = [piece], 0
                    flush()
                continue
            if cur and cur_tokens + n + prefix > self.max_tokens:
                flush()
            cur.append(block)
            cur_tokens += n
        flush()
        return chunks or [text]
