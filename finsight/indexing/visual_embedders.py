"""Visual page embedders (late interaction / multi-vector).

ColPali-style models turn a page IMAGE into many small vectors (one per image patch) and a
query into one vector per query token. Relevance = MaxSim: for each query token take its best
matching patch, then sum. That is why they find dense slides where a single CLIP vector fails.

* ColQwenEmbedder          : the real model (ColQwen2 / ColSmol via colpali-engine). Lazy-loaded.
* HashingVisualEmbedder    : deterministic stand-in that "embeds" the page TEXT token by token.
                             Tests the plumbing (indexing, MaxSim search, filters), not quality.
Both implement the same interface, so everything downstream is model-agnostic.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from finsight.indexing.sparse import tokenize
from finsight.vision.page_renderer import PageRecord

MultiVector = list[list[float]]


class VisualEmbedder(Protocol):
    name: str
    dim: int

    def embed_pages(self, pages: Sequence[PageRecord]) -> list[MultiVector]: ...

    def embed_query(self, text: str) -> MultiVector: ...


def _drop_zero_rows(rows: list[list[float]]) -> MultiVector:
    """ColQwen zeroes padding positions; padding must not be stored as vectors."""
    kept = [r for r in rows if any(x != 0.0 for x in r)]
    return kept or rows[:1]


class HashingVisualEmbedder:
    """One sparse-ish signed vector per distinct token. MaxSim then behaves like token overlap."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = f"hashing-visual-{dim}"

    def _tok_vec(self, tok: str) -> list[float]:
        v = [0.0] * self.dim
        digest = hashlib.blake2b(tok.encode(), digest_size=12).digest()
        for j in range(3):
            pos = int.from_bytes(digest[j * 4 : j * 4 + 4], "big")
            v[pos % self.dim] += 1.0 if (pos >> 31) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _multi(self, text: str) -> MultiVector:
        toks = list(dict.fromkeys(tokenize(text))) or ["<empty>"]
        return [self._tok_vec(t) for t in toks]

    def embed_pages(self, pages: Sequence[PageRecord]) -> list[MultiVector]:
        return [self._multi(p.text) for p in pages]

    def embed_query(self, text: str) -> MultiVector:
        return self._multi(text)


class ColQwenEmbedder:
    """ColQwen2 (default, ~2B params) or a ColSmol model, through colpali-engine.

    GPU strongly recommended for ColQwen2. On a CPU-only machine try
    model_name='vidore/colSmol-256M' (tiny, ~10x faster, lower accuracy).
    """

    dim = 128

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str | None = None,
        batch_size: int = 2,
    ):
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch  # heavy imports kept lazy

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if "smol" in self.name.lower():
            from colpali_engine.models import ColIdefics3 as Model
            from colpali_engine.models import ColIdefics3Processor as Processor
        else:
            from colpali_engine.models import ColQwen2 as Model
            from colpali_engine.models import ColQwen2Processor as Processor
        self._model = Model.from_pretrained(self.name, torch_dtype=dtype, device_map=device).eval()
        self._processor = Processor.from_pretrained(self.name)

    def embed_pages(self, pages: Sequence[PageRecord]) -> list[MultiVector]:
        import torch
        from PIL import Image

        self._load()
        out: list[MultiVector] = []
        for i in range(0, len(pages), self.batch_size):
            batch = pages[i : i + self.batch_size]
            images = [Image.open(p.image_path).convert("RGB") for p in batch]
            inputs = self._processor.process_images(images).to(self._model.device)
            with torch.no_grad():
                emb = self._model(**inputs)
            out.extend(_drop_zero_rows(e.float().cpu().tolist()) for e in emb)
        return out

    def embed_query(self, text: str) -> MultiVector:
        import torch

        self._load()
        inputs = self._processor.process_queries([text]).to(self._model.device)
        with torch.no_grad():
            emb = self._model(**inputs)
        return _drop_zero_rows(emb[0].float().cpu().tolist())


def build_visual_embedder(model_name: str, **kwargs) -> VisualEmbedder:
    """'hashing' / 'hashing-128' -> test double; anything else -> ColQwen/ColSmol."""
    if model_name.startswith("hashing"):
        dim = int(model_name.split("-")[1]) if "-" in model_name else 64
        return HashingVisualEmbedder(dim)
    return ColQwenEmbedder(model_name, **kwargs)
