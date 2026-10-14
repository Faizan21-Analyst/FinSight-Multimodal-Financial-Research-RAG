"""Dense embedders behind one small interface.

* SentenceTransformerEmbedder : the real thing (bge-m3, bge-small-en-v1.5, ...). Lazy-loaded, so
                                importing this module never downloads a model.
* HashingEmbedder             : deterministic bag-of-words stand-in for tests and offline dev.
                                NOT semantic. Use it to test plumbing, never to judge quality.

Each model gets its own Qdrant collection (collection_name), so different models/dimensions
never collide and Module 5 can compare them side by side.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from finsight.indexing.sparse import tokenize

# BGE English v1.5 models are trained to see this prefix on QUERIES (not on documents).
# bge-m3 needs no instruction.
_BGE_QUERY = "Represent this sentence for searching relevant passages: "
QUERY_INSTRUCTIONS = {
    "bge-small-en-v1.5": _BGE_QUERY,
    "bge-base-en-v1.5": _BGE_QUERY,
    "bge-large-en-v1.5": _BGE_QUERY,
}


class DenseEmbedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def collection_name(base: str, model_name: str) -> str:
    """finsight_text + BAAI/bge-m3 -> finsight_text__bge-m3"""
    slug = re.sub(r"[^a-z0-9._-]+", "-", model_name.split("/")[-1].lower()).strip("-")
    return f"{base}__{slug}"


class HashingEmbedder:
    """Signed feature hashing of tokens into `dim` dimensions, L2-normalised."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        max_seq_length: int | None = 512,
        query_instruction: str | None = None,
    ):
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        if query_instruction is None:
            query_instruction = QUERY_INSTRUCTIONS.get(model_name.split("/")[-1].lower(), "")
        self.query_instruction = query_instruction
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, kept lazy

            self._model = SentenceTransformer(self.name, device=self.device)
            if self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length
        return self._model

    @property
    def dim(self) -> int:
        m = self.model  # sentence-transformers renamed this method; support both names
        fn = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return int(fn())

    def _encode(self, texts: list[str]) -> list[list[float]]:
        out = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return out.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_instruction + text])[0]


def build_embedder(model_name: str, **kwargs) -> DenseEmbedder:
    """'hashing' or 'hashing-128' -> HashingEmbedder; anything else -> sentence-transformers."""
    if model_name.startswith("hashing"):
        dim = int(model_name.split("-")[1]) if "-" in model_name else 64
        return HashingEmbedder(dim)
    return SentenceTransformerEmbedder(model_name, **kwargs)