"""BM25-style sparse vectors for Qdrant. Pure functions, no model download.

Why not just embeddings? Financial questions hinge on exact tokens: "$22,563", "H20",
"Item 1A", "ASU 2023-08". Dense models blur those; lexical matching nails them. Hybrid
retrieval (dense + sparse, fused with RRF) gets both.

How it works:
  * Documents -> a sparse vector of BM25 term-frequency weights (saturating, length-normalised).
  * Queries   -> weight 1.0 per unique term.
  * Qdrant collection uses Modifier.IDF, so the server multiplies in each term's inverse
    document frequency at query time. Score = BM25.
  * Terms are hashed to u32 indices (crc32), so there is no vocabulary file to maintain.
    Collisions are possible but rare (~1 colliding pair expected per 100k distinct terms).

Numbers are normalised so "$22,563", "22,563" and "22563" are the same term.
"""
from __future__ import annotations

import re
import zlib
from collections import Counter

# number (optional thousands commas / decimals) OR word. Trailing "%" and "$" are ignored.
_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?|[a-z][a-z0-9&'\-]*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with "
    # question words carry no topical signal but are in nearly every natural-language query
    "how what why which who when where does did do can could would should about between".split()
)

K1 = 1.2
B = 0.75
AVG_DOC_LEN = 200.0  # fixed, as in Qdrant's own BM25 recipe; avoids a corpus-wide first pass


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _TOKEN.finditer(text.lower()):
        tok = m.group(0)
        if tok[0].isdigit():
            tok = tok.replace(",", "")
        else:
            tok = tok.strip("'-&")
            if tok.endswith("'s"):  # possessive: "nvidia's" must match "nvidia"
                tok = tok[:-2]
            if len(tok) < 2 or tok in STOPWORDS:
                continue
        tokens.append(tok)
    return tokens


def term_index(token: str) -> int:
    """Stable u32 index for a term (Qdrant sparse indices are unsigned 32-bit)."""
    return zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF


def document_vector(
    text: str, k1: float = K1, b: float = B, avg_len: float = AVG_DOC_LEN
) -> tuple[list[int], list[float]]:
    """BM25 term weights: tf*(k1+1) / (tf + k1*(1 - b + b*len/avg_len))."""
    tokens = tokenize(text)
    if not tokens:
        return [], []
    norm = k1 * (1 - b + b * len(tokens) / avg_len)
    weights: dict[int, float] = {}
    for tok, tf in Counter(tokens).items():
        idx = term_index(tok)
        weights[idx] = weights.get(idx, 0.0) + tf * (k1 + 1) / (tf + norm)
    indices = sorted(weights)
    return indices, [weights[i] for i in indices]


def query_vector(text: str) -> tuple[list[int], list[float]]:
    indices = sorted({term_index(t) for t in tokenize(text)})
    return indices, [1.0] * len(indices)