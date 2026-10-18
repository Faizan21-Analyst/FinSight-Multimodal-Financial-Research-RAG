"""Evidence assembly: merge results from text, visual and SQL retrievers into one package.

Rules:
  * SQL (XBRL) facts are pinned first: they are the source of truth for numbers (ADR 002).
  * Everything else is merged with reciprocal rank fusion, so scores from different retrievers
    never need to be comparable.
  * Duplicates (same evidence_id) collapse to one item.
  * The package is cut to a character budget so the prompt stays bounded.
Each item gets a stable label [E1], [E2], ... that the answerer cites and the claim verifier
later checks against.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from finsight.core.schemas import Citation, EvidenceItem, Modality
from finsight.retrieval.fusion import reciprocal_rank_fusion


@dataclass
class EvidencePackage:
    items: list[EvidenceItem] = field(default_factory=list)

    def label(self, index: int) -> str:
        return f"E{index + 1}"

    def by_label(self) -> dict[str, EvidenceItem]:
        return {self.label(i): item for i, item in enumerate(self.items)}

    def citations(self, snippet_chars: int = 160) -> list[Citation]:
        return [
            Citation(evidence_id=i.evidence_id, source=i.source, snippet=i.content[:snippet_chars])
            for i in self.items
        ]

    def render_context(self) -> str:
        """Prompt-ready context block with a labelled header per item."""
        blocks = []
        for n, item in enumerate(self.items):
            s = item.source
            bits = [s.ticker, s.doc_type.value]
            if s.fiscal_year:
                bits.append(f"FY{s.fiscal_year}{(' ' + s.fiscal_period) if s.fiscal_period else ''}")
            if s.section:
                bits.append(s.section)
            if s.page:
                bits.append(f"p.{s.page}")
            blocks.append(f"[{self.label(n)}] {' | '.join(bits)} ({item.modality.value})\n{item.content}")
        return "\n\n".join(blocks)

    def __len__(self) -> int:
        return len(self.items)


def assemble(
    results: Mapping[str, Sequence[EvidenceItem]],
    max_items: int = 8,
    max_chars: int = 12_000,
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> EvidencePackage:
    """`results` maps a retriever name ("text", "visual", "sql") to its ranked evidence list."""
    pinned: list[EvidenceItem] = []
    pinned_ids: set[str] = set()
    for item in results.get("sql", ()):
        if item.evidence_id not in pinned_ids:
            pinned.append(item)
            pinned_ids.add(item.evidence_id)

    names = [n for n in results if n != "sql"]
    lookup: dict[str, EvidenceItem] = {}
    for n in names:
        for item in results[n]:
            lookup.setdefault(item.evidence_id, item)
    rankings = [[i.evidence_id for i in results[n]] for n in names]
    w = [(weights or {}).get(n, 1.0) for n in names]
    fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=w) if rankings else []

    ordered = list(pinned)
    for eid, score in fused:
        if eid in pinned_ids:
            continue
        ordered.append(lookup[str(eid)].model_copy(update={"score": score}))

    package, used = [], 0
    for item in ordered:
        if len(package) >= max_items:
            break
        cost = len(item.content)
        # always keep at least one item, even if it alone exceeds the budget
        if package and used + cost > max_chars:
            continue
        package.append(item)
        used += cost
    return EvidencePackage(package)


def modalities(pkg: EvidencePackage) -> set[Modality]:
    return {i.modality for i in pkg.items}
