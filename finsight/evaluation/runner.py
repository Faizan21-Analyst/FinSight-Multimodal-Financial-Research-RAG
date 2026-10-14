"""Run retrieval configurations over an evaluation set and summarise the results.

Every config is scored on the SAME questions with the SAME gold, so differences
between rows are caused by the retrieval method alone. That is what makes it
an ablation.

Configs: mode in {dense, sparse, hybrid, hybrid+rerank} x oracle filter {off, on}.
- oracle off: the retriever sees only the question text.
- oracle on: it also gets the ticker / fiscal year / form a perfect
  query-understanding step would extract.
- split: with oracle enabled, run one search per named company and interleave
  results.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Protocol

from finsight.core.schemas import EvidenceItem
from finsight.evaluation.dataset import OracleFilter, Question, resolve_gold
from finsight.evaluation.metrics import (
    first_rank,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
)
from finsight.retrieval.text_retriever import SearchFilter


METRIC_KEYS = (
    "hit1",
    "hit3",
    "hit5",
    "hit10",
    "mrr",
    "ndcg",
    "latency_ms",
)

HEADERS = (
    "hit@1",
    "hit@3",
    "hit@5",
    "hit@10",
    "MRR@10",
    "nDCG@10",
    "ms/query",
)


class Backend(Protocol):
    def search(
        self,
        query: str,
        mode: str = ...,
        limit: int = ...,
        filt: SearchFilter | None = ...,
    ) -> list[EvidenceItem]: ...


@dataclass(frozen=True)
class EvalConfig:
    """mode: dense | sparse | hybrid | hybrid+rerank.

    oracle: also apply the question's oracle filter (company / year / form).

    split: with oracle, run ONE search per named company and interleave the
    results. This is the ceiling for entity-decomposed retrieval on comparison
    questions; it equals plain oracle on single-company questions.
    """

    mode: str
    oracle: bool = False
    split: bool = False

    def __post_init__(self):
        if self.split and not self.oracle:
            raise ValueError("split needs the oracle filter (it uses the company list)")

    @property
    def label(self) -> str:
        return f"{self.mode}{'+split' if self.split else ''}"

    @property
    def name(self) -> str:
        return f"{self.label}{' +oracle' if self.oracle else ''}"


def interleave(
    lists: list[list[EvidenceItem]],
    k: int,
) -> list[EvidenceItem]:
    """Round-robin merge, dropping duplicates, so every list is represented near the top."""
    out: list[EvidenceItem] = []
    seen: set[str] = set()

    for rank in range(max((len(items) for items in lists), default=0)):
        for items in lists:
            if (
                rank < len(items)
                and items[rank].evidence_id not in seen
            ):
                seen.add(items[rank].evidence_id)
                out.append(items[rank])

    return out[:k]


@dataclass
class Record:
    question_id: str
    category: str
    question: str
    mode: str
    oracle: bool
    hit1: float
    hit3: float
    hit5: float
    hit10: float
    mrr: float
    ndcg: float
    latency_ms: float
    first_ranks: list[int | None]  # per gold spec; None = not found in top 10


@dataclass
class EvalResult:
    records: list[Record] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)
    n_questions: int = 0
    gold_sizes: dict[str, list[int]] = field(default_factory=dict)


def to_filter(o: OracleFilter | None) -> SearchFilter | None:
    if o is None:
        return None

    return SearchFilter(
        tickers=tuple(o.tickers) if o.tickers else None,
        fiscal_years=tuple(o.fiscal_years) if o.fiscal_years else None,
        doc_types=tuple(o.doc_types) if o.doc_types else None,
    )


def evaluate(
    questions: list[Question],
    corpus: list[dict],
    backend: Backend,
    configs: list[EvalConfig],
    k: int = 10,
    progress=None,
) -> EvalResult:
    k = max(k, 10)
    result = EvalResult()

    usable: list[tuple[Question, list[set[str]]]] = []

    for q in questions:
        sets = [resolve_gold(corpus, spec) for spec in q.gold]
        empty = [i for i, gold_set in enumerate(sets) if not gold_set]

        if empty:
            result.dropped.append(
                (q.id, f"no gold chunk in corpus for spec(s) {empty}")
            )
            continue

        result.gold_sizes[q.id] = [len(gold_set) for gold_set in sets]
        usable.append((q, sets))

    result.n_questions = len(usable)

    for cfg in configs:
        for n, (q, sets) in enumerate(usable, 1):
            filt = to_filter(q.oracle) if cfg.oracle else None

            t0 = time.perf_counter()

            if (
                cfg.split
                and q.oracle
                and q.oracle.tickers
                and len(q.oracle.tickers) > 1
            ):
                per_company = [
                    backend.search(
                        q.question,
                        mode=cfg.mode,
                        limit=k,
                        filt=to_filter(
                            q.oracle.model_copy(update={"tickers": [ticker]})
                        ),
                    )
                    for ticker in q.oracle.tickers
                ]

                items = interleave(per_company, k)

            else:
                items = backend.search(
                    q.question,
                    mode=cfg.mode,
                    limit=k,
                    filt=filt,
                )

            ms = (time.perf_counter() - t0) * 1000

            ids = [item.evidence_id for item in items]
            per = {metric: [] for metric in METRIC_KEYS[:-1]}
            ranks = []

            for gold in sets:
                flags = [chunk_id in gold for chunk_id in ids]

                per["hit1"].append(hit_at_k(flags, 1))
                per["hit3"].append(hit_at_k(flags, 3))
                per["hit5"].append(hit_at_k(flags, 5))
                per["hit10"].append(hit_at_k(flags, 10))
                per["mrr"].append(reciprocal_rank(flags, 10))
                per["ndcg"].append(ndcg_at_k(flags, 10, len(gold)))

                ranks.append(first_rank(flags[:10]))

            mean = {
                metric: sum(values) / len(values)
                for metric, values in per.items()
            }

            result.records.append(
                Record(
                    question_id=q.id,
                    category=q.category,
                    question=q.question,
                    mode=cfg.label,
                    oracle=cfg.oracle,
                    latency_ms=ms,
                    first_ranks=ranks,
                    **mean,
                )
            )

            if progress:
                progress(cfg, n, len(usable))

    return result


def summarize(
    records: list[Record],
    by_category: bool = False,
) -> dict:
    """Return {(mode, oracle[, category]): {metric: mean, 'n': count}}."""
    groups: dict[tuple, list[Record]] = defaultdict(list)

    for record in records:
        key = (
            (record.mode, record.oracle, record.category)
            if by_category
            else (record.mode, record.oracle)
        )
        groups[key].append(record)

    out = {}

    for key, grouped_records in groups.items():
        out[key] = {
            metric: sum(getattr(record, metric) for record in grouped_records)
            / len(grouped_records)
            for metric in METRIC_KEYS
        }
        out[key]["n"] = len(grouped_records)

    return out


def misses(
    records: list[Record],
    mode: str,
    oracle: bool,
) -> list[Record]:
    return [
        record
        for record in records
        if record.mode == mode
        and record.oracle == oracle
        and any(rank is None for rank in record.first_ranks)
    ]


def _row(label: str, stats: dict) -> str:
    values = [
        f"{stats[metric]:.3f}"
        for metric in METRIC_KEYS[:-1]
    ] + [f"{stats['latency_ms']:.0f}"]

    return f"| {label} | " + " | ".join(values) + " |"


def format_markdown(
    result: EvalResult,
    meta: dict,
    modes: list[str],
    max_misses: int = 20,
) -> str:
    recs = result.records
    overall = summarize(recs)
    bycat = summarize(recs, by_category=True)
    cats = sorted({record.category for record in recs})

    lines = ["# Retrieval evaluation", ""]
    lines += [f"- {key}: {value}" for key, value in meta.items()]
    lines += [
        f"- questions evaluated: {result.n_questions}"
        f" (dropped, no gold in corpus: {len(result.dropped)})",
        "",
    ]

    # Overall metrics, separated by oracle-filter status.
    for oracle in (False, True):
        if not any(existing_oracle == oracle for _, existing_oracle in overall):
            continue

        title = (
            "With oracle filter (company / fiscal year / form known)"
            if oracle
            else "Unfiltered (question text only)"
        )

        lines += [
            f"## Overall: {title}",
            "",
            "| config | " + " | ".join(HEADERS) + " |",
            "|---|" + "---|" * len(HEADERS),
        ]

        lines += [
            _row(mode, overall[(mode, oracle)])
            for mode in modes
            if (mode, oracle) in overall
        ]
        lines.append("")

    # Category metrics, separately for unfiltered and oracle-filtered runs.
    for oracle in (False, True):
        if not any(record.oracle == oracle for record in recs):
            continue

        shown = [
            mode
            for mode in modes
            if any((mode, oracle, category) in bycat for category in cats)
        ]

        title = "oracle filter" if oracle else "unfiltered"

        lines += [
            f"## By category ({title}): hit@5 / MRR@10",
            "",
            "| category | n | " + " | ".join(shown) + " |",
            "|---|---|" + "---|" * len(shown),
        ]

        for category in cats:
            cells = []

            for mode in shown:
                stats = bycat.get((mode, oracle, category))
                cells.append(
                    f"{stats['hit5']:.2f} / {stats['mrr']:.2f}"
                    if stats
                    else "-"
                )

            n = next(
                (
                    bycat[(mode, oracle, category)]["n"]
                    for mode in shown
                    if (mode, oracle, category) in bycat
                ),
                0,
            )

            lines.append(
                f"| {category} | {n} | " + " | ".join(cells) + " |"
            )

        lines.append("")

    # Find misses using the best unfiltered configuration by MRR.
    best = max(
        (
            mode
            for mode in modes
            if (mode, False) in overall
        ),
        key=lambda mode: overall[(mode, False)]["mrr"],
        default=None,
    )

    if best:
        missed_records = misses(recs, best, False)

        lines += [
            f"## Misses in the top 10: {best} (unfiltered), "
            f"{len(missed_records)} questions",
            "",
        ]

        lines += [
            f"- `{record.question_id}`: {record.question}"
            for record in missed_records[:max_misses]
        ]

        if len(missed_records) > max_misses:
            lines.append(
                f"- ... and {len(missed_records) - max_misses} more "
                "(see the JSON file)"
            )

        lines.append("")

    if result.dropped:
        lines += [
            "## Dropped questions (gold predicate matched nothing)",
            "",
        ]

        lines += [
            f"- `{question_id}`: {reason}"
            for question_id, reason in result.dropped
        ]

        lines.append("")

    return "\n".join(lines)


def to_jsonable(result: EvalResult) -> dict:
    return {
        "n_questions": result.n_questions,
        "dropped": result.dropped,
        "gold_sizes": result.gold_sizes,
        "records": [asdict(record) for record in result.records],
    }
