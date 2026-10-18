# Failure log

| Date | Module | What broke | Root cause | Fix |
|------|--------|-----------|-----------|-----|
| 2026-10-16 | 5 | Retrieval hit@5 is only 0.12 on 129 `numeric_lookup` questions (hybrid, no filter); narrative is 1.00 | Numeric facts are answered from XBRL/SQL, not from prose chunks, so scoring them on text retrieval measures the wrong thing. They also swamp the overall average (129 of 154 questions) | Report text-retrieval metrics on narrative/table/comparison only; score numeric questions on the SQL path (exact match) in Module 7/8 |
| 2026-10-17 | 6 | Groq returned 404 `model_not_found` for `meta-llama/llama-4-scout-17b-16e-instruct` | Preview models are retired at short notice; `qwen/qwen3-32b` was also gone and is text-only | Switched to `qwen/qwen3.8-27b` (multimodal). Check `GET https://api.groq.com/openai/v1/models` when a model 404s |
| 2026-10-17 | 6 | Visual question file placed in `data/eval/` would have been loaded by `run_eval.py` as text questions | `run_eval.py` globs `data/eval/*.jsonl` | Moved to `data/eval/visual/` |

## Visual retrieval vs OCR baseline (2026-10-17, 20 hand-labelled slide questions, AMD + NVIDIA decks)

| retriever | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| ocr-bm25 (PDF text layer) | 0.80 | 0.95 | 0.95 | 0.87 |
| vidore/colSmol-256M | 0.70 | 0.95 | 0.95 | 0.81 |

Reading: no visual advantage on these decks. Both decks keep a text layer, so chart numbers are searchable text and BM25 is strong. 20 questions is also too few to call a 0.1 difference in hit@1 (2 questions). Both systems miss v01 ("revenue chart Q2 2025 vs Q2 2026", gold p.6): that slide has almost no text and revenue appears on 4 other slides, so the question is ambiguous by my design. Kept as is; changing a gold label after seeing results would be leakage. A fair test of visual retrieval needs image-only slides (decks without a text layer).

## Text retrieval: hybrid vs hybrid+rerank (2026-10-19, 25 non-numeric questions, no oracle filter)

| config | hit@1 | hit@5 | hit@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| hybrid | 0.68 | 0.74 | 0.74 | 0.704 | 0.559 |
| hybrid+rerank (bge-reranker-v2-m3, 30 candidates) | 0.72 | 0.74 | 0.76 | 0.729 | 0.677 |

Reading: reranking improves ordering (nDCG +0.12) but not recall (hit@5 unchanged). The same 6 questions miss before and after (segment-table and two-company comparison questions): the gold chunk is not among the 30 hybrid candidates, so the reranker cannot recover it. n=25, so one question = 4 points; treat as directional. Narrative questions are already at 1.00 hit@5 (no headroom). The rerank ms/query is not a real latency (scores came from the on-disk cache). The 154-question rerank run was abandoned (CPU too slow with v2-m3); numeric lookups belong to the SQL path (see the first row of the table above).

Next: fix recall for comparison questions (per-company retrieval) and table questions (SQL / table-aware chunks), not the ranker.
