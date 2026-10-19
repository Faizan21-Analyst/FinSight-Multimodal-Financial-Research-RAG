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

## Module 9: corrective agent loop (2026-10-19)

Question: "Compare NVIDIA and AMD research and development expense in fiscal 2024".

| run | evidence | grade | rounds | result |
|---|---|---|---|---|
| text only, heuristic grader | 7 text chunks, none with an R&D figure | 0.70 (heuristic only checks that both companies appear) | 1 | wrong: looks sufficient, is not |
| text only, Groq grader | same 7 chunks | 0.40 ("missing NVIDIA R&D for FY2024") | 3 | identical evidence every round: retries wasted time and API calls |
| text + SQL (`--sql`) | XBRL facts pinned first: NVDA $8.675B, AMD $6.456B | 1.00 | 1 | correct; answer quotes both exactly and cites [E1]/[E2], no arithmetic of its own |

Findings:
- Per-company sub-queries retrieve both companies (8 chunks each) but the income-statement figure still is not among them. Recall for table/number questions is a retrieval problem the loop cannot fix; the SQL path is the fix for consolidated metrics.
- Bug: a grader rewrite naming NVIDIA was also run for AMD's sub-query. Fixed: a rewrite only replaces the wording of the company it names.
- Bug: the loop kept retrying when a retry added no new evidence. Fixed: stop when a round adds nothing new (`new_items == 0`).
- Heuristic grader is only a fallback for when the LLM fails; it cannot see whether the figure is present.
- Groq free tier caps `qwen/qwen3.8-27b` at 1000 output tokens/minute; `max_tokens` lowered to 900 (text and VLM) and a "request too large" 429 now fails fast instead of retrying.
- Still open: 6 of 8 evidence items in the winning run are irrelevant text chunks (harmless because SQL is pinned first, but a risk for questions with no SQL answer). Reranking is not wired into the agent.

## Module 10: cross-modal verification (2026-10-20)

Verifier: `finsight/agents/cross_modal.py`; demo: `scripts/verify_chart.py`. Consolidated metrics are checked against XBRL (can say contradicted); everything else against 10-K/10-Q text (can only say supported or unverified).

| deck / slide | points | supported | contradicted | unverified | path |
|---|---|---|---|---|---|
| AMD Q2 FY26, p.12 segment table | 18 | 18 | 0 | 0 | text |
| NVDA Q2 FY27 CFO commentary, p.2 market platform | 15 | 15 | 0 | 0 | 12 text, 3 XBRL |
| NVDA p.2 with Total Q2 FY27 doctored to 99,000 and Data Center Q2 FY27 to 95,000 | 15 | 13 | 1 (Total, +2.9% vs XBRL) | 1 (Data Center) | |

Bugs found by real runs (each now has a regression test):

| symptom | root cause | fix |
|---|---|---|
| AMD Embedded Q2'25 chart 824 "supported" by 823 | 0.5% tolerance floor and first-match-in-range; 823 is the Q1'25 figure from another chunk | tolerance = half a unit of the last printed digit, floor 0.1%; take the closest number across all chunks, not the first in range |
| AMD Client & Gaming Q2'25 chart 3,621 "supported" by 3,603 | same | same |
| AMD Data Center Q2'25 operating loss (-155) unverified | text scan ignored sign, so a loss printed as "$155 million" never matched a negative value | compare by magnitude; note says "sign not checked" |
| NVDA Total row unverified (3 points) | a series named "Total" has no distinctive name for text matching, and no metric was inferred | a Total/Consolidated series is mapped to the metric named in the chart or slide text (only if unambiguous) and checked against XBRL |
| NVDA Q2 FY26 segment values unverified (3 points) | text search was limited to FY2026 filings, but NVIDIA recast prior-year market platforms in Q1 FY27, so the figures appear only in the FY2027 10-Q | search the label's year, the next fiscal year and the deck's year |

Known limits:
- A doctored segment number is reported as unverified, not contradicted: absence from retrieved text does not prove a different value. A period-aware parse of the 10-Q segment table could close this; it needs care to avoid false "contradicted".
- Text support means the figure appears next to the series name in the filing for that fiscal year, not for exactly that quarter.
- Percent series (growth labels such as "+107%") are not verified.
- Only two real decks tested, both with a text layer; visual retrieval was not exercised.

## Module 11: claim and citation verification, confidence (2026-10-20)

Verifier: `finsight/generation/claim_verifier.py` (+ `claims.py`, `confidence.py`); wired into the agent as a `verify` step. Citations and numbers are checked by rules; only the meaning of a claim whose numbers passed goes to the LLM judge (one batched call).

Same question and evidence as Module 9 (`--sql --verify` / `--answer-text`), 4 runs:

| answer verified | what it contained | result | confidence |
|---|---|---|---|
| the model's own answer | 9 claims: figures, periods, filing dates, an AMD text claim cited to [E4], a comparison | 9/9 supported (llm) | high |
| NVIDIA R&D "$9.5 billion [E1]" (true: $8.675B) | wrong number | claim: `number_not_in_evidence` (rules, no LLM call); AMD claim supported; overall partially_supported | low |
| NVIDIA "$8.675 billion [E2]" (E2 is AMD's fact) | right number, wrong citation | claim: `miscited`, partial (the figure exists in E1, not E2); 1/2 supported | low |
| "$8.675B [E1], lower than AMD at $6.456B [E2]" | right numbers, false comparison | numbers pass, LLM judge: contradicted ("NVIDIA R&D is higher than AMD, not lower") | low |

What the real run settled: the AMD claim from Module 9 that I could not see ($6.5B, +10% from $5.9B, cited [E4]) is supported: all three figures are in the full text of E4. The judge matched written dates ("January 30, 2023") to the ISO dates in the XBRL evidence.

No defect was found by the real runs. The only test failures were two stale assertions in `test_agent.py` (trace now ends `answer, verify`).

Known limits:
- The judge sees only the first 700 characters of each cited chunk, so a qualitative clause beyond that (the AMD claim's "due to higher employee-related costs") may not be checked; the numbers are checked against the full chunk. The judge's stated reason for that claim covers the numbers only.
- Judge and answerer are the same model family (Qwen), so their errors can be correlated. The false-comparison case is n=1: not evidence the judge is reliable on comparison wording. A deterministic check of "higher/lower/more than" against the cited numbers is the fallback if it proves lenient.
- A number match is by value in the cited item, not by entity or period: the same figure for the wrong company in the same chunk would pass the number check (the judge should catch it).
- A sentence holding two assertions is one claim. Splitting is heuristic (abbreviations handled, not all cases).
- Derived figures (differences, growth) are flagged `number_not_in_evidence` rather than recomputed; recomputing them through `numeric.py` would turn those flags into checks.
- Confidence is a rule set, not a calibrated probability; it has not been evaluated against labelled outcomes (Module 13).
