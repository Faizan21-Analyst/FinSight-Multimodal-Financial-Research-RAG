# Retrieval evaluation

- embedding model: BAAI/bge-small-en-v1.5
- collection: finsight_text__bge-small-en-v1.5 (7027 points)
- chunks in corpus: 7027
- reranker: -
- hybrid: RRF k=60, 30 dense + 30 sparse candidates
- question files: narrative.jsonl
- run time: 48s
- questions evaluated: 25 (dropped, no gold in corpus: 0)

## Overall: Unfiltered (question text only)

| config | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|---|
| dense | 0.660 | 0.660 | 0.700 | 0.720 | 0.670 | 0.531 | 1391 |
| sparse | 0.400 | 0.440 | 0.460 | 0.580 | 0.442 | 0.388 | 23 |
| hybrid | 0.580 | 0.660 | 0.660 | 0.720 | 0.629 | 0.493 | 98 |

## Overall: With oracle filter (company / fiscal year / form known)

| config | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|---|
| dense | 0.660 | 0.660 | 0.700 | 0.740 | 0.672 | 0.543 | 69 |
| sparse | 0.560 | 0.560 | 0.580 | 0.660 | 0.574 | 0.515 | 18 |
| hybrid | 0.660 | 0.720 | 0.720 | 0.720 | 0.690 | 0.592 | 89 |

## By category (unfiltered): hit@5 / MRR@10

| category | n | dense | sparse | hybrid |
|---|---|---|---|---|
| comparison | 5 | 0.10 / 0.11 | 0.10 / 0.02 | 0.10 / 0.11 |
| narrative | 15 | 1.00 / 0.95 | 0.73 / 0.72 | 1.00 / 0.97 |
| table_segment | 5 | 0.40 / 0.40 | 0.00 / 0.03 | 0.20 / 0.13 |

## Misses in the top 10: dense (unfiltered), 7 questions

- `nvda-segments`: How is NVIDIA's revenue split between Compute & Networking and Graphics?
- `nvda-market-platform`: NVIDIA revenue by market platform: Data Center, Gaming, Professional Visualization, Automotive
- `amd-segments`: What are AMD's reportable segments and their revenue?
- `cmp-rd-nvda-amd`: Compare NVIDIA's and AMD's research and development spending in fiscal 2024.
- `cmp-datacenter-nvda-amd`: Which has higher data center revenue, NVIDIA or AMD?
- `cmp-opinc-aapl-tsla`: Compare Apple's and Tesla's operating income.
- `cmp-supply-nvda-tsla`: How do NVIDIA and Tesla describe their supply chain risks?
