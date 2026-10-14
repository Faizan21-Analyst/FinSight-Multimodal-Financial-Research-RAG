# Retrieval evaluation

- embedding model: BAAI/bge-small-en-v1.5
- collection: finsight_text__bge-small-en-v1.5 (7027 points)
- chunks in corpus: 7027
- reranker: -
- hybrid: RRF k=60, 30 dense + 30 sparse candidates
- question files: narrative.jsonl, numeric.jsonl
- run time: 60s
- questions evaluated: 154 (dropped, no gold in corpus: 0)

## Overall: Unfiltered (question text only)

| config | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|---|
| dense | 0.114 | 0.146 | 0.172 | 0.227 | 0.141 | 0.115 | 46 |
| sparse | 0.107 | 0.136 | 0.143 | 0.201 | 0.131 | 0.108 | 14 |
| hybrid | 0.175 | 0.195 | 0.218 | 0.250 | 0.191 | 0.141 | 58 |

## Overall: With oracle filter (company / fiscal year / form known)

| config | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|---|
| dense | 0.114 | 0.179 | 0.192 | 0.289 | 0.155 | 0.123 | 43 |
| sparse | 0.127 | 0.149 | 0.201 | 0.260 | 0.157 | 0.124 | 12 |
| hybrid | 0.146 | 0.214 | 0.282 | 0.328 | 0.196 | 0.156 | 59 |
| dense+split | 0.114 | 0.185 | 0.198 | 0.289 | 0.158 | 0.124 | 43 |
| sparse+split | 0.127 | 0.149 | 0.201 | 0.260 | 0.157 | 0.125 | 12 |
| hybrid+split | 0.149 | 0.218 | 0.282 | 0.331 | 0.199 | 0.157 | 56 |

## By category (unfiltered): hit@5 / MRR@10

| category | n | dense | sparse | hybrid |
|---|---|---|---|---|
| comparison | 5 | 0.10 / 0.11 | 0.20 / 0.13 | 0.30 / 0.22 |
| narrative | 15 | 1.00 / 0.95 | 0.87 / 0.80 | 1.00 / 1.00 |
| numeric_lookup | 129 | 0.07 / 0.04 | 0.05 / 0.05 | 0.12 / 0.09 |
| table_segment | 5 | 0.40 / 0.40 | 0.20 / 0.23 | 0.40 / 0.30 |

## By category (oracle filter): hit@5 / MRR@10

| category | n | dense | sparse | hybrid | dense+split | sparse+split | hybrid+split |
|---|---|---|---|---|---|---|---|
| comparison | 5 | 0.10 / 0.12 | 0.20 / 0.13 | 0.30 / 0.17 | 0.30 / 0.20 | 0.20 / 0.15 | 0.30 / 0.26 |
| narrative | 15 | 1.00 / 0.95 | 0.93 / 0.93 | 1.00 / 1.00 | 1.00 / 0.95 | 0.93 / 0.93 | 1.00 / 1.00 |
| numeric_lookup | 129 | 0.09 / 0.05 | 0.12 / 0.06 | 0.19 / 0.10 | 0.09 / 0.05 | 0.12 / 0.06 | 0.19 / 0.10 |
| table_segment | 5 | 0.40 / 0.40 | 0.20 / 0.23 | 0.40 / 0.30 | 0.40 / 0.40 | 0.20 / 0.23 | 0.40 / 0.30 |

## Misses in the top 10: hybrid (unfiltered), 116 questions

- `nvda-segments`: How is NVIDIA's revenue split between Compute & Networking and Graphics?
- `nvda-market-platform`: NVIDIA revenue by market platform: Data Center, Gaming, Professional Visualization, Automotive
- `amd-segments`: What are AMD's reportable segments and their revenue?
- `cmp-rd-nvda-amd`: Compare NVIDIA's and AMD's research and development spending in fiscal 2024.
- `cmp-datacenter-nvda-amd`: Which has higher data center revenue, NVIDIA or AMD?
- `cmp-export-nvda-amd`: Compare the export-control risks NVIDIA and AMD describe.
- `cmp-opinc-aapl-tsla`: Compare Apple's and Tesla's operating income.
- `num-AAPL-net_income-fy2021`: What was Apple's net income in fiscal 2021?
- `num-AAPL-net_income-fy2022`: What was Apple's net income in fiscal 2022?
- `num-AAPL-net_income-fy2023`: What was Apple's net income in fiscal 2023?
- `num-AAPL-net_income-fy2024`: What was Apple's net income in fiscal 2024?
- `num-AAPL-net_income-fy2025`: What was Apple's net income in fiscal 2025?
- `num-AAPL-operating_income-fy2021`: What was Apple's operating income in fiscal 2021?
- `num-AAPL-operating_income-fy2022`: What was Apple's operating income in fiscal 2022?
- `num-AAPL-operating_income-fy2023`: What was Apple's operating income in fiscal 2023?
- `num-AAPL-operating_income-fy2024`: What was Apple's operating income in fiscal 2024?
- `num-AAPL-operating_income-fy2025`: What was Apple's operating income in fiscal 2025?
- `num-AAPL-rd_expense-fy2022`: What was Apple's research and development expense in fiscal 2022?
- `num-AAPL-rd_expense-fy2023`: What was Apple's research and development expense in fiscal 2023?
- `num-AAPL-rd_expense-fy2024`: What was Apple's research and development expense in fiscal 2024?
- ... and 96 more (see the JSON file)
