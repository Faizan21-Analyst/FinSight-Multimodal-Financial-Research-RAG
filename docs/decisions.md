# Decisions (ADR log)

## 001: ColPali/ColQwen for page retrieval instead of CLIP/SigLIP
CLIP-style models are weak on dense document pages. ColPali is built for them. Verify with an ablation in Module 6.

## 002: XBRL is the source of truth for numbers
PDF table extraction is lossy. Narrative text and charts are cross-checked against XBRL.

## 003: EDGAR as the primary data source
Official, free, reproducible, and provides structured XBRL data alongside filings.

## 004: RRF is fused client-side (tunable and testable) rather than with Qdrant’s server-side fusion.

## 005: BM25 sparse vectors via hashed terms plus Qdrant’s IDF modifier, instead of bge-m3’s learned sparse. No model is needed, exact numbers match, and there is no vocabulary file.

## 006: One collection per embedding model, so Module 5 can compare models without rebuilding.

## 007: Chunks are embedded with a “ticker | form | FY | section” header, but stored clean.

## 008: Groq (free tier) is the LLM/VLM provider instead of Anthropic
Cost. Text generation and chart reading both use Qwen 3.8 27B (`qwen/qwen3.8-27b`, a multimodal preview model), because Groq's earlier vision model (Llama 4 Scout) and Qwen3-32B are no longer listed. Instruct mode (`reasoning_effort: none`) is used for extraction and JSON output. Called over plain httpx (OpenAI-compatible API), no SDK, with disk caching and 429 backoff because the free tier is rate limited. Trade-off: weaker chart reading than frontier models, so extracted values carry a `values_are_estimates` flag and are always checked against XBRL in Module 10.