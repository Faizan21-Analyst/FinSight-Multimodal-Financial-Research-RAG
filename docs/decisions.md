# Decisions (ADR log)

## 001: ColPali/ColQwen for page retrieval instead of CLIP/SigLIP
CLIP-style models are weak on dense document pages. ColPali is built for them. Verify with an ablation in Module 6.

## 002: XBRL is the source of truth for numbers
PDF table extraction is lossy. Narrative text and charts are cross-checked against XBRL.

## 003: EDGAR as the primary data source
Official, free, reproducible, and provides structured XBRL data alongside filings.