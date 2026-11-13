# DATA 266 Lab 2 Part 1 — Multimodal RAG

Public repo for the Kaggle multimodal RAG lab (DOC1/DOC2/DOC3).

## What this builds
- PDF ingestion with text chunks + figure/table page renders
- Hybrid retrieval: dense embeddings + BM25 + CLIP for figures
- Exact figure/table forcing when a label is named in the question
- Conversation memory (never written into the vector DB)
- Status policy: `answered`, `not_found`, `conflict`, `false_premise`, `session_override`
- End-to-end `submission.csv` generation

## Local run (Windows / GPU)
```powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
python -m rag.ingest
python -m rag.index --  # or: python -c "from rag.index import MultimodalIndex; MultimodalIndex(rebuild=True)"
python -m rag.run_submission --rebuild-index
python data_export/validate_submission.py submission.csv data_export/Questions.csv
```

## Kaggle
1. Attach competition data under `/kaggle/input/`
2. Copy this repo into the notebook (or `%cd` to it)
3. Enable GPU
4. Run `python notebooks/kaggle_p1.py`
5. Submit `/kaggle/working/submission.csv`
6. Tag final notebook `p1-v1.0`

## Notes
- Appendix C (DOC3 p7) is low-trust and excluded from evidence.
- Do not hard-code question IDs.
- `.env` is gitignored (GitHub token for local push only).
