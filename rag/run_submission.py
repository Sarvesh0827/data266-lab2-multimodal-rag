from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from rag.answer_engine import StructuredAnswerer
from rag.config import ARTIFACTS, DATA_DIR, OUTPUT_DIR
from rag.extract_kb import build_kb
from rag.memory import MemoryBank

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

COLS = ["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"]


def run(questions_path: Path | None = None, out_path: Path | None = None, rebuild_index: bool = False) -> Path:
    # rebuild_index kept for CLI compatibility with Kaggle notebook; structured path does not need it.
    _ = rebuild_index
    qpath = questions_path or (DATA_DIR / "Questions.csv")
    out = out_path or (OUTPUT_DIR / "submission.csv")

    questions = pd.read_csv(qpath, dtype=str, keep_default_na=False)
    print(f"Loaded {len(questions)} questions from {qpath}", flush=True)

    print("Building auto-extracted KB (tables + OCR)...", flush=True)
    # Reuse figure OCR if present; always refresh table/claim facts from PDFs.
    prev_figs = []
    prev = ARTIFACTS / "kb_facts.json"
    if prev.exists():
        prev_figs = json.loads(prev.read_text(encoding="utf-8")).get("figures", [])
    do_ocr = len(prev_figs) == 0
    kb = build_kb(force=True, do_ocr=do_ocr)
    if prev_figs and not kb.get("figures"):
        kb["figures"] = prev_figs
        # Still parse A-6/A-7 bar labels when reusing cached figure OCR blobs.
        try:
            import easyocr
            from dataclasses import asdict
            from rag.extract_kb import (
                NumericFact,
                enrich_b4_cpi_facts,
                enrich_euro_area_bar_facts,
                enrich_stock_index_facts,
            )
            from rag.ingest import extract_knowledge_base

            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            _, assets = extract_knowledge_base(force=False)
            facts = [NumericFact(**f) for f in kb["numeric_facts"]]
            enrich_euro_area_bar_facts(facts, assets, reader)
            enrich_stock_index_facts(facts, assets, reader)
            enrich_b4_cpi_facts(facts, assets, reader)
            kb["numeric_facts"] = [asdict(f) for f in facts]
        except Exception as e:
            print("Bar-chart OCR enrich skipped:", e, flush=True)
        (ARTIFACTS / "kb_facts.json").write_text(json.dumps(kb, indent=2), encoding="utf-8")

    answerer = StructuredAnswerer()
    memory_bank = MemoryBank()

    rows = []
    t0 = time.time()
    for _, row in questions.iterrows():
        conv = row["conversation_id"]
        qid = row["question_id"]
        question = row["question"]
        mem = memory_bank.get(conv)

        t_q = time.time()
        result = answerer.answer(question, mem)
        mem.add_turn(qid, question, result["answer"], result["status"])

        rows.append(
            {
                "id": row["id"],
                "conversation_id": conv,
                "question_id": qid,
                "answer": result["answer"].replace("\n", " ").strip(),
                "figure_references": result["figure_references"],
                "citations": result["citations"],
                "status": result["status"],
            }
        )
        print(
            f"[{row['id']}/{len(questions)}] {qid} status={result['status']} "
            f"figs={result['figure_references']} ({time.time() - t_q:.2f}s)",
            flush=True,
        )

    sub = pd.DataFrame(rows, columns=COLS)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"Wrote {out} in {time.time() - t0:.1f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args()
    run(
        questions_path=Path(args.questions) if args.questions else None,
        out_path=Path(args.out) if args.out else None,
        rebuild_index=args.rebuild_index,
    )


if __name__ == "__main__":
    main()
