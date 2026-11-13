from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Unbuffered progress logs on Kaggle/Windows
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from rag.config import DATA_DIR, OUTPUT_DIR
from rag.generate import AnswerGenerator
from rag.index import MultimodalIndex
from rag.memory import MemoryBank
from rag.policy import decide_status
from rag.retrieve import HybridRetriever


COLS = ["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"]


def run(questions_path: Path | None = None, out_path: Path | None = None, rebuild_index: bool = False) -> Path:
    qpath = questions_path or (DATA_DIR / "Questions.csv")
    out = out_path or (OUTPUT_DIR / "submission.csv")

    questions = pd.read_csv(qpath, dtype=str, keep_default_na=False)
    print(f"Loaded {len(questions)} questions from {qpath}")

    index = MultimodalIndex(rebuild=rebuild_index)
    retriever = HybridRetriever(index)
    generator = AnswerGenerator()
    memory_bank = MemoryBank()

    rows = []
    t0 = time.time()
    for _, row in questions.iterrows():
        conv = row["conversation_id"]
        qid = row["question_id"]
        question = row["question"]
        mem = memory_bank.get(conv)

        t_q = time.time()
        evidence = retriever.retrieve(question)
        decision = decide_status(question, evidence, mem)
        result = generator.generate(question, evidence, mem, decision)
        mem.add_turn(qid, question, result["answer"], result["status"])

        rows.append(
            {
                "id": row["id"],
                "conversation_id": conv,
                "question_id": qid,
                "answer": result["answer"],
                "figure_references": result["figure_references"],
                "citations": result["citations"],
                "status": result["status"],
            }
        )
        print(
            f"[{row['id']}/{len(questions)}] {qid} status={result['status']} "
            f"figs={result['figure_references']} ({time.time() - t_q:.1f}s)"
        )

    sub = pd.DataFrame(rows, columns=COLS)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"Wrote {out} in {time.time() - t0:.1f}s")
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
