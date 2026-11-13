"""
Kaggle entrypoint (also importable from a notebook).

Usage in Kaggle notebook:
  !pip install -q -r /kaggle/working/repo/requirements.txt
  # or copy rag/ into /kaggle/working and run:
  %run kaggle_p1.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/kaggle/working")
# allow local testing
if not (ROOT / "rag").exists():
    ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.config import DATA_DIR
from rag.run_submission import run


def validate(submission: Path, questions: Path) -> None:
    script = DATA_DIR / "validate_submission.py"
    if script.exists():
        subprocess.check_call([sys.executable, str(script), str(submission), str(questions)])


if __name__ == "__main__":
    out = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else ROOT / "submission.csv"
    path = run(rebuild_index=True, out_path=out)
    validate(path, DATA_DIR / "Questions.csv")
    print("DONE", path)
