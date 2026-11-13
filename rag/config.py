from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_data_dir() -> Path:
    """Resolve Questions.csv / PDF folder.

    Priority: DATA_DIR env → local data_export → Kaggle /kaggle/input search.
    """
    env = os.environ.get("DATA_DIR")
    if env and (Path(env) / "Questions.csv").exists():
        return Path(env)

    local = ROOT / "data_export"
    if (local / "Questions.csv").exists():
        return local

    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in kaggle.rglob("Questions.csv"):
            return p.parent

    raise FileNotFoundError("Could not locate Questions.csv / knowledge-base data dir")


def refresh_paths() -> None:
    """Re-bind module paths after notebook sets DATA_DIR / ARTIFACTS_DIR / OUTPUT_DIR."""
    global DATA_DIR, ARTIFACTS, INDEX_DIR, OUTPUT_DIR
    DATA_DIR = resolve_data_dir()
    ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
    INDEX_DIR = Path(os.environ.get("INDEX_DIR", ROOT / "rag_index"))
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


try:
    DATA_DIR = resolve_data_dir()
except FileNotFoundError:
    # Allow importing the package before data is attached (notebook setup cells).
    DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data_export"))

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", ROOT / "rag_index"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT))

TEXT_MODEL = os.environ.get("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CLIP_MODEL = os.environ.get("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.environ.get("CLIP_PRETRAINED", "openai")
GEN_MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K_DENSE = 8
TOP_K_BM25 = 8
TOP_K_FINAL = 10


def load_manifest() -> dict:
    with open(DATA_DIR / "kb_manifest.json", encoding="utf-8") as f:
        return json.load(f)


def doc_paths(manifest: dict | None = None) -> dict[str, Path]:
    manifest = manifest or load_manifest()
    return {d["doc_id"]: DATA_DIR / d["file"] for d in manifest["documents"]}


def trust_map(manifest: dict | None = None) -> dict[str, str]:
    manifest = manifest or load_manifest()
    out = {d["doc_id"]: d["trust_level"] for d in manifest["documents"]}
    return out
