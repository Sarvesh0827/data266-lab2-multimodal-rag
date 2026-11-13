from __future__ import annotations

import json
import pickle
from pathlib import Path

import chromadb
import numpy as np
import open_clip
import torch
from chromadb.config import Settings
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.config import ARTIFACTS, CLIP_MODEL, CLIP_PRETRAINED, INDEX_DIR, TEXT_MODEL
from rag.ingest import TextChunk, VisualAsset, extract_knowledge_base


class MultimodalIndex:
    def __init__(self, rebuild: bool = False):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model = SentenceTransformer(TEXT_MODEL, device=self.device)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=self.device
        )
        self.clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.clip_model.eval()

        self.client = chromadb.PersistentClient(
            path=str(INDEX_DIR / "chroma"), settings=Settings(anonymized_telemetry=False)
        )
        self.text_col = self.client.get_or_create_collection("text_chunks", metadata={"hnsw:space": "cosine"})
        self.fig_col = self.client.get_or_create_collection("figures", metadata={"hnsw:space": "cosine"})

        self.bm25 = None
        self.bm25_docs: list[dict] = []
        self.label_index: dict[str, list[str]] = {}
        self.assets_by_id: dict[str, VisualAsset] = {}

        marker = INDEX_DIR / "ready.json"
        if rebuild or not marker.exists() or self.text_col.count() == 0:
            self.build()
        else:
            self._load_side()

    def _load_side(self) -> None:
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            payload = pickle.load(f)
        self.bm25 = payload["bm25"]
        self.bm25_docs = payload["docs"]
        self.label_index = json.loads((ARTIFACTS / "label_index.json").read_text(encoding="utf-8"))
        _, assets = extract_knowledge_base(force=False)
        self.assets_by_id = {a.asset_id: a for a in assets}

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        emb = self.text_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()

    @torch.no_grad()
    def _embed_image(self, path: str) -> list[float]:
        img = Image.open(path).convert("RGB")
        tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        feat = self.clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()

    @torch.no_grad()
    def embed_query_clip(self, text: str) -> list[float]:
        tokens = self.clip_tokenizer([text]).to(self.device)
        feat = self.clip_model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32).tolist()

    def build(self) -> None:
        chunks, assets = extract_knowledge_base(force=True)
        # reset collections
        try:
            self.client.delete_collection("text_chunks")
            self.client.delete_collection("figures")
        except Exception:
            pass
        self.text_col = self.client.get_or_create_collection("text_chunks", metadata={"hnsw:space": "cosine"})
        self.fig_col = self.client.get_or_create_collection("figures", metadata={"hnsw:space": "cosine"})

        # filter low-trust from searchable evidence index? keep but mark; retrieval will exclude
        batch = 64
        for i in tqdm(range(0, len(chunks), batch), desc="index-text"):
            part = chunks[i : i + batch]
            self.text_col.add(
                ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                embeddings=self._embed_texts([c.text for c in part]),
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "page": c.page,
                        "trust_level": c.trust_level,
                        "section": c.section,
                        "kind": "text",
                    }
                    for c in part
                ],
            )

        usable_assets = [a for a in assets if a.label.lower() != "page-"]
        for i in tqdm(range(0, len(usable_assets), 8), desc="index-figures"):
            part = usable_assets[i : i + 8]
            embeddings = []
            documents = []
            for a in part:
                try:
                    embeddings.append(self._embed_image(a.image_path))
                except Exception:
                    # fallback: embed caption text via clip text tower
                    embeddings.append(self.embed_query_clip(a.caption or a.label))
                documents.append(f"{a.label}. {a.caption}")
            self.fig_col.add(
                ids=[a.asset_id for a in part],
                documents=documents,
                embeddings=embeddings,
                metadatas=[
                    {
                        "doc_id": a.doc_id,
                        "page": a.page,
                        "trust_level": a.trust_level,
                        "figure_id": a.label,
                        "kind": a.kind,
                        "image_path": a.image_path,
                    }
                    for a in part
                ],
            )

        # BM25 over text + figure captions (exclude low trust)
        self.bm25_docs = []
        corpus = []
        for c in chunks:
            if c.trust_level == "low":
                continue
            self.bm25_docs.append(
                {
                    "id": c.chunk_id,
                    "text": c.text,
                    "doc_id": c.doc_id,
                    "page": c.page,
                    "trust_level": c.trust_level,
                    "kind": "text",
                    "figure_id": "",
                }
            )
            corpus.append(c.text.lower().split())
        for a in usable_assets:
            if a.trust_level == "low":
                continue
            text = f"{a.label} {a.caption}"
            self.bm25_docs.append(
                {
                    "id": a.asset_id,
                    "text": text,
                    "doc_id": a.doc_id,
                    "page": a.page,
                    "trust_level": a.trust_level,
                    "kind": a.kind,
                    "figure_id": a.label,
                }
            )
            corpus.append(text.lower().split())
        self.bm25 = BM25Okapi(corpus)
        with open(INDEX_DIR / "bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "docs": self.bm25_docs}, f)

        self.label_index = json.loads((ARTIFACTS / "label_index.json").read_text(encoding="utf-8"))
        self.assets_by_id = {a.asset_id: a for a in assets}
        (INDEX_DIR / "ready.json").write_text(json.dumps({"ok": True}), encoding="utf-8")


if __name__ == "__main__":
    MultimodalIndex(rebuild=True)
    print("index ready")
