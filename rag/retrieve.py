from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import TOP_K_BM25, TOP_K_DENSE, TOP_K_FINAL
from rag.index import MultimodalIndex

FIG_MENTION_RE = re.compile(
    r"(Figure|Table)\s+(([12]|[A-C])-\d+(?:\s*\(corrected\))?)",
    re.IGNORECASE,
)


@dataclass
class Evidence:
    evid_id: str
    doc_id: str
    page: int
    text: str
    trust_level: str
    kind: str
    figure_id: str
    score: float
    image_path: str = ""


def normalize_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label.strip())
    m = FIG_MENTION_RE.search(label)
    if not m:
        return label
    kind = "Figure" if m.group(1).lower() == "figure" else "Table"
    rest = m.group(2)
    rest = re.sub(r"\(\s*corrected\s*\)", "(corrected)", rest, flags=re.I)
    return f"{kind} {rest}"


def mentioned_labels(question: str) -> list[str]:
    return [normalize_label(m.group(0)) for m in FIG_MENTION_RE.finditer(question)]


def _rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    def __init__(self, index: MultimodalIndex):
        self.index = index

    def retrieve(self, question: str, top_k: int = TOP_K_FINAL) -> list[Evidence]:
        q = question.strip()
        forced_labels = mentioned_labels(q)

        # Dense text
        q_emb = self.index.text_model.encode([q], normalize_embeddings=True).tolist()
        text_res = self.index.text_col.query(query_embeddings=q_emb, n_results=TOP_K_DENSE)
        # Dense figures via CLIP text
        fig_emb = [self.index.embed_query_clip(q)]
        fig_n = min(TOP_K_DENSE, max(1, self.index.fig_col.count()))
        fig_res = self.index.fig_col.query(query_embeddings=fig_emb, n_results=fig_n) if self.index.fig_col.count() else None

        # BM25
        tokens = q.lower().split()
        bm25_scores = self.index.bm25.get_scores(tokens)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_BM25]

        id_to_ev: dict[str, Evidence] = {}

        def add_text_hit(eid, doc, meta, score):
            if meta.get("trust_level") == "low":
                return
            id_to_ev[eid] = Evidence(
                evid_id=eid,
                doc_id=meta["doc_id"],
                page=int(meta["page"]),
                text=doc,
                trust_level=meta["trust_level"],
                kind=meta.get("kind", "text"),
                figure_id=meta.get("figure_id", ""),
                score=float(score),
                image_path=meta.get("image_path", ""),
            )

        text_ids = text_res["ids"][0]
        for i, eid in enumerate(text_ids):
            add_text_hit(eid, text_res["documents"][0][i], text_res["metadatas"][0][i], 1.0 / (i + 1))

        fig_ids = []
        if fig_res:
            fig_ids = fig_res["ids"][0]
            for i, eid in enumerate(fig_ids):
                add_text_hit(eid, fig_res["documents"][0][i], fig_res["metadatas"][0][i], 1.0 / (i + 1))

        bm25_ids = []
        for rank, idx in enumerate(bm25_ranked):
            d = self.index.bm25_docs[idx]
            bm25_ids.append(d["id"])
            if d["id"] not in id_to_ev:
                add_text_hit(
                    d["id"],
                    d["text"],
                    {
                        "doc_id": d["doc_id"],
                        "page": d["page"],
                        "trust_level": d["trust_level"],
                        "kind": d["kind"],
                        "figure_id": d.get("figure_id", ""),
                    },
                    bm25_scores[idx],
                )

        fused = _rrf_fuse([text_ids, fig_ids, bm25_ids])
        for eid, sc in fused.items():
            if eid in id_to_ev:
                id_to_ev[eid].score = sc

        # Force exact figure/table mentions
        for label in forced_labels:
            key = label.lower()
            asset_ids = self.index.label_index.get(key, [])
            # also try without corrected / with corrected
            if not asset_ids and "(corrected)" not in key:
                asset_ids = self.index.label_index.get(key + " (corrected)".lower(), [])
            for aid in asset_ids:
                asset = self.index.assets_by_id.get(aid)
                if not asset or asset.trust_level == "low":
                    continue
                id_to_ev[aid] = Evidence(
                    evid_id=aid,
                    doc_id=asset.doc_id,
                    page=asset.page,
                    text=f"{asset.label}. {asset.caption}",
                    trust_level=asset.trust_level,
                    kind=asset.kind,
                    figure_id=asset.label,
                    score=1e6,  # force top
                    image_path=asset.image_path,
                )
                # also pull text chunks from same page
                page_hits = self.index.text_col.get(
                    where={"$and": [{"doc_id": asset.doc_id}, {"page": asset.page}]}
                )
                if page_hits and page_hits["ids"]:
                    for j, peid in enumerate(page_hits["ids"]):
                        meta = page_hits["metadatas"][j]
                        if meta.get("trust_level") == "low":
                            continue
                        id_to_ev[peid] = Evidence(
                            evid_id=peid,
                            doc_id=meta["doc_id"],
                            page=int(meta["page"]),
                            text=page_hits["documents"][j],
                            trust_level=meta["trust_level"],
                            kind="text",
                            figure_id=asset.label,
                            score=5e5,
                        )

        ranked = sorted(id_to_ev.values(), key=lambda e: e.score, reverse=True)
        # Prefer higher trust slightly
        trust_w = {"high": 1.05, "medium": 1.0, "low": 0.0}
        ranked.sort(key=lambda e: e.score * trust_w.get(e.trust_level, 1.0), reverse=True)
        return ranked[:top_k]
