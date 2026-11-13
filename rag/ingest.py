from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

import rag.config as config
from rag.config import doc_paths, load_manifest, trust_map

LABEL_RE = re.compile(
    r"(?P<label>(Figure|Table)\s+(([12]|[A-C])-\d+(?:\s*\(corrected\))?))",
    re.IGNORECASE,
)
APPENDIX_C_PAGE = {("DOC3", 7)}


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str
    trust_level: str
    section: str
    kind: str = "text"


@dataclass
class VisualAsset:
    asset_id: str
    doc_id: str
    page: int
    label: str
    caption: str
    image_path: str
    trust_level: str
    kind: str  # figure | table | page_render


def _normalize_label(label: str) -> str:
    label = re.sub(r"\s+", " ", label.strip())
    label = label.replace("figure", "Figure").replace("table", "Table")
    # standardize corrected
    label = re.sub(r"\(\s*corrected\s*\)", "(corrected)", label, flags=re.I)
    parts = label.split(" ", 1)
    if len(parts) == 2:
        kind, rest = parts
        kind = "Figure" if kind.lower() == "figure" else "Table"
        return f"{kind} {rest}"
    return label


def _page_trust(doc_id: str, page: int, base: str) -> str:
    if (doc_id, page) in APPENDIX_C_PAGE:
        return "low"
    return base


def _chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        if i + size >= len(text):
            break
        i += max(1, size - overlap)
    return chunks


def _caption_near(text: str, label: str) -> str:
    # take ~400 chars after first label mention
    m = re.search(re.escape(label), text, flags=re.I)
    if not m:
        return label
    start = m.start()
    return text[start : start + 420].strip()


def extract_knowledge_base(force: bool = False) -> tuple[list[TextChunk], list[VisualAsset]]:
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    figures_dir = config.ARTIFACTS / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    meta_path = config.ARTIFACTS / "ingest_meta.json"

    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [TextChunk(**c) for c in meta["chunks"]]
        assets = [VisualAsset(**a) for a in meta["assets"]]
        return chunks, assets

    manifest = load_manifest()
    trusts = trust_map(manifest)
    paths = doc_paths(manifest)

    chunks: list[TextChunk] = []
    assets: list[VisualAsset] = []
    label_index: dict[str, list[str]] = {}

    for doc_id, pdf_path in paths.items():
        doc = pymupdf.open(pdf_path)
        base_trust = trusts[doc_id]
        for page_idx in range(doc.page_count):
            page_no = page_idx + 1
            page = doc[page_idx]
            text = page.get_text("text")
            tlevel = _page_trust(doc_id, page_no, base_trust)
            section = "Appendix C" if tlevel == "low" else "body"

            for ci, piece in enumerate(_chunk_text(text)):
                cid = f"{doc_id}_p{page_no}_c{ci}"
                chunks.append(
                    TextChunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        page=page_no,
                        text=piece,
                        trust_level=tlevel,
                        section=section,
                    )
                )

            labels = [_normalize_label(m.group("label")) for m in LABEL_RE.finditer(text)]
            # unique preserve order
            seen = set()
            uniq_labels = []
            for lb in labels:
                key = lb.lower()
                if key not in seen:
                    seen.add(key)
                    uniq_labels.append(lb)

            # Always render page if it has a figure/table label, or embedded images
            images = page.get_images(full=True)
            need_render = bool(uniq_labels) or bool(images) or doc_id != "DOC1"
            if need_render and (uniq_labels or images):
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                if uniq_labels:
                    for lb in uniq_labels:
                        safe = re.sub(r"[^\w\-]+", "_", lb)
                        img_path = figures_dir / f"{doc_id}_p{page_no}_{safe}.png"
                        pix.save(str(img_path))
                        aid = f"{doc_id}:p{page_no}:{lb}"
                        kind = "table" if lb.lower().startswith("table") else "figure"
                        asset = VisualAsset(
                            asset_id=aid,
                            doc_id=doc_id,
                            page=page_no,
                            label=lb,
                            caption=_caption_near(text, lb),
                            image_path=str(img_path),
                            trust_level=tlevel,
                            kind=kind,
                        )
                        assets.append(asset)
                        label_index.setdefault(lb.lower(), []).append(aid)
                else:
                    img_path = figures_dir / f"{doc_id}_p{page_no}_page.png"
                    pix.save(str(img_path))
                    aid = f"{doc_id}:page{page_no}"
                    assets.append(
                        VisualAsset(
                            asset_id=aid,
                            doc_id=doc_id,
                            page=page_no,
                            label=f"page-{page_no}",
                            caption=text[:300],
                            image_path=str(img_path),
                            trust_level=tlevel,
                            kind="page_render",
                        )
                    )

            # Also export raw embedded images (extra evidence)
            for ii, img in enumerate(images):
                try:
                    xref = img[0]
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.n >= 5:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    img_path = figures_dir / f"{doc_id}_p{page_no}_embed{ii}.png"
                    pix.save(str(img_path))
                except Exception:
                    continue

    meta = {
        "chunks": [asdict(c) for c in chunks],
        "assets": [asdict(a) for a in assets],
        "label_index": label_index,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (config.ARTIFACTS / "label_index.json").write_text(json.dumps(label_index, indent=2), encoding="utf-8")
    return chunks, assets


if __name__ == "__main__":
    ch, as_ = extract_knowledge_base(force=True)
    print(f"chunks={len(ch)} assets={len(as_)}")
    for a in as_[:20]:
        print(a.doc_id, a.label, a.page, a.trust_level)
