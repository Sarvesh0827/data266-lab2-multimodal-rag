"""Auto-build structured KB from PDFs: table parsing + figure OCR. Values come only from PDFs."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import pymupdf

from rag.config import ARTIFACTS, doc_paths, load_manifest, trust_map
from rag.ingest import _page_trust, extract_knowledge_base

NUM = r"[-+]?(?:\d+\.\d+|\d+)"


def _normalize_text(text: str) -> str:
    """Map PDF unicode dashes/minuses to ASCII so numeric regexes match."""
    return (
        text.replace("\u2212", "-")  # minus
        .replace("\u2013", "-")  # en dash
        .replace("\u2014", "-")  # em dash
        .replace("\u00ad", "")  # soft hyphen
    )


@dataclass
class NumericFact:
    fact_id: str
    doc_id: str
    page: int
    trust_level: str
    source_label: str
    entity: str
    metric: str
    year: str
    value: float
    value_type: str
    raw: str


@dataclass
class FigureRecord:
    label: str
    doc_id: str
    page: int
    trust_level: str
    image_path: str
    caption: str
    ocr_text: str
    ocr_numbers: list[float]


def _f(tok: str) -> float:
    return float(tok.replace(",", "").replace("−", "-").replace("–", "-"))


def _add(facts: list[NumericFact], **kw) -> None:
    fid = f"{kw['doc_id']}:{kw['source_label']}:{kw['entity']}:{kw['metric']}:{kw['year']}:{kw['value_type']}"
    kw["fact_id"] = fid
    facts[:] = [x for x in facts if x.fact_id != fid]
    facts.append(NumericFact(**kw))


def parse_table_a1(doc_id, page, text, trust, facts):
    if "Table A-1" not in text:
        return
    # World / Advanced / Emerging blocks with 5 numbers
    patterns = [
        (r"World\s+((?:" + NUM + r"\s*){5})", "world"),
        (r"Advanced economies\s+((?:" + NUM + r"\s*){5})", "advanced_economies"),
        (r"Emerging and developing\s*economies\s+((?:" + NUM + r"\s*){5})", "emerging_economies"),
    ]
    for pat, ent in patterns:
        m = re.search(pat, text, flags=re.I)
        if not m:
            continue
        nums = [_f(x) for x in re.findall(NUM, m.group(1))]
        seq = [
            ("2011", nums[0], "forecast"),
            ("2011", nums[1], "outturn"),
            ("2012", nums[2], "forecast"),
            ("2012", nums[3], "outturn"),
            ("2013", nums[4], "outturn"),
        ]
        for y, v, vt in seq:
            _add(
                facts,
                doc_id=doc_id,
                page=page,
                trust_level=trust,
                source_label="Table A-1",
                entity=ent,
                metric="output_growth",
                year=y,
                value=v,
                value_type=vt,
                raw=m.group(0),
            )


def parse_table_a2(doc_id, page, text, trust, facts):
    if "Table A-2" not in text:
        return
    # Restrict to the table body (after header, before Source/Revision) so
    # section titles like "The United States, the Euro area and China" do not match.
    body_m = re.search(
        r"Table A-2\b.*?(?:Percent\s+)?(?:2011\s+2012\s+2013\s+)?(.*?)(?:Source:|Revision notice)",
        text,
        flags=re.S | re.I,
    )
    body = body_m.group(1) if body_m else text
    flat = re.sub(r"[ \t]+", " ", body)
    metric_map = {
        "Output growth": "output_growth",
        "Unemployment rate": "unemployment",
        "Inflation rate": "inflation",
    }
    for ent_name, ent in [
        ("United States", "united_states"),
        ("Euro area", "euro_area"),
        ("China", "china"),
    ]:
        m = re.search(
            rf"(?:^|\n)\s*{ent_name}\s*\n(.*?)(?=(?:^|\n)\s*(?:United States|Euro area|China)\s*\n|Source:|Revision notice|\Z)",
            flat,
            flags=re.S | re.I,
        )
        if not m:
            # Fallback: entity name immediately followed by a metric label
            m = re.search(
                rf"{ent_name}\s+(Output growth.*?)(?=United States|Euro area|China|Source:|Revision notice|\Z)",
                flat,
                flags=re.S | re.I,
            )
        if not m:
            continue
        block = m.group(1)
        for label, metric in metric_map.items():
            mm = re.search(
                rf"{label}\s+((?:{NUM}(?:\s*\(\s*{NUM}\s*\))?[\s]*){{1,3}})",
                block,
                flags=re.I,
            )
            if not mm:
                continue
            toks = list(re.finditer(rf"({NUM})(?:\s*\(\s*({NUM})\s*\))?", mm.group(1)))
            years = ["2011", "2012", "2013"]
            for i, tok in enumerate(toks[:3]):
                out_v = _f(tok.group(1))
                _add(
                    facts,
                    doc_id=doc_id,
                    page=page,
                    trust_level=trust,
                    source_label="Table A-2",
                    entity=ent,
                    metric=metric,
                    year=years[i],
                    value=out_v,
                    value_type="outturn",
                    raw=tok.group(0),
                )
                if tok.group(2):
                    _add(
                        facts,
                        doc_id=doc_id,
                        page=page,
                        trust_level=trust,
                        source_label="Table A-2",
                        entity=ent,
                        metric=metric,
                        year=years[i],
                        value=_f(tok.group(2)),
                        value_type="forecast",
                        raw=tok.group(0),
                    )
    rev = re.search(
        rf"revised from\s+({NUM})\s*%?\s*\([^)]*\)\s*to\s+({NUM})",
        text,
        flags=re.I,
    )
    if rev:
        _add(
            facts,
            doc_id=doc_id,
            page=page,
            trust_level=trust,
            source_label="Table A-2",
            entity="united_states",
            metric="output_growth",
            year="2010",
            value=_f(rev.group(2)),
            value_type="revision",
            raw=rev.group(0),
        )


def parse_table_a3(doc_id, page, text, trust, facts):
    if "Table A-3" not in text:
        return
    for m in re.finditer(
        rf"\b(20\d{{2}})\s+({NUM})\s+({NUM})\s+({NUM})",
        text,
    ):
        y, rev, span, bal = m.group(1), _f(m.group(2)), _f(m.group(3)), _f(m.group(4))
        for metric, val in [("revenues", rev), ("spending", span), ("budget_balance", bal)]:
            _add(
                facts,
                doc_id=doc_id,
                page=page,
                trust_level=trust,
                source_label="Table A-3",
                entity="united_states",
                metric=metric,
                year=y,
                value=val,
                value_type="outturn",
                raw=m.group(0),
            )


def parse_table_1_1(doc_id, page, text, trust, facts):
    """DOC1 Table 1-1: World / Advanced / Emerging output growth."""
    if "Table 1-1" not in text:
        return
    text = _normalize_text(text)
    flat = re.sub(r"[ \t]+", " ", text)
    years = ["2000-2007", "2008", "2009", "2010", "2011", "2012"]
    blocks = [
        (r"World\s+((?:" + NUM + r"\s*){6})", "world"),
        (r"Advanced economies\s+((?:" + NUM + r"\s*){6})", "advanced_economies"),
        (r"Emerging and developing economies\s+((?:" + NUM + r"\s*){6})", "emerging_economies"),
    ]
    for pat, ent in blocks:
        m = re.search(pat, flat, flags=re.I)
        if not m:
            continue
        nums = [_f(x) for x in re.findall(NUM, m.group(1))]
        if len(nums) < 6:
            continue
        for y, v in zip(years, nums[:6]):
            vtype = "forecast" if y in {"2011", "2012"} else "outturn"
            _add(
                facts,
                doc_id=doc_id,
                page=page,
                trust_level=trust,
                source_label="Table 1-1",
                entity=ent,
                metric="output_growth",
                year=y,
                value=v,
                value_type=vtype,
                raw=m.group(0),
            )


def parse_doc1_country_table(doc_id, page, text, trust, facts, table_label, entity):
    """Parse DOC1 Table 1-2/1-3/1-4 style: metrics with year columns 2008..2012."""
    if table_label not in text:
        return
    text = _normalize_text(text)
    flat = re.sub(r"[ \t]+", " ", text)
    metric_map = {
        "Output growth rate": "output_growth",
        "Unemployment rate": "unemployment",
        "Inflation rate": "inflation",
    }
    for label, metric in metric_map.items():
        mm = re.search(
            rf"{label}\s+((?:{NUM}[\s]+){{4,}}{NUM})",
            flat,
        )
        if not mm:
            continue
        nums = [_f(x) for x in re.findall(NUM, mm.group(1))]
        if len(nums) >= 5:
            year_vals = list(zip(["2008", "2009", "2010", "2011", "2012"], nums[-5:]))
        else:
            continue
        for y, v in year_vals:
            vtype = "forecast" if y in {"2011", "2012"} else "outturn"
            _add(
                facts,
                doc_id=doc_id,
                page=page,
                trust_level=trust,
                source_label=table_label,
                entity=entity,
                metric=metric,
                year=y,
                value=v,
                value_type=vtype,
                raw=mm.group(0),
            )
        # Also store 2000-2007 average when the row has 7 numbers
        # (1980-1999 avg, 2000-2007 avg, then 2008..2012).
        if len(nums) >= 7:
            _add(
                facts,
                doc_id=doc_id,
                page=page,
                trust_level=trust,
                source_label=table_label,
                entity=entity,
                metric=metric,
                year="2000-2007",
                value=nums[-6],
                value_type="outturn",
                raw=mm.group(0),
            )


def parse_doc1_averages_and_claims(doc_id, page, text, trust, facts):
    """Pull explicit prose statistics that appear as written percentages."""
    if trust == "low":
        return
    text = _normalize_text(text)
    patterns = [
        (
            rf"average growth in advanced economies was\s+({NUM})\s*%",
            "advanced_economies",
            "output_growth",
            "2009",
        ),
        (
            rf"from 2000 to\s*2007.*?world economy.*?Annual average world output growth was\s+({NUM})\s*%",
            "world",
            "output_growth",
            "2000-2007",
        ),
        (
            rf"Annual average world output growth was\s+({NUM})\s*%",
            "world",
            "output_growth",
            "2000-2007",
        ),
        (
            rf"advanced economies\s*\([^)]*\)\s*growing at\s+({NUM})\s*%",
            "advanced_economies",
            "output_growth",
            "2000-2007",
        ),
        (
            rf"emerging and developing economies.*?growing at an even faster\s+({NUM})\s*%",
            "emerging_economies",
            "output_growth",
            "2000-2007",
        ),
        (
            rf"pre-crisis level was already high,\s*({NUM})\s*%\s*for the Euro area over the\s*period 2000[–-]2007",
            "euro_area",
            "unemployment",
            "2000-2007",
        ),
        (
            rf"({NUM})\s*%\s*for the Euro area over the\s*period 2000[–-]2007",
            "euro_area",
            "unemployment",
            "2000-2007",
        ),
    ]
    for pat, ent, metric, year in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if m:
            _add(
                facts,
                doc_id=doc_id,
                page=page,
                trust_level=trust,
                source_label="text",
                entity=ent,
                metric=metric,
                year=year,
                value=_f(m.group(1)),
                value_type="outturn",
                raw=m.group(0)[:180],
            )


def parse_table_b2(doc_id, page, text, trust, facts):
    if "Table B-2" not in text:
        return
    for m in re.finditer(
        rf"\b(20\d{{2}})\s+({NUM})\s+({NUM})",
        text,
    ):
        y = m.group(1)
        if y not in {"2007", "2008", "2009", "2010"}:
            continue
        u, p = _f(m.group(2)), _f(m.group(3))
        # participation rates are ~60+, unemployment <20
        if u < 30:
            _add(
                facts,
                doc_id=doc_id,
                page=page,
                trust_level=trust,
                source_label="Table B-2",
                entity="united_states",
                metric="unemployment",
                year=y,
                value=u,
                value_type="outturn",
                raw=m.group(0),
            )


def extract_all_tables(doc_id, page, text, trust, facts):
    text = _normalize_text(text)
    parse_table_a1(doc_id, page, text, trust, facts)
    parse_table_a2(doc_id, page, text, trust, facts)
    parse_table_a3(doc_id, page, text, trust, facts)
    parse_table_b2(doc_id, page, text, trust, facts)
    if doc_id == "DOC1":
        parse_table_1_1(doc_id, page, text, trust, facts)
        parse_doc1_country_table(doc_id, page, text, trust, facts, "Table 1-2", "united_states")
        parse_doc1_country_table(doc_id, page, text, trust, facts, "Table 1-3", "euro_area")
        parse_doc1_country_table(doc_id, page, text, trust, facts, "Table 1-4", "china")
        parse_doc1_averages_and_claims(doc_id, page, text, trust, facts)


def ocr_figures(assets, reader) -> list[FigureRecord]:
    records: list[FigureRecord] = []
    for a in assets:
        if a.trust_level == "low":
            continue
        if not a.label.lower().startswith(("figure", "table")):
            continue
        try:
            texts = reader.readtext(a.image_path, detail=0, paragraph=False)
        except Exception:
            texts = []
        ocr_text = " | ".join(str(t) for t in texts)
        nums = []
        for t in texts:
            for n in re.findall(NUM, str(t).replace(",", "")):
                try:
                    v = float(n)
                except ValueError:
                    continue
                if abs(v) < 1e6:
                    nums.append(v)
        records.append(
            FigureRecord(
                label=a.label,
                doc_id=a.doc_id,
                page=a.page,
                trust_level=a.trust_level,
                image_path=a.image_path,
                caption=a.caption,
                ocr_text=ocr_text,
                ocr_numbers=nums,
            )
        )
    return records


def build_kb(force: bool = False, do_ocr: bool = True) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS / "kb_facts.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    chunks, assets = extract_knowledge_base(force=False)
    manifest = load_manifest()
    trusts = trust_map(manifest)
    paths = doc_paths(manifest)

    facts: list[NumericFact] = []
    for doc_id, pdf_path in paths.items():
        doc = pymupdf.open(pdf_path)
        base = trusts[doc_id]
        for i in range(doc.page_count):
            page_no = i + 1
            trust = _page_trust(doc_id, page_no, base)
            if trust == "low":
                continue
            text = doc[i].get_text("text")
            extract_all_tables(doc_id, page_no, text, trust, facts)

    figure_records: list[FigureRecord] = []
    if do_ocr:
        try:
            import easyocr

            reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            figure_records = ocr_figures(assets, reader)
        except Exception as e:
            print("OCR skipped:", e)

    payload = {
        "numeric_facts": [asdict(f) for f in facts],
        "figures": [asdict(f) for f in figure_records],
        "text_chunks": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page": c.page,
                "trust_level": c.trust_level,
                "text": c.text,
            }
            for c in chunks
            if c.trust_level != "low"
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"KB built: {len(facts)} numeric facts, {len(figure_records)} figures, "
        f"{len(payload['text_chunks'])} text chunks → {out_path}"
    )
    return payload


if __name__ == "__main__":
    build_kb(force=True, do_ocr=True)
