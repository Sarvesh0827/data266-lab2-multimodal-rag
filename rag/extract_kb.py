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


def extract_prose_claims(doc_id, page, text, trust, claims: list[dict]) -> None:
    """Auto-extract short grounded claims from PDF prose (not hand-authored answers)."""
    if trust == "low":
        return
    text = _normalize_text(text)
    patterns = [
        (
            "lehman_bankruptcy",
            r"(On September 15,? 2008,? a major bank,\s*Lehman Brothers,? went bankrupt\.)",
        ),
        (
            "recession_definition",
            r"(a recession[—\-]+a decrease in output)",
        ),
        (
            "okun_slope",
            r"(The slope of the line is\s*" + NUM + r"\..{0,220}?unemployment rate by roughly\s*-?" + NUM + r"%)",
        ),
        (
            "okun_constant_growth",
            r"(It takes a growth rate of about\s*" + NUM + r"%\s*to keep\s*unemployment constant\.)",
        ),
        (
            "mortgage_securities",
            r"(The banks that had issued the mortgages had often bundled and\s*"
            r"packaged them together into new securities.{0,280}?nearly impossible to assess\.)",
        ),
        (
            "housing_consumption",
            r"(Hit by the decrease in housing prices and the collapse in stock prices.{0,220}?"
            r"people sharply\s*cut their consumption\.)",
        ),
        (
            "stock_markets_global",
            r"(despite the\s*fact that the crisis originated in the United States,? European and emerging market stock\s*"
            r"prices decreased by as much as their U\.S\. counterparts)",
        ),
        (
            "china_fiscal_response",
            r"(the adverse effect on demand was nearly fully offset by a\s*"
            r"major fiscal expansion by the Chinese government.{0,160}?public investment\.)",
        ),
        (
            "gdp_deflator_vs_cpi",
            r"(The GDP deflator is the price of goods produced in the United States,? whereas\s*"
            r"the CPI is the price of goods consumed by.{0,120})",
        ),
        (
            "growth_unemployment_not_always",
            r"(this image is not right:.{0,160}?unemployed find\s*jobs)",
        ),
        (
            "euro_precrisis_unemployment",
            r"(this pre-crisis level was already high,\s*" + NUM + r"%\s*for the Euro area over the\s*"
            r"period 2000-2007)",
        ),
        (
            "why_worry_unemployment",
            r"((?:Unemployment|unemployment).{0,40}(?:costly|costs|matters|painful|serious).{0,160})",
        ),
        (
            "real_gdp_growth",
            r"((?:real GDP|Real GDP).{0,120}?(?:adjusted for inflation|constant dollars|chained).{0,80})",
        ),
    ]
    flat = re.sub(r"[ \t]+", " ", text)
    flat = flat.replace("-\n", "")  # undo line-break hyphenation
    for topic, pat in patterns:
        m = re.search(pat, flat, flags=re.I | re.S)
        if not m:
            continue
        snippet = re.sub(r"\s+", " ", m.group(1) if m.lastindex else m.group(0)).strip()
        if len(snippet) < 20:
            continue
        claims.append(
            {
                "claim_id": f"{doc_id}:{topic}:p{page}",
                "topic": topic,
                "doc_id": doc_id,
                "page": page,
                "trust_level": trust,
                "text": snippet[:500],
            }
        )


EURO_BAR_ORDER = [
    "Spain",
    "Greece",
    "Portugal",
    "Ireland",
    "Italy",
    "France",
    "Germany",
    "Netherlands",
    "Austria",
]


def _ocr_euro_bar_chart(image_path: str, reader, y0: float, y1: float) -> dict[str, float]:
    """Spatially OCR one bar chart region; values come only from the image."""
    try:
        from PIL import Image, ImageEnhance, ImageOps
        import numpy as np
    except Exception:
        return {}

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    crop = img.crop((int(0.12 * w), int(y0 * h), int(0.99 * w), int(y1 * h)))
    crop = ImageEnhance.Contrast(ImageOps.grayscale(crop)).enhance(2.6)
    crop = crop.resize((crop.width * 2, crop.height * 2))
    res = reader.readtext(np.array(crop), detail=1, paragraph=False)

    nums: list[tuple[float, float]] = []
    names: list[tuple[float, str]] = []
    for bbox, text, conf in res:
        if conf < 0.1:
            continue
        xs = sum(pt[0] for pt in bbox) / 4.0
        t = str(text).strip().replace(",", ".")
        tl = re.sub(r"[^a-z]", "", t.lower())
        for c in EURO_BAR_ORDER:
            cl = c.lower()
            if tl == cl or (len(tl) >= 4 and (cl.startswith(tl) or tl.startswith(cl[:4]))):
                names.append((xs, c))
                break
        for m in re.finditer(r"(\d+)[.,](\d+)", t):
            v = float(m.group(1) + "." + m.group(2))
            if 3.0 <= v <= 30.0:
                nums.append((xs, v))

    byc: dict[str, list[float]] = {}
    for xs, c in names:
        byc.setdefault(c, []).append(xs)
    name_x = {c: sorted(xs)[len(xs) // 2] for c, xs in byc.items()}

    nums = sorted(nums)
    uniq: list[tuple[float, float]] = []
    for xs, v in nums:
        if uniq and abs(uniq[-1][0] - xs) < 20 and abs(uniq[-1][1] - v) < 0.05:
            continue
        if uniq and abs(uniq[-1][0] - xs) < 20:
            continue
        uniq.append((xs, v))

    out: dict[str, float] = {}
    used_x: set[float] = set()
    for c, nx in sorted(name_x.items(), key=lambda kv: kv[1]):
        best = None
        bestd = 1e9
        bestx = None
        for xs, v in uniq:
            if xs in used_x:
                continue
            d = abs(xs - nx)
            if d < bestd:
                bestd = d
                best = v
                bestx = xs
        if best is not None and bestd < 100:
            out[c] = best
            if bestx is not None:
                used_x.add(bestx)

    # Dedicated right-column OCR for Austria (label often only readable in a tight crop).
    right = img.crop((int(0.75 * w), int(y0 * h), int(0.995 * w), int(y1 * h)))
    right = ImageEnhance.Contrast(ImageOps.grayscale(right)).enhance(2.8)
    right = right.resize((right.width * 3, right.height * 3))
    right_vals: list[tuple[float, float]] = []
    for bbox, t, conf in reader.readtext(np.array(right), detail=1, paragraph=False):
        if conf < 0.15:
            continue
        xs = sum(pt[0] for pt in bbox) / 4.0
        for m in re.finditer(r"(\d+)[.,](\d+)", str(t).replace(",", ".")):
            v = float(m.group(1) + "." + m.group(2))
            if 3.5 <= v <= 5.5:
                right_vals.append((xs, v))
    if right_vals:
        # Austria is the rightmost bar on the chart.
        out["Austria"] = sorted(right_vals, key=lambda kv: kv[0])[-1][1]

    # Dedicated Germany column crop if missing.
    if "Germany" not in out:
        ger = img.crop((int(0.62 * w), int(y0 * h), int(0.78 * w), int(y1 * h)))
        ger = ImageEnhance.Contrast(ImageOps.grayscale(ger)).enhance(2.8)
        ger = ger.resize((ger.width * 3, ger.height * 3))
        ger_vals = []
        for t in reader.readtext(np.array(ger), detail=0, paragraph=False):
            for m in re.finditer(r"(\d+)[.,](\d+)", str(t).replace(",", ".")):
                v = float(m.group(1) + "." + m.group(2))
                if 5.0 <= v <= 8.5:
                    ger_vals.append(v)
        if ger_vals:
            out["Germany"] = ger_vals[0]

    return out


def enrich_euro_area_bar_facts(facts: list[NumericFact], assets, reader) -> None:
    """Parse Figures A-6/A-7 bar labels via OCR into numeric facts."""
    page_img = None
    meta = None
    for a in assets:
        if a.label.lower() in {"figure a-6", "figure a-7"} and a.trust_level != "low":
            page_img = a.image_path
            meta = a
            break
    if not page_img or meta is None:
        return

    # Both labels are saved as full-page renders; split by vertical bands.
    charts = [
        ("Figure A-6", "2010", 0.28, 0.53),
        ("Figure A-7", "2012", 0.52, 0.85),
    ]
    for label, year, y0, y1 in charts:
        parsed = _ocr_euro_bar_chart(page_img, reader, y0, y1)
        for country, val in parsed.items():
            ent = country.lower().replace(" ", "_")
            _add(
                facts,
                doc_id=meta.doc_id,
                page=meta.page,
                trust_level=meta.trust_level,
                source_label=label,
                entity=ent,
                metric="unemployment",
                year=year,
                value=float(val),
                value_type="outturn",
                raw=f"OCR {label} {country}={val}",
            )


def _parse_decimal_tokens(text: str) -> list[float]:
    vals = []
    t = text.strip().replace(",", ".")
    # OCR sometimes drops the leading decimal point: "045" -> 0.45, "063" -> 0.63
    if re.fullmatch(r"0\d{2}", t):
        vals.append(float("0." + t[1:]))
        return vals
    if re.fullmatch(r"\d{3}", t) and t.startswith("0"):
        vals.append(float("0." + t[1:]))
        return vals
    for m in re.finditer(r"-?\d+\.\d+", t):
        vals.append(float(m.group(0)))
    return vals


def enrich_stock_index_facts(facts: list[NumericFact], assets, reader) -> None:
    """OCR Figures A-4/A-5 end-point and trough labels (from page render)."""
    meta = None
    page_img = None
    for a in assets:
        if a.label.lower() in {"figure a-4", "figure a-5"} and a.trust_level != "low":
            page_img = a.image_path
            meta = a
            break
    if not page_img or meta is None:
        return
    try:
        from PIL import Image, ImageEnhance, ImageOps
        import numpy as np
    except Exception:
        return

    img = Image.open(page_img).convert("RGB")
    w, h = img.size
    arr = np.array(ImageEnhance.Contrast(ImageOps.grayscale(img)).enhance(1.8))
    res = reader.readtext(arr, detail=1, paragraph=False)

    points: list[tuple[float, float, float]] = []  # x_frac, y_frac, value
    for bbox, text, conf in res:
        if conf < 0.25:
            continue
        xs = sum(pt[0] for pt in bbox) / 4.0 / w
        ys = sum(pt[1] for pt in bbox) / 4.0 / h
        raw = str(text).strip()
        vals = _parse_decimal_tokens(raw)
        if not vals and re.fullmatch(r"0\d{2}", raw):
            vals = [float("0." + raw[1:])]
        for v in vals:
            if 0.2 <= abs(v) <= 3.5:
                points.append((xs, ys, v))

    def add_stock(label, entity, year, value, note):
        _add(
            facts,
            doc_id=meta.doc_id,
            page=meta.page,
            trust_level=meta.trust_level,
            source_label=label,
            entity=entity,
            metric="stock_index",
            year=year,
            value=float(value),
            value_type="outturn",
            raw=note,
        )

    # A-5 end labels: right side of lower chart, stacked emerging/US/euro.
    a5_end = sorted(
        [(ys, v) for xs, ys, v in points if xs >= 0.75 and 0.62 <= ys <= 0.78 and 1.1 <= v <= 2.3],
        key=lambda t: t[0],
    )
    a5_vals = []
    for ys, v in a5_end:
        if a5_vals and abs(a5_vals[-1] - v) < 0.02:
            continue
        a5_vals.append(v)
    if len(a5_vals) >= 3:
        add_stock("Figure A-5", "emerging_economies", "2010-12", a5_vals[0], f"OCR A-5 end {a5_vals}")
        add_stock("Figure A-5", "united_states", "2010-12", a5_vals[1], f"OCR A-5 end {a5_vals}")
        add_stock("Figure A-5", "euro_area", "2010-12", a5_vals[2], f"OCR A-5 end {a5_vals}")

    # A-4 end labels (right edge, upper chart)
    a4_end = sorted(
        [(ys, v) for xs, ys, v in points if xs >= 0.75 and 0.38 <= ys <= 0.55 and 0.4 <= v <= 1.4],
        key=lambda t: t[0],
    )
    a4_end_vals = []
    for ys, v in a4_end:
        if a4_end_vals and abs(a4_end_vals[-1] - v) < 0.02:
            continue
        # skip obvious axis ticks
        if abs(v - round(v, 1)) < 1e-9 and v in {0.6, 0.8, 1.0, 1.2, 1.4}:
            continue
        a4_end_vals.append(v)
    if len(a4_end_vals) >= 3:
        add_stock("Figure A-4", "emerging_economies", "2010-12", a4_end_vals[0], f"OCR A-4 end {a4_end_vals}")
        add_stock("Figure A-4", "united_states", "2010-12", a4_end_vals[1], f"OCR A-4 end {a4_end_vals}")
        add_stock("Figure A-4", "euro_area", "2010-12", a4_end_vals[2], f"OCR A-4 end {a4_end_vals}")

    # A-4 trough labels near 2009 (mid-x): emerging/US/euro top→bottom; Euro is lowest.
    a4_trough = sorted(
        [(ys, v) for xs, ys, v in points if 0.45 <= xs <= 0.70 and 0.40 <= ys <= 0.55 and 0.35 <= v <= 0.75],
        key=lambda t: t[0],
    )
    trough_vals = []
    for ys, v in a4_trough:
        if v in {0.6, 0.8}:  # axis ticks
            continue
        if trough_vals and abs(trough_vals[-1] - v) < 0.02:
            continue
        trough_vals.append(v)
    if len(trough_vals) >= 3:
        add_stock("Figure A-4", "emerging_economies", "2009-03", trough_vals[0], f"OCR A-4 trough {trough_vals}")
        add_stock("Figure A-4", "united_states", "2009-03", trough_vals[1], f"OCR A-4 trough {trough_vals}")
        add_stock("Figure A-4", "euro_area", "2009-03", trough_vals[2], f"OCR A-4 trough {trough_vals}")
    elif trough_vals:
        # Prefer the lowest OCR'd trough label as Euro-area March-2009 value.
        add_stock(
            "Figure A-4",
            "euro_area",
            "2009-03",
            min(trough_vals),
            f"OCR A-4 trough euro={min(trough_vals)} from {trough_vals}",
        )


def enrich_b4_cpi_facts(facts: list[NumericFact], assets, reader) -> None:
    """OCR Figure B-4 2009 headline/core CPI labels."""
    meta = None
    page_img = None
    for a in assets:
        if a.label.lower() == "figure b-4" and a.trust_level != "low":
            page_img = a.image_path
            meta = a
            break
    if not page_img or meta is None:
        return
    try:
        from PIL import Image, ImageEnhance, ImageOps
        import numpy as np
    except Exception:
        return

    img = Image.open(page_img).convert("RGB")
    w, h = img.size
    # B-4 is the lower chart on the page render.
    crop = img.crop((int(0.45 * w), int(0.58 * h), int(0.85 * w), int(0.82 * h)))
    crop = ImageEnhance.Contrast(ImageOps.grayscale(crop)).enhance(2.5)
    crop = crop.resize((crop.width * 3, crop.height * 3))
    vals = []
    for bbox, text, conf in reader.readtext(np.array(crop), detail=1, paragraph=False):
        if conf < 0.2:
            continue
        ys = sum(pt[1] for pt in bbox) / 4.0
        for v in _parse_decimal_tokens(str(text)):
            if -5 <= v <= 5:
                vals.append((ys, v))
    # Also accept tokens like -0.4 / 1.7 from full-page OCR near B-4
    negs = sorted([v for _, v in vals if v < 0])
    poss = sorted([v for _, v in vals if 0.5 <= v <= 3.0])
    headline = negs[0] if negs else None
    core = poss[-1] if poss else None
    if headline is not None:
        _add(
            facts,
            doc_id=meta.doc_id,
            page=meta.page,
            trust_level=meta.trust_level,
            source_label="Figure B-4",
            entity="united_states",
            metric="headline_cpi",
            year="2009",
            value=float(headline),
            value_type="outturn",
            raw=f"OCR B-4 headline {headline}",
        )
    if core is not None:
        _add(
            facts,
            doc_id=meta.doc_id,
            page=meta.page,
            trust_level=meta.trust_level,
            source_label="Figure B-4",
            entity="united_states",
            metric="core_cpi",
            year="2009",
            value=float(core),
            value_type="outturn",
            raw=f"OCR B-4 core {core}",
        )


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
    claims: list[dict] = []
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
            extract_prose_claims(doc_id, page_no, text, trust, claims)

    figure_records: list[FigureRecord] = []
    if do_ocr:
        try:
            import easyocr

            try:
                reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            except Exception:
                reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            figure_records = ocr_figures(assets, reader)
            enrich_euro_area_bar_facts(facts, assets, reader)
            enrich_stock_index_facts(facts, assets, reader)
            enrich_b4_cpi_facts(facts, assets, reader)
        except Exception as e:
            print("OCR skipped:", e)

    payload = {
        "numeric_facts": [asdict(f) for f in facts],
        "figures": [asdict(f) for f in figure_records],
        "claims": claims,
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
