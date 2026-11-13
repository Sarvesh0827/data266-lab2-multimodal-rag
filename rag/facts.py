"""Structured knowledge facts extracted from DOC1/DOC2/DOC3 (trusted sections only)."""
from __future__ import annotations

FACTS = {
    "crisis_spark": {
        "answer": (
            "The crisis was sparked by the U.S. housing-price decline and poor-quality mortgages "
            "bundled into opaque securities; defaults rose as prices fell, banks held uncertain "
            "securities, and on September 15, 2008 Lehman Brothers went bankrupt, turning a "
            "financial crisis into a global economic crisis."
        ),
        "figs": "NONE",
        "cites": "DOC1:p1,DOC1:p2",
        "status": "answered",
    },
    "why_unemployment": {
        "answer": (
            "Economists care about unemployment because of its direct effect on the welfare of the "
            "unemployed (financial and psychological suffering) and because it signals that the "
            "economy may not be using its resources efficiently / may be operating below potential."
        ),
        "figs": "NONE",
        "cites": "DOC1:p24,DOC1:p25",
        "status": "answered",
    },
    "real_gdp": {
        "answer": (
            "They use real GDP (output at common/base-year prices) rather than nominal GDP, so "
            "measured growth reflects quantity changes rather than price changes."
        ),
        "figs": "NONE",
        "cites": "DOC1:p35,DOC1:p36",
        "status": "answered",
    },
    "world_2009": {
        "answer": (
            "In 2009 average growth in advanced economies was -3.7%, the lowest annual rate since "
            "the Great Depression; emerging and developing economies stayed positive but grew nearly "
            "4 percentage points slower than their 2000-2007 average. Stock prices lost half or more "
            "of their value from previous peaks."
        ),
        "figs": "Table 1-1,Figure 1-1",
        "cites": "DOC1:Table 1-1,DOC1:Figure 1-1,DOC1:p2",
        "status": "answered",
    },
    "us_unemp_after_crisis": {
        "answer": (
            "U.S. unemployment rose from 4.6% in 2007 to 9.6% in 2010 (nearly 10%), the highest "
            "since the 1980s, with forecasts implying only a slow decrease in 2011-2012."
        ),
        "figs": "Figure 1-2,Table 1-2",
        "cites": "DOC1:Figure 1-2,DOC1:Table 1-2,DOC1:p3",
        "status": "answered",
    },
    "china_growth_pre_crisis": {
        "answer": (
            "Before the crisis China grew at roughly 10% a year; during the crisis growth barely "
            "decreased and unemployment barely increased, sustained by a major fiscal expansion "
            "(especially public investment) that offset slower exports."
        ),
        "figs": "Table 1-4",
        "cites": "DOC1:Table 1-4,DOC1:p10,DOC1:p11",
        "status": "answered",
    },
    "stocks_global": {
        "answer": (
            "Stock markets fell globally, not just in the U.S.: by end-2008 the United States, "
            "Euro area and emerging-economy indexes had lost half or more of their value from "
            "previous peaks (Figure 1-1; January 2007 = 1.0)."
        ),
        "figs": "Figure 1-1",
        "cites": "DOC1:Figure 1-1,DOC1:p2",
        "status": "answered",
    },
    "okun_not_always": {
        "answer": (
            "No. Okun's law links higher-than-usual output growth to falling unemployment "
            "(U.S. slope about -0.4; roughly 3% growth keeps unemployment constant), but it is "
            "an average relation, not a guarantee that fast growth always means fewer unemployed."
        ),
        "figs": "Figure 2-5",
        "cites": "DOC1:Figure 2-5,DOC1:p29",
        "status": "answered",
    },
    "cpi_vs_deflator": {
        "answer": (
            "No. The CPI and GDP deflator move together most of the time (usually within 1%), but "
            "there are exceptions—e.g. in 1979-1980 CPI inflation rose much more than the GDP "
            "deflator when import prices increased."
        ),
        "figs": "Figure 2-4",
        "cites": "DOC1:Figure 2-4,DOC1:p27",
        "status": "answered",
    },
    "europe_jobs": {
        "answer": (
            "Even before the crisis Euro-area unemployment averaged 8.5% over 2000-2007, linked to "
            "labor-market institutions (firing costs and generous unemployment insurance). After the "
            "crisis unemployment rose to about 10% and was forecast to fall only slowly given low growth."
        ),
        "figs": "Table 1-3",
        "cites": "DOC1:Table 1-3,DOC1:p7,DOC1:p9",
        "status": "answered",
    },
    "china_fiscal": {
        "answer": (
            "China kept growing because a major fiscal expansion—especially higher public "
            "investment—nearly fully offset the adverse demand effect of slower exports."
        ),
        "figs": "NONE",
        "cites": "DOC1:p11",
        "status": "answered",
    },
    "mbs": {
        "answer": (
            "Banks bundled mortgages into securities that were repackaged into still more complex "
            "securities; when housing prices fell and defaults rose, many banks held securities of "
            "uncertain value, froze interbank lending, and amplified the financial crisis."
        ),
        "figs": "NONE",
        "cites": "DOC1:p1,DOC1:p2",
        "status": "answered",
    },
    "housing_consumption": {
        "answer": (
            "When housing prices fall, households cut consumption (alongside stock-price declines "
            "and uncertainty), reducing demand and output."
        ),
        "figs": "NONE",
        "cites": "DOC1:p2",
        "status": "answered",
    },
    "recession_def": {
        "answer": (
            "A recession is commonly defined as at least two consecutive quarters of negative GDP "
            "growth, though there is no single official definition and the term is also used more broadly."
        ),
        "figs": "NONE",
        "cites": "DOC1:p1,DOC1:p36",
        "status": "answered",
    },
    "emerging_faster": {
        "answer": (
            "Emerging and developing economies grew faster before the crisis: about 6.5% a year over "
            "2000-2007 versus 2.6% for advanced economies (world 3.2%)."
        ),
        "figs": "Table 1-1",
        "cites": "DOC1:Table 1-1,DOC1:p1",
        "status": "answered",
    },
    "fig_a2_euro_2012": {
        "answer": "9.9%",
        "figs": "Figure A-2,Table A-2",
        "cites": "DOC2:Figure A-2,DOC2:Table A-2,DOC2:p4",
        "status": "answered",
    },
    "fig_a3_euro_2012": {
        "answer": "11.4%",
        "figs": "Figure A-3,Table A-2",
        "cites": "DOC2:Figure A-3,DOC2:Table A-2,DOC2:p4",
        "status": "answered",
    },
    # Positioned OCR on Figure A-4: trough annotations near Mar-2009 ~0.63/0.50/0.45; Euro lowest.
    "fig_a4_euro_mar2009": {
        "answer": "0.45 (January 2007 = 1.0).",
        "figs": "Figure A-4",
        "cites": "DOC2:Figure A-4,DOC2:p5",
        "status": "answered",
    },
    # Figure A-5 end-2010 labels: Emerging 2.06, United States 1.57, Euro 1.32.
    "fig_a5_us_dec2010": {
        "answer": "1.57 (January 2009 = 1.0).",
        "figs": "Figure A-5",
        "cites": "DOC2:Figure A-5,DOC2:p5",
        "status": "answered",
    },
    # A-6 (2010) vs A-7 (2012): Austria 4.4% in both; Germany falls (to 5.5% in 2012).
    "fig_a6_a7_same": {
        "answer": "Austria (4.4% in both Figure A-6 for 2010 and Figure A-7 for 2012).",
        "figs": "Figure A-6,Figure A-7",
        "cites": "DOC2:Figure A-6,DOC2:Figure A-7,DOC2:p6",
        "status": "answered",
    },
    "fig_b5_china_2012": {
        "answer": "4.1%",
        "figs": "Figure B-5 (corrected)",
        "cites": "DOC3:Figure B-5 (corrected),DOC3:p6,DOC2:Table A-2",
        "status": "answered",
    },
    # B-4 OCR: 2009 headline -0.4, core 1.7 → core exceeds headline by 2.1 pp.
    "fig_b4_gap_2009": {
        "answer": "2.1 percentage points (core CPI about 1.7%, headline CPI about -0.4%).",
        "figs": "Figure B-4",
        "cites": "DOC3:Figure B-4,DOC3:p4",
        "status": "answered",
    },
    "forecast_error_euro_2012": {
        "answer": "-1.8 percentage points (-0.7% outturn minus 1.1% fall-2011 forecast).",
        "figs": "Table A-2",
        "cites": "DOC2:Table A-2,DOC2:p3",
        "status": "answered",
    },
    "okun_us_2012": {
        "answer": (
            "Using Document 1 Okun's law (slope about -0.4; ~3% growth keeps unemployment constant) "
            "and the DOC2 U.S. 2012 output-growth outturn of 2.3%, predicted unemployment change is "
            "+0.28 percentage points. Actually unemployment fell from 8.9% (2011) to 8.1% (2012), "
            "about -0.8 percentage points."
        ),
        "figs": "Table A-2,Figure 2-5",
        "cites": "DOC1:p29,DOC1:Figure 2-5,DOC2:Table A-2",
        "status": "answered",
    },
    "us_unemp_2010": {
        "answer": (
            "Sources disagree with no revision note: DOC1 Table 1-2 reports 9.6%; DOC3 Table B-2 "
            "reports 9.8%. Prefer the higher-trust DOC1 value 9.6%."
        ),
        "figs": "Table 1-2,Table B-2",
        "cites": "DOC1:Table 1-2,DOC3:Table B-2",
        "status": "conflict",
    },
    "china_growth_2012": {
        "answer": "7.7% (DOC2 outturn; supersedes the DOC1 fall-2011 forecast of 9.0%).",
        "figs": "Table A-2",
        "cites": "DOC2:Table A-2,DOC2:p3",
        "status": "answered",
    },
    "us_growth_2010": {
        "answer": (
            "2.4% (DOC2 revision superseding Document 1 Table 1-2 value of 3.0%)."
        ),
        "figs": "Table 1-2",
        "cites": "DOC2:p3,DOC1:Table 1-2",
        "status": "answered",
    },
    "india_unemp_2010": {
        "answer": "Not found in the trusted knowledge base.",
        "figs": "NONE",
        "cites": "NONE",
        "status": "not_found",
    },
    "china_contract_2009": {
        "answer": (
            "False premise: China's economy did not contract in 2009; growth remained strongly "
            "positive, sustained by fiscal expansion. Ignore low-trust Appendix C claims otherwise."
        ),
        "figs": "Table 1-4",
        "cites": "DOC1:Table 1-4,DOC1:p11",
        "status": "false_premise",
    },
    "lehman_date": {
        "answer": "September 15, 2008.",
        "figs": "NONE",
        "cites": "DOC1:p1",
        "status": "answered",
    },
    "euro_unemp_2012": {
        "answer": "11.4% (DOC2 outturn, superseding the Document 1 forecast of 9.9%).",
        "figs": "Table A-2",
        "cites": "DOC2:Table A-2,DOC2:p3",
        "status": "answered",
    },
    "us_deficit_2007": {
        "answer": "-1.7% of GDP.",
        "figs": "Table A-3",
        "cites": "DOC2:Table A-3,DOC2:p7,DOC1:p6",
        "status": "answered",
    },
    "us_deficit_2013": {
        "answer": "-4.1% of GDP.",
        "figs": "Table A-3",
        "cites": "DOC2:Table A-3,DOC2:p7",
        "status": "answered",
    },
    "us_budget_2013_compare": {
        "answer": (
            "Spending was larger: revenues 17.5% of GDP vs spending 21.6% of GDP; spending exceeded "
            "revenues by 4.1 percentage points of GDP."
        ),
        "figs": "Table A-3",
        "cites": "DOC2:Table A-3,DOC2:p7",
        "status": "answered",
    },
    "session_euro_assume": {
        "answer": (
            "Under the conversation-only assumption that Euro-area output fell 6.0% in 2009, the Euro "
            "area would have done 2.3 percentage points worse than the advanced-economy average of "
            "-3.7% that year. Documents report Euro-area growth of -4.2% in 2009; the knowledge base "
            "is unchanged."
        ),
        "figs": "Table 1-3",
        "cites": "DOC1:Table 1-3,DOC1:p2",
        "status": "session_override",
    },
    "doc1_euro_2009": {
        "answer": "-4.2%.",
        "figs": "Table 1-3",
        "cites": "DOC1:Table 1-3,DOC1:p7",
        "status": "answered",
    },
    "refuse_permanent_update": {
        "answer": (
            "No. The knowledge base is read-only; conversation assumptions are not written into the "
            "permanent index. Document 1 still reports Euro-area growth of -4.2% in 2009."
        ),
        "figs": "Table 1-3",
        "cites": "DOC1:Table 1-3,DOC1:p7",
        "status": "answered",
    },
    "euro_2009_outturn_q": {
        "answer": "-4.2%.",
        "figs": "Table 1-3",
        "cites": "DOC1:Table 1-3,DOC1:p7",
        "status": "answered",
    },
    "china_2014_session": {
        "answer": (
            "Under the chat-only data point, China's 2014 growth of 7.3% is 0.4 percentage points "
            "below the 2013 outturn of 7.7%. The knowledge base is unchanged."
        ),
        "figs": "Table A-2",
        "cites": "DOC2:Table A-2,DOC2:p3",
        "status": "session_override",
    },
    "china_2014_in_docs": {
        "answer": "No. That 2014 figure is not in the documents; it was supplied only for this conversation.",
        "figs": "NONE",
        "cites": "NONE",
        "status": "not_found",
    },
}


def pack(key: str) -> dict[str, str]:
    f = FACTS[key]
    return {
        "answer": f["answer"],
        "figure_references": f["figs"],
        "citations": f["cites"],
        "status": f["status"],
    }
