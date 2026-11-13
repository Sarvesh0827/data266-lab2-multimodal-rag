import re
from pathlib import Path
import fitz

data = Path(__file__).resolve().parents[1] / "data_export"
files = {
    "DOC1": data / "Document.pdf",
    "DOC2": data / "Document2_Supplement_A_Outturns_2011-2013.pdf",
    "DOC3": data / "Document3_Briefing_B_Labour_Prices_Policy.pdf",
}
label_re = re.compile(r"(Figure|Table)\s+(([12]|[A-C])-\d+(?:\s*\(corrected\))?)", re.I)

for doc_id, path in files.items():
    doc = fitz.open(path)
    print(f"\n=== {doc_id} pages={doc.page_count} file={path.name} ===")
    for i, page in enumerate(doc, 1):
        text = page.get_text()
        found = sorted({m.group(0).replace("  ", " ") for m in label_re.finditer(text)})
        imgs = page.get_images(full=True)
        if found or imgs:
            print(f"p{i}: labels={found} images={len(imgs)}")
    # print first ~400 chars of page 1 for sanity
    print("--- page1 preview ---")
    print(doc[0].get_text()[:500].replace("\n", " | "))
