"""Stdlib-only inspection of synthetic_b2b_transactions_2000.xlsx (audit phase)."""
import zipfile
import xml.etree.ElementTree as ET
import collections
import io
import sys

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
# Path updated to the copy shipped in this package; it is byte-identical to the
# handover original (verified by SHA-256), so the audit figures are unchanged.
P = "data/synthetic_b2b_transactions_2000.xlsx"

out = io.StringIO()


def w(*a):
    print(*a, file=out)


z = zipfile.ZipFile(P)
shared = []
root = ET.fromstring(z.read("xl/sharedStrings.xml"))
for si in root.findall(f"{NS}si"):
    shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))

root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
raw_rows = []
for r in root.iter(f"{NS}row"):
    cells = []
    for c in r:
        t = c.get("t")
        v = c.find(f"{NS}v")
        if v is None:
            isn = c.find(f"{NS}is")
            val = "".join(x.text or "" for x in isn.iter(f"{NS}t")) if isn is not None else ""
        else:
            val = shared[int(v.text)] if t == "s" else (v.text or "")
        cells.append(val)
    raw_rows.append(cells[0] if cells else "")

header = raw_rows[0].split(";")
w("N COLUMNS IN HEADER:", len(header))
w("COLUMNS:", header)

body = raw_rows[1:]
w("\nN DATA ROWS:", len(body))

field_counts = collections.Counter(len(r.split(";")) for r in body)
w("FIELD-COUNT DISTRIBUTION (semicolon split):", dict(field_counts))

recs = []
for r in body:
    parts = r.split(";")
    if len(parts) == len(header):
        recs.append(dict(zip(header, parts)))
w("PARSED CLEANLY:", len(recs), "of", len(body))

# non-empty column census
w("\n--- COLUMN FILL RATES (non-empty) ---")
for col in header:
    n = sum(1 for r in recs if r.get(col, "").strip() != "")
    uniq = len({r.get(col, "") for r in recs})
    if n > 0:
        w(f"  {col:35s} filled={n:5d}  unique={uniq:5d}")

w("\n--- SAMPLE ROWS (key fields) ---")
keys = ["description", "amount", "transactionDate", "parentCategoryName",
        "categoryName", "counterpartyNameRaw", "vendorName", "type"]
for i, r in enumerate(recs[:12]):
    w(f"[{i}]")
    for k in keys:
        w(f"     {k:22s} = {r.get(k, '')!r}")

w("\n--- categoryName DISTRIBUTION ---")
cat = collections.Counter(r["categoryName"] for r in recs)
w("n_categories:", len(cat))
for k, v in cat.most_common():
    w(f"  {v:5d}  {k}")

w("\n--- parentCategoryName DISTRIBUTION ---")
par = collections.Counter(r["parentCategoryName"] for r in recs)
w("n_parent_categories:", len(par))
for k, v in par.most_common():
    w(f"  {v:5d}  {k}")

w("\n--- parent -> child MAPPING ---")
pc = collections.defaultdict(set)
for r in recs:
    pc[r["parentCategoryName"]].add(r["categoryName"])
for p in sorted(pc):
    w(f"  {p}:")
    for c in sorted(pc[p]):
        w(f"      - {c}")

w("\n--- DESCRIPTION ANALYSIS ---")
descs = [r["description"] for r in recs]
w("n:", len(descs))
w("unique raw:", len(set(descs)))
lens = [len(d) for d in descs]
w("char len min/mean/max:", min(lens), sum(lens) / len(lens), max(lens))
wc = [len(d.split()) for d in descs]
w("word count min/mean/max:", min(wc), sum(wc) / len(wc), max(wc))

dupc = collections.Counter(descs)
w("\nTOP 25 MOST REPEATED RAW DESCRIPTIONS:")
for k, v in dupc.most_common(25):
    w(f"  {v:5d}  {k!r}")

# how many descriptions appear with >1 category?
d2c = collections.defaultdict(set)
for r in recs:
    d2c[r["description"]].add(r["categoryName"])
ambig = {d: cs for d, cs in d2c.items() if len(cs) > 1}
w("\nRAW DESCRIPTIONS MAPPED TO >1 CATEGORY:", len(ambig))
for d, cs in list(ambig.items())[:20]:
    w(f"  {d!r} -> {sorted(cs)}")

w("\n--- ALL UNIQUE RAW DESCRIPTIONS (with count & category) ---")
seen = {}
for r in recs:
    seen.setdefault(r["description"], r["categoryName"])
for d, c in sorted(seen.items(), key=lambda x: -dupc[x[0]]):
    w(f"  {dupc[d]:5d} | {c:32s} | {d!r}")

sys.stdout.reconfigure(encoding="utf-8")
open("_xlsx_report.txt", "w", encoding="utf-8").write(out.getvalue())
print("report written:", len(out.getvalue()), "chars")
