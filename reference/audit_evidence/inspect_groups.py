"""Reproduce the notebook's clean_description + grouping to quantify the split problem."""
import zipfile
import xml.etree.ElementTree as ET
import collections
import io
import re
import sys

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
out = io.StringIO()


def w(*a):
    print(*a, file=out)


def read_xlsx_rows(path, sheet="xl/worksheets/sheet1.xml"):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    root = ET.fromstring(z.read(sheet))
    rows = []
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
        rows.append(cells)
    return rows


# ---- notebook's clean_description, verbatim ----
def clean_description(text):
    text = str(text).lower().strip()
    text = re.sub(r"\b20\d{2}[-/.]\d{1,2}\b", " ", text)
    text = re.sub(r"\b[a-z]{1,4}-?\d{3,}\b", " ", text)
    text = re.sub(r"[^a-zäöüß\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Path updated to the copy shipped in this package; it is byte-identical to the
# handover original (verified by SHA-256), so the audit figures are unchanged.
rows = read_xlsx_rows("data/synthetic_b2b_transactions_2000.xlsx")
header = rows[0][0].split(";")
recs = [dict(zip(header, r[0].split(";"))) for r in rows[1:]]

di, ci = "description", "categoryName"
for r in recs:
    r["clean"] = clean_description(r[di])

w("=" * 70)
w("GROUPING ANALYSIS (notebook clean_description, exact-match grouping)")
w("=" * 70)
w("transactions:", len(recs))
w("unique RAW descriptions:", len({r[di] for r in recs}))
w("unique CLEANED descriptions (= groups):", len({r["clean"] for r in recs}))
w("categories:", len({r[ci] for r in recs}))
w("empty after cleaning:", sum(1 for r in recs if r["clean"] == ""))

# clean -> categories (label consistency of the grouping)
c2cat = collections.defaultdict(set)
for r in recs:
    c2cat[r["clean"]].add(r[ci])
ambig = {k: v for k, v in c2cat.items() if len(v) > 1}
w("\ncleaned patterns mapping to >1 category:", len(ambig))
for k, v in ambig.items():
    w(f"   {k!r} -> {sorted(v)}")

# patterns per category
pat_per_cat = collections.defaultdict(set)
rows_per_cat = collections.Counter()
for r in recs:
    pat_per_cat[r[ci]].add(r["clean"])
    rows_per_cat[r[ci]] += 1

w("\n" + "-" * 70)
w("UNIQUE CLEANED PATTERNS PER CATEGORY  (rows | patterns)")
w("-" * 70)
for cat in sorted(pat_per_cat, key=lambda c: len(pat_per_cat[c])):
    w(f"  {rows_per_cat[cat]:5d} | {len(pat_per_cat[cat]):3d}  {cat}")

npat = [len(v) for v in pat_per_cat.values()]
w("\npatterns/category  min=%d  median=%d  max=%d  total=%d"
  % (min(npat), sorted(npat)[len(npat) // 2], max(npat), sum(npat)))
w("categories with 1 pattern :", sum(1 for n in npat if n == 1))
w("categories with <3 patterns:", sum(1 for n in npat if n < 3))
w("categories with <5 patterns:", sum(1 for n in npat if n < 5))

# group sizes
gs = collections.Counter(r["clean"] for r in recs)
w("\ngroup size: min=%d mean=%.2f max=%d" % (min(gs.values()), sum(gs.values()) / len(gs), max(gs.values())))
w("largest groups:")
for k, v in gs.most_common(15):
    w(f"   {v:4d}  {k!r}")

# ---- what the notebook's "unseen pattern" holdout does (cell 126) ----
w("\n" + "=" * 70)
w("NOTEBOOK CELL-126 HOLDOUT SIMULATION (1 unseen pattern per category)")
w("=" * 70)
n_test_pat = sum(1 for v in pat_per_cat.values() if len(v) >= 2)
w("categories contributing a test pattern:", n_test_pat)
w("categories fully kept in train (only 1 pattern):", sum(1 for v in pat_per_cat.values() if len(v) < 2))
w("=> test set = %d unique patterns; NO validation split at all" % n_test_pat)

# ---- effect of cleaning: information loss examples ----
w("\n" + "=" * 70)
w("CLEANING SIDE EFFECTS (raw -> cleaned) where digits/tokens were destroyed")
w("=" * 70)
seen = set()
shown = 0
for r in recs:
    if r[di] in seen:
        continue
    seen.add(r[di])
    raw = r[di]
    cl = r["clean"]
    # flag cases where a digit-bearing token carried meaning
    if re.search(r"\b(b2b|365|24|4\.0|q[1-4])\b", raw.lower()) or re.search(r"\b\w*\d\w*\b", raw) and len(cl.split()) < len(raw.split()) - 1:
        w(f"  RAW: {raw!r}\n  CLN: {cl!r}\n")
        shown += 1
        if shown > 25:
            break

open("_group_report.txt", "w", encoding="utf-8").write(out.getvalue())
print("group report written")
