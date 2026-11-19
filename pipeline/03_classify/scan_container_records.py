#!/usr/bin/env python3
"""
Thorough scan of ALL kept safety papers for non-paper container / front-matter
records (proceedings volumes, workshop banners, prefaces, special issues, etc.),
across both sources (pool + snowball).

Tiering (precision-first):
  A = container-title pattern AND empty abstract AND no DOI   -> auto-droppable
  B = container-title pattern but has an abstract or a DOI     -> manual review
  (also reports C = empty abstract & no DOI but NO container pattern -> just
   metadata-poor papers, NOT dropped, count only)

Outputs:
  paper_survey/_pool_audit/container_scan_all.csv   (tiers A+B, with reasons)
Then applies Tier A to papers_llm_cleaned.json (originals untouched).
"""
import csv, io, json, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
OUT = os.path.join(PS, "_pool_audit")
REASON = "container/front-matter record (not a paper)"

# ---- container-title patterns (each returns a short reason) ----
PATTERNS = [
    ("proceedings", re.compile(r"\bproceedings\b", re.I)),
    ("co-located", re.compile(r"\bco-located with\b", re.I)),
    ("workshop-banner", re.compile(r"\bworkshop on .+?(20\d\d|co-located|@)", re.I)),
    ("conference-frontmatter", re.compile(r"\bconference on .+?,\s*20\d\d", re.I)),
    ("symposium", re.compile(r"\bsymposium on .+?20\d\d", re.I)),
    ("companion", re.compile(r"\bcompanion (of|proceedings|volume)\b", re.I)),
    ("selected-papers", re.compile(r"\b(revised )?selected papers\b", re.I)),
    ("preface/toc", re.compile(r"\b(preface|front[- ]matter|table of contents|"
                               r"editorial|message from the (chairs|program))\b", re.I)),
    ("special-issue", re.compile(r"\bspecial issue\b", re.I)),
    ("dagstuhl", re.compile(r"\bdagstuhl (reports?|seminar)\b", re.I)),
    ("volume", re.compile(r"\b(vol\.|volume)\s*\d+\b", re.I)),
    ("safeai-tag", re.compile(r"\((safeai|de-factify|avss|aisafety)\s*20\d\d\)", re.I)),
    ("venue+daterange", re.compile(r"\b\d{1,2}\s*[-–]\s*\d{1,2}\s+\w+\s+20\d\d\b")),
    ("city-date-frontmatter", re.compile(r",\s+(january|february|march|april|may|june|july|"
                                         r"august|september|october|november|december)\s+\d{1,2},?\s+20\d\d", re.I)),
]


def container_reason(title):
    hits = [name for name, rx in PATTERNS if rx.search(title or "")]
    # require the strong banner signals, OR 'proceedings' alone, OR two weaker hits
    strong = {"proceedings", "co-located", "selected-papers", "preface/toc",
              "companion", "dagstuhl", "safeai-tag", "conference-frontmatter"}
    if any(h in strong for h in hits) or len(hits) >= 2:
        return "+".join(hits)
    return None


papers = json.load(io.open(os.path.join(PS, "papers_llm.json"), encoding="utf-8"))
kept = [p for p in papers if p.get("is_safety")]

tierA, tierB, tierC = [], [], 0
for p in kept:
    title = p.get("title") or ""
    empty = not (p.get("abstract") or "").strip()
    nodoi = not (p.get("doi") or "").strip()
    reason = container_reason(title)
    if reason:
        row = {"id": p["id"], "title": title, "venue": p.get("venue"), "year": p.get("year"),
               "source": p.get("source"), "category": p["category"],
               "conf": p.get("llm_confidence"), "has_abstract": not empty,
               "has_doi": not nodoi, "pattern": reason}
        (tierA if (empty and nodoi) else tierB).append(row)
    elif empty and nodoi:
        tierC += 1

with io.open(os.path.join(OUT, "container_scan_all.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tier", "id", "title", "venue", "year", "source", "category",
                "conf", "has_abstract", "has_doi", "pattern"])
    for t, rows in (("A-autodrop", tierA), ("B-review", tierB)):
        for r in sorted(rows, key=lambda r: (r["source"], r["id"])):
            w.writerow([t, r["id"], r["title"], r["venue"], r["year"], r["source"],
                        r["category"], r["conf"], r["has_abstract"], r["has_doi"], r["pattern"]])

print(f"kept safety papers scanned: {len(kept)}")
print(f"Tier A (auto-drop: container + empty abstract + no DOI): {len(tierA)}")
print(f"   by source: {dict(Counter(r['source'] for r in tierA))}")
print(f"Tier B (review: container pattern but has abstract/DOI): {len(tierB)}")
print(f"   by source: {dict(Counter(r['source'] for r in tierB))}")
print(f"Tier C (empty abstract+no DOI, NO container pattern, NOT dropped): {tierC}")

print("\n--- Tier A records ---")
for r in sorted(tierA, key=lambda r: (r["source"], r["id"])):
    print(f"  [{r['id']}] {r['source']:8s} ({r['category']}) {r['title'][:66]}")

print("\n--- Tier B records (need a human/LLM look) ---")
for r in sorted(tierB, key=lambda r: (r["source"], r["id"]))[:40]:
    print(f"  [{r['id']}] {r['source']:8s} abs={r['has_abstract']} doi={r['has_doi']} "
          f"pat={r['pattern']}\n        {r['title'][:80]}")

# ---- apply Tier A to the cleaned corpus (build on existing cleaned if present) ----
cleaned_path = os.path.join(PS, "papers_llm_cleaned.json")
base = json.load(io.open(cleaned_path, encoding="utf-8")) if os.path.exists(cleaned_path) else papers
dropA = {r["id"] for r in tierA}
n_flipped = 0
out = []
for p in base:
    q = dict(p)
    if q["id"] in dropA and q.get("is_safety"):
        q["is_safety"] = False
        q["drop_reason"] = REASON
        n_flipped += 1
    out.append(q)
json.dump(out, io.open(cleaned_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n_kept_after = sum(1 for p in out if p.get("is_safety"))
print(f"\napplied Tier A to papers_llm_cleaned.json: flipped {n_flipped} more "
      f"(kept safety now {n_kept_after}); originals untouched")
print("wrote _pool_audit/container_scan_all.csv")
