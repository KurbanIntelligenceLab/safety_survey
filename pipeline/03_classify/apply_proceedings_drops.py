#!/usr/bin/env python3
"""
Non-destructively drop proceedings / front-matter container records (not papers)
from the pool. Reads the audited candidate list, flips is_safety=false on a COPY,
and reports the before/after delta. Originals (papers_llm.json) are untouched.

Inputs : paper_survey/papers_llm.json
         paper_survey/_pool_audit/proceedings_drop_candidates.csv
Outputs: paper_survey/papers_llm_cleaned.json   (copy with 7 records flipped)
         paper_survey/_pool_audit/proceedings_drops.json  (authoritative drop list)
"""
import csv, io, json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
OUT = os.path.join(PS, "_pool_audit")
REASON = "proceedings/front-matter container record (no abstract, no DOI, not a paper)"

# audited candidate ids (all verified: empty abstract + no DOI + workshop-banner title)
cands = list(csv.DictReader(io.open(os.path.join(OUT, "proceedings_drop_candidates.csv"),
                                    encoding="utf-8")))
drop_ids = {int(c["id"]) for c in cands}

papers = json.load(io.open(os.path.join(PS, "papers_llm.json"), encoding="utf-8"))

# safety guard: only drop records that really are empty container entries
verified, skipped = set(), []
for p in papers:
    if p["id"] in drop_ids:
        empty = not (p.get("abstract") or "").strip()
        nodoi = not (p.get("doi") or "").strip()
        if empty and nodoi:
            verified.add(p["id"])
        else:
            skipped.append((p["id"], p["title"][:60], f"abstract={not empty} doi={not nodoi}"))

def cat_counts(ps):
    return Counter(p["category"] for p in ps if p.get("is_safety"))

pool_before = [p for p in papers if p.get("source") == "pool" and p.get("is_safety")]
before = cat_counts(pool_before)
n_kept_before = sum(1 for p in papers if p.get("is_safety"))

# build cleaned copy
cleaned = []
for p in papers:
    q = dict(p)
    if q["id"] in verified:
        q["is_safety"] = False
        q["drop_reason"] = REASON
    cleaned.append(q)
json.dump(cleaned, io.open(os.path.join(PS, "papers_llm_cleaned.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)

drop_list = [{"id": p["id"], "title": p["title"], "venue": p.get("venue"),
              "year": p.get("year"), "was_category": p["category"], "reason": REASON}
             for p in papers if p["id"] in verified]
json.dump(drop_list, io.open(os.path.join(OUT, "proceedings_drops.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=2)

pool_after = [p for p in cleaned if p.get("source") == "pool" and p.get("is_safety")]
after = cat_counts(pool_after)
n_kept_after = sum(1 for p in cleaned if p.get("is_safety"))

print(f"dropped {len(verified)} container records (skipped {len(skipped)} that had content)")
for s in skipped:
    print("  SKIP", s)
print(f"\ntotal kept (all sources): {n_kept_before} -> {n_kept_after}  (-{n_kept_before-n_kept_after})")
print(f"pool kept                : {len(pool_before)} -> {len(pool_after)}  (-{len(pool_before)-len(pool_after)})")
print("\naffected categories (pool):")
for c in sorted(set(before) | set(after), key=lambda c: -(before[c]-after[c])):
    if before[c] != after[c]:
        print(f"  {c}: {before[c]} -> {after[c]}")
print("\ndropped records:")
for d in drop_list:
    print(f"  [{d['id']}] ({d['was_category']}) {d['title'][:72]}")
print("\nwrote papers_llm_cleaned.json + _pool_audit/proceedings_drops.json (originals untouched)")
