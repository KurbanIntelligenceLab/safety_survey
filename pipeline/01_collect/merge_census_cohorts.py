#!/usr/bin/env python3
"""
Fold the manually-scraped NeurIPS-2025 and AIES-2025 cohorts into the
title-screened census, producing ONE source-of-truth candidate file for the
downstream full-text + classify pipeline.

Why this exists: DBLP has a recency hole -- NeurIPS-2025 and AIES-2025 are
near-zero in the full-proceedings census (census_candidates.json). Their real
cohorts were scraped separately to paper_survey/_neurips2025 and
paper_survey/_aies2025 and are ALREADY title-screened + abstract-enriched. To
avoid a confusing parallel backfill path, we merge them here so every venue x
year flows through the identical screen -> enrich -> gate -> full-text ->
classify pipeline from a single file. A `source` field records provenance.

The cohorts skip title-screen (already screened) and carry their abstracts
forward (so enrich_abstracts only needs to fill the census rows).

  python3 pipeline/01_collect/merge_census_cohorts.py
    in : paper_survey/_census/census_titlescreened.json   (10,423 census survivors)
         paper_survey/_neurips2025/neurips2025_candidates.json  (2,839)
         paper_survey/_aies2025/aies2025_candidates.json        (227)
    out: paper_survey/_census/census_screened_all.json    (single source of truth)
"""
import io, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(".")
SCREENED = ROOT / "paper_survey/_census/census_titlescreened.json"
COHORTS = [
    ("neurips2025_backfill", ROOT / "paper_survey/_neurips2025/neurips2025_candidates.json"),
    ("aies2025_backfill", ROOT / "paper_survey/_aies2025/aies2025_candidates.json"),
]
OUT = ROOT / "paper_survey/_census/census_screened_all.json"

FIELDS = ["title", "year", "venue", "authors", "url", "doi", "dblp_key",
          "abstract", "screen_hits", "source"]


def L(p):
    return json.load(io.open(p, encoding="utf-8"))


def norm_title(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (t or "").lower())).strip()


def normalize(rec, source):
    q = {k: rec.get(k) for k in FIELDS}
    q["source"] = source
    q["year"] = str(q["year"]) if q["year"] is not None else None
    return q


def main():
    merged, seen, dups = [], set(), Counter()

    # census survivors first (no abstracts yet -- enrich fills them)
    for r in L(SCREENED):
        nt = norm_title(r.get("title"))
        if nt in seen:
            dups["census"] += 1
            continue
        seen.add(nt)
        merged.append(normalize(r, "census"))

    # then the manually-scraped 2025 cohorts (already enriched)
    for source, path in COHORTS:
        rows = L(path)
        added = 0
        for r in rows:
            nt = norm_title(r.get("title"))
            if nt in seen:
                dups[source] += 1
                continue
            seen.add(nt)
            merged.append(normalize(r, source))
            added += 1
        print(f"{source}: {len(rows)} in, {added} added, {dups[source]} dup vs already-merged")

    json.dump(merged, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    have_abs = sum(1 for r in merged if r.get("abstract"))
    print(f"\n==== MERGED SOURCE OF TRUTH ====")
    print(f"total candidates: {len(merged)}")
    print("by source:", dict(Counter(r["source"] for r in merged)))
    print("by year:", dict(sorted(Counter(r["year"] for r in merged).items())))
    print("by venue:", dict(Counter(r["venue"] for r in merged).most_common()))
    print(f"already have abstract: {have_abs}  (need enrich: {len(merged) - have_abs})")
    print(f"wrote {OUT}")
    print("\nnext: S2_API_KEY=... python3 pipeline/02_enrich/enrich_abstracts.py "
          f"--in {OUT} --out paper_survey/_census/census_enriched.json")


if __name__ == "__main__":
    main()
