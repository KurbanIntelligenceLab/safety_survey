#!/usr/bin/env python3
"""
Title-screen the full proceedings census and dedup against the already-classified
corpus, so only NEW (not-yet-seen) safety candidates go forward to abstract
enrichment + LLM classification.

Uses the SAME screen as the NeurIPS/AIES-2025 backfill (collect_neurips2025.
screen_terms / matches) so every year is screened identically. Two-stage by
design: title-screen first (no abstracts needed, cuts the ~100k census to a few
thousand), then enrich_abstracts.py fetches abstracts only for survivors, then
re-screen on title+abstract before classify.

  python3 pipeline/01_collect/screen_census.py
    in : paper_survey/_census/census_candidates.json   (full census)
         paper_survey/papers_llm_13cat.json             (already-classified base)
         paper_survey/_neurips2025 + _aies2025 candidates (already-done 2025 backfill)
    out: paper_survey/_census/census_titlescreened.json (NEW candidates only)
"""
import io, json, re, sys
from pathlib import Path

sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from collect_neurips2025 import screen_terms, matches  # identical screen

ROOT = Path(".")
CENSUS = ROOT / "paper_survey/_census/census_candidates.json"
BASE = ROOT / "paper_survey/papers_llm_13cat.json"
DONE_2025 = [
    ROOT / "paper_survey/_neurips2025/neurips2025_candidates.json",
    ROOT / "paper_survey/_aies2025/aies2025_candidates.json",
]
OUT = ROOT / "paper_survey/_census/census_titlescreened.json"


def L(p):
    return json.load(io.open(p, encoding="utf-8"))


def norm_title(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (t or "").lower())).strip()


def main():
    census = L(CENSUS)
    terms = screen_terms()

    # already-seen titles: classified base + the 2025 backfill candidate sets
    seen = {norm_title(p.get("title")) for p in L(BASE)}
    for d in DONE_2025:
        if d.exists():
            seen |= {norm_title(p.get("title")) for p in L(d)}
    print(f"census={len(census)}  already-seen titles={len(seen)}")

    kept, dup = [], 0
    seen_local = set()
    for p in census:
        hits = matches(p["title"], terms)
        if not hits:
            continue
        nt = norm_title(p["title"])
        if nt in seen or nt in seen_local:
            dup += 1
            continue
        seen_local.add(nt)
        q = dict(p)
        q["screen_hits"] = sorted(set(" ".join(h) if isinstance(h, list) else h for h in hits))
        kept.append(q)

    json.dump(kept, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    from collections import Counter
    print(f"title-screened NEW candidates: {len(kept)}  (dropped {dup} already-seen)")
    print("by year:", dict(sorted(Counter(str(p['year']) for p in kept).items())))
    print("by venue:", dict(Counter(p['venue'] for p in kept).most_common()))
    print(f"wrote {OUT}")
    print("\nnext: S2_API_KEY=... python3 pipeline/02_enrich/enrich_abstracts.py "
          f"--in {OUT} --out paper_survey/_census/census_enriched.json")


if __name__ == "__main__":
    main()
