#!/usr/bin/env python
"""Build the full-text acquisition index for the CENSUS safety corpus.

Unlike build_fulltext_index.py (which re-joins the old paper_survey corpus to S2
via DOI/title), the census survivors already carry a stored `s2_id` from the
enrich step, so we use those directly and only batch-query Semantic Scholar for
openAccessPdf URLs + arXiv ids. Output schema matches the original index so the
fetcher consumes it unchanged (via FT_INDEX/FT_ROOT env overrides).

  source .env   # for S2_API_KEY
  python build_census_fulltext_index.py
"""
import json, io, os, sys
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from build_fulltext_index import s2_batch, pdf_candidates

GATED = "paper_survey/_census/census_gated.json"
OUT = "paper_survey/_census/fulltext_index.json"


def main():
    key = os.environ.get("S2_API_KEY")
    if not key:
        print("WARN: no S2_API_KEY in env (source .env)")
    papers = json.load(io.open(GATED, encoding="utf-8"))
    safe = [p for p in papers if p.get("is_safety")]

    rec, ids = {}, []
    for p in safe:
        s2 = p.get("s2_id")
        rec[p["id"]] = {"id": p["id"], "title": p.get("title"), "venue": p.get("venue"),
                        "year": p.get("year"), "category": p.get("category"),
                        "s2_id": s2, "pdf_urls": []}
        if s2:
            ids.append(s2)
    ids = sorted(set(ids))
    print("survivors: %d   unique s2_ids to resolve: %d" % (len(safe), len(ids)))

    meta = s2_batch(ids, key)
    for r in rec.values():
        if r["s2_id"] and r["s2_id"] in meta:
            r["pdf_urls"] = pdf_candidates(meta[r["s2_id"]])

    out = list(rec.values())
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    have = sum(1 for r in out if r["pdf_urls"])
    print("\n==== CENSUS INDEX ====")
    print("total safety: %d" % len(out))
    print("mapped to s2: %d" % sum(1 for r in out if r["s2_id"]))
    print("have >=1 pdf url: %d (%.0f%%)" % (have, 100 * have / len(out)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
