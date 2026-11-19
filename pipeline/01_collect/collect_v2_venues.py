#!/usr/bin/env python3
"""
v2 venue-expansion collector: NLP + security venues, query-pool, keyless DBLP.
Same 72 queries / parse as v1 (one query pass returns all venues; we filter
after). Trailing-slash prefixes avoid false matches (conf/sp vs conf/spaa).

  python3 pipeline/01_collect/collect_v2_venues.py --out paper_survey_v2/v2_candidates.json
"""
import argparse, json, sys, time
from pathlib import Path
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
import scrape_ai_safety_papers as s

NEW_VENUES = {
    "conf/acl/":    "ACL",
    "conf/emnlp/":  "EMNLP",
    "conf/naacl/":  "NAACL",
    "conf/uss/":    "USENIX-Sec",
    "conf/sp/":     "IEEE-S&P",
    "conf/ccs/":    "CCS",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper_survey_v2/v2_candidates.json")
    ap.add_argument("--sleep", type=float, default=4.0)
    ap.add_argument("--hits", type=int, default=1000)
    args = ap.parse_args()

    s.VENUE_KEYS.update(NEW_VENUES)
    new_names = set(NEW_VENUES.values())

    seen, papers = set(), []
    for qi, q in enumerate(s.QUERIES, 1):
        print(f"[{qi}/{len(s.QUERIES)}] {q!r}", flush=True)
        try:
            hits = s.fetch_dblp(q, num_hits=args.hits)
        except Exception as e:
            print(f"  WARN: {e}", flush=True); continue
        for h in hits:
            p = s.parse_dblp_hit(h)
            if not p or p["venue"] not in new_names or p["dblp_key"] in seen:
                continue
            seen.add(p["dblp_key"]); papers.append(p)
        time.sleep(args.sleep)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(papers, indent=2))
    by_v, by_y = {}, {}
    for p in papers:
        by_v[p["venue"]] = by_v.get(p["venue"], 0) + 1
        by_y[p["year"]] = by_y.get(p["year"], 0) + 1
    print(f"\nTOTAL: {len(papers)}")
    print("by venue:", dict(sorted(by_v.items())))
    print("by year :", dict(sorted(by_y.items())))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
