#!/usr/bin/env python3
"""
PILOT venue-expansion collector (v2 paper, sensitivity check for review concern #1).

Replicates the ORIGINAL query-pool method exactly -- same 72 DBLP queries,
same parse/filter -- but adds NLP and security venues so we can test whether the
robustness->alignment trend survives a broader venue set. Query-pool only
(keyless DBLP), which is precisely comparable to Table 2 in v1 (also query-pool).

Usage:
  python3 pipeline/01_collect/pilot_expand.py --out paper_survey_v2/pilot_candidates.json
"""
import argparse, json, sys, time
from pathlib import Path

sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
import scrape_ai_safety_papers as s

# DBLP conf prefixes for the pilot venues the review flagged.
PILOT_VENUES = {
    "conf/acl":  "ACL",        # NLP cluster (alignment / hallucination / toxicity)
    "conf/uss":  "USENIX-Sec", # security cluster (adversarial / backdoor / privacy)
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper_survey_v2/pilot_candidates.json")
    ap.add_argument("--sleep", type=float, default=4.0, help="DBLP throttle (s)")
    ap.add_argument("--hits", type=int, default=1000)
    args = ap.parse_args()

    # Extend the venue map so parse_dblp_hit keeps the new venues; we filter to
    # ONLY the pilot venues afterward (the original 8 are already in v1).
    s.VENUE_KEYS.update(PILOT_VENUES)
    pilot_names = set(PILOT_VENUES.values())

    seen, papers = set(), []
    for qi, q in enumerate(s.QUERIES, 1):
        print(f"[{qi}/{len(s.QUERIES)}] {q!r}", flush=True)
        try:
            hits = s.fetch_dblp(q, num_hits=args.hits)
        except Exception as e:
            print(f"  WARN fetch failed: {e}", flush=True)
            continue
        for h in hits:
            p = s.parse_dblp_hit(h)
            if not p or p["venue"] not in pilot_names:
                continue
            if p["dblp_key"] in seen:
                continue
            seen.add(p["dblp_key"])
            papers.append(p)
        time.sleep(args.sleep)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(papers, indent=2))

    # quick composition report
    by_venue, by_year = {}, {}
    for p in papers:
        by_venue[p["venue"]] = by_venue.get(p["venue"], 0) + 1
        by_year[p["year"]] = by_year.get(p["year"], 0) + 1
    print(f"\nTOTAL pilot candidates: {len(papers)}")
    print("by venue:", dict(sorted(by_venue.items())))
    print("by year :", dict(sorted(by_year.items())))
    print(f"written -> {outp}")


if __name__ == "__main__":
    main()
