#!/usr/bin/env python3
"""
Re-run DBLP collection with the EXPANDED query list and dedup against the
existing 1,941-paper set. Outputs only the NEW venue-matched candidates so we
can measure the recall gain from broader queries. Keyless (DBLP only).

Writes paper_survey/candidates_expanded.json (new papers, no category yet -
final labels come from the LLM classifier on title+abstract).
"""
import argparse, io, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from scrape_ai_safety_papers import fetch_dblp, parse_dblp_hit, QUERIES as ORIG_Q
from taxonomy_expanded import NEW_QUERIES

EXISTING = os.path.join(HERE, "..", "..", "paper_survey", "papers_all.json")
OUT = os.path.join(HERE, "..", "..", "paper_survey", "candidates_expanded.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="seconds between queries (DBLP IP-throttles; keep >=4)")
    args = ap.parse_args()

    existing = json.load(io.open(EXISTING, encoding="utf-8"))
    seen = {p["dblp_key"] for p in existing}
    print(f"existing papers: {len(existing)}  (dedup against these keys)")

    # Run the NEW queries (originals already collected, but include them as a
    # safety net is wasteful; NEW_QUERIES is the delta we care about).
    queries = NEW_QUERIES
    new_papers = []
    new_keys = set()
    for qi, q in enumerate(queries, 1):
        hits = fetch_dblp(q, num_hits=1000)
        added = 0
        for hit in hits:
            p = parse_dblp_hit(hit)
            if p is None:
                continue
            k = p["dblp_key"]
            if k in seen or k in new_keys:
                continue
            new_keys.add(k)
            new_papers.append(p)
            added += 1
        sys.stdout.write(f"[{qi:3d}/{len(queries)}] {q!r:50s} "
                         f"{len(hits):4d} hits, +{added} new\n")
        sys.stdout.flush()
        time.sleep(args.sleep)

    json.dump(new_papers, io.open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n{'='*60}")
    print(f"NEW candidate papers from expanded queries: {len(new_papers)}")
    print(f"(would grow set {len(existing)} -> {len(existing)+len(new_papers)}, "
          f"+{100*len(new_papers)/len(existing):.1f}%)")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
