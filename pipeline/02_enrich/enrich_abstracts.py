#!/usr/bin/env python3
"""
Full abstract enrichment via Semantic Scholar. Uses the /paper/batch endpoint
(up to 500 IDs/request) keyed on DOI, falling back to title search for papers
without a DOI. Reads an S2 API key from the S2_API_KEY env var if set (kills the
429s); works keyless but slow.

Usage:
  S2_API_KEY=xxx python3 enrich_abstracts.py --in paper_survey/papers_all.json \
       --out paper_survey/papers_enriched.json

Writes the input records with added "abstract" and "s2_id" fields (null when
unresolved). Caches resolved records so re-runs are cheap and resumable.
"""
import argparse, io, json, os, sys, time
import requests

S2 = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,externalIds,venue,year"


def session():
    s = requests.Session()
    key = os.environ.get("S2_API_KEY")
    if key:
        s.headers["x-api-key"] = key
    return s, bool(key)


def batch_by_doi(s, dois, sleep):
    """Resolve a list of DOIs via the batch endpoint. Returns {doi: record}."""
    out = {}
    for i in range(0, len(dois), 500):
        chunk = dois[i:i + 500]
        for attempt in range(5):
            r = s.post(f"{S2}/paper/batch", params={"fields": FIELDS},
                       json={"ids": [f"DOI:{d}" for d in chunk]}, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2); continue
            if r.status_code == 200:
                for d, rec in zip(chunk, r.json()):
                    if rec:
                        out[d] = rec
                break
            time.sleep(2 ** attempt)
        time.sleep(sleep)
        sys.stdout.write(f"\r  batch {min(i+500,len(dois))}/{len(dois)} "
                         f"resolved={len(out)}"); sys.stdout.flush()
    print()
    return out


def title_search(s, title, sleep):
    for attempt in range(4):
        r = s.get(f"{S2}/paper/search",
                  params={"query": title, "fields": FIELDS, "limit": 1}, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** attempt * 2); continue
        if r.status_code == 200:
            hits = r.json().get("data", [])
            time.sleep(sleep)
            return hits[0] if hits else None
        time.sleep(2 ** attempt)
    time.sleep(sleep)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=None,
                    help="per-request sleep; default 0.1 keyed, 1.1 keyless")
    args = ap.parse_args()

    papers = json.load(io.open(args.inp, encoding="utf-8"))
    s, keyed = session()
    # S2 introductory keys are still 1 req/sec cumulative, so keep >=1.1s either way.
    sleep = args.sleep if args.sleep is not None else 1.1
    print(f"enriching {len(papers)} papers  keyed={keyed}  sleep={sleep}s")

    # resume from existing output
    cache = {}
    if os.path.exists(args.out):
        for p in json.load(io.open(args.out, encoding="utf-8")):
            if p.get("abstract") is not None or p.get("s2_id"):
                cache[p["dblp_key"]] = p

    dois = [p["doi"] for p in papers
            if p.get("doi") and not p.get("abstract") and p["dblp_key"] not in cache]
    by_doi = batch_by_doi(s, dois, sleep) if dois else {}

    resolved = no_abs = kept = 0
    for p in papers:
        if p.get("abstract"):  # already enriched upstream (e.g. 2025 cohorts) -- never clobber
            p.setdefault("s2_id", None)
            kept += 1; resolved += 1; continue
        if p["dblp_key"] and p["dblp_key"] in cache:
            p.update({k: cache[p["dblp_key"]].get(k)
                      for k in ("abstract", "s2_id")})
            resolved += 1; continue
        rec = by_doi.get(p.get("doi"))
        if rec is None and p.get("title"):
            rec = title_search(s, p["title"], sleep)
        if rec:
            p["abstract"] = rec.get("abstract")
            p["s2_id"] = rec.get("paperId")
            resolved += 1
            if not rec.get("abstract"):
                no_abs += 1
        else:
            p["abstract"] = None
            p["s2_id"] = None

    json.dump(papers, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with_abs = sum(1 for p in papers if p.get("abstract"))
    print(f"\nresolved={resolved}/{len(papers)}  with_abstract={with_abs} "
          f"({100*with_abs/len(papers):.1f}%)  kept_preexisting={kept}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
