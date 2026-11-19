#!/usr/bin/env python3
"""
Abstract enrichment for the title-screened snowball subset.

Unlike enrich_abstracts.py (keyed on DOID / title search), snowball candidates
already carry an S2 paperId from the references endpoint, so we resolve abstracts
through /paper/batch keyed on paperId (500/request) -- ~5 requests for the whole
safety subset instead of thousands of per-paper lookups. Candidates with neither
s2_id nor doi fall back to title search.

Reads S2_API_KEY from env (source .env first). Resumable: re-runs
reuse any record that already has an abstract. Writes the input records with
added/overwritten "abstract" field.

  set -a; . .env; set +a
  python3 pipeline/02_enrich/enrich_snowball.py \
      --in  paper_survey/snowball_safety_titlematched.json \
      --out paper_survey/snowball_enriched.json --sleep 1.1
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


def batch_by_id(s, ids, sleep):
    """Resolve paperIds via /paper/batch. Returns {paperId: record}."""
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for attempt in range(5):
            r = s.post(f"{S2}/paper/batch", params={"fields": FIELDS},
                       json={"ids": chunk}, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** attempt * 2); continue
            if r.status_code == 200:
                for pid, rec in zip(chunk, r.json()):
                    if rec:
                        out[pid] = rec
                break
            time.sleep(2 ** attempt)
        time.sleep(sleep)
        sys.stdout.write(f"\r  batch {min(i+500,len(ids))}/{len(ids)} "
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
    ap.add_argument("--sleep", type=float, default=1.1)
    args = ap.parse_args()

    papers = json.load(io.open(args.inp, encoding="utf-8"))
    s, keyed = session()
    if not keyed:
        print("WARNING: no S2_API_KEY in env; source .env first")
    print(f"enriching {len(papers)} papers  keyed={keyed}  sleep={args.sleep}s")

    # resume: keep abstracts already resolved in a prior run (cache by s2_id/title)
    cache = {}
    if os.path.exists(args.out):
        for p in json.load(io.open(args.out, encoding="utf-8")):
            if p.get("abstract"):
                cache[p.get("s2_id") or p["title"].lower()] = p["abstract"]

    todo = [p for p in papers
            if not cache.get(p.get("s2_id") or p["title"].lower())]
    ids = [p["s2_id"] for p in todo if p.get("s2_id")]
    by_id = batch_by_id(s, ids, args.sleep) if ids else {}

    no_id = [p for p in todo if not p.get("s2_id")]
    for p in no_id:
        rec = title_search(s, p["title"], args.sleep)
        if rec:
            by_id[p["title"].lower()] = rec  # store under title key

    abs_n = 0
    for p in papers:
        ck = p.get("s2_id") or p["title"].lower()
        if cache.get(ck):
            p["abstract"] = cache[ck]
        else:
            rec = by_id.get(p.get("s2_id")) or by_id.get(p["title"].lower())
            p["abstract"] = rec.get("abstract") if rec else None
        if p.get("abstract"):
            abs_n += 1

    json.dump(papers, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nwith_abstract={abs_n}/{len(papers)} ({100*abs_n/len(papers):.1f}%)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
