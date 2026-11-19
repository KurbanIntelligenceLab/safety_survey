#!/usr/bin/env python3
"""
Citation snowball (review follow-up): pull the references of every paper in our
set, keep those published at our eight venues in-window that are NOT already in
the set. Papers heavily cited *by* safety papers are likely safety papers too -
this catches the "unusual title" recall gap that keyword search misses.

Uses S2 /paper/{id}/references. Needs s2_id (run enrich_abstracts.py first) or
falls back to DOI. Reads S2_API_KEY from env. Ranks new candidates by how many
of our papers cite them (citation frequency = relevance signal).

Writes paper_survey/snowball_candidates.json (new venue-matched papers + cite
count), sorted by cite count desc.
"""
import argparse, io, json, os, re, sys, time
from collections import defaultdict
import requests

S2 = "https://api.semanticscholar.org/graph/v1"
REF_FIELDS = "title,year,venue,externalIds"

# venue string (S2 'venue' field) -> our canonical venue
VENUE_PAT = [
    (re.compile(r"\bcvpr\b|computer vision and pattern", re.I), "CVPR"),
    (re.compile(r"\biccv\b|international conference on computer vision", re.I), "ICCV"),
    (re.compile(r"\bicml\b|international conference on machine learning", re.I), "ICML"),
    (re.compile(r"\biclr\b|learning representations", re.I), "ICLR"),
    (re.compile(r"neurips|nips|neural information processing", re.I), "NeurIPS"),
    (re.compile(r"\baaai\b", re.I), "AAAI"),
    (re.compile(r"\baies\b|aaai/acm conference on ai", re.I), "AIES"),
    (re.compile(r"facct|fairness, accountability", re.I), "FAccT"),
]


def map_venue(v):
    if not v:
        return None
    for pat, name in VENUE_PAT:
        if pat.search(v):
            return name
    return None


def session():
    s = requests.Session()
    key = os.environ.get("S2_API_KEY")
    if key:
        s.headers["x-api-key"] = key
    return s, bool(key)


def get_refs(s, pid, sleep):
    for attempt in range(4):
        r = s.get(f"{S2}/paper/{pid}/references",
                  params={"fields": REF_FIELDS, "limit": 1000}, timeout=60)
        if r.status_code == 429:
            time.sleep(2 ** attempt * 2); continue
        if r.status_code == 200:
            time.sleep(sleep)
            data = r.json()
            rows = data.get("data") if isinstance(data, dict) else None
            return [(d.get("citedPaper") or {}) for d in (rows or [])]
        time.sleep(2 ** attempt)
    time.sleep(sleep)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="enriched json (must have s2_id) or papers_all.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sleep", type=float, default=None)
    ap.add_argument("--ymin", type=int, default=2018)
    ap.add_argument("--ymax", type=int, default=2026)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.inp), "snowball_candidates.json")

    papers = json.load(io.open(args.inp, encoding="utf-8"))
    have_doi = {(p.get("doi") or "").lower() for p in papers if p.get("doi")}
    have_title = {p["title"].lower() for p in papers}
    s, keyed = session()
    # S2 introductory keys are still 1 req/sec cumulative, so keep >=1.1s either way.
    sleep = args.sleep if args.sleep is not None else 1.1

    seeds = [p for p in papers if p.get("s2_id") or p.get("doi")]
    print(f"snowballing from {len(seeds)} seeds  keyed={keyed}  sleep={sleep}s")

    cite_count = defaultdict(int)
    cand = {}
    for i, p in enumerate(seeds, 1):
        pid = p.get("s2_id") or f"DOI:{p['doi']}"
        for ref in get_refs(s, pid, sleep):
            v = map_venue(ref.get("venue"))
            y = ref.get("year")
            if v is None or not y or y < args.ymin or y > args.ymax:
                continue
            title = (ref.get("title") or "").strip()
            doi = (ref.get("externalIds") or {}).get("DOI", "")
            if not title or title.lower() in have_title or \
               (doi and doi.lower() in have_doi):
                continue
            key = doi.lower() or title.lower()
            cite_count[key] += 1
            cand[key] = {"title": title, "year": y, "venue": v, "doi": doi,
                         "s2_id": ref.get("paperId")}
        if i % 25 == 0:
            sys.stdout.write(f"\r  {i}/{len(seeds)} seeds  candidates={len(cand)}")
            sys.stdout.flush()
    print()

    ranked = sorted(({**cand[k], "cited_by_n": cite_count[k]} for k in cand),
                    key=lambda c: -c["cited_by_n"])
    json.dump(ranked, io.open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"new venue-matched snowball candidates: {len(ranked)}")
    print(f"  cited by >=3 of our papers: {sum(1 for c in ranked if c['cited_by_n']>=3)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
