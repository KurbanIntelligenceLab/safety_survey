#!/usr/bin/env python3
"""
NLP-venue snowball (clean-fix follow-up). The 10-venue merge added ACL and EMNLP
through the query pool only; this runs the citation snowball seeded by the
ACL/EMNLP papers so those venues get the same citation-graph recovery the other
eight already had. Keeps reference neighbors at ANY of the ten venues (so it also
surfaces new NeurIPS/ICLR/etc. neighbors that ACL/EMNLP papers cite) that are NOT
already in the corpus.

Mirrors snowball.py (S2 /paper/{id}/references), but:
  - venue map includes ACL and EMNLP,
  - seeds come from --seeds, dedup is against the FULL corpus --dedup.

  set -a; . .env; set +a
  python3 pipeline/01_collect/snowball_nlp.py \
      --seeds paper_survey_v2/v2_enriched.json \
      --dedup paper_survey/papers_llm.json \
      --out paper_survey/snowball_nlp_candidates.json
"""
import argparse, io, json, os, re, sys, time
from collections import defaultdict
import requests

S2 = "https://api.semanticscholar.org/graph/v1"
REF_FIELDS = "title,year,venue,externalIds"
SEED_VENUES = {"ACL", "EMNLP"}

# EMNLP before ACL so "Findings ... : EMNLP" maps to EMNLP, not ACL.
VENUE_PAT = [
    (re.compile(r"\bcvpr\b|computer vision and pattern", re.I), "CVPR"),
    (re.compile(r"\biccv\b|international conference on computer vision", re.I), "ICCV"),
    (re.compile(r"\bicml\b|international conference on machine learning", re.I), "ICML"),
    (re.compile(r"\biclr\b|learning representations", re.I), "ICLR"),
    (re.compile(r"neurips|nips|neural information processing", re.I), "NeurIPS"),
    (re.compile(r"\baaai\b", re.I), "AAAI"),
    (re.compile(r"\baies\b|aaai/acm conference on ai", re.I), "AIES"),
    (re.compile(r"facct|fairness, accountability", re.I), "FAccT"),
    (re.compile(r"emnlp|empirical methods in natural language processing", re.I), "EMNLP"),
    (re.compile(r"\bacl\b|annual meeting of the association for computational linguistics", re.I), "ACL"),
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
        try:
            r = s.get(f"{S2}/paper/{pid}/references",
                      params={"fields": REF_FIELDS, "limit": 1000}, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt); continue
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
    ap.add_argument("--seeds", required=True, help="enriched json with s2_id (ACL/EMNLP)")
    ap.add_argument("--dedup", required=True, help="full corpus to dedup against")
    ap.add_argument("--out", default="paper_survey/snowball_nlp_candidates.json")
    ap.add_argument("--sleep", type=float, default=1.1)
    ap.add_argument("--ymin", type=int, default=2018)
    ap.add_argument("--ymax", type=int, default=2026)
    args = ap.parse_args()

    seeds_all = json.load(io.open(args.seeds, encoding="utf-8"))
    seeds = [p for p in seeds_all if p.get("venue") in SEED_VENUES
             and (p.get("s2_id") or p.get("doi"))]
    corpus = json.load(io.open(args.dedup, encoding="utf-8"))
    have_doi = {(p.get("doi") or "").lower() for p in corpus if p.get("doi")}
    have_title = {(p.get("title") or "").lower() for p in corpus}
    # also dedup against seeds themselves
    have_title |= {(p.get("title") or "").lower() for p in seeds}

    s, keyed = session()
    print(f"snowballing from {len(seeds)} ACL/EMNLP seeds  keyed={keyed}  "
          f"sleep={args.sleep}s  dedup-corpus={len(corpus)}", flush=True)

    cite_count = defaultdict(int)
    cand = {}
    by_v = defaultdict(int)
    for i, p in enumerate(seeds, 1):
        pid = p.get("s2_id") or f"DOI:{p['doi']}"
        for ref in get_refs(s, pid, args.sleep):
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
            if key not in cand:
                by_v[v] += 1
            cite_count[key] += 1
            cand[key] = {"title": title, "year": y, "venue": v, "doi": doi,
                         "s2_id": ref.get("paperId")}
        if i % 25 == 0:
            sys.stdout.write(f"\r  {i}/{len(seeds)} seeds  candidates={len(cand)}")
            sys.stdout.flush()
    print()

    ranked = sorted(({**cand[k], "cited_by_n": cite_count[k]} for k in cand),
                    key=lambda c: -c["cited_by_n"])
    json.dump(ranked, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"new venue-matched snowball candidates: {len(ranked)}")
    print("  by venue:", dict(sorted(by_v.items())))
    print(f"  cited by >=2 of the seeds: {sum(1 for c in ranked if c['cited_by_n']>=2)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
