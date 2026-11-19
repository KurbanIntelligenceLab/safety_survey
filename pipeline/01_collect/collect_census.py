#!/usr/bin/env python3
"""
Full per-venue-year proceedings CENSUS via DBLP stream queries.

Unlike scrape_ai_safety_papers.py (72 safety KEYWORD queries -> recall-limited
"narrow net"), this pulls the COMPLETE table of contents for every venue x year
and applies NO safety filter at collection time. The safety screen + LLM classify
happen downstream, exactly as they did for the NeurIPS/AIES-2025 backfill. This
is what makes 2025 comparable to 2018-2024: every year is censused identically.

Source: DBLP publ search API with `stream:streams/conf/<key>: year:<Y>:`, which
returns the full venue-year population (verified: NeurIPS-2023 -> 3541, identical
to the toc: query). Venue assignment reuses parse_dblp_hit / VENUE_KEYS so the
census uses the SAME venue definition as the original pool.

Resumable: one shard json per (venue,year) under _census/shards/; reruns skip
shards already on disk. Merge step writes _census/census_candidates.json.

  python3 pipeline/01_collect/collect_census.py collect   # paginate DBLP, write shards
  python3 pipeline/01_collect/collect_census.py merge      # merge shards -> candidates
"""
import io, json, os, sys, time
from pathlib import Path

import requests

sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from scrape_ai_safety_papers import parse_dblp_hit, venue_from_dblp_key  # noqa: E402

DBLP = "https://dblp.org/search/publ/api"
YEARS = list(range(2018, 2026))                 # 2018..2025 inclusive
# venue display name -> DBLP stream key (all verified 2026-06-11)
VENUE_STREAMS = {
    "CVPR": "cvpr", "ICCV": "iccv", "ICML": "icml", "ICLR": "iclr",
    "NeurIPS": "nips", "AAAI": "aaai", "AIES": "aies", "FAccT": "fat",
    "ACL": "acl", "EMNLP": "emnlp",
}
PAGE = 100                                       # DBLP caps hits/page at 100 (verified)
SLEEP = 3.0                                       # DBLP IP-throttles; keep >=2.5
OUTDIR = Path("paper_survey/_census")
SHARDS = OUTDIR / "shards"

S = requests.Session()
S.headers["User-Agent"] = "safety-survey-census/1.0 (research; %s)" % os.environ.get("CONTACT_EMAIL", "set CONTACT_EMAIL")


def dblp_page(stream, year, offset, tries=8):
    """One page of a venue-year stream query. Returns (total, hits) or (None, [])."""
    params = {"q": f"stream:streams/conf/{stream}: year:{year}:",
              "h": PAGE, "f": offset, "format": "json"}
    for a in range(tries):
        try:
            r = S.get(DBLP, params=params, timeout=60)
            if r.status_code == 429:
                w = 6 * 2 ** a
                print(f"    429, wait {w}s", flush=True); time.sleep(w); continue
            r.raise_for_status()
            hits = r.json()["result"]["hits"]
            total = int(hits.get("@total", 0))
            return total, hits.get("hit", [])
        except Exception as e:
            w = 5 * 2 ** a
            print(f"    err {str(e)[:60]} retry {a} in {w}s", flush=True); time.sleep(w)
    return None, []


def collect_venue_year(venue, stream, year):
    """Paginate the full stream for one venue-year. Returns list of parsed records."""
    recs, seen, offset, total = [], set(), 0, None
    while True:
        t, hits = dblp_page(stream, year, offset)
        if t is None:
            raise RuntimeError(f"{venue} {year}: page at f={offset} failed after retries")
        total = t
        if not hits:
            break
        for h in hits:
            p = parse_dblp_hit(h)               # filters year range + assigns venue
            if p is None or p["venue"] != venue:
                continue                        # keep only THIS venue's papers
            k = p["dblp_key"]
            if k in seen:
                continue
            seen.add(k); recs.append(p)
        offset += PAGE
        time.sleep(SLEEP)
        if offset >= total:
            break
    return recs, total


def collect():
    SHARDS.mkdir(parents=True, exist_ok=True)
    summary, failed = [], []
    for venue, stream in VENUE_STREAMS.items():
        for year in YEARS:
            shard = SHARDS / f"{venue}_{year}.json"
            if shard.exists():
                n = len(json.load(io.open(shard, encoding="utf-8")))
                print(f"[skip] {venue} {year}: shard exists ({n})", flush=True)
                summary.append((venue, year, n, "cached"))
                continue
            print(f"[get ] {venue} {year} ...", flush=True)
            try:
                recs, total = collect_venue_year(venue, stream, year)
            except RuntimeError as e:
                # Non-fatal: skip this venue-year (no shard written, so it is
                # retried on the next pass) and keep collecting the rest. DBLP
                # throttling should not abort the whole 80-shard census.
                print(f"    FAIL {e} -- skipping, will retry next pass", flush=True)
                failed.append((venue, year))
                continue
            json.dump(recs, io.open(shard, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"       {venue} {year}: stream_total={total} kept={len(recs)}", flush=True)
            summary.append((venue, year, len(recs), total))
    print("\n=== COLLECT SUMMARY (kept / stream_total) ===")
    for v, y, k, t in summary:
        print(f"  {v:8} {y}: {k}")
    if failed:
        print(f"\n=== {len(failed)} venue-years FAILED (rerun to retry) ===")
        for v, y in failed:
            print(f"  {v} {y}")
        sys.exit(2)                                  # signal incomplete to wrapper


def merge():
    shards = sorted(SHARDS.glob("*.json"))
    allrecs, seen = [], set()
    for sh in shards:
        for p in json.load(io.open(sh, encoding="utf-8")):
            k = p["dblp_key"]
            if k in seen:
                continue
            seen.add(k); allrecs.append(p)
    out = OUTDIR / "census_candidates.json"
    json.dump(allrecs, io.open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    from collections import Counter
    print(f"merged {len(shards)} shards -> {len(allrecs)} unique census papers")
    print("by year:", dict(sorted(Counter(str(p['year']) for p in allrecs).items())))
    print("by venue:", dict(Counter(p['venue'] for p in allrecs).most_common()))
    print(f"wrote {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "collect"
    {"collect": collect, "merge": merge}[cmd]()
