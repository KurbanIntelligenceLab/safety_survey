#!/usr/bin/env python
"""Build the full-text acquisition index for the safety corpus.

Joins each classified safety paper (stable `id` in papers_llm_cleaned.json) to an
S2 paper id (via DOI, then normalized title, against the enriched pools), then
batch-queries Semantic Scholar for openAccessPdf URLs + arXiv ids. Output is a
resumable index: id -> {title, venue, year, s2_id, pdf_urls:[...]} where pdf_urls
is an ordered, deduped list of candidate PDF URLs (diverse hosts first, arXiv
constructed from the external id last). The fetcher consumes this; no PDFs are
downloaded here.
"""
import json, io, re, os, sys, time
import requests

SRC = "paper_survey/papers_llm_cleaned.json"
ENRICHED = ["papers_enriched.json", "snowball_enriched.json", "snowball_nlp_enriched.json"]
OUT = "paper_survey/fulltext_index.json"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"


def load(f):
    d = json.load(io.open(os.path.join("paper_survey", f), encoding="utf-8"))
    return d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def build_s2_index():
    doi2, title2 = {}, {}
    for f in ENRICHED:
        for r in load(f):
            s2 = r.get("s2_id")
            if not s2:
                continue
            if r.get("doi"):
                doi2[r["doi"].lower()] = s2
            title2[norm(r.get("title"))] = s2
    return doi2, title2


def s2_batch(ids, key, fields="openAccessPdf,externalIds"):
    hdr = {"x-api-key": key} if key else {}
    out = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        for attempt in range(6):
            try:
                r = requests.post(S2_BATCH, params={"fields": fields},
                                  json={"ids": chunk}, headers=hdr, timeout=90)
            except Exception as e:
                print("  batch req_err %s (retry %d)" % (str(e)[:50], attempt))
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                for sid, p in zip(chunk, r.json()):
                    out[sid] = p or {}
                break
            if r.status_code in (429, 503):
                wait = 2 ** attempt
                print("  http %d, backoff %ds" % (r.status_code, wait))
                time.sleep(wait)
                continue
            print("  http %d: %s" % (r.status_code, r.text[:120]))
            break
        print("  resolved %d/%d" % (min(i + 100, len(ids)), len(ids)))
        time.sleep(1.2)
    return out


def pdf_candidates(p):
    urls = []
    oa = (p.get("openAccessPdf") or {}).get("url")
    if oa:
        urls.append(oa)
    arx = (p.get("externalIds") or {}).get("ArXiv")
    if arx:
        urls.append("https://arxiv.org/pdf/%s" % arx)
    # dedup preserving order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def main():
    key = os.environ.get("S2_API_KEY")
    if not key:
        print("WARN: no S2_API_KEY in env (source .env)")
    doi2, title2 = build_s2_index()
    safe = [r for r in load(os.path.basename(SRC)) if r.get("is_safety")]
    print("safety papers: %d" % len(safe))

    # join to s2
    rec, ids = {}, []
    for r in safe:
        s2 = doi2.get((r.get("doi") or "").lower()) or title2.get(norm(r.get("title")))
        rec[r["id"]] = {"id": r["id"], "title": r.get("title"), "venue": r.get("venue"),
                        "year": r.get("year"), "category": r.get("category"),
                        "s2_id": s2, "pdf_urls": []}
        if s2:
            ids.append(s2)
    ids = sorted(set(ids))
    print("unique s2_ids to resolve: %d" % len(ids))

    meta = s2_batch(ids, key)
    for r in rec.values():
        if r["s2_id"] and r["s2_id"] in meta:
            r["pdf_urls"] = pdf_candidates(meta[r["s2_id"]])

    out = list(rec.values())
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    have = sum(1 for r in out if r["pdf_urls"])
    print("\n==== INDEX ====")
    print("total safety: %d" % len(out))
    print("mapped to s2: %d" % sum(1 for r in out if r["s2_id"]))
    print("have >=1 pdf url: %d (%.0f%%)" % (have, 100 * have / len(out)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
