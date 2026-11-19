#!/usr/bin/env python
"""Build the full-text acquisition index for the FULL 12,987-paper corpus.

Keyed by the new corpus id (corpus_clusterlabeled.json, 0..12986). Re-uses the
5,354 closing sections already fetched for the old query-pool/snowball stream by
title-joining them into the new id space (pre-populated into the new sections dir
so the fetcher's resume logic skips them). For the ~7,600 papers without a section
it attaches pdf_urls: via Semantic Scholar batch for those with an s2_id, and a
url/doi fallback (arXiv / OpenReview / ACL Anthology) for those without.

Outputs:
  paper_survey/_fulltext12987/fulltext_index.json   (id -> {.., s2_id, pdf_urls})
  paper_survey/_fulltext12987/sections/<id>.json     (pre-copied existing sections)
"""
import json, io, os, re, sys, collections
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from build_fulltext_index import s2_batch, pdf_candidates

PS = "paper_survey"
CORPUS = os.path.join(PS, "_census/corpus_clusterlabeled.json")
OLD_IDX = os.path.join(PS, "fulltext_index.json")
OLD_SEC = os.path.join(PS, "_fulltext/sections")
NEW_ROOT = os.path.join(PS, "_fulltext12987")
NEW_SEC = os.path.join(NEW_ROOT, "sections")
OUT = os.path.join(NEW_ROOT, "fulltext_index.json")

# files that carry s2_id, for title-join resolution
S2_SOURCES = ["_census/census_enriched.json", "fulltext_index.json"]
# files that carry url/doi, for the no-s2 fallback
URL_SOURCES = ["_census/census_gated_dedup.json", "_neurips2025/all_papers.json",
               "_aies2025/all_papers.json", "_neurips2025/neurips2025_candidates.json",
               "_aies2025/aies2025_candidates.json"]


def norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:80]


def jload(p):
    try:
        return json.load(io.open(os.path.join(PS, p), encoding="utf-8"))
    except Exception as e:
        print("  skip %s: %s" % (p, str(e)[:50]))
        return []


def arxiv_from_url(u):
    if not u:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", u)
    if m:
        return "https://arxiv.org/pdf/%s" % m.group(1)
    m = re.search(r"openreview\.net/forum\?id=([\w-]+)", u)
    if m:
        return "https://openreview.net/pdf?id=%s" % m.group(1)
    m = re.search(r"openreview\.net/pdf\?id=([\w-]+)", u)
    if m:
        return u
    m = re.search(r"aclanthology\.org/([\w.-]+?)/?$", u)
    if m:
        return "https://aclanthology.org/%s.pdf" % m.group(1).rstrip("/")
    if u.lower().endswith(".pdf"):
        return u
    return None


def main():
    key = os.environ.get("S2_API_KEY")
    os.makedirs(NEW_SEC, exist_ok=True)
    corpus = jload("_census/corpus_clusterlabeled.json")
    print("corpus N = %d" % len(corpus))

    # 1) title -> s2_id
    title2s2 = {}
    for f in S2_SOURCES:
        for r in jload(f):
            s2, t = r.get("s2_id"), norm(r.get("title"))
            if s2 and t and t not in title2s2:
                title2s2[t] = s2
    print("title->s2 entries: %d" % len(title2s2))

    # 2) title -> url/doi (fallback)
    title2url, title2doi = {}, {}
    for f in URL_SOURCES:
        for r in jload(f):
            t = norm(r.get("title"))
            if not t:
                continue
            if r.get("url") and t not in title2url:
                title2url[t] = r["url"]
            if r.get("doi") and t not in title2doi:
                title2doi[t] = r["doi"]

    # 3) title -> already-fetched section (old id space)
    old_idx = jload("fulltext_index.json")
    oldid2title = {r["id"]: norm(r["title"]) for r in old_idx}
    title2sec = {}
    have_secfiles = set(int(f[:-5]) for f in os.listdir(OLD_SEC) if f.endswith(".json"))
    for oid, t in oldid2title.items():
        if oid in have_secfiles and t and t not in title2sec:
            title2sec[t] = oid

    # 4) walk corpus; pre-copy existing sections, build fetch index for the rest
    out, need_s2, fallback = [], [], 0
    reused = 0
    for p in corpus:
        cid, t = p["id"], norm(p["title"])
        rec = {"id": cid, "title": p.get("title"), "venue": p.get("venue"),
               "year": p.get("year"), "category": p.get("category"),
               "source": p.get("source"), "s2_id": title2s2.get(t), "pdf_urls": []}
        if t in title2sec:
            # re-key the existing section into the new id space
            old = json.load(io.open(os.path.join(OLD_SEC, "%d.json" % title2sec[t]), encoding="utf-8"))
            old["id"] = cid
            json.dump(old, io.open(os.path.join(NEW_SEC, "%d.json" % cid), "w", encoding="utf-8"),
                      ensure_ascii=False)
            reused += 1
            out.append(rec)
            continue
        # needs fetching
        if rec["s2_id"]:
            need_s2.append(rec)
        else:
            u = arxiv_from_url(title2url.get(t)) or arxiv_from_url(title2doi.get(t) and "https://doi.org/" + title2doi[t])
            if u:
                rec["pdf_urls"] = [u]
                fallback += 1
        out.append(rec)

    print("reused existing sections: %d" % reused)
    print("need fetch w/ s2_id: %d   url/doi fallback urls: %d" % (len(need_s2), fallback))

    # 5) resolve pdf_urls for the s2 set
    ids = sorted({r["s2_id"] for r in need_s2})
    print("unique s2_ids to resolve: %d" % len(ids))
    meta = s2_batch(ids, key)
    for r in need_s2:
        if r["s2_id"] in meta:
            r["pdf_urls"] = pdf_candidates(meta[r["s2_id"]])

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    to_fetch = [r for r in out if not (norm(r["title"]) in title2sec) and r["pdf_urls"]]
    no_url = [r for r in out if not (norm(r["title"]) in title2sec) and not r["pdf_urls"]]
    print("\n==== 12,987 INDEX ====")
    print("total: %d  reused sections: %d  fetchable (have url): %d  no url: %d"
          % (len(out), reused, len(to_fetch), len(no_url)))
    print("no-url by source:", dict(collections.Counter(str(r["source"]) for r in no_url)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
