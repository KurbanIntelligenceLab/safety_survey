#!/usr/bin/env python3
"""
SMOKE TEST for abstract enrichment via the Semantic Scholar Graph API
(review follow-up: "search abstracts, not just titles").

Goal: measure, on a small sample, what fraction of our papers S2 can resolve
(by DOI, else by title) and how many actually carry an abstract. This gives a
real estimate of the recall upside BEFORE committing to a full 1,941-paper or
30k-venue run.

Keyless S2 access is heavily rate-limited; we sleep between calls and only do
--n papers. Writes a small JSON report; does not modify the dataset.
"""
import argparse, io, json, os, sys, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "paper_survey", "papers_all.json")
S2 = "https://api.semanticscholar.org/graph/v1/paper"


def lookup(paper, fields="title,abstract,externalIds"):
    doi = (paper.get("doi") or "").strip()
    try:
        if doi:
            r = requests.get(f"{S2}/DOI:{doi}", params={"fields": fields}, timeout=30)
            if r.status_code == 200:
                return r.json(), "doi"
        # fall back to title search
        r = requests.get(f"{S2}/search", params={
            "query": paper["title"], "fields": fields, "limit": 1}, timeout=30)
        if r.status_code == 200:
            hits = r.json().get("data", [])
            if hits:
                return hits[0], "title"
        return None, f"status_{r.status_code}"
    except requests.RequestException as e:
        return None, f"err_{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sleep", type=float, default=1.2)
    args = ap.parse_args()

    papers = json.load(io.open(DATA, encoding="utf-8"))[: args.n]
    resolved = with_abstract = 0
    via = {"doi": 0, "title": 0}
    fails = []
    for i, p in enumerate(papers, 1):
        rec, how = lookup(p)
        if rec is not None:
            resolved += 1
            via[how] = via.get(how, 0) + 1
            if rec.get("abstract"):
                with_abstract += 1
        else:
            fails.append(how)
        sys.stdout.write(f"\r  {i}/{len(papers)} resolved={resolved} "
                         f"abstract={with_abstract}")
        sys.stdout.flush()
        time.sleep(args.sleep)
    print()
    rep = {
        "sampled": len(papers),
        "resolved": resolved,
        "resolved_pct": round(100 * resolved / len(papers), 1),
        "with_abstract": with_abstract,
        "with_abstract_pct": round(100 * with_abstract / len(papers), 1),
        "via": via,
        "fail_reasons": dict(__import__("collections").Counter(fails)),
    }
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
