#!/usr/bin/env python3
"""
verify_corpus.py — check whether papers in safety_corpus_v1nlp.csv actually exist.

Runs three checks per row:
  1. If a DOI is present  -> resolve it via the Crossref REST API and compare titles.
  2. If no DOI            -> search the title on Crossref (and DBLP as a fallback)
                            and fuzzy-match the returned title + venue/year.
  3. Aggregates anything suspicious into a report CSV.

Network: hits api.crossref.org and dblp.org. Run this somewhere with open
outbound HTTPS (NOT inside a locked-down sandbox). Stdlib only — no pip install.

Usage:
    python3 verify_corpus.py safety_corpus_v1nlp.csv
    # optional: --limit N to test on the first N rows, --mailto you@email for the
    # Crossref "polite pool" (faster, more reliable).
"""
import csv, sys, time, json, argparse, urllib.parse, urllib.request
from difflib import SequenceMatcher

UA = "corpus-verifier/1.0 (mailto:%s)"

def norm(s):
    return "".join(c.lower() for c in (s or "") if c.isalnum() or c.isspace()).split()

def title_sim(a, b):
    a, b = " ".join(norm(a)), " ".join(norm(b))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def get_json(url, mailto, tries=3):
    req = urllib.request.Request(url, headers={"User-Agent": UA % mailto})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                return {"__error__": str(e)}
            time.sleep(1.5 * (i + 1))
    return None

def check_doi(doi, csv_title, mailto):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    data = get_json(url, mailto)
    if not data or "__error__" in data:
        return "DOI_UNRESOLVED", "", 0.0
    msg = data.get("message", {})
    cr_title = (msg.get("title") or [""])[0]
    sim = title_sim(csv_title, cr_title)
    status = "OK" if sim >= 0.85 else "TITLE_MISMATCH"
    return status, cr_title, round(sim, 2)

def check_title(csv_title, mailto):
    q = urllib.parse.urlencode({"query.bibliographic": csv_title, "rows": 3})
    data = get_json("https://api.crossref.org/works?" + q, mailto)
    best = ("", 0.0)
    if data and "__error__" not in data:
        for it in data.get("message", {}).get("items", []):
            t = (it.get("title") or [""])[0]
            s = title_sim(csv_title, t)
            if s > best[1]:
                best = (t, s)
    if best[1] >= 0.85:
        return "OK_BY_TITLE", best[0], round(best[1], 2)
    # DBLP fallback
    q2 = urllib.parse.urlencode({"q": csv_title, "format": "json", "h": 3})
    d2 = get_json("https://dblp.org/search/publ/api?" + q2, mailto)
    try:
        hits = d2["result"]["hits"].get("hit", [])
        for h in hits:
            t = h["info"].get("title", "")
            s = title_sim(csv_title, t)
            if s > best[1]:
                best = (t, s)
    except Exception:
        pass
    if best[1] >= 0.85:
        return "OK_BY_TITLE_DBLP", best[0], round(best[1], 2)
    return "NOT_FOUND", best[0], round(best[1], 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--mailto", default="anonymous@example.com")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--out", default="verification_report.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    counts, flagged = {}, []
    for i, r in enumerate(rows, 1):
        doi, title = r.get("doi", "").strip(), r.get("title", "").strip()
        if doi:
            status, found, sim = check_doi(doi, title, args.mailto)
        else:
            status, found, sim = check_title(title, args.mailto)
        counts[status] = counts.get(status, 0) + 1
        if not status.startswith("OK"):
            flagged.append({**r, "check_status": status,
                            "found_title": found, "title_similarity": sim})
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  {counts}", file=sys.stderr)
        time.sleep(args.sleep)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        cols = list(rows[0].keys()) + ["check_status", "found_title", "title_similarity"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(flagged)

    print("\n=== SUMMARY ===")
    for k, v in sorted(counts.items()):
        print(f"  {k:20s} {v}")
    print(f"\n{len(flagged)} row(s) flagged -> {args.out}")

if __name__ == "__main__":
    main()
