#!/usr/bin/env python
"""Fetch + section-extract full text for the safety corpus.

Consumes paper_survey/fulltext_index.json (id -> pdf_urls). For each paper it
tries each candidate URL in order with per-host rate limiting and exponential
backoff, downloads the PDF to a cache (resumable -- skips if already fetched),
parses with PyMuPDF, and heuristically extracts Method/Approach and
Conclusion/Discussion sections. Writes one JSON per paper plus a coverage report.

Usage:
  fetch_fulltext.py --pilot 40      # stratified sample across venue/year
  fetch_fulltext.py --limit 500     # first N of the index
  fetch_fulltext.py                 # full run (resumable)
"""
import json, io, os, re, sys, time, argparse
from collections import defaultdict, Counter
import requests
import fitz  # PyMuPDF

IDX = os.environ.get("FT_INDEX", "paper_survey/fulltext_index.json")
ROOT = os.environ.get("FT_ROOT", "paper_survey/_fulltext")
PDFDIR = os.path.join(ROOT, "pdf")
SECDIR = os.path.join(ROOT, "sections")
REPORT = os.path.join(ROOT, "coverage.json")

UA = "Mozilla/5.0 (safety-survey academic fetch; mailto:%s)" % os.environ.get("CONTACT_EMAIL", "set CONTACT_EMAIL")
MIN_HOST_GAP = 2.5            # seconds between hits to the same host
_last_host = defaultdict(float)


def host_of(url):
    m = re.match(r"https?://([^/]+)/", url)
    return m.group(1).replace("www.", "") if m else url


def polite(url):
    h = host_of(url)
    dt = time.time() - _last_host[h]
    if dt < MIN_HOST_GAP:
        time.sleep(MIN_HOST_GAP - dt)
    _last_host[h] = time.time()


def fetch_pdf(urls, dest):
    """Try each url with backoff; save first valid PDF to dest. Return (ok, info)."""
    for url in urls:
        for attempt in range(4):
            polite(url)
            try:
                r = requests.get(url, timeout=90, headers={"User-Agent": UA},
                                 allow_redirects=True)
            except Exception as e:
                info = "req_err:%s" % str(e)[:40]
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                open(dest, "wb").write(r.content)
                return True, "ok:%s" % host_of(url)
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt + 1)
                info = "http_%d" % r.status_code
                continue
            info = "http_%d" % r.status_code
            break  # non-retryable for this url; try next url
    return False, info


# --- section extraction -----------------------------------------------------
METH = re.compile(r"^\s*(\d+\.?\d*\.?\s*)?(method(s|ology)?|approach|proposed\s+\w+|"
                  r"our\s+(method|approach|model|framework)|model\s+architecture)\b", re.I)
CONC = re.compile(r"^\s*(\d+\.?\d*\.?\s*)?(conclusion(s)?|concluding\s+remarks|"
                  r"discussion\s+and\s+conclusion|summary\s+and\s+conclusion)\b", re.I)
HEAD = re.compile(r"^\s*(\d+\.?\d*\.?\s+[A-Z]|[A-Z][A-Za-z ]{3,40}$)")


def extract(text):
    lines = [ln.strip() for ln in text.split("\n")]
    def grab(startre):
        for i, ln in enumerate(lines):
            if startre.match(ln) and len(ln) < 60:
                buf = []
                for ln2 in lines[i + 1:i + 200]:
                    if HEAD.match(ln2) and len(buf) > 8:
                        break
                    if ln2:
                        buf.append(ln2)
                return " ".join(buf).strip()
        return ""
    return grab(METH), grab(CONC)


def stratified(idx, n):
    by = defaultdict(list)
    for r in idx:
        if r["pdf_urls"]:
            by[r["venue"]].append(r)
    out = []
    per = max(1, n // max(1, len(by)))
    for v, rows in sorted(by.items()):
        rows.sort(key=lambda r: str(r.get("year")))
        step = max(1, len(rows) // per)
        out += rows[::step][:per]
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(PDFDIR, exist_ok=True)
    os.makedirs(SECDIR, exist_ok=True)
    idx = json.load(io.open(IDX, encoding="utf-8"))

    if args.pilot:
        work = stratified(idx, args.pilot)
        tag = "PILOT(%d)" % args.pilot
    else:
        work = [r for r in idx if r["pdf_urls"]]
        if args.limit:
            work = work[:args.limit]
        tag = "FULL(%d)" % len(work)
    print("== fetch_fulltext %s ==" % tag)

    res, reasons = [], Counter()
    for i, r in enumerate(work):
        pid = r["id"]
        pdf = os.path.join(PDFDIR, "%s.pdf" % pid)
        secf = os.path.join(SECDIR, "%s.json" % pid)
        if os.path.exists(secf):  # resume
            rec = json.load(io.open(secf, encoding="utf-8"))
            res.append(rec); reasons[rec.get("info", "cached")] += 1
            continue
        if os.path.exists(pdf) and os.path.getsize(pdf) > 1000:
            ok, info = True, "cached_pdf"
        else:
            ok, info = fetch_pdf(r["pdf_urls"], pdf)
        rec = {"id": pid, "venue": r["venue"], "year": r["year"],
               "title": (r["title"] or "")[:90], "ok": ok, "info": info}
        if ok:
            try:
                doc = fitz.open(pdf)
                txt = "\n".join(pg.get_text() for pg in doc)
                doc.close()
                meth, conc = extract(txt)
                rec.update(chars=len(txt), has_method=bool(meth),
                           has_conclusion=bool(conc), method=meth, conclusion=conc)
            except Exception as e:
                rec["ok"] = False; rec["info"] = "parse_err:%s" % str(e)[:40]
        json.dump(rec, io.open(secf, "w", encoding="utf-8"), ensure_ascii=False)
        res.append(rec); reasons[rec["info"]] += 1
        if (i + 1) % 10 == 0 or args.pilot:
            print("  [%4d/%4d] %-7s ok=%s meth=%s conc=%s %s"
                  % (i + 1, len(work), rec["venue"], rec["ok"],
                     rec.get("has_method"), rec.get("has_conclusion"), rec["info"][:24]))

    n = len(res)
    got = sum(1 for r in res if r["ok"])
    meth = sum(1 for r in res if r.get("has_method"))
    conc = sum(1 for r in res if r.get("has_conclusion"))
    both = sum(1 for r in res if r.get("has_method") and r.get("has_conclusion"))
    rep = {"tag": tag, "n": n, "pdf_ok": got, "has_method": meth,
           "has_conclusion": conc, "has_both": both, "reasons": dict(reasons)}
    json.dump(rep, io.open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n==== SUMMARY %s ====" % tag)
    print("PDF fetched:    %d/%d (%.0f%%)" % (got, n, 100 * got / n))
    print("method section: %d/%d (%.0f%%)" % (meth, n, 100 * meth / n))
    print("conclusion:     %d/%d (%.0f%%)" % (conc, n, 100 * conc / n))
    print("both:           %d/%d (%.0f%%)" % (both, n, 100 * both / n))
    print("reasons:", reasons.most_common(10))


if __name__ == "__main__":
    main()
