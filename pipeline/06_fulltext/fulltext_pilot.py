#!/usr/bin/env python
"""Full-text acquisition PILOT.

Goal: measure feasibility of getting methods+conclusion full text for the
safety corpus, before committing to all 6,827 papers. For a stratified sample
it (1) resolves a PDF via the arXiv API by title match, (2) parses with PyMuPDF,
(3) extracts Method/Approach and Conclusion/Discussion sections heuristically,
and (4) reports coverage. No LLM calls here -- that is the next stage.
"""
import io, json, re, sys, time, urllib.parse, os
import requests
import fitz  # PyMuPDF

SRC = "paper_survey/papers_llm_cleaned.json"
OUTDIR = "paper_survey/_fulltext_pilot"
PER_VENUE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "pdf"), exist_ok=True)

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def tok(s):
    return set(norm(s).split())

def load():
    d = json.load(io.open(SRC, encoding="utf-8"))
    d = d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))
    return [r for r in d if r.get("is_safety")]

def stratified(safe, per):
    by = {}
    for r in safe:
        by.setdefault(r.get("venue"), []).append(r)
    out = []
    for v, rows in sorted(by.items()):
        # spread across years
        rows = sorted(rows, key=lambda r: str(r.get("year")))
        step = max(1, len(rows) // per)
        out += rows[::step][:per]
    return out

def arxiv_pdf_url(title):
    """Query arXiv API by title; return (pdf_url, arxiv_title) if a good match."""
    q = 'ti:"%s"' % re.sub(r'["]', "", title)[:240]
    url = "http://export.arxiv.org/api/query?search_query=%s&max_results=5" % urllib.parse.quote(q)
    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        return None, "req_err:%s" % e
    if r.status_code != 200:
        return None, "http_%d" % r.status_code
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
    want = tok(title)
    best = None
    for e in entries:
        mt = re.search(r"<title>(.*?)</title>", e, re.S)
        mid = re.search(r"<id>(.*?)</id>", e, re.S)
        if not mt or not mid:
            continue
        at = re.sub(r"\s+", " ", mt.group(1)).strip()
        got = tok(at)
        if not want or not got:
            continue
        j = len(want & got) / len(want | got)
        if best is None or j > best[0]:
            pid = mid.group(1).strip().rsplit("/", 1)[-1]
            best = (j, "https://arxiv.org/pdf/%s" % pid, at)
    if best and best[0] >= 0.75:   # strong title overlap
        return best[1], best[2]
    return None, "no_match(best=%.2f)" % (best[0] if best else 0.0)

SEC_START = re.compile(r"^\s*\d*\.?\s*(method|methodology|approach|our (method|approach|model)|proposed)\b", re.I)
CONC_START = re.compile(r"^\s*\d*\.?\s*(conclusion|conclusions|discussion and conclusion|concluding)\b", re.I)
NEXT_SEC = re.compile(r"^\s*\d+\.?\s+[A-Z]")

def extract_sections(text):
    lines = text.split("\n")
    def grab(startre):
        for i, ln in enumerate(lines):
            if startre.match(ln):
                buf = []
                for ln2 in lines[i+1:i+120]:
                    if NEXT_SEC.match(ln2) and len(buf) > 5:
                        break
                    buf.append(ln2)
                return " ".join(buf).strip()
        return ""
    return grab(SEC_START), grab(CONC_START)

def main():
    safe = load()
    sample = stratified(safe, PER_VENUE)
    print("pilot sample: %d papers (%d per venue)" % (len(sample), PER_VENUE))
    results = []
    for i, r in enumerate(sample):
        title = r.get("title", "")
        pdf_url, info = arxiv_pdf_url(title)
        rec = {"id": r.get("id"), "venue": r.get("venue"), "year": r.get("year"),
               "title": title[:80], "pdf": bool(pdf_url), "info": info[:60]}
        if pdf_url:
            try:
                pr = requests.get(pdf_url, timeout=60)
                if pr.status_code == 200 and pr.content[:4] == b"%PDF":
                    p = os.path.join(OUTDIR, "pdf", "%s.pdf" % r.get("id"))
                    open(p, "wb").write(pr.content)
                    doc = fitz.open(p)
                    txt = "\n".join(pg.get_text() for pg in doc)
                    doc.close()
                    meth, conc = extract_sections(txt)
                    rec.update(chars=len(txt), has_method=bool(meth),
                               has_conclusion=bool(conc),
                               method_preview=meth[:160], conc_preview=conc[:160])
                else:
                    rec["info"] = "dl_fail_%d" % pr.status_code
            except Exception as e:
                rec["info"] = "parse_err:%s" % str(e)[:40]
        results.append(rec)
        print("  [%2d/%2d] %-7s pdf=%s meth=%s conc=%s  %s"
              % (i+1, len(sample), rec["venue"], rec["pdf"],
                 rec.get("has_method"), rec.get("has_conclusion"), title[:46]))
        time.sleep(3)  # arXiv politeness
    json.dump(results, io.open(os.path.join(OUTDIR, "pilot_results.json"), "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)
    n = len(results)
    got = sum(1 for r in results if r["pdf"])
    meth = sum(1 for r in results if r.get("has_method"))
    conc = sum(1 for r in results if r.get("has_conclusion"))
    print("\n==== PILOT SUMMARY ====")
    print("PDF found (arXiv only): %d/%d (%.0f%%)" % (got, n, 100*got/n))
    print("methods extracted:      %d/%d" % (meth, n))
    print("conclusion extracted:   %d/%d" % (conc, n))
    from collections import Counter
    cov = Counter()
    for r in results:
        cov[r["venue"]] += 1 if r["pdf"] else 0
    print("by-venue PDF hits:", dict(cov))

if __name__ == "__main__":
    main()
