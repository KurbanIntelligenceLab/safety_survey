#!/usr/bin/env python3
"""
AIES 2025 candidate collector for the AI-safety survey corpus.

Like NeurIPS 2025, AIES 2025 is not in DBLP yet (DBLP's latest AIES is 2024),
so the venue-stream collector missed it. AIES 2025 (Eighth AAAI/ACM Conf. on
AI, Ethics & Society, Madrid, Oct 2025) is published on AAAI OJS as Vol. 8,
issues 677/678/679/682. We scrape those, emit records in the corpus schema
({title, year, venue, authors, url, doi, dblp_key}) plus abstract, and screen
against the safety keyword pool.

Usage:
  python3 pipeline/01_collect/collect_aies2025.py collect   # scrape issues + articles
  python3 pipeline/01_collect/collect_aies2025.py screen     # title+abstract screen
"""
import argparse, gzip, io, json, re, sys, time, urllib.request, zlib, html as htmllib
from pathlib import Path

sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
import scrape_ai_safety_papers as s
import collect_neurips2025 as nips  # reuse screen_terms / matches

ISSUES = [677, 678, 679, 682]  # AIES Vol. 8 (2025) tracks
OUTDIR = Path("paper_survey/_aies2025")
ALL = OUTDIR / "all_papers.json"
CANDIDATES = OUTDIR / "aies2025_candidates.json"
UA = {"User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
      "Accept-Encoding": "gzip, deflate"}


def get(u, tries=4):
    last = None
    for a in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60)
            data = r.read()
            enc = r.headers.get("Content-Encoding", "")
            if enc == "gzip":
                data = gzip.decompress(data)
            elif enc == "deflate":
                data = zlib.decompress(data)
            return data.decode("utf-8", "replace")
        except Exception as e:
            last = e; time.sleep(2 ** a)
    raise last


def meta(html, name, many=False):
    pat = r'<meta name="%s" content="([^"]*)"' % re.escape(name)
    ms = [htmllib.unescape(x).strip() for x in re.findall(pat, html)]
    return ms if many else (ms[0] if ms else "")


def abstract(html):
    m = (re.search(r'class="item abstract">(.*?)</div>\s*</div>', html, re.DOTALL)
         or re.search(r'>Abstract</h2>(.*?)</section>', html, re.DOTALL))
    if not m:
        return ""
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", m.group(1)))).strip()


def cmd_collect(args):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    art_urls = []
    for iid in ISSUES:
        page = get(f"https://ojs.aaai.org/index.php/AIES/issue/view/{iid}")
        urls = sorted(set(re.findall(r'/index\.php/AIES/article/view/\d+', page)),
                      key=lambda u: int(u.rsplit("/", 1)[1]))
        full = ["https://ojs.aaai.org" + u for u in urls]
        art_urls += full
        print(f"  issue {iid}: {len(full)} articles", flush=True)
        time.sleep(args.sleep)
    art_urls = list(dict.fromkeys(art_urls))
    print(f"total {len(art_urls)} article pages to fetch")
    rows = []
    for i, u in enumerate(art_urls, 1):
        try:
            h = get(u)
            title = meta(h, "DC.Title") or ""
            authors = meta(h, "DC.Creator.PersonalName", many=True)
            doi = meta(h, "DC.Identifier.DOI")
            rows.append({
                "title": re.sub(r"\s+", " ", title).strip(),
                "year": 2025, "venue": "AIES",
                "authors": authors,
                "url": meta(h, "DC.Identifier.URI") or u,
                "doi": doi, "dblp_key": "",
                "abstract": abstract(h),
            })
        except Exception as e:
            print(f"  WARN {u}: {e}")
        if i % 40 == 0:
            nips.wj(ALL, rows); print(f"  {i}/{len(art_urls)} fetched", flush=True)
        time.sleep(args.sleep)
    nips.wj(ALL, rows)
    n_ab = sum(1 for r in rows if r["abstract"])
    print(f"collected {len(rows)} AIES 2025 papers ({n_ab} with abstract) -> {ALL}")


def cmd_screen(args):
    rows = nips.rj(ALL)
    terms = nips.screen_terms()
    kept = []
    for r in rows:
        h = nips.matches(r["title"] + " " + r.get("abstract", ""), terms)
        if h:
            r2 = dict(r); r2["screen_hits"] = sorted(set(h)); kept.append(r2)
    nips.wj(CANDIDATES, kept)
    print(f"screened: {len(kept)}/{len(rows)} kept -> {CANDIDATES}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    pc = sub.add_parser("collect"); pc.add_argument("--sleep", type=float, default=0.4)
    sub.add_parser("screen")
    args = ap.parse_args()
    {"collect": cmd_collect, "screen": cmd_screen}[args.cmd](args)


if __name__ == "__main__":
    main()
