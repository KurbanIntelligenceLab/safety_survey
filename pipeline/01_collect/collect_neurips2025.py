#!/usr/bin/env python3
"""
NeurIPS 2025 candidate collector for the AI-safety survey corpus.

NeurIPS 2025 is not yet in DBLP (the normal corpus source) as of 2026-06,
so this scrapes the official proceedings index instead and emits records in
the SAME schema as scrape_ai_safety_papers.parse_dblp_hit:

    {title, year, venue, authors, url, doi, dblp_key}

Stages:
  index    -> scrape proceedings index into all_papers.json (5.8k records)
  screen   -> title-screen against the safety keyword pool -> screened.json
  abstracts-> scrape abstracts from each proceedings page (resumable cache)
  rescreen -> re-screen on title+abstract -> neurips2025_candidates.json

Usage:
  python3 pipeline/01_collect/collect_neurips2025.py index
  python3 pipeline/01_collect/collect_neurips2025.py screen
  python3 pipeline/01_collect/collect_neurips2025.py abstracts        # long, resumable
  python3 pipeline/01_collect/collect_neurips2025.py rescreen
"""
import argparse, io, json, re, sys, time, urllib.request, html as htmllib
from pathlib import Path

sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
import scrape_ai_safety_papers as s  # reuse QUERIES keyword pool

INDEX_URL = "https://proceedings.neurips.cc/paper_files/paper/2025"
OUTDIR = Path("paper_survey/_neurips2025")
ALL = OUTDIR / "all_papers.json"
SCREENED = OUTDIR / "screened.json"
ABSTRACTS = OUTDIR / "abstracts_cache.json"
CANDIDATES = OUTDIR / "neurips2025_candidates.json"

UA = {"User-Agent": "Mozilla/5.0 (safety-survey corpus collector)"}


def rj(path):
    return json.loads(path.read_text(encoding="utf-8"))


def wj(path, obj, indent=1):
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")

# Screening terms derived from the DBLP query pool. Multi-word queries become
# AND-of-tokens; we drop generic stopword-ish tokens so "AI safety" doesn't
# match every paper containing "ai".
_STOP = {"ai", "deep", "learning", "machine", "neural", "network", "networks",
         "model", "models", "image", "classification"}


def screen_terms():
    terms = []
    for q in s.QUERIES:
        toks = [t for t in re.findall(r"[a-z\-]+", q.lower())]
        strong = [t for t in toks if t not in _STOP and len(t) > 2]
        if strong:
            terms.append(strong)
    # a few high-precision single tokens always count
    for t in ["adversarial", "backdoor", "jailbreak", "deepfake", "fairness",
              "robustness", "alignment", "hallucination", "toxicity", "watermark",
              "interpretability", "explainability", "privacy", "unlearning",
              "poisoning", "calibration", "out-of-distribution", "ood",
              "red-teaming", "redteam", "safety", "trustworthy"]:
        terms.append([t])
    return terms


def matches(text, terms):
    t = text.lower()
    hit = []
    for grp in terms:
        if all(tok in t for tok in grp):
            hit.append(" ".join(grp))
    return hit


def fetch(url, tries=4):
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
        except Exception as e:
            last = e
            time.sleep(2 ** a)
    raise last


def cmd_index(args):
    html = fetch(INDEX_URL)
    li_re = re.compile(
        r'<li class="[^"]*" data-track="([^"]*)">.*?'
        r'<a title="paper title" href="([^"]+)">(.*?)</a>.*?'
        r'<span class="paper-authors">(.*?)</span>', re.DOTALL)
    rows = []
    for m in li_re.finditer(html):
        track, href, title, authors = m.groups()
        title = htmllib.unescape(re.sub(r"<[^>]+>", "", title)).strip()
        authors = htmllib.unescape(re.sub(r"<[^>]+>", "", authors)).strip()
        rows.append({
            "title": title,
            "year": 2025,
            "venue": "NeurIPS",
            "authors": [a.strip() for a in authors.split(",") if a.strip()],
            "url": "https://proceedings.neurips.cc" + href,
            "doi": "",
            "dblp_key": "",
            "track": track,
        })
    OUTDIR.mkdir(parents=True, exist_ok=True)
    wj(ALL, rows)
    from collections import Counter
    print(f"indexed {len(rows)} papers -> {ALL}")
    print("by track:", dict(Counter(r["track"] for r in rows)))


def cmd_screen(args):
    rows = rj(ALL)
    terms = screen_terms()
    kept = []
    for r in rows:
        h = matches(r["title"], terms)
        if h:
            r2 = dict(r); r2["screen_hits"] = sorted(set(h)); kept.append(r2)
    wj(SCREENED, kept)
    print(f"title-screened: {len(kept)}/{len(rows)} kept -> {SCREENED}")


def cmd_abstracts(args):
    rows = rj(SCREENED if args.screened_only else ALL)
    cache = rj(ABSTRACTS) if ABSTRACTS.exists() else {}
    abst_re = re.compile(r'<p class="paper-abstract">(.*?)</p>\s*</div>', re.DOTALL)
    abst_re2 = re.compile(r'paper-abstract">(.*)', re.DOTALL)
    todo = [r for r in rows if r["url"] not in cache]
    print(f"{len(cache)} cached, {len(todo)} to fetch")
    for i, r in enumerate(todo, 1):
        try:
            page = fetch(r["url"])
            m = abst_re.search(page) or abst_re2.search(page)
            ab = re.sub(r"\s+", " ",
                        htmllib.unescape(re.sub(r"<[^>]+>", " ", m.group(1)))).strip() if m else ""
            cache[r["url"]] = ab
        except Exception as e:
            cache[r["url"]] = ""
            print(f"  WARN {r['url']}: {e}")
        if i % 50 == 0:
            wj(ABSTRACTS, cache, indent=None)
            print(f"  {i}/{len(todo)} fetched", flush=True)
        time.sleep(args.sleep)
    wj(ABSTRACTS, cache, indent=None)
    got = sum(1 for v in cache.values() if v)
    print(f"done: {got}/{len(cache)} have abstracts -> {ABSTRACTS}")


def cmd_rescreen(args):
    rows = rj(ALL)
    cache = rj(ABSTRACTS) if ABSTRACTS.exists() else {}
    terms = screen_terms()
    kept = []
    for r in rows:
        ab = cache.get(r["url"], "")
        h = matches(r["title"] + " " + ab, terms)
        if h:
            r2 = {k: r[k] for k in ("title", "year", "venue", "authors", "url", "doi", "dblp_key")}
            r2["abstract"] = ab
            r2["screen_hits"] = sorted(set(h))
            kept.append(r2)
    wj(CANDIDATES, kept)
    n_ab = sum(1 for r in kept if r["abstract"])
    print(f"re-screened on title+abstract: {len(kept)}/{len(rows)} kept "
          f"({n_ab} with abstract) -> {CANDIDATES}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("index")
    sub.add_parser("screen")
    pa = sub.add_parser("abstracts")
    pa.add_argument("--sleep", type=float, default=0.4)
    pa.add_argument("--screened-only", action="store_true",
                    help="only fetch abstracts for title-screened subset")
    sub.add_parser("rescreen")
    args = ap.parse_args()
    {"index": cmd_index, "screen": cmd_screen,
     "abstracts": cmd_abstracts, "rescreen": cmd_rescreen}[args.cmd](args)


if __name__ == "__main__":
    main()
