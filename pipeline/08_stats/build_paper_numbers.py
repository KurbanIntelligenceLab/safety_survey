#!/usr/bin/env python3
"""
Authoritative numbers for the 10-venue (8 ML/vision/ethics + ACL + EMNLP) merged
AI-safety review. Recomputes every derived figure the prose cites so each is
traceable, plus a controlled title-keyword vs abstract-LLM recall comparison on
the *identical* candidate pool, and exports a paper-level CSV for verification.

  python3 pipeline/08_stats/build_paper_numbers.py

Reads:  paper_survey/papers_llm.json        (augmented 13,101-classified corpus)
        paper_survey/classify_input.json    (8-venue candidate titles, 12,630)
        paper_survey/v1_nlp_collect.json     (ACL+EMNLP candidate titles, 471)
Writes: paper_survey/paper_numbers_v1nlp.json
        paper_survey/safety_corpus_v1nlp.csv (6,066 kept-safety rows)
"""
import csv, io, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from taxonomy_expanded import classify_multi

NLP = {"ACL", "EMNLP"}
VENUES10 = ["CVPR", "ICCV", "ICML", "ICLR", "NeurIPS", "AAAI", "AIES", "FAccT",
            "ACL", "EMNLP"]


def load(p):
    return json.load(io.open(ROOT / p, encoding="utf-8"))


def main():
    corpus = load("paper_survey/papers_llm.json")
    kept = [p for p in corpus if p.get("is_safety")]
    pool = [p for p in kept if p.get("source") == "pool"]
    n_kept = len(kept)

    out = {}
    out["n_classified"] = len(corpus)
    out["n_kept"] = n_kept
    out["n_pool"] = len(pool)

    # ---- composition (full corpus, primary label) ----
    cat = Counter(p["category"] for p in kept if p.get("category"))
    out["category_primary"] = [
        {"category": c, "count": k, "pct": round(100 * k / n_kept, 1)}
        for c, k in cat.most_common()]
    out["n_multi"] = sum(1 for p in kept if p.get("category_secondary"))
    out["pct_multi"] = round(100 * out["n_multi"] / n_kept, 1)

    # ---- co-occurrence ----
    cooc = Counter()
    for p in kept:
        s = p.get("category_secondary")
        if s and p.get("category"):
            cooc[tuple(sorted([p["category"], s]))] += 1
    out["cooccurrence_top"] = [{"pair": list(pr), "count": c}
                               for pr, c in cooc.most_common(15)]

    # ---- venue (full corpus, of kept) ----
    ven = Counter(p.get("venue") for p in kept)
    out["venue"] = [{"venue": v, "count": ven[v], "pct": round(100 * ven[v] / n_kept, 1)}
                    for v in VENUES10 if ven.get(v)]
    gen4 = sum(ven[v] for v in ["NeurIPS", "ICML", "ICLR", "AAAI"])
    out["generalist4_pct"] = round(100 * gen4 / n_kept, 1)
    out["vision2_pct"] = round(100 * (ven["CVPR"] + ven["ICCV"]) / n_kept, 1)
    out["ethics2_pct"] = round(100 * (ven["FAccT"] + ven["AIES"]) / n_kept, 1)
    out["nlp2_pct"] = round(100 * (ven["ACL"] + ven["EMNLP"]) / n_kept, 1)

    # ---- pool year table + growth ----
    years = list(range(2018, 2027))
    cat_year = defaultdict(Counter)
    yt = Counter()
    for p in pool:
        yt[p.get("year")] += 1
        if p.get("category"):
            cat_year[p["category"]][p.get("year")] += 1
    out["pool_year_totals"] = {str(y): yt.get(y, 0) for y in years}
    out["pool_cat_year"] = {c: {str(y): cat_year[c].get(y, 0) for y in years} for c in cat}
    growth = {}
    for c in cat:
        a, b = cat_year[c].get(2023, 0), cat_year[c].get(2025, 0)
        growth[c] = {"y2023": a, "y2025": b,
                     "pct": (round(100 * (b - a) / a) if a else None)}
    out["pool_growth_2023_2025"] = growth

    # ---- adversarial share of pool, early vs recent ----
    def share(cat_name, yrs):
        num = sum(cat_year[cat_name].get(y, 0) for y in yrs)
        den = sum(yt.get(y, 0) for y in yrs)
        return round(100 * num / den, 1) if den else None
    out["adv_share_2018_2022"] = share("Adversarial Robustness", range(2018, 2023))
    out["adv_share_2024_2025"] = share("Adversarial Robustness", [2024, 2025])

    # ---- fairness decline detail (pool) ----
    fy = cat_year["Fairness & Bias"]
    out["fairness_pool_by_year"] = {str(y): fy.get(y, 0) for y in years}
    # fairness at ethics venues vs generalist (kept, primary=fairness)
    fair = [p for p in kept if p.get("category") == "Fairness & Bias"]
    def fair_year(venue_set, y):
        return sum(1 for p in fair if p.get("venue") in venue_set and p.get("year") == y)
    out["fairness_faact"] = {str(y): fair_year({"FAccT"}, y) for y in years}
    out["fairness_generalist"] = {str(y): fair_year({"NeurIPS","ICML","ICLR","AAAI"}, y) for y in years}

    # ---- controlled recall comparison on the IDENTICAL candidate pool ----
    # candidate universe = the full classified corpus (every candidate has a title)
    cand = corpus
    title_primary = Counter()       # first-match single label, "safety" only
    title_total = 0
    title_by_venue = Counter()
    for p in cand:
        hits = classify_multi(p.get("title", "") or "")
        if hits and hits != ["Miscellaneous Safety"]:
            title_total += 1
            title_primary[hits[0]] += 1
            if p.get("venue"):
                title_by_venue[p["venue"]] += 1
    out["recall"] = {
        "n_candidates": len(cand),
        "title_keyword_safety": title_total,
        "abstract_llm_safety": n_kept,
        "ratio": round(n_kept / title_total, 2),
    }
    # per-category recall gain (abstract primary / title primary)
    gain = {}
    for c in cat:
        t = title_primary.get(c, 0)
        gain[c] = {"title": t, "abstract": cat[c],
                   "pct_gain": (round(100 * (cat[c] - t) / t) if t else None)}
    out["recall_per_category"] = gain
    out["recall_title_by_venue"] = dict(title_by_venue.most_common())
    # keyword venue ranking (% of title-keyword safety) vs abstract ranking
    out["keyword_venue_pct"] = {v: round(100 * title_by_venue[v] / title_total, 1)
                                for v, _ in title_by_venue.most_common()}

    # confidence
    confs = [p["llm_confidence"] for p in kept if p.get("llm_confidence") is not None]
    out["conf_mean"] = round(sum(confs) / len(confs), 3)
    out["n_low_conf"] = sum(1 for c in confs if c <= 0.6)
    out["pct_low_conf"] = round(100 * out["n_low_conf"] / n_kept, 1)

    json.dump(out, io.open(ROOT / "paper_survey/paper_numbers_v1nlp.json", "w",
              encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- CSV export (kept-safety rows) ----
    with io.open(ROOT / "paper_survey/safety_corpus_v1nlp.csv", "w",
                 encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "venue", "year", "source", "primary_category",
                    "secondary_category", "is_safety", "llm_confidence", "doi"])
        for p in sorted(kept, key=lambda x: (x.get("venue") or "", -(x.get("year") or 0))):
            w.writerow([p.get("id"), (p.get("title") or "").replace("\n", " "),
                        p.get("venue"), p.get("year"), p.get("source"),
                        p.get("category"), p.get("category_secondary"),
                        p.get("is_safety"), p.get("llm_confidence"), p.get("doi")])

    # ---- console summary ----
    print(f"kept safety        : {n_kept}   pool {len(pool)}")
    print(f"multi-label        : {out['n_multi']} ({out['pct_multi']}%)")
    print(f"conf mean / low<=.6: {out['conf_mean']} / {out['n_low_conf']} ({out['pct_low_conf']}%)")
    print(f"generalist4 / vis2 / ethics2 / nlp2 : "
          f"{out['generalist4_pct']} / {out['vision2_pct']} / {out['ethics2_pct']} / {out['nlp2_pct']}%")
    print(f"adv share 2018-22 -> 2024-25 : {out['adv_share_2018_2022']}% -> {out['adv_share_2024_2025']}%")
    print(f"\nRECALL (identical {len(cand)}-candidate pool):")
    print(f"  title-keyword safety = {title_total}   abstract-LLM safety = {n_kept}   "
          f"ratio = {out['recall']['ratio']}x")
    print("  keyword venue ranking:", list(out['keyword_venue_pct'].items())[:4])
    print("\nVENUE (abstract, kept):")
    for v in out["venue"]:
        print(f"  {v['count']:5d}  {v['pct']:5.1f}%  {v['venue']}")
    print("\nPOOL year totals:", out["pool_year_totals"])
    print("\nPer-category recall gain (title -> abstract):")
    for c in [x["category"] for x in out["category_primary"]]:
        g = gain[c]
        print(f"  {c:38s} {g['title']:4d} -> {g['abstract']:4d}  ({g['pct_gain']}%)")
    print("\nWrote paper_numbers_v1nlp.json + safety_corpus_v1nlp.csv")


if __name__ == "__main__":
    main()
