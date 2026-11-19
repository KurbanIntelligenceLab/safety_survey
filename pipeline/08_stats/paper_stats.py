#!/usr/bin/env python3
"""
Authoritative statistics for the re-collected AI-safety survey, computed from
paper_survey/papers_llm.json (12,630 LLM-classified papers, abstract-based,
multi-label). Produces every count the rewritten paper needs:

  - total kept-safety papers and drop count
  - primary-category distribution (+ % of kept)
  - multi-label rate and category co-occurrence (primary x secondary)
  - per-year totals and per-category x year matrix
  - per-venue distribution
  - 2023->2025 growth rates per category
  - confidence distribution / low-confidence count
  - source breakdown (pool vs snowball)

Writes paper_survey/paper_stats.json and prints a human summary.
Pure analysis; reads only, mutates nothing.
"""
import argparse, io, json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "paper_survey", "papers_llm.json")
OUT = os.path.join(HERE, "..", "..", "paper_survey", "paper_stats.json")

VENUES = ["CVPR", "ICCV", "ICML", "ICLR", "NeurIPS", "AAAI", "AIES", "FAccT",
          "ACL", "EMNLP"]


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DATA)
    ap.add_argument("--out", dest="out", default=OUT)
    args = ap.parse_args()
    OUT = args.out
    papers = json.load(io.open(args.inp, encoding="utf-8"))
    n_all = len(papers)
    kept = [p for p in papers if p.get("is_safety")]
    n_kept = len(kept)
    n_drop = n_all - n_kept

    # source breakdown (of kept)
    src = Counter(p.get("source", "?") for p in kept)

    # primary-category distribution
    cat = Counter(p["category"] for p in kept if p.get("category"))

    # multi-label
    n_multi = sum(1 for p in kept if p.get("category_secondary"))
    cooc = Counter()
    for p in kept:
        sec = p.get("category_secondary")
        if sec and p.get("category"):
            pair = tuple(sorted([p["category"], sec]))
            cooc[pair] += 1

    # per-year totals + category x year (primary label)
    year_total = Counter()
    cat_year = defaultdict(Counter)
    for p in kept:
        y = p.get("year")
        year_total[y] += 1
        if p.get("category"):
            cat_year[p["category"]][y] += 1

    # per-venue (of kept)
    ven = Counter(p.get("venue") for p in kept)

    # growth 2023 -> 2025 per category
    growth = {}
    for c in cat:
        a, b = cat_year[c].get(2023, 0), cat_year[c].get(2025, 0)
        growth[c] = {
            "y2023": a, "y2025": b,
            "pct": (round(100 * (b - a) / a, 1) if a else None),
        }

    # confidence
    confs = [p.get("llm_confidence") for p in kept if p.get("llm_confidence") is not None]
    low_conf = sum(1 for c in confs if c <= 0.6)

    years = sorted(y for y in year_total if y is not None)

    # ---- POOL-ONLY temporal series (recency-complete; snowball is citation-
    # biased toward older papers, so it is excluded from all year/growth claims).
    pool = [p for p in kept if p.get("source") == "pool"]
    pool_year_total = Counter(p.get("year") for p in pool)
    pool_cat_year = defaultdict(Counter)
    for p in pool:
        if p.get("category"):
            pool_cat_year[p["category"]][p.get("year")] += 1
    pool_growth = {}
    for c in cat:
        a, b = pool_cat_year[c].get(2023, 0), pool_cat_year[c].get(2025, 0)
        pool_growth[c] = {"y2023": a, "y2025": b,
                          "pct": (round(100 * (b - a) / a, 1) if a else None)}

    summary = {
        "n_all_classified": n_all,
        "n_kept_safety": n_kept,
        "n_dropped": n_drop,
        "pct_kept": round(100 * n_kept / n_all, 1),
        "source_breakdown": dict(src),
        "n_categories": len(cat),
        "category_primary": [
            {"category": c, "count": k, "pct_of_kept": round(100 * k / n_kept, 1)}
            for c, k in cat.most_common()
        ],
        "n_multi_label": n_multi,
        "pct_multi_label": round(100 * n_multi / n_kept, 1),
        "top_cooccurring_pairs": [
            {"pair": list(pr), "count": c} for pr, c in cooc.most_common(15)
        ],
        "years": years,
        "year_totals": {str(y): year_total[y] for y in years},
        "category_x_year": {
            c: {str(y): cat_year[c].get(y, 0) for y in years} for c in cat
        },
        "venue": [
            {"venue": v, "count": ven.get(v, 0),
             "pct": round(100 * ven.get(v, 0) / n_kept, 1)}
            for v, _ in ven.most_common()
        ],
        "growth_2023_2025": growth,
        "n_low_confidence_le_0.6": low_conf,
        "confidence_mean": round(sum(confs) / len(confs), 3) if confs else None,
        "pool": {
            "n": len(pool),
            "year_totals": {str(y): pool_year_total.get(y, 0) for y in years},
            "category_x_year": {
                c: {str(y): pool_cat_year[c].get(y, 0) for y in years} for c in cat
            },
            "growth_2023_2025": pool_growth,
        },
    }
    json.dump(summary, io.open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- human summary ----
    print(f"ALL classified : {n_all}")
    print(f"KEPT safety    : {n_kept}  ({summary['pct_kept']}%)")
    print(f"DROPPED        : {n_drop}")
    print(f"source (kept)  : {dict(src)}")
    print(f"multi-label    : {n_multi}  ({summary['pct_multi_label']}% of kept)")
    print(f"low-conf <=0.6 : {low_conf}   mean conf {summary['confidence_mean']}")
    print(f"\nCATEGORY (primary), ranked:")
    for r in summary["category_primary"]:
        g = growth[r["category"]]["pct"]
        print(f"  {r['count']:5d}  {r['pct_of_kept']:5.1f}%  {r['category']:40s}"
              f"  2023->25: {growth[r['category']]['y2023']}->{growth[r['category']]['y2025']}"
              f" ({g}%)")
    print(f"\nYEAR totals:")
    for y in years:
        print(f"  {y}: {year_total[y]}")
    print(f"\nVENUE:")
    for v in summary["venue"]:
        print(f"  {v['count']:5d}  {v['pct']:5.1f}%  {v['venue']}")
    print(f"\nTOP CO-OCCURRING PAIRS (primary x secondary):")
    for pr in summary["top_cooccurring_pairs"][:12]:
        print(f"  {pr['count']:4d}  {pr['pair'][0]}  +  {pr['pair'][1]}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
