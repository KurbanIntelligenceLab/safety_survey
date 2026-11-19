#!/usr/bin/env python3
"""
Offline diagnostics on the existing 1,941-paper survey set, to quantify the
methodological concerns raised in review WITHOUT any new data collection:

  1. Multi-label overlap  - how many titles match >1 category keyword set
                            (quantifies single-label distortion / "convergence").
  2. First-match-wins bias - how often the assigned label differs from the set
                            of all matching labels, and which categories "win"
                            because they are checked earlier.
  3. Miscellaneous audit   - the 311 Misc papers matched a query but no
                            classification keyword: how many could keyword
                            expansion plausibly rescue (heuristic flag only).
  4. Co-occurrence matrix  - which category pairs co-occur most in titles
                            (feeds data-driven "integrated evaluation" recs).

Reads paper_survey/papers_all.json. Writes paper_survey/classification_audit.json
and prints a summary. Pure analysis; does not modify the dataset.
"""
import io, json, os
from collections import Counter, defaultdict
from itertools import combinations

# Reuse the exact taxonomy + matching logic from the scraper so the audit
# reflects the real classifier, not a re-implementation.
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "scraper", os.path.join(HERE, "scrape_ai_safety_papers.py"))
scraper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scraper)
TAXONOMY = scraper.TAXONOMY

DATA = os.path.join(HERE, "..", "..", "paper_survey", "papers_all.json")
OUT = os.path.join(HERE, "..", "..", "paper_survey", "classification_audit.json")


def all_matches(title):
    """Return list of ALL categories whose keyword set matches the title."""
    blob = f" {title} ".lower()
    hits = []
    for category, keywords in TAXONOMY:
        if keywords and any(kw in blob for kw in keywords):
            hits.append(category)
    return hits


def main():
    papers = json.load(io.open(DATA, encoding="utf-8"))
    n = len(papers)

    nlabels = Counter()
    cooc = Counter()
    multi_examples = []
    assigned_vs_first = 0   # assigned label != first match (shouldn't happen, sanity)
    misc_total = 0
    pair_to_papers = defaultdict(list)

    for p in papers:
        matches = all_matches(p["title"])
        k = len(matches)
        nlabels[k] += 1
        if p["category"] == "Miscellaneous Safety":
            misc_total += 1
        if k >= 2:
            for a, b in combinations(sorted(matches), 2):
                cooc[(a, b)] += 1
                pair_to_papers[(a, b)].append(p["title"])
            if len(multi_examples) < 25:
                multi_examples.append({"title": p["title"],
                                       "assigned": p["category"],
                                       "all_matches": matches})
        # sanity: assigned should equal first match (or Misc when k==0)
        first = matches[0] if matches else "Miscellaneous Safety"
        if first != p["category"]:
            assigned_vs_first += 1

    multi = sum(c for k, c in nlabels.items() if k >= 2)
    classified = n - misc_total

    summary = {
        "n_papers": n,
        "n_misc": misc_total,
        "n_classified": classified,
        "label_count_distribution": dict(sorted(nlabels.items())),
        "n_multi_label": multi,
        "pct_multi_label_of_classified": round(100 * multi / classified, 1),
        "sanity_assigned_ne_firstmatch": assigned_vs_first,
        "top_cooccurring_pairs": [
            {"pair": list(pair), "count": c}
            for pair, c in cooc.most_common(15)
        ],
        "multi_label_examples": multi_examples,
    }
    json.dump(summary, io.open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"papers={n}  classified={classified}  miscellaneous={misc_total} "
          f"({100*misc_total/n:.1f}%)")
    print(f"\nlabels-per-title distribution (multi-label view):")
    for k in sorted(nlabels):
        print(f"  {k} matching categor{'y' if k==1 else 'ies'}: {nlabels[k]:5d} "
              f"papers ({100*nlabels[k]/n:.1f}%)")
    print(f"\n>=2 categories: {multi} papers = {100*multi/classified:.1f}% of "
          f"the {classified} classified  <-- hidden by single-label")
    print(f"sanity (assigned != first match): {assigned_vs_first} "
          f"(should be 0 if audit matches scraper)")
    print(f"\nTop co-occurring category pairs (title-level):")
    for pair, c in cooc.most_common(12):
        print(f"  {c:4d}  {pair[0]}  +  {pair[1]}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
