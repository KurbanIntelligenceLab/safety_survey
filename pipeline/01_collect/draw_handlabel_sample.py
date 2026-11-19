#!/usr/bin/env python3
"""
Draw a reproducible random sample of papers for manual labeling, so we can
measure classifier precision against ground truth (review concern #1).

Output CSV has the model's current label plus BLANK columns for the human:
  is_safety        - y/n: is this genuinely an AI-safety paper at all?
  correct_category - the category you judge correct (or 'none')
  notes

Fill it in by reading the actual paper (title+abstract+skim), then run
score_handlabels.py to get precision per category and overall.
"""
import csv, io, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "paper_survey", "papers_all.json")
OUT = os.path.join(HERE, "..", "..", "paper_survey", "handlabel_sample.csv")
N = 200
SEED = 20260528  # fixed for reproducibility


def main():
    papers = json.load(io.open(DATA, encoding="utf-8"))
    rng = random.Random(SEED)
    sample = rng.sample(papers, N)
    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "title", "venue", "year", "model_category",
                    "url", "doi", "is_safety", "correct_category", "notes"])
        for i, p in enumerate(sample):
            w.writerow([i, p["title"], p["venue"], p["year"], p["category"],
                        p.get("url", ""), p.get("doi", ""), "", "", ""])
    print(f"Wrote {OUT}  ({N} papers, seed={SEED})")
    print("Fill is_safety (y/n) and correct_category, then run score_handlabels.py")


if __name__ == "__main__":
    main()
