#!/usr/bin/env python3
"""
Pre-enrichment title screen for snowball candidates.

The snowball ranks by citation-overlap, so the top hits are canonical ML refs
(CLIP/ViT/DDPM/InstructGPT) that the user does NOT want in the safety survey.
Enriching all ~9.8k abstracts via S2 (1.1s each) would burn ~3h mostly on
non-safety papers. This screen keeps only candidates whose *title* matches an
expanded-taxonomy safety keyword, so enrich_abstracts.py runs on the safety
subset only.

Tradeoff (reported, not hidden): a title screen re-introduces the keyword-recall
gap snowball was meant to close (unusual-title safety papers with no keyword in
the title are dropped). The abstract-based LLM classifier in stage 4 is the final
arbiter; this screen only bounds enrichment cost. Adjust by lowering the bar
(e.g. --keep-misc) if recall matters more than S2 budget.

Writes snowball_safety_titlematched.json (kept subset, with matched categories)
and prints what is dropped so the ML-ref exclusion can be eyeballed.
"""
import argparse, io, json, os, sys
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from taxonomy_expanded import classify_multi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="paper_survey/snowball_candidates.json")
    ap.add_argument("--out", default="paper_survey/snowball_safety_titlematched.json")
    ap.add_argument("--min-cited", type=int, default=1,
                    help="drop candidates cited by fewer than N of our seeds")
    args = ap.parse_args()

    cands = json.load(io.open(args.inp, encoding="utf-8"))
    kept, dropped = [], []
    for c in cands:
        cats = classify_multi(c.get("title", ""))
        is_safety = cats != ["Miscellaneous Safety"]
        if is_safety and c.get("cited_by_n", 0) >= args.min_cited:
            kept.append({**c, "title_cats": cats})
        else:
            dropped.append(c)

    json.dump(kept, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"total candidates : {len(cands)}")
    print(f"kept (safety title): {len(kept)}")
    print(f"dropped            : {len(dropped)}")
    print(f"\nTop 15 DROPPED by cite count (should be ML refs, not safety):")
    for c in sorted(dropped, key=lambda x: -x.get("cited_by_n", 0))[:15]:
        print(f"  [{c.get('cited_by_n')}x {c.get('venue')}] {c.get('title')[:80]}")
    print(f"\nTop 15 KEPT by cite count (should be safety):")
    for c in sorted(kept, key=lambda x: -x.get("cited_by_n", 0))[:15]:
        print(f"  [{c.get('cited_by_n')}x] {c.get('title')[:70]} -> {c['title_cats']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
