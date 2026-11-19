#!/usr/bin/env python3
"""
Derive the 13-category corpus from the 16-category LLM-classified corpus.

The 13-cat taxonomy (used by _jair_revision/SafetyReview_2026_13cat) is the
16-cat taxonomy with three categories folded out of the safety set:
Watermarking & IP Protection, Agentic & Frontier-Model Safety, and
Miscellaneous Safety. Concretely the derivation is:

  * drop every record whose PRIMARY category is one of the three and is_safety
  * null any SECONDARY label that is one of the three
  * keep all other records and labels identical

This script is verified to reproduce paper_survey/papers_llm_13cat.json exactly
from paper_survey/papers_llm.json (0 field mismatches). Running it on the
2025-augmented 16-cat corpus therefore extends the 13-cat corpus with the
AIES+NeurIPS 2025 backfill under identical rules, leaving the pre-existing
papers byte-identical.

  python3 pipeline/03_classify/derive_13cat.py            # validate + write augmented
  python3 pipeline/03_classify/derive_13cat.py --check     # validate reproduction only
"""
import io, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC16 = ROOT / "paper_survey/papers_llm.json"
REF13 = ROOT / "paper_survey/papers_llm_13cat.json"
AUG16 = ROOT / "paper_survey/papers_llm_2025augmented.json"
OUT13 = ROOT / "paper_survey/papers_llm_13cat_2025augmented.json"

DROP = {
    "Watermarking & IP Protection",
    "Agentic & Frontier-Model Safety",
    "Miscellaneous Safety",
}


def L(p):
    return json.load(io.open(p, encoding="utf-8"))


def transform(recs):
    out = []
    for p in recs:
        if p.get("is_safety") and p.get("category") in DROP:
            continue
        q = dict(p)
        if q.get("category_secondary") in DROP:
            q["category_secondary"] = None
        out.append(q)
    # reindex ids contiguously, matching how the corpus is built
    for i, p in enumerate(out):
        p["id"] = i
    return out


def reproduces_reference():
    cand = transform(L(SRC16))
    ref = L(REF13)
    if len(cand) != len(ref):
        return False, f"len {len(cand)} != {len(ref)}"
    ct = {p.get("title"): p for p in cand}
    dt = {p.get("title"): p for p in ref}
    if set(ct) != set(dt):
        return False, "title set differs"
    keys = ("is_safety", "category", "category_secondary")
    mism = [t for t in ct if any(ct[t].get(k) != dt[t].get(k) for k in keys)]
    return (not mism), f"{len(mism)} field mismatches"


def main():
    ok, msg = reproduces_reference()
    print(f"reproduction check vs papers_llm_13cat.json: {'PASS' if ok else 'FAIL'} ({msg})")
    if not ok:
        sys.exit("Refusing to derive: transform no longer reproduces the reference 13-cat corpus.")
    if "--check" in sys.argv:
        return

    aug = L(AUG16)
    out = transform(aug)
    n_safe = sum(bool(p.get("is_safety")) for p in out)
    json.dump(out, io.open(OUT13, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"augmented 16-cat {len(aug)} -> 13-cat {len(out)}  (safety: {n_safe})")
    print(f"wrote {OUT13}")


if __name__ == "__main__":
    main()
