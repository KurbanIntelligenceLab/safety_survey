#!/usr/bin/env python3
"""Build the combined SPECTER-clustering corpus = old safety corpus + new census.

The clustering pipeline (embed_cluster / bertopic_compare / bertopic_tier2) reads
is_safety rows of papers_llm_cleaned.json (6,827) and a positionally-aligned
embedding cache (_specter_emb.npy). To re-cluster on the post-census corpus we
keep those 6,827 rows FIRST and IN ORDER (so the cached embeddings stay valid),
then append the 6,283 deduped census safety papers. New SPECTER embeddings are
only needed for the appended block.

Writes:
  paper_survey/_census/cluster_corpus_census.json   (combined records)
Order invariant: rows [0:6827] == is_safety rows of papers_llm_cleaned.json, same
order; rows [6827:] == census_gated_dedup.json, same order.
"""
import io, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PS = os.path.join(ROOT, "paper_survey")
OLD = os.path.join(PS, "papers_llm_cleaned.json")
NEW = os.path.join(PS, "_census", "census_gated_dedup.json")
OUT = os.path.join(PS, "_census", "cluster_corpus_census.json")
FIELDS = ("id", "title", "abstract", "category", "category_secondary",
          "is_safety", "llm_confidence", "venue", "year", "source", "doi")


def L(p):
    return json.load(io.open(p, encoding="utf-8"))


def proj(p, origin):
    r = {k: p.get(k) for k in FIELDS}
    r["is_safety"] = True
    r["cluster_origin"] = origin
    return r


def main():
    old = [proj(p, "old") for p in L(OLD) if p.get("is_safety")]
    new = [proj(p, "census") for p in L(NEW)]
    combined = old + new

    # invariants
    assert len(old) == 6827, f"old safety count drifted: {len(old)} != 6827"
    assert all(r.get("category") for r in combined), "missing category"
    cats_old = {r["category"] for r in old}
    cats_new = {r["category"] for r in new}
    assert cats_old == cats_new, f"category space mismatch: {cats_old ^ cats_new}"

    json.dump(combined, io.open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    n_abs = sum(1 for r in combined if r.get("abstract"))
    print(f"old safety   : {len(old)}")
    print(f"new census   : {len(new)}")
    print(f"combined     : {len(combined)}  ({len(cats_old)} categories)")
    print(f"with abstract: {n_abs}/{len(combined)}")
    print(f"embed needed : rows [{len(old)}:{len(combined)}] = {len(new)} new")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
