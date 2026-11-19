#!/usr/bin/env python3
"""
Stage 1 of the pooled-paper re-classification audit.

Select the query-pool safety papers (source=="pool" and is_safety) from
papers_llm.json, stash their STORED labels in a gold file (held out from the
classifier), and emit BLIND batch-prompt files using the *canonical* prompt from
classify_llm.py (so the re-run is methodologically identical to the original),
plus one additive instruction: also return a one-sentence `summary`.

Outputs (all under paper_survey/_pool_audit/):
  _pool_gold.json      {id: {title, venue, year, is_safety, category,
                              category_secondary, llm_confidence, has_abstract}}
  batch_000.txt ...    blind prompts, ~BATCH papers each (no stored labels)
  batch_index.json     {batch_file: [ids]} for coverage checking in Stage 3

Read-only w.r.t. the corpus; never modifies papers_llm.json.
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from classify_llm import build_batch_prompt, BATCH  # canonical prompt + batch size

PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
SRC = os.path.join(PS, "papers_llm.json")
OUT = os.path.join(PS, "_pool_audit")

# Additive instruction appended AFTER the canonical prompt so the original
# is_safety/primary/secondary text stays byte-identical. Adds the `summary` field.
SUMMARY_ADDENDUM = """

ADDITIONAL FIELD - for every paper also return:
4. summary: ONE sentence stating the paper's method and its main finding or
   conclusion, as reported in the abstract. If the abstract is "(none)", write a
   best-effort one-line summary from the title alone and lower the confidence.

So each result object must have: id, is_safety, primary, secondary, confidence,
summary.
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    papers = json.load(io.open(SRC, encoding="utf-8"))
    pool = [p for p in papers if p.get("source") == "pool" and p.get("is_safety")]
    pool.sort(key=lambda p: p["id"])
    print(f"pool safety papers: {len(pool)}")

    # ---- gold (held-out stored labels) ----
    gold = {}
    n_no_abs = 0
    for p in pool:
        has_abs = bool((p.get("abstract") or "").strip())
        if not has_abs:
            n_no_abs += 1
        gold[str(p["id"])] = {
            "title": p["title"],
            "venue": p.get("venue"),
            "year": p.get("year"),
            "is_safety": p.get("is_safety"),
            "category": p.get("category"),
            "category_secondary": p.get("category_secondary"),
            "llm_confidence": p.get("llm_confidence"),
            "has_abstract": has_abs,
        }
    json.dump(gold, io.open(os.path.join(OUT, "_pool_gold.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  title-only (no abstract): {n_no_abs}")

    # ---- blind batch prompts ----
    index = {}
    n_batches = (len(pool) + BATCH - 1) // BATCH
    for b in range(n_batches):
        batch = pool[b * BATCH:(b + 1) * BATCH]
        # build_batch_prompt only reads id/title/abstract -> no labels leak.
        slim = [{"id": p["id"], "title": p["title"],
                 "abstract": p.get("abstract")} for p in batch]
        prompt = build_batch_prompt(slim) + SUMMARY_ADDENDUM
        fn = f"batch_{b:03d}.txt"
        io.open(os.path.join(OUT, fn), "w", encoding="utf-8").write(prompt)
        index[fn] = [p["id"] for p in batch]

    json.dump(index, io.open(os.path.join(OUT, "batch_index.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)
    total = sum(len(v) for v in index.values())
    print(f"wrote {n_batches} batch files covering {total} ids -> {OUT}")
    assert total == len(pool), "batch coverage mismatch"
    assert len(set(i for v in index.values() for i in v)) == len(pool), "dup ids"


if __name__ == "__main__":
    main()
