#!/usr/bin/env python3
"""
Merge in-agent gate/classify results back onto the enriched papers.

The gate workflow fans out one subagent per batch_NNN.txt prompt (built by
build_batches.py) and each writes a batch_NNN.json result file of the canonical
{"results":[{id,is_safety,primary,secondary,confidence}, ...]} shape. This
script reads every result file, applies labels onto the papers by integer id
(same ids build_batches assigned by enumerate order), and writes the labeled
corpus. Robust to missing/partial batches: it reports any ids never labeled.

  python3 pipeline/03_classify/merge_gate_results.py \
      --papers paper_survey/_census/census_enriched.json \
      --results-dir paper_survey/_census/gate_out \
      --out paper_survey/_census/census_gated.json
"""
import argparse, io, json, glob, os
from collections import Counter


def L(p):
    return json.load(io.open(p, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    papers = L(args.papers)
    for i, p in enumerate(papers):
        p["id"] = i
    by_id = {p["id"]: p for p in papers}

    labeled, bad_cat, dup = set(), 0, 0
    files = sorted(glob.glob(os.path.join(args.results_dir, "batch_*.json")))
    for f in files:
        try:
            obj = L(f)
        except Exception as e:
            print(f"  WARN unreadable {os.path.basename(f)}: {str(e)[:50]}")
            continue
        for r in obj.get("results", []):
            p = by_id.get(r.get("id"))
            if p is None:
                continue
            if r["id"] in labeled:
                dup += 1
                continue
            labeled.add(r["id"])
            p["is_safety"] = bool(r.get("is_safety"))
            p["category"] = r.get("primary")
            p["category_secondary"] = r.get("secondary")
            p["llm_confidence"] = r.get("confidence")

    missing = [i for i in by_id if i not in labeled]
    json.dump(papers, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    kept = [p for p in papers if p.get("is_safety")]
    print(f"result files: {len(files)}")
    print(f"labeled: {len(labeled)}/{len(papers)}  dup_ids={dup}  missing={len(missing)}")
    if missing:
        print(f"  first missing ids: {missing[:20]}")
    print(f"is_safety kept: {len(kept)}  dropped: {len(labeled) - len(kept)}")
    print("kept by primary:", dict(Counter(p['category'] for p in kept).most_common()))
    print("kept by source:", dict(Counter(p.get('source') for p in kept)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
