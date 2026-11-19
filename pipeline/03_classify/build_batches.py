#!/usr/bin/env python3
"""
Prep step for the in-agent/workflow classification route. Reads the enriched
papers, assigns stable integer ids, and writes one self-contained prompt file
per batch (using the CANONICAL build_batch_prompt so labels match route A), plus
an index mapping id -> paper metadata for merging the results back.

  python3 pipeline/03_classify/build_batches.py --in paper_survey_v2/v2_enriched.json \
      --prompt-dir paper_survey_v2/prompts --index paper_survey_v2/batch_index.json
"""
import argparse, io, json
from pathlib import Path
import sys
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from classify_llm import build_batch_prompt, BATCH


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--prompt-dir", default="paper_survey_v2/prompts")
    ap.add_argument("--index", default="paper_survey_v2/batch_index.json")
    args = ap.parse_args()

    papers = json.load(io.open(args.inp, encoding="utf-8"))
    for i, p in enumerate(papers):
        p["id"] = i

    pdir = Path(args.prompt_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    # clear stale prompt/out files
    for f in pdir.glob("batch_*.txt"):
        f.unlink()

    n_batches = 0
    for i in range(0, len(papers), BATCH):
        batch = papers[i:i + BATCH]
        prompt = build_batch_prompt(batch)
        prompt += ("\n\nReturn a JSON object {\"results\": [...]}, one entry per "
                   "paper above, each with: id (int), is_safety (bool), primary "
                   "(category string), secondary (category string or null), "
                   "confidence (0-1).")
        (pdir / f"batch_{n_batches:03d}.txt").write_text(prompt, encoding="utf-8")
        n_batches += 1

    # index for merge (keep everything except the bulky abstract)
    index = {str(p["id"]): {k: p.get(k) for k in
             ("title", "venue", "year", "dblp_key", "doi")} for p in papers}
    json.dump(index, io.open(args.index, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    with_abs = sum(1 for p in papers if p.get("abstract"))
    print(f"papers={len(papers)}  with_abstract={with_abs} "
          f"({100*with_abs/len(papers):.0f}%)")
    print(f"wrote {n_batches} batch prompts -> {pdir}")
    print(f"wrote index -> {args.index}")


if __name__ == "__main__":
    main()
