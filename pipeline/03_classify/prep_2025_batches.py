#!/usr/bin/env python3
"""
One-off prep for the 2025 backfill (AIES + NeurIPS). Combines the two screened
candidate sets into a single stable-id input, writes canonical per-batch prompt
files (reusing build_batch_prompt so labels match the rest of the corpus), and
writes an index that ALSO carries url + source so the merge step can key NeurIPS
records by url (they have no DOI).

  python3 pipeline/03_classify/prep_2025_batches.py
"""
import io, json
from pathlib import Path
import sys
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from classify_llm import build_batch_prompt, BATCH

SOURCES = [
    ("paper_survey/_aies2025/aies2025_candidates.json", "aies2025"),
    ("paper_survey/_neurips2025/neurips2025_candidates.json", "neurips2025"),
]
OUTDIR = Path("paper_survey/_2025merge")
PROMPT_DIR = OUTDIR / "prompts"
INDEX = OUTDIR / "batch_index.json"


def main():
    papers = []
    for path, src in SOURCES:
        recs = json.load(io.open(path, encoding="utf-8"))
        for p in recs:
            p["source"] = src
            papers.append(p)
    for i, p in enumerate(papers):
        p["id"] = i

    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    for f in PROMPT_DIR.glob("batch_*.txt"):
        f.unlink()
    for f in PROMPT_DIR.glob("out_*.json"):
        f.unlink()

    n_batches = 0
    for i in range(0, len(papers), BATCH):
        batch = papers[i:i + BATCH]
        prompt = build_batch_prompt(batch)
        prompt += ("\n\nReturn a JSON object {\"results\": [...]}, one entry per "
                   "paper above, each with: id (int), is_safety (bool), primary "
                   "(category string), secondary (category string or null), "
                   "confidence (0-1).")
        (PROMPT_DIR / f"batch_{n_batches:03d}.txt").write_text(prompt, encoding="utf-8")
        n_batches += 1

    index = {str(p["id"]): {k: p.get(k) for k in
             ("title", "venue", "year", "dblp_key", "doi", "url", "source")}
             for p in papers}
    json.dump(index, io.open(INDEX, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    with_abs = sum(1 for p in papers if p.get("abstract"))
    print(f"papers={len(papers)}  with_abstract={with_abs} "
          f"({100*with_abs/len(papers):.0f}%)")
    print(f"wrote {n_batches} batch prompts -> {PROMPT_DIR}")
    print(f"wrote index -> {INDEX}")
    print(f"N_BATCHES={n_batches}")


if __name__ == "__main__":
    main()
