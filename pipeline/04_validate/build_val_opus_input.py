#!/usr/bin/env python3
"""
Build a stratified validation sample (~250) from the 2,572 pool safety papers and
emit BOTH:
  (1) blind batch prompts for an independent Claude-Opus re-annotation, and
  (2) a blinded human-labeling sheet for the SAME papers (true-gold collection),

so we can compare stored (Sonnet 4.6) vs Opus vs human on one sample.

Deterministic (seeded) stratified-by-stored-category sampling: a floor per
category so small categories are represented, remainder allocated proportionally.

Outputs under paper_survey/_val_opus/:
  _gold.json            {id: stored labels + meta}  (held out from annotators)
  batch_000.txt ...     blind Opus prompts (~20 papers each)
  batch_index.json      {batch_file: [ids]}
  human_label_sheet.csv blinded: id,title,abstract,venue,year + empty label cols
  TAXONOMY.md           the 16 categories + definitions, for the human labeler
"""
import csv, io, json, os, random, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from classify_llm import build_batch_prompt, BATCH, CATEGORIES, DEFINITIONS

PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
SRC = os.path.join(PS, "papers_llm.json")
OUT = os.path.join(PS, "_val_opus")
TARGET = 250
FLOOR = 8
SEED = 42

SUMMARY_ADDENDUM = """

ADDITIONAL FIELD - for every paper also return:
4. summary: ONE sentence stating the paper's method and its main finding or
   conclusion, as reported in the abstract. If the abstract is "(none)", write a
   best-effort one-line summary from the title alone and lower the confidence.

So each result object must have: id, is_safety, primary, secondary, confidence,
summary.
"""


def allocate(counts, target, floor):
    """floor per category (capped at availability), remainder proportional."""
    alloc = {c: min(floor, counts[c]) for c in counts}
    used = sum(alloc.values())
    rem = target - used
    room = {c: counts[c] - alloc[c] for c in counts}
    total_room = sum(room.values())
    if rem > 0 and total_room > 0:
        # proportional to remaining size of each category
        extras = {c: int(rem * room[c] / total_room) for c in counts}
        for c in counts:
            add = min(extras[c], room[c])
            alloc[c] += add
        # top up any rounding shortfall by largest-room categories
        short = target - sum(alloc.values())
        for c in sorted(counts, key=lambda c: -(counts[c] - alloc[c])):
            if short <= 0:
                break
            if alloc[c] < counts[c]:
                alloc[c] += 1
                short -= 1
    return alloc


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    papers = json.load(io.open(SRC, encoding="utf-8"))
    pool = [p for p in papers if p.get("source") == "pool" and p.get("is_safety")]
    by_cat = {}
    for p in pool:
        by_cat.setdefault(p["category"], []).append(p)
    counts = {c: len(v) for c, v in by_cat.items()}
    alloc = allocate(counts, TARGET, FLOOR)

    sample = []
    for c, v in by_cat.items():
        k = alloc[c]
        sample.extend(rng.sample(v, k) if k < len(v) else list(v))
    sample.sort(key=lambda p: p["id"])
    print(f"stratified sample: {len(sample)} papers across {len(by_cat)} categories")
    for c in sorted(alloc, key=lambda c: -alloc[c]):
        print(f"  {alloc[c]:3d}/{counts[c]:4d}  {c}")

    # ---- gold (held out) ----
    gold = {}
    for p in sample:
        gold[str(p["id"])] = {
            "title": p["title"], "venue": p.get("venue"), "year": p.get("year"),
            "is_safety": p.get("is_safety"), "category": p.get("category"),
            "category_secondary": p.get("category_secondary"),
            "llm_confidence": p.get("llm_confidence"),
            "has_abstract": bool((p.get("abstract") or "").strip()),
        }
    json.dump(gold, io.open(os.path.join(OUT, "_gold.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---- blind Opus batch prompts ----
    index = {}
    n_batches = (len(sample) + BATCH - 1) // BATCH
    for b in range(n_batches):
        batch = sample[b * BATCH:(b + 1) * BATCH]
        slim = [{"id": p["id"], "title": p["title"], "abstract": p.get("abstract")}
                for p in batch]
        prompt = build_batch_prompt(slim) + SUMMARY_ADDENDUM
        fn = f"batch_{b:03d}.txt"
        io.open(os.path.join(OUT, fn), "w", encoding="utf-8").write(prompt)
        index[fn] = [p["id"] for p in batch]
    json.dump(index, io.open(os.path.join(OUT, "batch_index.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- human-labeling sheet (blinded) ----
    with io.open(os.path.join(OUT, "human_label_sheet.csv"), "w",
                 encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "venue", "year", "abstract",
                    "human_is_safety (yes/no)", "human_primary",
                    "human_secondary (or blank)", "human_notes"])
        for p in sample:
            w.writerow([p["id"], p["title"], p.get("venue"), p.get("year"),
                        (p.get("abstract") or "").replace("\n", " "), "", "", "", ""])

    # ---- taxonomy reference for the human labeler ----
    with io.open(os.path.join(OUT, "TAXONOMY.md"), "w", encoding="utf-8") as f:
        f.write("# Human labeling guide\n\n")
        f.write("For each row in `human_label_sheet.csv`, fill in:\n\n")
        f.write("- **human_is_safety**: `yes` only if the paper's PRIMARY contribution is "
                "about AI safety (broad sense); `no` if safety is only mentioned in passing.\n")
        f.write("- **human_primary**: the single best-fit category (exact name from the list).\n")
        f.write("- **human_secondary**: a second category only if the paper genuinely spans "
                "two; else leave blank.\n")
        f.write("- **human_notes**: optional, e.g. why it's a borderline case.\n\n")
        f.write("Do not look at our model labels while labeling — that's the point.\n\n")
        f.write("## The 16 categories\n\n")
        for line in DEFINITIONS.strip().splitlines():
            f.write(line + "\n")
        f.write("\n## Allowed category names (copy exactly)\n\n")
        for c in CATEGORIES:
            f.write(f"- {c}\n")

    print(f"\nwrote {n_batches} Opus batch files, human_label_sheet.csv ({len(sample)} rows), "
          f"TAXONOMY.md -> {OUT}")


if __name__ == "__main__":
    main()
