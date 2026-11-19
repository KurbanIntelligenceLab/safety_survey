#!/usr/bin/env python3
"""Build the FULL-corpus full-text re-classification batches.

Companion to build_reclassify_sample.py, but instead of a stratified 191-paper
sample this emits batches over EVERY safety paper that has full text (method or
conclusion extracted). Used to test whether the paper's corpus-weighted Table-1
distribution -- not just per-category stability -- reproduces when the classifier
is given method+conclusion instead of title+abstract only.

Reuses the canonical PROMPT_TEMPLATE / DEFINITIONS / SCHEMA from classify_llm so
labels are produced by the exact same instrument as route A and the stratified
check. The only delta is that each paper block carries METHOD and CONCLUSION
lines after ABSTRACT (mirroring the stratified reclass_batches format).

Outputs (under paper_survey/_fulltext/):
  reclass_full_batches/batch_NNN.txt  - one self-contained prompt per batch
  reclass_full_gold.json              - id -> stored {category, secondary, title}
                                        for the post-run comparison
  reclass_full_manifest.json          - batch -> [ids], counts, config

Run the batches through the workflow harness (one subagent per batch, SCHEMA
forced), drop each result JSON in reclass_full_out/batch_NNN.json, then score
with score_fulltext_reclass_full.py.
"""
import io, json, os, argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from classify_llm import PROMPT_TEMPLATE, DEFINITIONS, BATCH  # canonical instrument

ROOT = HERE.parent.parent
# Paths overridable via env so the same builder serves the census corpus
# (RC_LAB/RC_SEC/RC_OUTDIR) without clobbering the old paper_survey run.
LAB = Path(os.environ.get("RC_LAB", ROOT / "paper_survey" / "papers_llm_cleaned.json"))
SEC = Path(os.environ.get("RC_SEC", ROOT / "paper_survey" / "_fulltext" / "sections"))
OUTDIR = Path(os.environ.get("RC_OUTDIR", ROOT / "paper_survey" / "_fulltext"))

TAIL = ('\n\nReturn a JSON object {"results": [...]}, one entry per paper above, '
        'each with: id (int), is_safety (bool), primary (category string), '
        'secondary (category string or null), confidence (0-1).')


def load_section(pid):
    f = SEC / ("%s.json" % pid)
    if not f.exists():
        return None
    return json.load(io.open(f, encoding="utf-8"))


def build_block(p):
    """One paper block: TITLE / ABSTRACT / METHOD / CONCLUSION (canonical truncation)."""
    abs_ = (p.get("abstract") or "")[:1200]
    meth = (p.get("_method") or "")[:1500]
    conc = (p.get("_conclusion") or "")[:1500]
    return (f"[{p['id']}] TITLE: {p['title']}\n"
            f"ABSTRACT: {abs_ or '(none)'}\n"
            f"METHOD: {meth or '(none)'}\n"
            f"CONCLUSION: {conc or '(none)'}")


def build_prompt(batch):
    body = "\n\n".join(build_block(p) for p in batch)
    return PROMPT_TEMPLATE.format(definitions=DEFINITIONS, papers=body) + TAIL


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="papers per batch (default: canonical BATCH=%d)" % BATCH)
    args = ap.parse_args()

    papers = json.load(io.open(LAB, encoding="utf-8"))
    saf = [p for p in papers if p.get("is_safety")]

    rows = []
    n_no_ft = 0
    for p in saf:
        sec = load_section(p["id"])
        if not sec or not (sec.get("has_method") or sec.get("has_conclusion")):
            n_no_ft += 1
            continue
        q = dict(p)
        q["_method"] = sec.get("method") or ""
        q["_conclusion"] = sec.get("conclusion") or ""
        rows.append(q)
    rows.sort(key=lambda r: r["id"])

    pdir = OUTDIR / "reclass_full_batches"
    pdir.mkdir(parents=True, exist_ok=True)
    for f in pdir.glob("batch_*.txt"):
        f.unlink()

    manifest = {"batch_size": args.batch, "n_safety": len(saf),
                "n_fulltext": len(rows), "n_no_fulltext": n_no_ft, "batches": {}}
    nb = 0
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        (pdir / f"batch_{nb:03d}.txt").write_text(build_prompt(batch), encoding="utf-8")
        manifest["batches"][f"batch_{nb:03d}"] = [p["id"] for p in batch]
        nb += 1
    manifest["n_batches"] = nb

    gold = {str(p["id"]): {"category": p.get("category"),
                           "secondary": p.get("category_secondary"),
                           "title": p.get("title")} for p in rows}
    json.dump(gold, io.open(OUTDIR / "reclass_full_gold.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(manifest, io.open(OUTDIR / "reclass_full_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"safety papers          : {len(saf)}")
    print(f"  with full text       : {len(rows)}  -> {nb} batches of {args.batch}")
    print(f"  without full text    : {n_no_ft}  (keep abstract label, not re-run)")
    print(f"wrote batches -> {pdir}")
    print(f"wrote gold    -> {OUTDIR/'reclass_full_gold.json'}")
    print(f"wrote manifest-> {OUTDIR/'reclass_full_manifest.json'}")


if __name__ == "__main__":
    main()
