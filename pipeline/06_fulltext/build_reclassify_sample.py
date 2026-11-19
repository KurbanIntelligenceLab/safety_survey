#!/usr/bin/env python
"""Build a stratified sample for a FULL-TEXT re-classification check.

The paper's labels were assigned from title+abstract. Now that method and
conclusion sections are cached, we re-classify a stratified sample using the
richer text and compare to the stored (abstract-only) primary label. This tests
whether the taxonomy assignment is stable under more context -- i.e. whether the
paper's distribution reproduces.

Emits paper_survey/_fulltext/reclass_sample.json and prints batch files for the
in-agent classifier route.
"""
import json, io, os, glob, argparse
from collections import defaultdict

LAB = "paper_survey/papers_llm_cleaned.json"
SEC = "paper_survey/_fulltext/sections"
OUT = "paper_survey/_fulltext/reclass_sample.json"


def load_section(pid):
    f = os.path.join(SEC, "%s.json" % pid)
    if not os.path.exists(f):
        return None
    return json.load(io.open(f, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=12, help="papers per category")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    papers = json.load(io.open(LAB, encoding="utf-8"))
    kept = [p for p in papers if p.get("is_safety")]

    # only papers that actually have full text (method or conclusion extracted)
    pool = defaultdict(list)
    for p in kept:
        sec = load_section(p["id"])
        if not sec or not (sec.get("has_method") or sec.get("has_conclusion")):
            continue
        p = dict(p)
        p["_method"] = (sec.get("method") or "")[:1500]
        p["_conclusion"] = (sec.get("conclusion") or "")[:1500]
        pool[p["category"]].append(p)

    # deterministic stratified pick: evenly spaced through each category list
    sample = []
    for cat, rows in sorted(pool.items()):
        rows.sort(key=lambda r: r["id"])
        if not rows:
            continue
        step = max(1, len(rows) // args.per)
        sample += rows[::step][:args.per]

    json.dump(sample, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("sample size: %d  (from %d full-text safety papers)" %
          (len(sample), sum(len(v) for v in pool.values())))
    print("per-category available / sampled:")
    chosen = defaultdict(int)
    for s in sample:
        chosen[s["category"]] += 1
    for cat in sorted(pool):
        print("  %-34s avail=%-4d sampled=%d" % (cat, len(pool[cat]), chosen[cat]))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
