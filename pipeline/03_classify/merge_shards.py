#!/usr/bin/env python3
"""
Merge per-agent classification shards (paper_survey/llm_shards/shard_*.json) onto
classify_input.json, producing the stage-4 output papers_llm.json. Reports
coverage so missing/failed shards are visible (no silent truncation).

Each shard is a list of {id, is_safety, category, category_secondary, llm_confidence}.
"""
import glob, io, json, os

INP = "paper_survey/classify_input.json"
OUT = "paper_survey/papers_llm.json"
SHARDS = "paper_survey/llm_shards/shard_*.json"


def main():
    papers = {p["id"]: p for p in json.load(io.open(INP, encoding="utf-8"))}
    labels = {}
    bad = []
    for f in sorted(glob.glob(SHARDS)):
        try:
            for r in json.load(io.open(f, encoding="utf-8")):
                labels[r["id"]] = r
        except Exception as e:
            bad.append((f, str(e)))

    out = []
    for pid, p in papers.items():
        r = labels.get(pid)
        if not r:
            continue
        q = dict(p)
        q["is_safety"] = r.get("is_safety")
        q["category"] = r.get("category")
        q["category_secondary"] = r.get("category_secondary")
        q["llm_confidence"] = r.get("llm_confidence")
        out.append(q)

    json.dump(out, io.open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    covered = len(out)
    missing = sorted(set(papers) - set(labels))
    kept = sum(1 for p in out if p.get("is_safety"))
    print(f"shards merged   : {len(glob.glob(SHARDS))}")
    if bad:
        print(f"UNREADABLE shards: {len(bad)} -> {bad[:5]}")
    print(f"papers labeled  : {covered}/{len(papers)}")
    print(f"  kept(is_safety): {kept}")
    print(f"  dropped        : {covered - kept}")
    if missing:
        print(f"MISSING ids     : {len(missing)} (e.g. {missing[:10]})")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
