#!/usr/bin/env python
"""Merge the per-batch <tag>.result.json files (written by the census workflow
agents) into op_results.json, the format score_op_census.py consumes:
  [ {"subfield":..., "file":..., "results":[{id,labels,off_topic,other}]} , ... ]
Reports how many batches are still missing/unparseable so we know coverage.
"""
import json, io, os

PS = "paper_survey"
ROOT = os.path.join(PS, "_fulltext12987")
OUTDIR = os.path.join(ROOT, "op_batches")
MANIFEST = os.path.join(OUTDIR, "manifest.json")
OUT = os.path.join(ROOT, "op_results.json")


def main():
    manifest = json.load(io.open(MANIFEST, encoding="utf-8"))
    merged, missing, bad = [], [], []
    n_papers = 0
    for b in manifest:
        tag = os.path.basename(b["file"])[:-4]
        rf = os.path.join(OUTDIR, tag + ".result.json")
        if not os.path.exists(rf):
            missing.append(tag)
            continue
        try:
            obj = json.load(io.open(rf, encoding="utf-8"))
            res = obj.get("results") if isinstance(obj, dict) else obj
            if not isinstance(res, list):
                raise ValueError("no results list")
        except Exception as e:
            bad.append((tag, str(e)))
            continue
        merged.append({"subfield": b["subfield"], "file": b["file"], "results": res})
        n_papers += len(res)

    json.dump(merged, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("merged batches: %d / %d   papers: %d" % (len(merged), len(manifest), n_papers))
    if missing:
        print("MISSING (%d): %s%s" % (len(missing), ", ".join(missing[:15]),
                                      " ..." if len(missing) > 15 else ""))
    if bad:
        print("UNPARSEABLE (%d): %s" % (len(bad), bad[:10]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
