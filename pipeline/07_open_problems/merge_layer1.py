#!/usr/bin/env python
"""Merge per-batch layer1/<tag>.result.json files into layer1_results.json."""
import json, io, os

PS = "paper_survey"
ROOT = os.path.join(PS, "_fulltext12987")
OUTDIR = os.path.join(ROOT, "layer1")
MANIFEST = os.path.join(OUTDIR, "manifest.json")
OUT = os.path.join(ROOT, "layer1_results.json")


def main():
    manifest = json.load(io.open(MANIFEST, encoding="utf-8"))
    merged, missing, bad = [], [], []
    n = 0
    for b in manifest:
        tag = os.path.basename(b["file"])[:-4]
        rf = os.path.join(OUTDIR, tag + ".result.json")
        if not os.path.exists(rf):
            missing.append(tag); continue
        try:
            obj = json.load(io.open(rf, encoding="utf-8"))
            res = obj.get("results") if isinstance(obj, dict) else obj
            assert isinstance(res, list)
        except Exception as e:
            bad.append((tag, str(e))); continue
        merged.append({"subfield": b["subfield"], "file": b["file"], "results": res})
        n += len(res)
    json.dump(merged, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("merged %d/%d batches, %d papers" % (len(merged), len(manifest), n))
    if missing:
        print("MISSING (%d): %s" % (len(missing), ", ".join(missing[:15])))
    if bad:
        print("BAD (%d): %s" % (len(bad), bad[:10]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
