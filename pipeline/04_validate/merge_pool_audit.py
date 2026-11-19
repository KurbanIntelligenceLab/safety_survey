#!/usr/bin/env python3
"""
Stage 2b: merge the per-batch agent outputs into one independent-labels file and
report coverage. Validates each result against the canonical taxonomy enum and
the batch_index, so any missing / malformed / short batch is surfaced (and can be
re-run) rather than silently dropped.

Reads : paper_survey/_pool_audit/out/batch_*.json   ({"results":[...]})
        paper_survey/_pool_audit/batch_index.json   (expected ids per batch)
Writes: paper_survey/_pool_audit/_pool_independent.json  (flat list of results)
Prints: per-batch coverage, any missing/extra/invalid ids, and bad batch list.
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from classify_llm import CATEGORIES

PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
OUT = os.path.join(PS, "_pool_audit")
OUTDIR = os.path.join(OUT, "out")
CATSET = set(CATEGORIES)
REQ = {"id", "is_safety", "primary", "secondary", "confidence", "summary"}


def main():
    index = json.load(io.open(os.path.join(OUT, "batch_index.json"), encoding="utf-8"))
    expected_ids = {i for v in index.values() for i in v}

    merged, seen = [], set()
    bad_batches, invalid = [], []
    for fn in sorted(index):
        path = os.path.join(OUTDIR, fn.replace(".txt", ".json"))
        if not os.path.exists(path):
            bad_batches.append((fn, "missing-file"))
            continue
        try:
            obj = json.load(io.open(path, encoding="utf-8"))
            results = obj["results"] if isinstance(obj, dict) else obj
        except Exception as e:
            bad_batches.append((fn, f"parse-error:{e}"))
            continue
        got = {r.get("id") for r in results if isinstance(r, dict)}
        want = set(index[fn])
        if got != want:
            bad_batches.append((fn, f"id-mismatch miss={sorted(want-got)} extra={sorted(got-want)}"))
        for r in results:
            if not isinstance(r, dict) or not REQ.issubset(r):
                invalid.append((fn, "missing-keys", r.get("id") if isinstance(r, dict) else None))
                continue
            if r["primary"] not in CATSET or (r["secondary"] not in CATSET and r["secondary"] is not None):
                invalid.append((fn, "bad-category", r.get("id")))
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            merged.append(r)

    json.dump(merged, io.open(os.path.join(OUT, "_pool_independent.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=1)

    missing_ids = sorted(expected_ids - seen)
    print(f"merged results : {len(merged)}")
    print(f"expected ids   : {len(expected_ids)}")
    print(f"covered ids    : {len(seen)}")
    print(f"MISSING ids    : {len(missing_ids)}")
    if bad_batches:
        print(f"\nBAD BATCHES ({len(bad_batches)}) -- re-run these:")
        for fn, why in bad_batches:
            print(f"  {fn}: {why}")
    if invalid:
        print(f"\nINVALID RESULTS ({len(invalid)}):")
        for x in invalid[:50]:
            print(f"  {x}")
    if missing_ids:
        print(f"\nfirst missing ids: {missing_ids[:30]}")
    ok = (not bad_batches) and (not missing_ids) and (not invalid)
    print(f"\nCOVERAGE COMPLETE: {ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
