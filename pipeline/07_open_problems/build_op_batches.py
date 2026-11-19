#!/usr/bin/env python
"""Build per-subfield open-problem classification batches for the Layer-2 census.

Groups every 12,987-corpus paper that has a usable closing section by its subfield,
and writes batch prompt files (~BATCH papers each) listing that subfield's open-
problem label set plus the papers' [id]/title/conclusion excerpt. Each batch's
companion agent assigns, per paper, the matching label ids + off_topic/other.

Outputs:
  _fulltext12987/op_batches/<Subfield_slug>_NNN.txt   prompt files
  _fulltext12987/op_batches/manifest.json             {file, subfield, ids[]}
"""
import json, io, os, re, collections

PS = "paper_survey"
ROOT = os.path.join(PS, "_fulltext12987")
SEC = os.path.join(ROOT, "sections")
IDX = os.path.join(ROOT, "fulltext_index.json")
LABELS = os.path.join(ROOT, "open_problem_labels.json")
OUTDIR = os.path.join(ROOT, "op_batches")
BATCH = 25
MAXC = 2600  # chars of conclusion (+method fallback) per paper


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    idx = {r["id"]: r for r in json.load(io.open(IDX, encoding="utf-8"))}
    labels = json.load(io.open(LABELS, encoding="utf-8"))

    # gather usable papers by subfield
    bysub = collections.defaultdict(list)
    n_sec = n_usable = 0
    for fn in os.listdir(SEC):
        if not fn.endswith(".json"):
            continue
        n_sec += 1
        rec = json.load(io.open(os.path.join(SEC, fn), encoding="utf-8"))
        if not rec.get("ok"):
            continue
        text = (rec.get("conclusion") or "").strip()
        if len(text) < 200:  # fall back to method tail if conclusion thin
            text = (text + " " + (rec.get("method") or "")).strip()
        if len(text) < 200:
            continue
        pid = rec["id"]
        sub = (idx.get(pid) or {}).get("category")
        if sub not in labels:
            continue
        bysub[sub].append({"id": pid, "title": (rec.get("title") or "")[:160],
                            "text": text[:MAXC]})
        n_usable += 1

    manifest = []
    for sub, papers in sorted(bysub.items()):
        papers.sort(key=lambda p: p["id"])
        lab = labels[sub]
        lab_lines = "\n".join("  - %s: %s" % (k, v) for k, v in lab.items())
        for bi in range(0, len(papers), BATCH):
            chunk = papers[bi:bi + BATCH]
            tag = "%s_%03d" % (slug(sub), bi // BATCH)
            fn = os.path.join(OUTDIR, tag + ".txt")
            body = []
            body.append("SUBFIELD: %s\n" % sub)
            body.append(
                "TASK: For each paper below, read its CLOSING TEXT (conclusion / "
                "discussion / limitations / future work) and decide which of this "
                "subfield's OPEN-PROBLEM labels the paper itself states as open, "
                "unresolved, or future work. Assign every label that applies "
                "(multi-label; may be none). Judge only what the authors actually "
                "raise as open, not what the paper solves.\n")
            body.append("ALLOWED LABELS (id: meaning):\n%s\n" % lab_lines)
            body.append(
                "Also set: off_topic=true if the closing text is missing, garbled, "
                "or the paper is not actually about %s; other=true if the paper "
                "raises a substantive open problem that none of the labels above "
                "capture.\n" % sub)
            body.append(
                "OUTPUT: write ONLY a JSON object {\"results\":[{\"id\":<int>,"
                "\"labels\":[<label-id strings>],\"off_topic\":<bool>,"
                "\"other\":<bool>}]} with one entry per paper, ids matching [id].\n")
            body.append("=" * 60 + "\nPAPERS:\n")
            for p in chunk:
                body.append("[id] %d\nTITLE: %s\nCLOSING TEXT: %s\n---"
                            % (p["id"], p["title"], p["text"]))
            io.open(fn, "w", encoding="utf-8").write("\n".join(body))
            manifest.append({"file": fn, "subfield": sub,
                             "ids": [p["id"] for p in chunk]})

    json.dump(manifest, io.open(os.path.join(OUTDIR, "manifest.json"), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=1)
    print("sections seen: %d  usable: %d" % (n_sec, n_usable))
    print("subfield sizes:", {k: len(v) for k, v in sorted(bysub.items())})
    print("batches: %d  -> %s" % (len(manifest), OUTDIR))


if __name__ == "__main__":
    main()
