#!/usr/bin/env python
"""Regenerate the Layer-1 deep-read prompt files for the SAME fixed sample.

After folding the 14 recurring gap themes into open_problem_labels.json, the
deep-read agreement must be re-measured against the IDENTICAL label set the
census now uses. To keep the reference sample fixed (the same 700 papers already
selected), we do NOT re-select: we read the existing layer1/manifest.json
(file, subfield, ids[]) and rewrite each prompt .txt with the EXPANDED label set
and the same papers' full closing text. Batching and ids are untouched.
"""
import json, io, os, re

PS = "paper_survey"
ROOT = os.path.join(PS, "_fulltext12987")
SEC = os.path.join(ROOT, "sections")
IDX = os.path.join(ROOT, "fulltext_index.json")
LABELS = os.path.join(ROOT, "open_problem_labels.json")
OUTDIR = os.path.join(ROOT, "layer1")
MANIFEST = os.path.join(OUTDIR, "manifest.json")
MAXC = 6000


def closing(pid):
    fn = os.path.join(SEC, "%d.json" % pid)
    if not os.path.exists(fn):
        return ""
    rec = json.load(io.open(fn, encoding="utf-8"))
    if not rec.get("ok"):
        return ""
    t = (rec.get("conclusion") or "").strip()
    if len(t) < 200:
        t = (t + " " + (rec.get("method") or "")).strip()
    return t


def main():
    idx = {r["id"]: r for r in json.load(io.open(IDX, encoding="utf-8"))}
    labels = json.load(io.open(LABELS, encoding="utf-8"))
    manifest = json.load(io.open(MANIFEST, encoding="utf-8"))

    n = 0
    for b in manifest:
        sub = b["subfield"]
        lab = labels[sub]
        lab_lines = "\n".join("  - %s: %s" % (k, v) for k, v in lab.items())
        body = []
        body.append("SUBFIELD: %s\n" % sub)
        body.append(
            "TASK (deep coding / ground truth): For each paper below, read its "
            "FULL closing text carefully and decide which of this subfield's "
            "OPEN-PROBLEM labels the paper's authors themselves state as open, "
            "unresolved, or future work. Assign every label that genuinely "
            "applies (multi-label; may be none). Be precise: code only what the "
            "authors actually raise as open, not what they solve or merely "
            "mention. ALSO record, in 'novel_gaps', any substantive open problem "
            "the authors raise that NONE of the labels capture (short phrase "
            "each; empty list if none).\n")
        body.append("ALLOWED LABELS (id: meaning):\n%s\n" % lab_lines)
        body.append(
            "Set off_topic=true only if the closing text is missing/garbled or "
            "the paper is not actually about %s.\n" % sub)
        body.append(
            "OUTPUT: write ONLY a JSON object {\"results\":[{\"id\":<int>,"
            "\"labels\":[<label-id strings>],\"novel_gaps\":[<short strings>],"
            "\"off_topic\":<bool>}]} with one entry per paper, ids matching [id].\n")
        body.append("=" * 60 + "\nPAPERS:\n")
        for pid in b["ids"]:
            title = (idx.get(pid) or {}).get("title", "")[:200]
            t = closing(pid)
            body.append("[id] %d\nTITLE: %s\nFULL CLOSING TEXT: %s\n---"
                        % (pid, title, t[:MAXC]))
        io.open(b["file"], "w", encoding="utf-8").write("\n".join(body))
        n += 1

    print("rewrote %d layer1 prompt files (fixed sample, expanded labels)" % n)


if __name__ == "__main__":
    main()
