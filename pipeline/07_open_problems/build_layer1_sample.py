#!/usr/bin/env python
"""Build the Layer-1 deep-coding sample (~50 papers/subfield) for ground truth.

Draws from the SAME papers the Layer-2 census classified (op_results.json), so
every Layer-1 paper has a Layer-2 label set to compare against (per-paper
agreement / Jaccard). Within each subfield we deterministically pick ~TARGET
papers, preferring those with substantial closing text and spreading across the
id range (every k-th after sorting by id). Writes small prompt files (BATCH
papers, FULL closing text) that ask for careful per-paper deep coding plus
free-text novel gaps.

Outputs:
  _fulltext12987/layer1/<Subfield_slug>_NNN.txt   prompt files
  _fulltext12987/layer1/manifest.json             {file, subfield, ids[]}
  _fulltext12987/layer1/sample_ids.json           {subfield: [ids]}
"""
import json, io, os, re, collections

PS = "paper_survey"
ROOT = os.path.join(PS, "_fulltext12987")
SEC = os.path.join(ROOT, "sections")
IDX = os.path.join(ROOT, "fulltext_index.json")
LABELS = os.path.join(ROOT, "open_problem_labels.json")
RESULTS = os.path.join(ROOT, "op_results.json")
OUTDIR = os.path.join(ROOT, "layer1")
TARGET = 50      # papers per subfield (capped at availability)
BATCH = 5        # papers per prompt file (full text -> small batches)
MAXC = 6000      # chars of closing text per paper (full, not truncated)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    idx = {r["id"]: r for r in json.load(io.open(IDX, encoding="utf-8"))}
    labels = json.load(io.open(LABELS, encoding="utf-8"))
    census = json.load(io.open(RESULTS, encoding="utf-8"))

    # census papers per subfield that were NOT off_topic (real, codeable papers)
    classified = collections.defaultdict(set)
    for batch in census:
        sub = batch["subfield"]
        for r in batch["results"]:
            if not r.get("off_topic"):
                classified[sub].add(r.get("id"))

    # load closing text for those ids
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

    sample_ids = {}
    manifest = []
    for sub in sorted(labels):
        if sub.startswith("_"):
            continue
        ids = sorted(classified.get(sub, []))
        # keep only ones with substantial closing text, prefer longer
        cand = [(pid, closing(pid)) for pid in ids]
        cand = [(pid, t) for pid, t in cand if len(t) >= 400]
        cand.sort(key=lambda x: x[0])  # stable by id
        if len(cand) <= TARGET:
            pick = cand
        else:
            # deterministic stratified stride across the id-sorted list
            step = len(cand) / float(TARGET)
            pick = [cand[int(i * step)] for i in range(TARGET)]
        sample_ids[sub] = [pid for pid, _ in pick]

        lab = labels[sub]
        lab_lines = "\n".join("  - %s: %s" % (k, v) for k, v in lab.items())
        for bi in range(0, len(pick), BATCH):
            chunk = pick[bi:bi + BATCH]
            tag = "%s_%03d" % (slug(sub), bi // BATCH)
            fn = os.path.join(OUTDIR, tag + ".txt")
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
            for pid, t in chunk:
                title = (idx.get(pid) or {}).get("title", "")[:200]
                body.append("[id] %d\nTITLE: %s\nFULL CLOSING TEXT: %s\n---"
                            % (pid, title, t[:MAXC]))
            io.open(fn, "w", encoding="utf-8").write("\n".join(body))
            manifest.append({"file": fn, "subfield": sub,
                             "ids": [pid for pid, _ in chunk]})

    json.dump(manifest, io.open(os.path.join(OUTDIR, "manifest.json"), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(sample_ids, io.open(os.path.join(OUTDIR, "sample_ids.json"), "w",
                                  encoding="utf-8"), ensure_ascii=False, indent=1)
    tot = sum(len(v) for v in sample_ids.values())
    print("sampled per subfield:", {k: len(v) for k, v in sorted(sample_ids.items())})
    print("total sampled: %d  batches: %d  -> %s" % (tot, len(manifest), OUTDIR))


if __name__ == "__main__":
    main()
