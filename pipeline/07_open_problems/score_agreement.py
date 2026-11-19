#!/usr/bin/env python
"""Layer-1 (deep-coded ground truth) vs Layer-2 (full census) agreement.

For the papers that appear in BOTH (the Layer-1 sample is drawn from census
papers), compute:
  - per-paper Jaccard of the assigned label sets, averaged (paper-level reliability)
  - per-paper precision/recall/F1 treating Layer-1 as truth, Layer-2 as prediction
  - theme-level (per subfield x label) agreement: does Layer-2's prevalence rank
    /presence of each label track Layer-1's on the shared sample (presence corr)
Also reports, per subfield, the most common novel_gaps Layer-1 surfaced that are
not in the label set (candidate new open problems).

Writes _fulltext12987/agreement.json and prints a summary.
"""
import json, io, os, collections

PS = "paper_survey"
ROOT = os.path.join(PS, "_fulltext12987")
L1 = os.path.join(ROOT, "layer1_results.json")
L2 = os.path.join(ROOT, "op_results.json")
LABELS = os.path.join(ROOT, "open_problem_labels.json")
OUT = os.path.join(ROOT, "agreement.json")


def load_paper_labels(path):
    """id -> (subfield, set(labels), off_topic)."""
    data = json.load(io.open(path, encoding="utf-8"))
    out = {}
    for batch in data:
        sub = batch["subfield"]
        for r in batch["results"]:
            pid = r.get("id")
            out[pid] = (sub, set(r.get("labels") or []), bool(r.get("off_topic")))
    return out


def jac(a, b):
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / float(len(u)) if u else 1.0


def main():
    labels = json.load(io.open(LABELS, encoding="utf-8"))
    l1 = load_paper_labels(L1)
    l2 = load_paper_labels(L2)
    shared = [pid for pid in l1 if pid in l2]

    # paper-level (skip papers either side marked off_topic)
    js, tp, fp, fn = [], 0, 0, 0
    persub = collections.defaultdict(lambda: {"jac": [], "n": 0})
    for pid in shared:
        s1, a, ot1 = l1[pid]
        s2, b, ot2 = l2[pid]
        if ot1 or ot2:
            continue
        valid = set(labels.get(s1, {}))
        a = a & valid
        b = b & valid
        js.append(jac(a, b))
        persub[s1]["jac"].append(jac(a, b))
        persub[s1]["n"] += 1
        tp += len(a & b); fp += len(b - a); fn += len(a - b)

    prec = tp / float(tp + fp) if (tp + fp) else 0.0
    rec = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    mean_jac = sum(js) / len(js) if js else 0.0

    # theme-level: per subfield, share of each label under L1 vs L2 on shared set
    theme = {}
    for sub in labels:
        ids = [pid for pid in shared if l1[pid][0] == sub and not l1[pid][2]
               and not l2[pid][2]]
        n = len(ids)
        if not n:
            continue
        rows = {}
        for L in labels[sub]:
            s1 = sum(1 for pid in ids if L in l1[pid][1]) / float(n)
            s2 = sum(1 for pid in ids if L in l2[pid][1]) / float(n)
            rows[L] = {"l1": round(100 * s1, 1), "l2": round(100 * s2, 1)}
        theme[sub] = {"n": n, "labels": rows}

    # novel gaps from layer-1
    gaps = collections.defaultdict(collections.Counter)
    for batch in json.load(io.open(L1, encoding="utf-8")):
        for r in batch["results"]:
            for g in (r.get("novel_gaps") or []):
                gaps[batch["subfield"]][g.strip().lower()] += 1

    summary = {
        "shared_papers": len(shared),
        "scored_papers": len(js),
        "mean_paper_jaccard": round(mean_jac, 3),
        "label_precision_L2_vs_L1": round(prec, 3),
        "label_recall_L2_vs_L1": round(rec, 3),
        "label_f1": round(f1, 3),
        "per_subfield_jaccard": {s: round(sum(d["jac"]) / len(d["jac"]), 3)
                                 for s, d in persub.items() if d["jac"]},
        "theme_level": theme,
        "top_novel_gaps": {s: c.most_common(8) for s, c in gaps.items()},
    }
    json.dump(summary, io.open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("shared=%d scored=%d  mean paper Jaccard=%.3f  F1=%.3f (P=%.3f R=%.3f)"
          % (len(shared), len(js), mean_jac, f1, prec, rec))
    print("per-subfield Jaccard:")
    for s, j in sorted(summary["per_subfield_jaccard"].items(),
                       key=lambda kv: -kv[1]):
        print("  %-34s %.3f" % (s[:34], j))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
