#!/usr/bin/env python3
"""Per-LABEL recoverability probe for the open-problem taxonomy.

Successor to op_cluster_validate.py, which had two measurement problems:
  1. It used only single-label papers (1364 of 1637), discarding 17% of the data.
  2. It scored multi-class accuracy against a majority baseline. With 4-9 labels
     per subfield and one label often dominating, that baseline ran as high as
     0.771 (Watermarking), so a high accuracy could mean nothing and a low one
     could be an artifact of class fragmentation rather than a bad label.

This version fixes both:
  - Uses ALL labeled papers via one-vs-rest, so multi-label papers count as a
    positive for each label they carry (open problems genuinely co-occur; forcing
    a single label was throwing away real signal).
  - Scores each label by ROC-AUC, which is prevalence-independent: chance is 0.5
    regardless of how rare or dominant the label is. This makes labels comparable
    across subfields of very different sizes, which accuracy never was.

Output is per-LABEL rather than per-subfield, because that is the actionable unit:
it says which specific open problem is not recoverable from the closing text, not
merely that some subfield scored low. For weak labels it also reports the nearest
other label by centroid cosine, as a merge candidate.

  sbatch pipeline/07_open_problems/op_label_probe.sbatch
"""
import io, json, os, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey", "_fulltext12987"))
OUTDIR = os.path.join(FT, "opcluster")
EMB = os.path.join(OUTDIR, "_op_emb.npy")
META = os.path.join(OUTDIR, "_op_meta.json")
OUT = os.path.join(OUTDIR, "op_label_probe.json")

MIN_POS = 8          # need this many positives to cross-validate at all
WEAK_AUC = 0.65      # below this, a label is not reliably recoverable


def ovr_auc(Z, pos, seed=0):
    """Cross-validated one-vs-rest ROC-AUC for one label."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = pos.astype(int)
    npos = int(y.sum())
    if npos < MIN_POS or npos == len(y):
        return None
    k = int(min(5, npos))
    if k < 3:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(Z, y):
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(Z[tr], y[tr])
        scores[te] = clf.predict_proba(Z[te])[:, 1]
    return dict(n_pos=npos, n=len(y),
                auc=round(float(roc_auc_score(y, scores)), 3),
                ap=round(float(average_precision_score(y, scores)), 3),
                base_rate=round(float(y.mean()), 3))


def main():
    Z = np.load(EMB).astype(np.float32)
    meta = json.load(io.open(META, encoding="utf-8"))
    assert len(meta) == Z.shape[0], (len(meta), Z.shape)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    bysf = collections.defaultdict(list)
    for i, m in enumerate(meta):
        bysf[m["subfield"]].append(i)

    out = {"n_docs": len(meta), "min_pos": MIN_POS, "weak_auc": WEAK_AUC, "subfields": {}}
    all_auc, weak, untested = [], [], []
    for sf in sorted(bysf):
        idx = np.array(bysf[sf])
        Zs, Zsn = Z[idx], Zn[idx]
        labs = sorted({l for i in idx for l in meta[i]["labels"]})
        # label -> boolean membership over this subfield's docs
        memb = {l: np.array([l in meta[i]["labels"] for i in idx]) for l in labs}
        # centroids for merge-candidate lookup
        cent = {l: Zsn[memb[l]].mean(0) for l in labs if memb[l].sum() > 0}
        for l in cent:
            cent[l] = cent[l] / (np.linalg.norm(cent[l]) + 1e-9)

        rec, aucs = {}, []
        for l in labs:
            r = ovr_auc(Zs, memb[l])
            if r is None:
                rec[l] = dict(n_pos=int(memb[l].sum()), auc=None, note="too few positives")
                untested.append((sf, l, int(memb[l].sum())))
                continue
            # nearest other label by centroid cosine = merge candidate
            sims = [(float(cent[l] @ cent[o]), o) for o in cent if o != l]
            sims.sort(reverse=True)
            r["nearest_label"] = sims[0][1] if sims else None
            r["nearest_cos"] = round(sims[0][0], 3) if sims else None
            rec[l] = r
            aucs.append(r["auc"])
            all_auc.append(r["auc"])
            if r["auc"] < WEAK_AUC:
                weak.append((sf, l, r["auc"], r["n_pos"], r.get("nearest_label")))
        out["subfields"][sf] = dict(
            n_docs=len(idx), n_labels=len(labs), n_tested=len(aucs),
            macro_auc=round(float(np.mean(aucs)), 3) if aucs else None,
            labels=rec)
        ms = ("macro-AUC %.3f over %d/%d labels" % (np.mean(aucs), len(aucs), len(labs))
              if aucs else "no label testable")
        print("%-34s docs=%3d | %s" % (sf, len(idx), ms), flush=True)

    out["aggregate"] = dict(
        n_labels_tested=len(all_auc),
        macro_auc_mean=round(float(np.mean(all_auc)), 3),
        frac_auc_ge_070=round(float(np.mean([a >= 0.70 for a in all_auc])), 3),
        frac_auc_ge_065=round(float(np.mean([a >= WEAK_AUC for a in all_auc])), 3),
        n_weak=len(weak), n_untested=len(untested))
    out["weak_labels"] = [dict(subfield=s, label=l, auc=a, n_pos=n, merge_candidate=m)
                          for s, l, a, n, m in sorted(weak, key=lambda x: x[2])]
    out["untested_labels"] = [dict(subfield=s, label=l, n_pos=n)
                              for s, l, n in sorted(untested, key=lambda x: -x[2])]

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("\nAGG:", out["aggregate"])
    print("\nWEAK LABELS (AUC < %.2f):" % WEAK_AUC)
    for w in out["weak_labels"]:
        print("  %-34s %-30s AUC=%.3f n=%3d  merge? %s"
              % (w["subfield"], w["label"], w["auc"], w["n_pos"], w["merge_candidate"]))
    print("\nUNTESTED (n_pos < %d):" % MIN_POS)
    for u in out["untested_labels"]:
        print("  %-34s %-30s n=%d" % (u["subfield"], u["label"], u["n_pos"]))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
