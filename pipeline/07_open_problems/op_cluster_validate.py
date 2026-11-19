#!/usr/bin/env python3
"""Bottom-up validation of the OPEN-PROBLEM label taxonomy, mirroring the subfield
derivation (app:bottomup / cluster_stability_census.py).

For the subfields we asked: do the 14 categories emerge from clustering SPECTER
abstract embeddings? Here we ask the parallel question per subfield: do the curated
open-problem labels emerge from clustering SPECTER embeddings of the CLOSING
sections that state them?

Inputs (from embed_op_sections.py): _op_emb.npy, _op_meta.json.
Evaluation set: single-label on-topic papers (clean ground truth), per subfield.

Three experiments, per subfield and aggregated:
  (A) CLUSTERING RECOVERY   UMAP->HDBSCAN vs the labels: NMI / ARI / purity, noise%.
  (B) SEPARABILITY PROBE     5-fold CV logistic regression on frozen SPECTER embeddings
                             predicting the label: accuracy, macro-F1, vs majority.
  (C) SEED STABILITY         clustering NMI over 5 UMAP seeds + pairwise run NMI.

  python pipeline/07_open_problems/op_cluster_validate.py
"""
import io, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey", "_fulltext12987"))
OUTDIR = os.path.join(FT, "opcluster")
EMB = os.path.join(OUTDIR, "_op_emb.npy")
META = os.path.join(OUTDIR, "_op_meta.json")
OUT = os.path.join(OUTDIR, "op_cluster_validate.json")


def nmi_ari_purity(c, y):
    ks = sorted(set(c)); kidx = {k: i for i, k in enumerate(ks)}
    ys = sorted(set(y)); yidx = {k: i for i, k in enumerate(ys)}
    M = np.zeros((len(ks), len(ys)), int)
    for ci, yi in zip(c, y):
        M[kidx[ci], yi if isinstance(yi, int) else yidx[yi]] += 1
    n = M.sum(); comb = lambda x: x * (x - 1) / 2.0
    a, b = M.sum(1), M.sum(0)
    idx, ea, eb = comb(M).sum(), comb(a).sum(), comb(b).sum()
    exp, mx = ea * eb / comb(n) if comb(n) else 0, 0.5 * (ea + eb)
    ari = (idx - exp) / (mx - exp) if mx != exp else 0.0
    Pij, Pi, Pj = M / n, a / n, b / n
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(Pij * np.log(Pij / (np.outer(Pi, Pj) + 1e-12) + 1e-12))
        Hi = -np.nansum(Pi * np.log(Pi + 1e-12)); Hj = -np.nansum(Pj * np.log(Pj + 1e-12))
    nmi = mi / (0.5 * (Hi + Hj)) if (Hi + Hj) > 0 else 0.0
    purity = M.max(1).sum() / n
    return float(nmi), float(ari), float(purity)


def label_nmi(c1, c2):
    k1 = {k: i for i, k in enumerate(sorted(set(c1)))}
    k2 = {k: i for i, k in enumerate(sorted(set(c2)))}
    M = np.zeros((len(k1), len(k2)), int)
    for x, z in zip(c1, c2):
        M[k1[x], k2[z]] += 1
    n = M.sum(); Pij, Pi, Pj = M / n, M.sum(1) / n, M.sum(0) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(Pij * np.log(Pij / (np.outer(Pi, Pj) + 1e-12) + 1e-12))
        Hi = -np.nansum(Pi * np.log(Pi + 1e-12)); Hj = -np.nansum(Pj * np.log(Pj + 1e-12))
    return float(mi / (0.5 * (Hi + Hj))) if (Hi + Hj) > 0 else 0.0


def cluster(Z, nn, mcs, seed):
    import umap, hdbscan
    U = umap.UMAP(n_neighbors=nn, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=seed).fit_transform(Z)
    lab = hdbscan.HDBSCAN(min_cluster_size=int(mcs), metric="euclidean",
                          cluster_selection_method="eom").fit_predict(U)
    return lab


def probe(Z, y, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score
    yv = np.array(y)
    # keep classes with >=8 members so 4-fold CV is valid
    keep = np.array([c for c in set(yv) if (yv == c).sum() >= 8])
    m = np.isin(yv, keep)
    Zc, yc = Z[m], yv[m]
    if len(set(yc)) < 2 or len(yc) < 20:
        return None
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    accs, f1s = [], []
    for tr, te in skf.split(Zc, yc):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(Zc[tr], yc[tr])
        p = clf.predict(Zc[te])
        accs.append(accuracy_score(yc[te], p))
        f1s.append(f1_score(yc[te], p, average="macro"))
    _, cnts = np.unique(yc, return_counts=True)
    return dict(n=int(len(yc)), n_classes=int(len(set(yc))),
                acc=round(float(np.mean(accs)), 3), acc_std=round(float(np.std(accs)), 3),
                macro_f1=round(float(np.mean(f1s)), 3),
                majority=round(float(cnts.max() / cnts.sum()), 3))


def main():
    Z = np.load(EMB).astype(np.float32)
    meta = json.load(io.open(META, encoding="utf-8"))
    assert len(meta) == Z.shape[0], (len(meta), Z.shape)
    print("total labeled docs:", len(meta), "dim", Z.shape[1])

    # group row indices by subfield, single-label only
    bysf = {}
    for i, m in enumerate(meta):
        if m["n_labels"] == 1:
            bysf.setdefault(m["subfield"], []).append(i)

    out = {"n_docs_total": len(meta), "subfields": {}}
    agg_probe_acc, agg_probe_f1, agg_maj, agg_nmi = [], [], [], []
    for sf in sorted(bysf):
        idx = np.array(bysf[sf])
        Zs = Z[idx]
        ys = [meta[i]["labels"][0] for i in idx]
        nlab = len(set(ys))
        N = len(idx)
        rec = {"n_single": N, "n_labels": nlab}

        # adaptive clustering config for this subfield size
        mcs = max(5, int(round(N * 0.04)))
        nn = min(15, max(5, N // 6))

        # (B) separability probe
        pr = probe(Zs, ys)
        rec["probe"] = pr
        if pr:
            agg_probe_acc.append(pr["acc"]); agg_probe_f1.append(pr["macro_f1"]); agg_maj.append(pr["majority"])

        # (A) clustering recovery + (C) seed stability
        labs, rows = [], []
        for s in range(5):
            try:
                lab = cluster(Zs, nn, mcs, s)
            except Exception as e:
                lab = np.full(N, -1)
            asg = lab >= 0
            nclu = int(lab.max() + 1) if asg.any() else 0
            if nclu >= 2 and asg.sum() > 10:
                nmi, ari, pur = nmi_ari_purity(lab[asg].tolist(),
                                               [ys[j] for j in range(N) if asg[j]])
            else:
                nmi, ari, pur = 0.0, 0.0, 0.0
            labs.append(lab)
            rows.append(dict(seed=s, n_clusters=nclu, noise=round(float((~asg).mean()), 3),
                             nmi=round(nmi, 3), ari=round(ari, 3), purity=round(pur, 3)))
        nmi_vals = [r["nmi"] for r in rows]
        pair = [label_nmi(labs[i], labs[j]) for i in range(5) for j in range(i + 1, 5)]
        rec["cluster"] = dict(
            per_seed=rows, mcs=mcs, n_neighbors=nn,
            nmi_mean=round(float(np.mean(nmi_vals)), 3), nmi_std=round(float(np.std(nmi_vals)), 3),
            ari_mean=round(float(np.mean([r["ari"] for r in rows])), 3),
            purity_mean=round(float(np.mean([r["purity"] for r in rows])), 3),
            noise_mean=round(float(np.mean([r["noise"] for r in rows])), 3),
            nclusters_mean=round(float(np.mean([r["n_clusters"] for r in rows])), 1),
            pairwise_run_nmi=round(float(np.mean(pair)), 3) if pair else 0.0)
        agg_nmi.append(rec["cluster"]["nmi_mean"])
        out["subfields"][sf] = rec
        pstr = ("acc=%.3f f1=%.3f maj=%.3f" % (pr["acc"], pr["macro_f1"], pr["majority"])) if pr else "probe:skip(small)"
        print("%-34s N=%3d L=%2d | clusterNMI=%.3f pur=%.3f noise=%.0f%% | %s"
              % (sf, N, nlab, rec["cluster"]["nmi_mean"], rec["cluster"]["purity_mean"],
                 rec["cluster"]["noise_mean"] * 100, pstr), flush=True)

    out["aggregate"] = dict(
        probe_acc_mean=round(float(np.mean(agg_probe_acc)), 3),
        probe_macro_f1_mean=round(float(np.mean(agg_probe_f1)), 3),
        probe_majority_mean=round(float(np.mean(agg_maj)), 3),
        cluster_nmi_mean=round(float(np.mean(agg_nmi)), 3),
        n_subfields=len(out["subfields"]))
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("\nAGG:", out["aggregate"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
