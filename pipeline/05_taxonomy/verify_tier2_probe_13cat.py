#!/usr/bin/env python3
"""
Verify Appendix G's TIER-2 recursion + linear-probe numbers on the 13-scope
mega-cluster (NOT the saved 16-cat artifacts). Replicates bertopic_tier2.py
(DBCV-blind selection) and c0_separability_probe.py exactly, but on the 1,758-
member 13-cat mega reproduced from bertopic_tier1_13cat.py.

Expected from Appendix G: 16 sub-clusters, purity 0.19 -> 0.72, NMI 0.52;
probe n=1724 over 9 categories, acc 77.9%, macro-F1 0.766, majority 19.0%.
"""
import io, json, os
import numpy as np
from bertopic_compare import cluster_metrics, DOMAIN_STOP, SRC, EMB, SEED

OUT_OF_SCOPE = {
    "Watermarking & IP Protection",
    "Agentic & Frontier-Model Safety",
    "Miscellaneous Safety",
}

def main():
    import umap, hdbscan
    recs_all = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    Z_all = np.load(EMB).astype(np.float32)
    keep = np.array([p["category"] not in OUT_OF_SCOPE for p in recs_all])
    recs = [p for p, k in zip(recs_all, keep) if k]
    Z = Z_all[keep]
    cats = sorted({p["category"] for p in recs})
    cidx = {c: i for i, c in enumerate(cats)}
    y = np.array([cidx[p["category"]] for p in recs])

    # ---- tier-1 (13-cat) to recover the mega-cluster membership ----
    U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=SEED).fit_transform(Z)
    lab1 = hdbscan.HDBSCAN(min_cluster_size=40, metric="euclidean",
                           cluster_selection_method="eom",
                           prediction_data=True).fit_predict(U)
    # mega = largest, lowest-purity assigned cluster
    best, score = None, -1
    for k in sorted(set(lab1.tolist())):
        if k < 0: continue
        m = lab1 == k
        n = int(m.sum())
        dom = np.bincount(y[m], minlength=len(cats)).max() / n
        s = n * (1.0 - dom)
        if s > score: best, score = k, s
    sel = np.where(lab1 == best)[0]
    sub = [recs[i] for i in sel]
    Zs = Z[sel]
    N = len(sel)
    catsub = sorted({p["category"] for p in sub})
    sidx = {c: i for i, c in enumerate(catsub)}
    ys = np.array([sidx[p["category"]] for p in sub])
    pre_purity = np.bincount(ys, minlength=len(catsub)).max() / N
    print(f"mega-cluster c{best}: n={N}  tier-1 purity={pre_purity:.3f}  cats present={len(catsub)}")

    # ---- TIER-2 recursion: same params, refit on mega only, DBCV-blind select ----
    print("\nTIER-2 recursion (DBCV-blind selection):")
    Ur = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                   metric="cosine", random_state=SEED).fit_transform(Zs)
    runs = {}
    for mcs in (10, 15, 20, 25, 30, 40, 60):
        cl = hdbscan.HDBSCAN(min_cluster_size=int(mcs), metric="euclidean",
                             cluster_selection_method="eom",
                             prediction_data=True, gen_min_span_tree=True)
        lab = cl.fit_predict(Ur)
        a = lab >= 0
        nclu = int(lab.max() + 1) if a.any() else 0
        try: dbcv = float(cl.relative_validity_)
        except Exception: dbcv = float("nan")
        if nclu >= 2 and a.sum() > 10:
            _, _, mm = cluster_metrics(lab[a], ys[a], len(catsub))
        else:
            mm = dict(purity=0, ari=0, nmi=0, matched_acc=0)
        ok = (~a).mean() <= 0.40
        runs[int(mcs)] = (lab, dbcv, mm, ok)
        print(f"  mcs={mcs:3d} -> sub-clusters={nclu:3d}  noise={(~a).mean()*100:4.1f}%  "
              f"DBCV={dbcv:+.3f}  NMI={mm['nmi']:.3f}  purity={mm['purity']:.3f}")
    pick = max(runs, key=lambda m: (runs[m][1] if (runs[m][3] and runs[m][1]==runs[m][1]) else -1e9,))
    lab, dbcv, mm, ok = runs[pick]
    nclu = int(lab.max() + 1)
    print(f"\n=== DBCV-selected mcs={pick}: {nclu} sub-clusters  "
          f"NMI={mm['nmi']:.3f}  purity {pre_purity:.2f} -> {mm['purity']:.3f}  DBCV={dbcv:+.3f} ===")

    # ---- LINEAR PROBE: c0_separability_probe.py logic on the mega ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import f1_score, accuracy_score
    ycat = np.array([p["category"] for p in sub])
    uniq, cnt = np.unique(ycat, return_counts=True)
    keepc = set(uniq[cnt >= 25])
    km = np.array([c in keepc for c in ycat])
    X, yp = Zs[km], ycat[km]
    _, c2 = np.unique(yp, return_counts=True)
    maj = c2.max() / c2.sum()
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    yhat = cross_val_predict(clf, X, yp, cv=cv, n_jobs=-1)
    print(f"\n=== linear probe: n={len(yp)}  classes={len(keepc)}  "
          f"majority={maj*100:.1f}%  acc={accuracy_score(yp,yhat)*100:.1f}%  "
          f"macro-F1={f1_score(yp,yhat,average='macro'):.3f} ===")

if __name__ == "__main__":
    main()
