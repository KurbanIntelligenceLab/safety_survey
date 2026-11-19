#!/usr/bin/env python3
"""Clustering stability & hyperparameter sensitivity for the CENSUS taxonomy-
derivation pipeline (reviewer R2: SPECTER->UMAP->HDBSCAN stability not quantified).

Census variant: runs on the 13,110-paper census corpus that the 13/14-category
revision is built on, reusing the EXACT cached embeddings and pipeline used to
derive the taxonomy (cluster_census_sweep.py / apply_cluster_labels.py):
  SPECTER (_specter_emb_census.npy, 13110x768) -> UMAP(n_neighbors,n_components=5,
  min_dist=0.0,cosine) -> HDBSCAN(min_cluster_size, euclidean, eom).

The fixed operating point matches the manuscript's tier-1 (19 clusters, NMI~0.437
against the stored categories): n_neighbors=15, min_cluster_size=90.

Two experiments:
  (A) SEED STABILITY  — fixed operating config (n_neighbors=15, mcs=90), 5 UMAP seeds.
      Reports mean +/- std of n_clusters, noise%, and agreement-with-taxonomy
      (NMI/ARI) plus pairwise-run label stability (NMI between seed pairs).
  (B) HYPERPARAM SENSITIVITY — grid over n_neighbors x min_cluster_size at one
      seed. Shows the headline (bottom-up recovers the taxonomy) is not a
      single-config artifact.

Run: python pipeline/05_taxonomy/cluster_stability_census.py
"""
import io, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
SRC = os.path.join(PS, "_census", "cluster_corpus_census.json")
OUTDIR = os.path.join(PS, "_pool_audit")
EMB = os.path.join(OUTDIR, "_specter_emb_census.npy")
OUT = os.path.join(OUTDIR, "cluster_stability_census.json")


def nmi_ari(c, y, n_cats):
    ks = sorted(set(c)); kidx = {k: i for i, k in enumerate(ks)}; K = len(ks)
    M = np.zeros((K, n_cats), int)
    for ci, yi in zip(c, y):
        M[kidx[ci], yi] += 1
    n = M.sum(); comb = lambda x: x * (x - 1) / 2.0
    a, b = M.sum(1), M.sum(0)
    idx, ea, eb = comb(M).sum(), comb(a).sum(), comb(b).sum()
    exp, mx = ea * eb / comb(n), 0.5 * (ea + eb)
    ari = (idx - exp) / (mx - exp) if mx != exp else 0.0
    Pij, Pi, Pj = M / n, a / n, b / n
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(Pij * np.log(Pij / (np.outer(Pi, Pj) + 1e-12) + 1e-12))
        Hi = -np.nansum(Pi * np.log(Pi + 1e-12)); Hj = -np.nansum(Pj * np.log(Pj + 1e-12))
    nmi = mi / (0.5 * (Hi + Hj)) if (Hi + Hj) > 0 else 0.0
    return float(nmi), float(ari)


def label_nmi(c1, c2):
    """NMI between two full cluster labelings (run-to-run stability)."""
    ks1 = {k: i for i, k in enumerate(sorted(set(c1)))}
    ks2 = {k: i for i, k in enumerate(sorted(set(c2)))}
    M = np.zeros((len(ks1), len(ks2)), int)
    for x, z in zip(c1, c2):
        M[ks1[x], ks2[z]] += 1
    n = M.sum(); Pij, Pi, Pj = M / n, M.sum(1) / n, M.sum(0) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(Pij * np.log(Pij / (np.outer(Pi, Pj) + 1e-12) + 1e-12))
        Hi = -np.nansum(Pi * np.log(Pi + 1e-12)); Hj = -np.nansum(Pj * np.log(Pj + 1e-12))
    return float(mi / (0.5 * (Hi + Hj))) if (Hi + Hj) > 0 else 0.0


def run(Z, y, n_cats, n_neighbors, mcs, seed):
    import umap, hdbscan
    U = umap.UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=seed).fit_transform(Z)
    lab = hdbscan.HDBSCAN(min_cluster_size=int(mcs), metric="euclidean",
                          cluster_selection_method="eom").fit_predict(U)
    asg = lab >= 0
    nclu = int(lab.max() + 1) if asg.any() else 0
    nmi, ari = nmi_ari(lab[asg], y[asg], n_cats) if (nclu >= 2 and asg.sum() > 10) else (0., 0.)
    return lab, dict(n_clusters=nclu, noise_frac=float((~asg).mean()),
                     nmi=round(nmi, 3), ari=round(ari, 3))


def main():
    recs = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    Z = np.load(EMB).astype(np.float32)
    assert len(recs) == Z.shape[0], (len(recs), Z.shape)
    cats = sorted({p["category"] for p in recs}); cidx = {c: i for i, c in enumerate(cats)}
    y = np.array([cidx[p["category"]] for p in recs])
    print(f"docs={len(recs)} dim={Z.shape[1]} cats={len(cats)}")

    out = {"n_docs": len(recs), "n_categories": len(cats)}

    # (A) seed stability at the chosen config
    print("\n=== (A) SEED STABILITY  n_neighbors=15, min_cluster_size=90 ===")
    labs, rows = [], []
    for s in range(5):
        lab, m = run(Z, y, len(cats), 15, 90, s)
        labs.append(lab); rows.append(m)
        print(f"  seed={s}: clusters={m['n_clusters']:3d} noise={m['noise_frac']*100:4.1f}% "
              f"NMI={m['nmi']:.3f} ARI={m['ari']:.3f}", flush=True)
    agg = {k: [r[k] for r in rows] for k in rows[0]}
    summ = {k: dict(mean=round(float(np.mean(v)), 3), std=round(float(np.std(v)), 3),
                    min=round(float(np.min(v)), 3), max=round(float(np.max(v)), 3))
            for k, v in agg.items()}
    pair = [round(label_nmi(labs[i], labs[j]), 3)
            for i in range(len(labs)) for j in range(i + 1, len(labs))]
    summ["pairwise_run_nmi"] = dict(mean=round(float(np.mean(pair)), 3),
                                    std=round(float(np.std(pair)), 3),
                                    min=round(float(np.min(pair)), 3))
    out["seed_stability"] = {"per_seed": rows, "summary": summ}
    print("  summary:", {k: summ[k] for k in ("n_clusters", "nmi", "ari", "pairwise_run_nmi")})

    # (B) hyperparameter sensitivity (one seed)
    print("\n=== (B) HYPERPARAMETER SENSITIVITY (seed=42) ===")
    grid = []
    for nn in (10, 15, 30):
        for mcs in (60, 90, 120):
            _, m = run(Z, y, len(cats), nn, mcs, 42)
            m["n_neighbors"] = nn; m["min_cluster_size"] = mcs
            grid.append(m)
            print(f"  nn={nn:2d} mcs={mcs:3d}: clusters={m['n_clusters']:3d} "
                  f"noise={m['noise_frac']*100:4.1f}% NMI={m['nmi']:.3f} ARI={m['ari']:.3f}", flush=True)
    nmis = [g["nmi"] for g in grid]
    out["hyperparam_grid"] = {"grid": grid,
                              "nmi_mean": round(float(np.mean(nmis)), 3),
                              "nmi_std": round(float(np.std(nmis)), 3),
                              "nmi_min": round(float(np.min(nmis)), 3),
                              "nmi_max": round(float(np.max(nmis)), 3)}
    print(f"  NMI across 9 configs: mean={np.mean(nmis):.3f} "
          f"sd={np.std(nmis):.3f} range=[{min(nmis):.3f},{max(nmis):.3f}]")

    json.dump(out, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
