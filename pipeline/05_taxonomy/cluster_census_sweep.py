#!/usr/bin/env python3
"""Granularity sweep for the census BERTopic tier-1, to find where Verification /
Watermarking / Agentic rejoin as dedicated clusters.

UMAP is computed ONCE (same params as bertopic_compare); only HDBSCAN's
min_cluster_size is swept. For each mcs we report n_clusters, noise, NMI, and the
SET of top-down categories that own a dedicated (dominant) cluster -- flagging the
three the 13-cat paper excluded plus Verification. Picks the mcs that recovers the
most categories at tier-1 (tie-break: fewer clusters; require noise < 0.40), writes
it to _pool_audit/census_sweep_pick.txt for the detailing run.

  module load python/gpu/3.11.5 ; python \
      pipeline/05_taxonomy/cluster_census_sweep.py 40 50 60 70 80 90
"""
import io, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
OUTDIR = os.path.join(PS, "_pool_audit")
SRC = os.environ.get("CLUSTER_SRC") or os.path.join(PS, "_census", "cluster_corpus_census.json")
EMB = os.environ.get("CLUSTER_EMB") or os.path.join(OUTDIR, "_specter_emb_census.npy")
SEED = 42
WATCH = ["Verification & Formal Methods", "Watermarking & IP Protection",
         "Agentic & Frontier-Model Safety"]


def main(sizes):
    import umap, hdbscan
    recs = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    Z = np.load(EMB).astype(np.float32)
    assert len(recs) == Z.shape[0], (len(recs), Z.shape)
    cats = sorted({p["category"] for p in recs})
    cidx = {c: i for i, c in enumerate(cats)}
    y = np.array([cidx[p["category"]] for p in recs])

    print(f"docs={len(recs)} dim={Z.shape[1]} cats={len(cats)}  UMAP(5d,cosine,nn15)...", flush=True)
    U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=SEED).fit_transform(Z)

    def nmi(c, yy):
        ks = sorted(set(c)); K = len(ks); ki = {k: i for i, k in enumerate(ks)}
        M = np.zeros((K, len(cats)))
        for ci, yi in zip(c, yy):
            M[ki[ci], yi] += 1
        n = M.sum(); Pij = M / n; Pi = M.sum(1) / n; Pj = M.sum(0) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(Pij * np.log(Pij / (np.outer(Pi, Pj) + 1e-12) + 1e-12))
            Hi = -np.nansum(Pi * np.log(Pi + 1e-12)); Hj = -np.nansum(Pj * np.log(Pj + 1e-12))
        return mi / (0.5 * (Hi + Hj)) if (Hi + Hj) > 0 else 0.0

    rows = []
    print("\n%5s %9s %7s %6s  %-3s  %s" % ("mcs", "clusters", "noise", "NMI", "rec", "watch: Verif/Water/Agent"))
    print("-" * 78)
    for mcs in sizes:
        lab = hdbscan.HDBSCAN(min_cluster_size=int(mcs), metric="euclidean",
                              cluster_selection_method="eom").fit_predict(U)
        a = lab >= 0
        nclu = int(lab.max() + 1) if a.any() else 0
        noise = float((~a).mean())
        recovered = set()
        for k in range(nclu):
            m = lab == k
            if m.sum() == 0:
                continue
            dom = np.bincount(y[m], minlength=len(cats)).argmax()
            recovered.add(cats[dom])
        watch = " ".join(("Y" if w in recovered else ".") for w in WATCH)
        nm = nmi(lab[a], y[a]) if a.sum() > 10 and nclu >= 2 else 0.0
        rows.append(dict(mcs=int(mcs), n_clusters=nclu, noise=noise, nmi=nm,
                         recovered=sorted(recovered), n_recovered=len(recovered)))
        print("%5d %9d %6.1f%% %6.3f  %2d   %s" % (mcs, nclu, noise * 100, nm, len(recovered), watch))

    # pick: max categories recovered, noise<0.40, tie-break fewer clusters
    elig = [r for r in rows if r["noise"] < 0.40]
    pick = max(elig or rows, key=lambda r: (r["n_recovered"], -r["n_clusters"]))
    io.open(os.path.join(OUTDIR, "census_sweep_pick.txt"), "w").write(str(pick["mcs"]))
    json.dump(rows, io.open(os.path.join(OUTDIR, "census_sweep.json"), "w"), indent=1)
    print(f"\nPICK mcs={pick['mcs']}  ({pick['n_recovered']} cats, {pick['n_clusters']} clusters, "
          f"noise {pick['noise']*100:.1f}%)  -> _pool_audit/census_sweep_pick.txt")
    miss = [c for c in cats if c not in pick["recovered"]]
    print("tier-1 not recovered at pick:", miss)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [40, 50, 60, 70, 80, 90])
