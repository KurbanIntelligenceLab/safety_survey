#!/usr/bin/env python3
"""
Verification re-run of the TIER-1 bottom-up taxonomy derivation on the 13-scope
corpus (the 16 LLM categories minus the three out-of-scope ones: Watermarking,
Agentic & Frontier-Model Safety, Miscellaneous Safety).

Identical pipeline to bertopic_compare.py (SPECTER -> UMAP 5d/cosine ->
soft HDBSCAN/eom -> metrics), just restricted to the 13 in-scope categories and
scored against those 13. Purpose: confirm whether Appendix G's numbers
(6,628 embedded, 18 clusters @ mcs=40, NMI 0.497, purity 0.598, mega-cluster
n=1,758 @ 26.5%) are reproducible, before touching the paper.

Run: python pipeline/05_taxonomy/bertopic_tier1_13cat.py
"""
import io, json, os
import numpy as np
from bertopic_compare import cluster_metrics, SRC, EMB, SEED

OUT_OF_SCOPE = {
    "Watermarking & IP Protection",
    "Agentic & Frontier-Model Safety",
    "Miscellaneous Safety",
}

def main():
    import umap, hdbscan
    recs_all = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    Z_all = np.load(EMB).astype(np.float32)
    assert len(recs_all) == Z_all.shape[0], (len(recs_all), Z_all.shape)

    keep = np.array([p["category"] not in OUT_OF_SCOPE for p in recs_all])
    recs = [p for p, k in zip(recs_all, keep) if k]
    Z = Z_all[keep]
    print(f"full is_safety={len(recs_all)}  in-scope(13-cat)={len(recs)}  dropped={len(recs_all)-len(recs)}")

    cats = sorted({p["category"] for p in recs})
    cidx = {c: i for i, c in enumerate(cats)}
    y = np.array([cidx[p["category"]] for p in recs])
    print(f"n categories={len(cats)}  embdim={Z.shape[1]}")

    print("UMAP reducing...", flush=True)
    U = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=SEED).fit_transform(Z)

    print("\nHDBSCAN sweep (vs 13 categories):")
    runs = {}
    for mcs in (25, 40, 60, 90, 120, 160):
        cl = hdbscan.HDBSCAN(min_cluster_size=int(mcs), metric="euclidean",
                             cluster_selection_method="eom", prediction_data=True)
        lab = cl.fit_predict(U)
        asg = lab >= 0
        nclu = int(lab.max() + 1) if asg.any() else 0
        noise = float((~asg).mean())
        if nclu >= 2 and asg.sum() > 10:
            _, _, mm = cluster_metrics(lab[asg], y[asg], len(cats))
        else:
            mm = dict(purity=0, ari=0, nmi=0, matched_acc=0)
        runs[int(mcs)] = (lab, cl)
        print(f"  mcs={mcs:4d} -> clusters={nclu:3d}  noise={noise*100:4.1f}%  "
              f"NMI={mm['nmi']:.3f}  purity={mm['purity']:.3f}  ARI={mm['ari']:.3f}")

    # detail mcs=40 (the run Appendix G describes)
    lab, cl = runs[40]
    asg = lab >= 0
    nclu = int(lab.max() + 1)
    M, ks, mm = cluster_metrics(lab[asg], y[asg], len(cats))
    print(f"\n=== mcs=40 detail: clusters={nclu}  noise={(~asg).mean():.3f}  "
          f"NMI={mm['nmi']:.3f}  purity={mm['purity']:.3f}  ARI={mm['ari']:.3f} ===")

    # per-cluster dominant category + size (to identify the mega-cluster)
    rows = []
    for k in range(nclu):
        mem = np.where(lab == k)[0]
        if not len(mem):
            continue
        cnt = np.bincount(y[mem], minlength=len(cats))
        dom = int(cnt.argmax())
        share = cnt[dom] / len(mem)
        mix = {cats[i]: int(cnt[i]) for i in np.argsort(-cnt)[:4] if cnt[i]}
        rows.append((k, len(mem), cats[dom], share, mix))
    rows.sort(key=lambda r: -r[1])
    print("\nclusters by size (k, n, dominant, dom_share):")
    for k, n, dom, share, mix in rows:
        flag = "  <-- MEGA" if share < 0.5 and n > 500 else ""
        print(f"  c{k:2d}  n={n:4d}  {dom[:30]:30s}  {share*100:4.0f}%{flag}")
        if flag:
            tot = len(recs)
            print(f"        mega-cluster = {n} papers = {n/tot*100:.1f}% of corpus; mix={mix}")

    # categories with their own dominant cluster
    dom_of = {r[0]: r[2] for r in rows}
    covered = {c: sum(1 for k in dom_of if dom_of[k] == c) for c in cats}
    clean = [c for c in cats if covered[c] == 1]
    frag = {c: covered[c] for c in cats if covered[c] >= 2}
    absent = [c for c in cats if covered[c] == 0]
    print(f"\nclean (1 cluster): {len(clean)} -> {clean}")
    print(f"fragmented (>1):   {frag}")
    print(f"absent (in mega):  {absent}")

if __name__ == "__main__":
    main()
