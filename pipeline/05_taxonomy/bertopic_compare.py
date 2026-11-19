#!/usr/bin/env python3
"""
Bottom-up taxonomy derivation (the BERTopic-style pipeline) vs. the top-down 16-category
LLM taxonomy.

Pipeline (== BERTopic internals): cached SPECTER embeddings -> UMAP -> soft
HDBSCAN -> class-based c-TF-IDF cluster labels. We then measure how well the
*auto-derived* clusters recover the 16 hand-set categories, using the SAME
metrics already in the paper (NMI/ARI/purity/Hungarian-1:1), so the result sits
directly next to the existing TF-IDF (NMI 0.55) and SPECTER-kmeans (NMI 0.38)
numbers in unsupervised_metrics.json / embed_metrics.json.

Reuses the SAME embedding cache (_specter_emb.npy) the paper's SPECTER-kmeans
used, so this is an apples-to-apples test of "does bottom-up reproduce the
top-down taxonomy?"

Run:  python pipeline/05_taxonomy/bertopic_compare.py [min_cluster_size ...]
Default: sweep {25,40,60,90,120,160} then detail the run nearest ~22 clusters.
"""
import io, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
OUTDIR = os.path.join(PS, "_pool_audit")
# Additive env overrides (defaults reproduce the canonical 6,827 run):
#   CLUSTER_SRC  alt corpus json (is_safety rows + category)
#   CLUSTER_EMB  alt SPECTER embedding cache, positionally aligned to CLUSTER_SRC
#   CLUSTER_TAG  suffix on every output file (e.g. "_census") to avoid clobbering
SRC = os.environ.get("CLUSTER_SRC") or os.path.join(PS, "papers_llm_cleaned.json")
if not os.path.exists(SRC):
    SRC = os.path.join(PS, "papers_llm.json")
EMB = os.environ.get("CLUSTER_EMB") or os.path.join(OUTDIR, "_specter_emb.npy")
TAG = os.environ.get("CLUSTER_TAG", "")
SEED = 42

DOMAIN_STOP = """model models method methods approach approaches propose proposed
paper using used use based novel new results result show shows demonstrate
learning deep neural network networks training train data dataset datasets
task tasks performance state art achieve our we present study work framework
algorithm via toward towards large language""".split()


def cluster_metrics(c, y, n_cats):
    """contingency + ARI/NMI/purity/Hungarian-1:1 over assigned (non-noise) docs."""
    from scipy.optimize import linear_sum_assignment
    ks = sorted(set(c))
    kidx = {k: i for i, k in enumerate(ks)}
    K = len(ks)
    M = np.zeros((K, n_cats), int)
    for ci, yi in zip(c, y):
        M[kidx[ci], yi] += 1
    n = M.sum()
    comb = lambda x: x * (x - 1) / 2.0
    a, b = M.sum(1), M.sum(0)
    idx, ea, eb = comb(M).sum(), comb(a).sum(), comb(b).sum()
    exp, mx = ea * eb / comb(n), 0.5 * (ea + eb)
    ari = (idx - exp) / (mx - exp) if mx != exp else 0.0
    Pij, Pi, Pj = M / n, a / n, b / n
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(Pij * np.log(Pij / (np.outer(Pi, Pj) + 1e-12) + 1e-12))
        Hi = -np.nansum(Pi * np.log(Pi + 1e-12))
        Hj = -np.nansum(Pj * np.log(Pj + 1e-12))
    nmi = mi / (0.5 * (Hi + Hj)) if (Hi + Hj) > 0 else 0.0
    ri, cj = linear_sum_assignment(-M)
    return M, ks, dict(purity=float(M.max(1).sum() / n), ari=float(ari),
                       nmi=float(nmi), matched_acc=float(M[ri, cj].sum() / n))


def main(sizes):
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer

    recs = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    Z = np.load(EMB).astype(np.float32)
    assert len(recs) == Z.shape[0], (len(recs), Z.shape)
    cats = sorted({p["category"] for p in recs})
    cidx = {c: i for i, c in enumerate(cats)}
    y = np.array([cidx[p["category"]] for p in recs])
    sec = [p.get("category_secondary") for p in recs]
    conf = np.array([p.get("llm_confidence") or 1.0 for p in recs])
    texts = [((p.get("title") or "") + ". " + (p.get("abstract") or "")) for p in recs]
    N = len(recs)
    print(f"docs={N}  embdim={Z.shape[1]}  categories={len(cats)}")

    # ---- 1-2. UMAP (BERTopic defaults: cosine, 5d, n_neighbors=15) ----
    print("UMAP reducing...", flush=True)
    reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=SEED)
    U = reducer.fit_transform(Z)

    # ---- 3. HDBSCAN sweep over min_cluster_size ----
    sweep = []
    runs = {}
    for mcs in sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(mcs), metric="euclidean",
                                    cluster_selection_method="eom",
                                    prediction_data=True)
        lab = clusterer.fit_predict(U)
        assigned = lab >= 0
        nclu = int(lab.max() + 1) if assigned.any() else 0
        noise = float((~assigned).mean())
        if nclu >= 2 and assigned.sum() > 10:
            _, _, mm = cluster_metrics(lab[assigned], y[assigned], len(cats))
        else:
            mm = dict(purity=0, ari=0, nmi=0, matched_acc=0)
        row = dict(min_cluster_size=int(mcs), n_clusters=nclu, noise_frac=noise,
                   **{k: round(v, 3) for k, v in mm.items()})
        sweep.append(row)
        runs[int(mcs)] = (lab, clusterer)
        print(f"  mcs={mcs:4d} -> clusters={nclu:3d}  noise={noise*100:4.1f}%  "
              f"NMI={mm['nmi']:.3f}  purity={mm['purity']:.3f}  "
              f"ARI={mm['ari']:.3f}  Hung={mm['matched_acc']:.3f}", flush=True)

    # ---- choose the run whose cluster count is nearest the 16-category target band (~22) ----
    # CLUSTER_FORCE_MCS overrides the auto-pick with an explicit min_cluster_size
    # (used to detail a granularity chosen for best taxonomy recovery on the sweep).
    target = 22
    force = os.environ.get("CLUSTER_FORCE_MCS")
    if force and int(force) in runs:
        pick = int(force)
    else:
        pick = min(runs, key=lambda m: abs(runs[m][0].max() + 1 - target))
    lab, clusterer = runs[pick]
    assigned = lab >= 0
    nclu = int(lab.max() + 1)
    print(f"\n=== detailing min_cluster_size={pick}  ({nclu} clusters, "
          f"{(~assigned).mean()*100:.1f}% noise) ===")

    M, ks, mm = cluster_metrics(lab[assigned], y[assigned], len(cats))

    # ---- 4. soft membership -> secondary topic (soft clustering) ----
    soft = hdbscan.all_points_membership_vectors(clusterer)  # N x nclu
    if soft.ndim == 1:
        soft = soft.reshape(N, -1)
    np.save(os.path.join(OUTDIR, f"bertopic_soft{TAG}.npy"), soft)  # sparse multi-label signal
    order = np.argsort(-soft, axis=1)
    prim_soft = order[:, 0]
    sec_soft = order[:, 1] if soft.shape[1] > 1 else order[:, 0]
    sec_strength = soft[np.arange(N), sec_soft]
    multi = sec_strength > 0.10  # doc meaningfully shares a 2nd topic
    print(f"soft multi-label rate (2nd topic >0.10): {multi.mean()*100:.1f}%  "
          f"(top-down secondary rate: {np.mean([s is not None for s in sec])*100:.1f}%)")

    # ---- class-based c-TF-IDF top terms per cluster (BERTopic-style) ----
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                         min_df=5, max_df=0.5)
    Xc = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    drop = np.array([all(w not in DOMAIN_STOP for w in t.split()) for t in vocab])
    per_cluster = {}
    for k in range(nclu):
        mem = np.where(lab == k)[0]
        if len(mem) == 0:
            continue
        tf = np.asarray(Xc[mem].sum(0)).ravel().astype(float)
        tf_all = np.asarray(Xc.sum(0)).ravel().astype(float)
        ctfidf = (tf / max(tf.sum(), 1)) * np.log(1 + tf_all.sum() / (tf_all + 1))
        ctfidf[~drop] *= 0.2
        top = vocab[np.argsort(-ctfidf)[:10]].tolist()
        cnt = np.bincount(y[mem], minlength=len(cats))
        dom = int(cnt.argmax())
        share = cnt[dom] / len(mem)
        rep_idx = mem[np.argsort(-clusterer.probabilities_[mem])[:8]]
        reps = [recs[i]["title"] for i in rep_idx]
        rep_cards = [dict(title=recs[i].get("title") or "",
                          abstract=(recs[i].get("abstract") or "")[:300],
                          topdown=recs[i]["category"]) for i in rep_idx]
        per_cluster[k] = dict(n=int(len(mem)), dom_category=cats[dom],
                              dom_share=round(float(share), 3),
                              terms=top, reps=reps, rep_cards=rep_cards,
                              cat_mix={cats[i]: int(cnt[i]) for i in np.argsort(-cnt)[:4] if cnt[i]})

    # ---- taxonomy reconciliation: fragmentation / merging vs the 16 categories ----
    dom_of = {k: per_cluster[k]["dom_category"] for k in per_cluster}
    clusters_per_cat = {c: [k for k in dom_of if dom_of[k] == c] for c in cats}
    covered = {c: len(v) for c, v in clusters_per_cat.items()}
    fragmented = {c: n for c, n in covered.items() if n >= 2}
    absent = [c for c, n in covered.items() if n == 0]
    merged = {k: per_cluster[k]["cat_mix"] for k in per_cluster
              if per_cluster[k]["dom_share"] < 0.5}

    print(f"\nglobal vs 16-cat:  NMI={mm['nmi']:.3f}  ARI={mm['ari']:.3f}  "
          f"purity={mm['purity']:.3f}  Hungarian-1:1={mm['matched_acc']:.3f}")
    print(f"categories with their own cluster: {sum(1 for c in covered if covered[c]>=1)}/{len(cats)}")
    print(f"categories FRAGMENTED across >=2 clusters: {fragmented}")
    print(f"categories with NO dominant cluster (merged/absorbed): {absent}")

    print("\nper-cluster:")
    for k in sorted(per_cluster):
        pc = per_cluster[k]
        print(f"  c{k:2d} n={pc['n']:4d} ->{pc['dom_category'][:26]:26s} "
              f"({pc['dom_share']*100:3.0f}%)  {', '.join(pc['terms'][:6])}")

    # ---- contingency heatmap (clusters x 16 categories) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # order rows by dominant category, then size, for readability
    row_dom = M.argmax(1)
    order = sorted(range(len(ks)), key=lambda r: (row_dom[r], -M[r].sum()))
    Mo = M[order]
    klabels = [f"c{ks[r]} (n={M[r].sum()})" for r in order]
    rownorm = Mo / Mo.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(rownorm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace(" & ", "\n& ") for c in cats], rotation=90, fontsize=7)
    ax.set_yticks(range(len(klabels)))
    ax.set_yticklabels(klabels, fontsize=7)
    for i in range(Mo.shape[0]):
        for j in range(Mo.shape[1]):
            if Mo[i, j] >= max(8, 0.04 * Mo[i].sum()):
                ax.text(j, i, str(Mo[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if rownorm[i, j] < 0.55 else "black")
    ax.set_xlabel("Top-down LLM category", fontsize=9)
    ax.set_ylabel("Bottom-up HDBSCAN cluster (row-normalized share)", fontsize=9)
    ax.set_title(f"Bottom-up clusters vs. top-down taxonomy  "
                 f"(NMI={mm['nmi']:.2f}, purity={mm['purity']:.2f})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, label="share of cluster")
    fig.tight_layout()
    fig_pdf = os.path.join(OUTDIR, f"bertopic_contingency{TAG}.pdf")
    fig.savefig(fig_pdf, bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, f"bertopic_contingency{TAG}.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_pdf} (+ .png)")

    # ---- cluster cards for LLM naming (step 3) ----
    cards = {str(k): dict(n=per_cluster[k]["n"], top_terms=per_cluster[k]["terms"],
                          dom_topdown=per_cluster[k]["dom_category"],
                          dom_share=per_cluster[k]["dom_share"],
                          cat_mix=per_cluster[k]["cat_mix"],
                          representative_papers=per_cluster[k]["rep_cards"])
             for k in sorted(per_cluster)}
    cp = os.path.join(OUTDIR, f"cluster_cards{TAG}.json")
    json.dump(cards, io.open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {cp}")
    np.save(os.path.join(OUTDIR, f"bertopic_labels{TAG}.npy"), lab)

    # ---- low-confidence cohort (route these to human review) ----
    low = conf <= 0.6
    print(f"\nlow-confidence top-down labels (<=0.6): {int(low.sum())} "
          f"({low.mean()*100:.1f}%) -> human-verify cohort")

    out = dict(docs=N, embedding="SPECTER (cached, same as embed_metrics.json)",
               pipeline="SPECTER -> UMAP(5d,cosine) -> soft HDBSCAN -> c-TF-IDF",
               sweep=sweep, chosen_min_cluster_size=int(pick), n_clusters=nclu,
               noise_frac=float((~assigned).mean()),
               global_vs_taxonomy=mm,
               baseline_reference=dict(tfidf_kmeans_nmi=0.55, specter_kmeans_nmi=0.38),
               categories_covered=covered, fragmented=fragmented,
               categories_no_cluster=absent,
               soft_multilabel_rate=float(multi.mean()),
               topdown_secondary_rate=float(np.mean([s is not None for s in sec])),
               low_conf_count=int(low.sum()),
               per_cluster={str(k): v for k, v in per_cluster.items()})
    op = os.path.join(OUTDIR, f"bertopic_compare{TAG}.json")
    json.dump(out, io.open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [25, 40, 60, 90, 120, 160])
