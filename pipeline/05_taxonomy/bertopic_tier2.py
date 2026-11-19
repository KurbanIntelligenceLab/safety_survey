#!/usr/bin/env python3
"""
Tier-2 (recursive) clustering of the LLM-safety mega-cluster.

Advisor's request: tier-1 bottom-up clustering (bertopic_compare.py) collapses
alignment / toxicity / hallucination / agentic into ONE high-density super-
cluster (c0, n=1827, ~27% of the corpus, top-down purity only 0.18) because
those frontiers share heavily overlapping LLM vocabulary, so global HDBSCAN
sees a single density mountain. The fix is a second tier: re-run the EXACT same
pipeline (UMAP 5d/cosine -> soft HDBSCAN -> c-TF-IDF) on ONLY that subset, where
the algorithm is no longer dominated by the inter-cluster distances and should
fracture the super-cluster into its internal frontiers.

This script reuses the SAME embedding cache (_specter_emb.npy), the SAME tier-1
labels (bertopic_labels.npy), and the SAME UMAP/HDBSCAN params and metric code
(imported from bertopic_compare) so it is provably the identical pipeline, just
restricted to the super-cluster's members.

Run:
  python pipeline/05_taxonomy/bertopic_tier2.py [min_cluster_size ...]
Default sweep: {10,15,20,25,30,40,60}; details the run nearest the # of
top-down frontiers present in the super-cluster.
"""
import io, json, os, sys
import numpy as np

import bertopic_compare as t1  # same dir; reuse SRC/EMB/OUTDIR/SEED/metrics/stopwords
from bertopic_compare import cluster_metrics, DOMAIN_STOP, SRC, EMB, OUTDIR, SEED

# which label file + cluster id to recurse INTO. Defaults select c0 from tier-1.
# For a tier-3 recursion, point LABELS at a tier-2 label file and SUPER at the
# residual giant sub-cluster (e.g. bertopic_tier2_blind_labels.npy, id=1).
LABELS = os.path.join(OUTDIR, os.environ.get("BERTOPIC_LABELS_FILE", "bertopic_labels.npy"))
SUPER = int(os.environ.get("BERTOPIC_SUPERCLUSTER", "0"))
# free-text tag appended to output filenames so recursions don't clobber tier-2.
TAG = os.environ.get("BERTOPIC_TAG", "")


def pick_supercluster(lab, y, n_cats):
    """If SUPER<0, auto-detect: the assigned cluster that is largest AND lowest
    top-down purity (the density mountain). Otherwise trust SUPER."""
    if SUPER >= 0:
        return SUPER
    best, score = None, -1
    for k in sorted(set(lab.tolist())):
        if k < 0:
            continue
        m = lab == k
        n = int(m.sum())
        dom = np.bincount(y[m], minlength=n_cats).max() / n
        s = n * (1.0 - dom)  # big and impure
        if s > score:
            best, score = k, s
    return best


def main(sizes):
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer

    recs = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    Z = np.load(EMB).astype(np.float32)
    lab1 = np.load(LABELS)
    assert len(recs) == Z.shape[0] == lab1.shape[0], (len(recs), Z.shape, lab1.shape)
    cats_all = sorted({p["category"] for p in recs})
    yall = np.array([cats_all.index(p["category"]) for p in recs])

    sup = pick_supercluster(lab1, yall, len(cats_all))
    sel = np.where(lab1 == sup)[0]
    N = len(sel)
    sub = [recs[i] for i in sel]
    Zs = Z[sel]

    # top-down categories present in the super-cluster (metrics computed over these)
    cats = sorted({p["category"] for p in sub})
    cidx = {c: i for i, c in enumerate(cats)}
    y = np.array([cidx[p["category"]] for p in sub])
    conf = np.array([p.get("llm_confidence") or 1.0 for p in sub])
    sec = [p.get("category_secondary") for p in sub]
    texts = [((p.get("title") or "") + ". " + (p.get("abstract") or "")) for p in sub]
    print(f"super-cluster c{sup}: n={N}  ({N/len(recs)*100:.1f}% of corpus)  "
          f"top-down categories present={len(cats)}")
    pre = np.bincount(y, minlength=len(cats))
    print("  top-down mix:", ", ".join(
        f"{cats[i]} {pre[i]} ({pre[i]/N*100:.1f}%)" for i in np.argsort(-pre)[:8]))

    # ---- 1-2. UMAP: IDENTICAL params to tier-1, refit on the subset only ----
    print("\nUMAP reducing (subset only, same params as tier-1)...", flush=True)
    reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=SEED)
    U = reducer.fit_transform(Zs)

    # ---- 3. soft HDBSCAN sweep over min_cluster_size (subset is smaller) ----
    # gen_min_span_tree=True so HDBSCAN exposes relative_validity_ (DBCV), a
    # LABEL-FREE internal cluster-quality score used for blind run selection.
    sweep, runs = [], {}
    for mcs in sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(mcs), metric="euclidean",
                                    cluster_selection_method="eom",
                                    prediction_data=True, gen_min_span_tree=True)
        lab = clusterer.fit_predict(U)
        assigned = lab >= 0
        nclu = int(lab.max() + 1) if assigned.any() else 0
        noise = float((~assigned).mean())
        try:
            dbcv = float(clusterer.relative_validity_)
        except Exception:
            dbcv = float("nan")
        if nclu >= 2 and assigned.sum() > 10:
            _, _, mm = cluster_metrics(lab[assigned], y[assigned], len(cats))
        else:
            mm = dict(purity=0, ari=0, nmi=0, matched_acc=0)
        sweep.append(dict(min_cluster_size=int(mcs), n_clusters=nclu, noise_frac=noise,
                          dbcv=round(dbcv, 4), **{k: round(v, 3) for k, v in mm.items()}))
        runs[int(mcs)] = (lab, clusterer, dbcv)
        print(f"  mcs={mcs:4d} -> sub-clusters={nclu:3d}  noise={noise*100:4.1f}%  "
              f"DBCV={dbcv:+.3f}  NMI={mm['nmi']:.3f}  purity={mm['purity']:.3f}  "
              f"ARI={mm['ari']:.3f}  Hung={mm['matched_acc']:.3f}", flush=True)

    # ---- run selection ----
    # BERTOPIC_SELECT=dbcv (default): LABEL-BLIND. Pick the mcs with the best
    #   internal DBCV (relative_validity_) -- never looks at the taxonomy. Taxonomy
    #   agreement is then reported purely as an after-the-fact check, so "the
    #   frontiers emerge bottom-up" is a clean, non-circular claim.
    # BERTOPIC_SELECT=frontiers: LABEL-INFORMED (the original validation selector).
    #   Picks the mcs that maximizes the # of distinct dominant top-down frontiers.
    SELECT = os.environ.get("BERTOPIC_SELECT", "dbcv").lower()

    def run_quality_frontiers(m):
        lab_, clu_, _dbcv = runs[m]
        a = lab_ >= 0
        nclu_ = int(lab_.max() + 1)
        if nclu_ < 2 or a.sum() <= 10:
            return (-1, -1.0)
        doms = {int(np.bincount(y[lab_ == k], minlength=len(cats)).argmax())
                for k in range(nclu_) if (lab_ == k).any()}
        _, _, q = cluster_metrics(lab_[a], y[a], len(cats))
        # reject runs that over-fragment (>2.5 sub-clusters per recovered frontier)
        # or leave too much noise -- we want ~one clean cluster per frontier.
        ok = (~a).mean() <= 0.40 and nclu_ <= 2.5 * len(doms)
        return (len(doms) if ok else len(doms) - 100, q["purity"])

    def run_quality_dbcv(m):
        lab_, clu_, dbcv = runs[m]
        a = lab_ >= 0
        nclu_ = int(lab_.max() + 1)
        if nclu_ < 2 or a.sum() <= 10 or (~a).mean() > 0.40:
            return (-1e9,)
        # purely internal: DBCV (higher = denser, better-separated clusters). NaN
        # (degenerate single-cluster MST) sorts to the bottom.
        return (dbcv if dbcv == dbcv else -1e9,)

    SUF = ("_blind" if SELECT == "dbcv" else "") + TAG  # keep recursions separate
    selector = run_quality_dbcv if SELECT == "dbcv" else run_quality_frontiers
    pick = max(runs, key=selector)
    print(f"\n[run selection mode = {SELECT.upper()}"
          f"{' -- LABEL-BLIND (DBCV internal score, taxonomy not used to choose)' if SELECT=='dbcv' else ' -- label-informed'}]")
    lab, clusterer, pick_dbcv = runs[pick]
    assigned = lab >= 0
    nclu = int(lab.max() + 1)
    _why = (f"selected by internal DBCV={pick_dbcv:+.3f}, taxonomy NOT used"
            if SELECT == "dbcv" else "selected to maximize distinct frontiers recovered")
    print(f"\n=== detailing min_cluster_size={pick}  ({nclu} sub-clusters, "
          f"{(~assigned).mean()*100:.1f}% noise; {_why}) ===")
    M, ks, mm = cluster_metrics(lab[assigned], y[assigned], len(cats))

    # ---- c-TF-IDF per sub-cluster (fit on subset so terms are discriminative WITHIN
    #      the super-cluster -> this is what makes the frontiers pop apart) ----
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.5)
    Xc = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    drop = np.array([all(w not in DOMAIN_STOP for w in t.split()) for t in vocab])
    tf_all = np.asarray(Xc.sum(0)).ravel().astype(float)
    per_cluster = {}
    for k in range(nclu):
        mem = np.where(lab == k)[0]
        if len(mem) == 0:
            continue
        tf = np.asarray(Xc[mem].sum(0)).ravel().astype(float)
        ctfidf = (tf / max(tf.sum(), 1)) * np.log(1 + tf_all.sum() / (tf_all + 1))
        ctfidf[~drop] *= 0.2
        top = vocab[np.argsort(-ctfidf)[:12]].tolist()
        cnt = np.bincount(y[mem], minlength=len(cats))
        dom = int(cnt.argmax())
        rep_idx = mem[np.argsort(-clusterer.probabilities_[mem])[:8]]
        per_cluster[k] = dict(
            n=int(len(mem)), dom_category=cats[dom],
            dom_share=round(float(cnt[dom] / len(mem)), 3), terms=top,
            reps=[sub[i].get("title") or "" for i in rep_idx],
            rep_cards=[dict(title=sub[i].get("title") or "",
                            abstract=(sub[i].get("abstract") or "")[:300],
                            topdown=sub[i]["category"]) for i in rep_idx],
            cat_mix={cats[i]: int(cnt[i]) for i in np.argsort(-cnt)[:5] if cnt[i]})

    # ---- soft membership (soft clustering) ----
    soft = hdbscan.all_points_membership_vectors(clusterer)
    if soft.ndim == 1:
        soft = soft.reshape(N, -1)
    np.save(os.path.join(OUTDIR, f"bertopic_tier2{SUF}_soft.npy"), soft)  # sub-cluster membership
    if soft.shape[1] > 1:
        order = np.argsort(-soft, axis=1)
        sec_strength = soft[np.arange(N), order[:, 1]]
        multi = float((sec_strength > 0.10).mean())
    else:
        multi = 0.0

    # ---- did it fracture? which top-down frontiers now own a sub-cluster ----
    dom_of = {k: per_cluster[k]["dom_category"] for k in per_cluster}
    frontiers_recovered = sorted({v for v in dom_of.values()})
    clusters_per_cat = {c: [k for k in dom_of if dom_of[k] == c] for c in cats}
    fragmented = {c: len(v) for c, v in clusters_per_cat.items() if len(v) >= 2}
    still_absent = [c for c in cats if len(clusters_per_cat[c]) == 0]

    print(f"\nTIER-2 vs top-down (within super-cluster):  NMI={mm['nmi']:.3f}  "
          f"ARI={mm['ari']:.3f}  purity={mm['purity']:.3f}  "
          f"Hungarian-1:1={mm['matched_acc']:.3f}")
    print(f"tier-1 super-cluster purity was 0.18 -> tier-2 purity {mm['purity']:.3f}")
    print(f"frontiers that gained their own sub-cluster: {len(frontiers_recovered)}/"
          f"{len(cats)} -> {frontiers_recovered}")
    print(f"frontiers still with NO dominant sub-cluster: {still_absent}")
    print(f"frontiers split across >=2 sub-clusters: {fragmented}")
    print(f"soft multi-label rate (2nd topic >0.10): {multi*100:.1f}%")

    print("\nper sub-cluster (dominant top-down frontier | c-TF-IDF terms):")
    for k in sorted(per_cluster):
        pc = per_cluster[k]
        print(f"  s{k:2d} n={pc['n']:4d} ->{pc['dom_category'][:30]:30s} "
              f"({pc['dom_share']*100:3.0f}%)  {', '.join(pc['terms'][:7])}")

    # ---- contingency heatmap: tier-2 sub-clusters x top-down frontiers ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    row_dom = M.argmax(1)
    order = sorted(range(len(ks)), key=lambda r: (row_dom[r], -M[r].sum()))
    Mo = M[order]
    klabels = [f"s{ks[r]} (n={M[r].sum()})" for r in order]
    rownorm = Mo / Mo.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(rownorm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace(" & ", "\n& ") for c in cats], rotation=90, fontsize=7)
    ax.set_yticks(range(len(klabels)))
    ax.set_yticklabels(klabels, fontsize=7)
    for i in range(Mo.shape[0]):
        for j in range(Mo.shape[1]):
            if Mo[i, j] >= max(5, 0.04 * Mo[i].sum()):
                ax.text(j, i, str(Mo[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if rownorm[i, j] < 0.55 else "black")
    ax.set_xlabel("Top-down LLM frontier (within super-cluster)", fontsize=9)
    ax.set_ylabel("Tier-2 sub-cluster (row-normalized share)", fontsize=9)
    ax.set_title(f"Tier-2 recursive clustering of c{sup} mega-cluster  "
                 f"(NMI={mm['nmi']:.2f}, purity={mm['purity']:.2f} vs tier-1 0.18)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, label="share of sub-cluster")
    fig.tight_layout()
    fig_pdf = os.path.join(OUTDIR, f"bertopic_tier2{SUF}_contingency.pdf")
    fig.savefig(fig_pdf, bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, f"bertopic_tier2{SUF}_contingency.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {fig_pdf} (+ .png)")

    # ---- cluster cards + global-index labels for downstream LLM naming ----
    cards = {str(k): dict(n=per_cluster[k]["n"], top_terms=per_cluster[k]["terms"],
                          dom_topdown=per_cluster[k]["dom_category"],
                          dom_share=per_cluster[k]["dom_share"],
                          cat_mix=per_cluster[k]["cat_mix"],
                          representative_papers=per_cluster[k]["rep_cards"])
             for k in sorted(per_cluster)}
    json.dump(cards, io.open(os.path.join(OUTDIR, f"tier2{SUF}_cluster_cards.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=2)
    # map tier-2 labels back onto the full-corpus index (others = -2 = not in super-cluster)
    full = np.full(len(recs), -2, dtype=int)
    full[sel] = lab
    np.save(os.path.join(OUTDIR, f"bertopic_tier2{SUF}_labels.npy"), full)

    out = dict(
        super_cluster=int(sup), super_cluster_name="LLM-safety mega-cluster",
        n=N, frac_of_corpus=float(N / len(recs)),
        pipeline="IDENTICAL to tier-1: SPECTER -> UMAP(5d,cosine,nn15) -> soft "
                 "HDBSCAN(eom) -> c-TF-IDF, refit on super-cluster members only",
        sweep=sweep, selection_mode=SELECT,
        selection_is_label_blind=(SELECT == "dbcv"),
        chosen_min_cluster_size=int(pick),
        chosen_dbcv=(float(pick_dbcv) if pick_dbcv == pick_dbcv else None),
        n_subclusters=nclu, noise_frac=float((~assigned).mean()),
        tier1_supercluster_purity=0.18,
        tier2_vs_topdown=mm,
        frontiers_present=cats,
        frontiers_recovered=frontiers_recovered,
        frontiers_still_absent=still_absent,
        frontiers_fragmented=fragmented,
        soft_multilabel_rate=multi,
        per_subcluster={str(k): v for k, v in per_cluster.items()})
    op = os.path.join(OUTDIR, f"bertopic_tier2{SUF}.json")
    json.dump(out, io.open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {op}")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [10, 15, 20, 25, 30, 40, 60])
