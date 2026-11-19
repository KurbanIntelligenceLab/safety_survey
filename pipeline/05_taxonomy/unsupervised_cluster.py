#!/usr/bin/env python3
"""
Unsupervised clustering of the kept safety papers (title+abstract), compared to
the LLM taxonomy labels. Pure numpy/scipy (no sklearn): TF-IDF -> LSA (truncated
SVD) -> spherical k-means. Reports ARI, NMI, purity, Hungarian-matched accuracy,
the cluster->category contingency, and top terms per cluster.

Usage: python3 unsupervised_cluster.py [K ...]   (default K=16 and 24)
"""
import io, json, os, re, sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
SRC = os.path.join(PS, "papers_llm_cleaned.json")
if not os.path.exists(SRC):
    SRC = os.path.join(PS, "papers_llm.json")
OUTDIR = os.path.join(PS, "_pool_audit")
SEED = 0

STOP = set("""a an the of to in on for and or is are be as at by with from this that these those we our
it its their they them he she his her you your i me my ma was were has have had will would can could
should may might do does did not no nor only own same so than too very s t just but if then because
while which who whom whose what when where why how all any both each few more most other some such only
into through during before after above below up down out off over under again further once here there
been being also use used using uses paper method methods approach approaches propose proposed propose
result results show shows shown new novel work works study studies based via toward towards et al
""".split())

TOK = re.compile(r"[a-z][a-z\-]{2,}")


def main(Ks):
    recs = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    texts, labels, ids = [], [], []
    for p in recs:
        t = ((p.get("title") or "") + ". " + (p.get("abstract") or "")).lower()
        texts.append([w for w in TOK.findall(t) if w not in STOP and len(w) < 30])
        labels.append(p["category"])
        ids.append(p["id"])
    N = len(texts)
    cats = sorted(set(labels))
    cat_idx = {c: i for i, c in enumerate(cats)}
    y = np.array([cat_idx[c] for c in labels])
    print(f"docs={N}  categories={len(cats)}")

    # ---- vocab: df filtering ----
    from collections import Counter
    df = Counter()
    for toks in texts:
        df.update(set(toks))
    min_df, max_df = 5, int(0.5 * N)
    vocab = sorted([w for w, c in df.items() if min_df <= c <= max_df])
    vidx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"vocab={V}")

    # ---- sparse TF-IDF (sublinear tf, smoothed idf, L2-norm rows) ----
    idf = np.array([np.log(N / df[w]) + 1.0 for w in vocab])
    rows, cols, vals = [], [], []
    for r, toks in enumerate(texts):
        tf = Counter(w for w in toks if w in vidx)
        if not tf:
            continue
        idxs = np.array([vidx[w] for w in tf])
        wts = np.array([(1.0 + np.log(c)) for c in tf.values()]) * idf[idxs]
        wts /= (np.linalg.norm(wts) + 1e-9)
        rows.extend([r] * len(idxs)); cols.extend(idxs); vals.extend(wts)
    X = csr_matrix((vals, (rows, cols)), shape=(N, V))

    # ---- LSA: truncated SVD to 100 dims, row-normalize ----
    k_svd = 100
    U, S, Vt = svds(X, k=k_svd)
    Z = U * S
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)  # spherical

    def contingency(c, y, K):
        M = np.zeros((K, len(cats)), int)
        for ci, yi in zip(c, y):
            M[ci, yi] += 1
        return M

    def purity(M):
        return M.max(axis=1).sum() / M.sum()

    def ari(M):
        n = M.sum()
        a = M.sum(1); b = M.sum(0)
        comb = lambda x: x * (x - 1) / 2.0
        idx = comb(M).sum()
        ea = comb(a).sum(); eb = comb(b).sum()
        exp = ea * eb / comb(n)
        mx = 0.5 * (ea + eb)
        return (idx - exp) / (mx - exp) if mx != exp else 0.0

    def nmi(M):
        n = M.sum()
        Pij = M / n
        Pi = M.sum(1) / n; Pj = M.sum(0) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(Pij * np.log(Pij / (np.outer(Pi, Pj) + 1e-12) + 1e-12))
            Hi = -np.nansum(Pi * np.log(Pi + 1e-12))
            Hj = -np.nansum(Pj * np.log(Pj + 1e-12))
        return mi / (0.5 * (Hi + Hj)) if (Hi + Hj) > 0 else 0.0

    # mean tf-idf per cluster for top terms (use original sparse X)
    Xc = X.tocsc()
    results = {}
    for K in Ks:
        np.random.seed(SEED)
        try:
            cent, c = kmeans2(Z, K, minit="++", missing="warn")
        except (TypeError, ValueError):
            np.random.seed(SEED)
            cent, c = kmeans2(Z, K, minit="points", missing="warn")
        M = contingency(c, y, K)
        # Hungarian best 1:1 cluster->category accuracy
        ri, cj = linear_sum_assignment(-M)
        matched_acc = M[ri, cj].sum() / N
        pur, a, nm = purity(M), ari(M), nmi(M)
        results[K] = dict(purity=pur, ari=a, nmi=nm, matched_acc=matched_acc)
        print(f"\n=== K={K} ===  purity={pur:.3f}  ARI={a:.3f}  NMI={nm:.3f}  "
              f"Hungarian-1:1 acc={matched_acc:.3f}")
        # per-cluster: size, dominant category + share, top terms
        for ci in range(K):
            members = np.where(c == ci)[0]
            if len(members) == 0:
                continue
            dom = M[ci].argmax(); share = M[ci, dom] / max(M[ci].sum(), 1)
            sub = Xc[members]
            term_sum = np.asarray(sub.sum(axis=0)).ravel()
            top = [vocab[j] for j in term_sum.argsort()[::-1][:8]]
            print(f"  c{ci:2d} n={len(members):4d}  ->{cats[dom][:26]:26s} ({share*100:4.0f}%)  "
                  f"{' '.join(top)}")

    out = {"docs": N, "vocab": V, "K_results": {str(k): v for k, v in results.items()}}
    json.dump(out, io.open(os.path.join(OUTDIR, "unsupervised_metrics.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote _pool_audit/unsupervised_metrics.json")


if __name__ == "__main__":
    Ks = [int(x) for x in sys.argv[1:]] or [16, 24]
    main(Ks)
