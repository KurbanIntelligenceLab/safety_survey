#!/usr/bin/env python3
"""
Embedding-based unsupervised clustering of kept safety papers, compared to the
LLM taxonomy. Uses a scientific-paper sentence encoder (SPECTER) on title+abstract,
then spherical k-means. Run with the .venv_embed interpreter.

Reports ARI/NMI/purity/Hungarian-1:1 accuracy vs the LLM primary category, and the
cluster->category contingency, so it can be put next to the TF-IDF baseline.
"""
import io, json, os, sys
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
SRC = os.path.join(PS, "papers_llm_cleaned.json")
if not os.path.exists(SRC):
    SRC = os.path.join(PS, "papers_llm.json")
OUTDIR = os.path.join(PS, "_pool_audit")
EMB_CACHE = os.path.join(OUTDIR, "_specter_emb.npy")
MODEL = "sentence-transformers/allenai-specter"
SEED = 0


def metrics(c, y, K, cats, N):
    M = np.zeros((K, len(cats)), int)
    for ci, yi in zip(c, y):
        M[ci, yi] += 1
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
    return M, dict(purity=M.max(1).sum() / N, ari=ari, nmi=nmi,
                   matched_acc=M[ri, cj].sum() / N)


def main(Ks):
    recs = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    cats = sorted({p["category"] for p in recs})
    ci = {c: i for i, c in enumerate(cats)}
    y = np.array([ci[p["category"]] for p in recs])
    N = len(recs)

    if os.path.exists(EMB_CACHE):
        Z = np.load(EMB_CACHE)
        print(f"loaded cached embeddings {Z.shape}")
    else:
        import torch
        torch.set_num_threads(int(os.environ.get("EMB_THREADS", "8")))
        from sentence_transformers import SentenceTransformer
        # SPECTER expects title[SEP]abstract
        texts = [((p.get("title") or "") + "[SEP]" + (p.get("abstract") or "")) for p in recs]
        model = SentenceTransformer(MODEL)
        model.max_seq_length = int(os.environ.get("EMB_SEQ", "192"))
        bs = int(os.environ.get("EMB_BATCH", "32"))
        chunk = int(os.environ.get("EMB_CHUNK", "500"))
        cdir = os.path.join(OUTDIR, "_emb_chunks"); os.makedirs(cdir, exist_ok=True)
        nchunks = (N + chunk - 1) // chunk
        for ci0 in range(nchunks):
            cf = os.path.join(cdir, f"chunk_{ci0:04d}.npy")
            if os.path.exists(cf):
                continue  # resume
            seg = texts[ci0 * chunk:(ci0 + 1) * chunk]
            e = model.encode(seg, batch_size=bs, show_progress_bar=False,
                             convert_to_numpy=True)
            np.save(cf, e)
            print(f"  chunk {ci0+1}/{nchunks} done ({(ci0+1)*chunk}/{N})", flush=True)
        Z = np.concatenate([np.load(os.path.join(cdir, f"chunk_{i:04d}.npy"))
                            for i in range(nchunks)], axis=0)
        np.save(EMB_CACHE, Z)
        print(f"embeddings {Z.shape} cached -> {EMB_CACHE}")

    pca = int(os.environ.get("EMB_PCA", "0"))
    if pca and pca < Z.shape[1]:
        Zc = Z - Z.mean(0, keepdims=True)
        # top-`pca` right singular vectors (PCA) via thin SVD
        from scipy.sparse.linalg import svds
        _, _, Vt = svds(Zc.astype(np.float64), k=pca)
        Z = Zc @ Vt.T
        print(f"PCA-reduced embeddings to {pca} dims")
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)  # spherical
    print(f"docs={N} dim={Z.shape[1]} categories={len(cats)}")

    results = {}
    for K in Ks:
        np.random.seed(SEED)
        try:
            _, c = kmeans2(Z, K, minit="++", missing="warn")
        except (TypeError, ValueError):
            np.random.seed(SEED)
            _, c = kmeans2(Z, K, minit="points", missing="warn")
        M, mm = metrics(c, y, K, cats, N)
        results[K] = mm
        print(f"\n=== K={K} (SPECTER) ===  purity={mm['purity']:.3f}  "
              f"ARI={mm['ari']:.3f}  NMI={mm['nmi']:.3f}  Hungarian-1:1={mm['matched_acc']:.3f}")
        for k in range(K):
            mem = np.where(c == k)[0]
            if len(mem) == 0:
                continue
            dom = M[k].argmax(); share = M[k, dom] / max(M[k].sum(), 1)
            print(f"  c{k:2d} n={len(mem):4d}  ->{cats[dom][:30]:30s} ({share*100:4.0f}%)")
    json.dump({"model": MODEL, "docs": N, "dim": int(Z.shape[1]),
               "K_results": {str(k): v for k, v in results.items()}},
              io.open(os.path.join(OUTDIR, "embed_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nwrote _pool_audit/embed_metrics.json")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [16, 24])
