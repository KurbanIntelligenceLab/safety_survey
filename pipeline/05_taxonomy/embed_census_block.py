#!/usr/bin/env python3
"""SPECTER-embed the new census block and concatenate onto the cached old block.

Combined corpus (cluster_corpus_census.json) is [old 6827 | new 6283]. The old
block already has cached SPECTER vectors (_specter_emb.npy); only the new block
needs embedding. Same recipe as embed_cluster.py: allenai-specter, text =
"title[SEP]abstract", max_seq_length 192, batch 32, chunked + resumable.

Step 0 is a reproducibility guard: re-embed a sample of OLD rows in THIS env and
assert mean cosine vs the cached vectors > 0.999, so the freshly-embedded new
block is comparable to the cached old block ("same as before").

Writes paper_survey/_pool_audit/_specter_emb_census.npy  (13110 x 768).

  module load python/gpu/3.11.5 && pip install -r requirements.txt
  python pipeline/05_taxonomy/embed_census_block.py
"""
import io, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PS = os.path.join(ROOT, "paper_survey")
SRC = os.path.join(PS, "_census", "cluster_corpus_census.json")
OUTDIR = os.path.join(PS, "_pool_audit")
OLD_EMB = os.path.join(OUTDIR, "_specter_emb.npy")
OUT_EMB = os.path.join(OUTDIR, "_specter_emb_census.npy")
CDIR = os.path.join(OUTDIR, "_emb_chunks_census")
MODEL = "sentence-transformers/allenai-specter"
N_OLD = 6827


def text_of(p):
    return (p.get("title") or "") + "[SEP]" + (p.get("abstract") or "")


def cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a * b).sum(1)


def main():
    recs = json.load(io.open(SRC, encoding="utf-8"))
    assert len(recs) > N_OLD, f"corpus too small: {len(recs)}"
    old, new = recs[:N_OLD], recs[N_OLD:]
    old_emb = np.load(OLD_EMB)
    assert old_emb.shape[0] == N_OLD, f"cached emb rows {old_emb.shape[0]} != {N_OLD}"

    import torch
    torch.set_num_threads(int(os.environ.get("EMB_THREADS", "8")))
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    model.max_seq_length = int(os.environ.get("EMB_SEQ", "192"))
    bs = int(os.environ.get("EMB_BATCH", "32"))

    # --- step 0: reproducibility guard on a sample of OLD rows ---
    ns = int(os.environ.get("REPRO_N", "24"))
    samp = model.encode([text_of(p) for p in old[:ns]], batch_size=bs,
                        show_progress_bar=False, convert_to_numpy=True)
    c = cos(samp, old_emb[:ns])
    print(f"repro guard: {ns} old rows, mean cos={c.mean():.5f} min={c.min():.5f}")
    assert c.mean() > 0.999, ("new env does NOT reproduce cached SPECTER vectors "
                              f"(mean cos {c.mean():.4f}); aborting before embed")

    # --- embed new block, chunked + resumable ---
    os.makedirs(CDIR, exist_ok=True)
    chunk = int(os.environ.get("EMB_CHUNK", "500"))
    texts = [text_of(p) for p in new]
    N = len(texts)
    nchunks = (N + chunk - 1) // chunk
    for ci in range(nchunks):
        cf = os.path.join(CDIR, f"chunk_{ci:04d}.npy")
        if os.path.exists(cf):
            continue
        e = model.encode(texts[ci * chunk:(ci + 1) * chunk], batch_size=bs,
                         show_progress_bar=False, convert_to_numpy=True)
        np.save(cf, e)
        print(f"  chunk {ci+1}/{nchunks} ({min((ci+1)*chunk, N)}/{N})", flush=True)
    new_emb = np.concatenate(
        [np.load(os.path.join(CDIR, f"chunk_{i:04d}.npy")) for i in range(nchunks)], 0)
    assert new_emb.shape[0] == N, f"new emb {new_emb.shape[0]} != {N}"

    combined = np.concatenate([old_emb, new_emb], 0).astype(np.float32)
    np.save(OUT_EMB, combined)
    print(f"combined embeddings {combined.shape} -> {OUT_EMB}")


if __name__ == "__main__":
    main()
