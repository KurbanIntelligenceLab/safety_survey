#!/usr/bin/env python3
"""SPECTER-embed the CLOSING SECTIONS of the open-problem-labeled census papers.

Parallel to the subfield derivation (embed_census_block.py / app:bottomup), which
SPECTER-embeds each paper's title[SEP]abstract to test whether the 14 subfields
emerge from clustering. Here the "document" whose label we validate is the paper's
STATED open problem = its closing section, so we embed title[SEP]closing_section
(closing capped at 2600 chars, the same window the LLM classifier saw).

Doc set: every on-topic censused paper that carries >=1 open-problem label
(op_results.json). Emits, aligned row-for-row:
  _op_emb.npy         (N x 768 float32)  SPECTER vectors
  _op_meta.json       [{id, subfield, labels, n_labels, src, chars}]

  pip install -r requirements.txt
  python pipeline/07_open_problems/embed_op_sections.py
"""
import io, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FT = os.path.join(ROOT, "paper_survey", "_fulltext12987")
RES = os.path.join(FT, "op_results.json")
OUTDIR = os.path.join(FT, "opcluster")
IDTEXT = os.path.join(OUTDIR, "_idtext.json")
EMB = os.path.join(OUTDIR, "_op_emb.npy")
META = os.path.join(OUTDIR, "_op_meta.json")
MODEL = "sentence-transformers/allenai-specter"
MAXC = 2600


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    res = json.load(io.open(RES, encoding="utf-8"))
    idtext = json.load(io.open(IDTEXT, encoding="utf-8"))  # {str(id): [title, closing]}
    meta, texts = [], []
    for rec in res:
        sf = rec["subfield"]
        for r in rec["results"]:
            if r.get("off_topic"):
                continue
            labs = r.get("labels") or []
            if not labs:
                continue
            pid = r["id"]
            tt = idtext.get(str(pid))
            if not tt or not tt[1]:
                continue
            title, body = tt[0], tt[1]
            texts.append((title or "") + "[SEP]" + body[:MAXC])
            meta.append(dict(id=pid, subfield=sf, labels=labs,
                             n_labels=len(labs), chars=min(len(body), MAXC)))
    print("docs to embed:", len(texts), flush=True)

    MAXTOK = 384
    # STEP 1: tokenize everything BEFORE importing torch. The fast tokenizer's rayon
    # threadpool works only while there is thread budget; torch/OpenMP exhausts it, so
    # tokenize first (return numpy, no torch), then load the model. SPECTER = CLS pooling.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id or 0
    bt = tok.backend_tokenizer            # raw tokenizers.Tokenizer
    bt.enable_truncation(max_length=MAXTOK)
    bt.no_padding()
    ids_rows, am_rows = [], []
    for i, t in enumerate(texts):
        ii = bt.encode(t).ids[:MAXTOK]    # single-seq encode => no rayon batch pool
        am = [1] * len(ii) + [0] * (MAXTOK - len(ii))
        ii = ii + [pad_id] * (MAXTOK - len(ii))
        ids_rows.append(ii); am_rows.append(am)
        if i % 400 == 0:
            print("  tokenized %d/%d" % (i, len(texts)), flush=True)
    input_ids_all = np.array(ids_rows, dtype=np.int64)
    attn_all = np.array(am_rows, dtype=np.int64)
    print("tokenized", input_ids_all.shape, flush=True)

    # STEP 2: now torch + model forward on the pre-tokenized ids.
    import torch
    torch.set_num_threads(int(os.environ.get("EMB_THREADS", "4")))
    from transformers import AutoModel
    mdl = AutoModel.from_pretrained(MODEL)
    mdl.eval()
    print("model loaded", flush=True)
    bs = int(os.environ.get("EMB_BATCH", "16"))

    N = len(texts)
    parts = []
    CDIR = os.path.join(OUTDIR, "_chunks")
    os.makedirs(CDIR, exist_ok=True)
    chunk = 300
    nchunks = (N + chunk - 1) // chunk
    for ci in range(nchunks):
        cf = os.path.join(CDIR, "chunk_%04d.npy" % ci)
        if not os.path.exists(cf):
            lo, hi = ci * chunk, min((ci + 1) * chunk, N)
            vecs = []
            for b in range(lo, hi, bs):
                e2 = min(b + bs, hi)          # clamp: never spill into the next chunk
                ii = torch.tensor(input_ids_all[b:e2])
                am = torch.tensor(attn_all[b:e2])
                with torch.no_grad():
                    out = mdl(input_ids=ii, attention_mask=am)
                vecs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
            e = np.concatenate(vecs, 0)
            assert e.shape[0] == hi - lo, (ci, e.shape, hi - lo)
            np.save(cf, e)
            print("  chunk %d/%d (%d/%d)" % (ci + 1, nchunks, hi, N), flush=True)
        parts.append(np.load(cf))
    emb = np.concatenate(parts, 0).astype(np.float32)
    assert emb.shape[0] == N, (emb.shape, N)
    np.save(EMB, emb)
    json.dump(meta, io.open(META, "w", encoding="utf-8"))
    print("wrote", EMB, emb.shape, "and", META, flush=True)


if __name__ == "__main__":
    main()
