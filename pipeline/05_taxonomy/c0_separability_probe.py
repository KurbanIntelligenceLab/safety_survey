#!/usr/bin/env python
"""Within-mega-cluster separability probe.

Diagnostic for the BERTopic comparison: the bottom-up (SPECTER->UMAP->soft
HDBSCAN) pipeline collapses the LLM-centric safety concerns (alignment,
hallucination, toxicity, agentic, ...) into a single ~27% mega-cluster (c0).

Question: is that collapse a property of the *data* (these concerns are
genuinely inseparable) or of the *unsupervised objective* (the signal is there
but HDBSCAN's density objective doesn't surface it)?

Test: restrict to the c0 mega-cluster ONLY, and ask whether a *supervised*
linear probe can recover the top-down categories from the SAME SPECTER
embeddings HDBSCAN saw. High cross-validated macro-F1 >> chance => the signal
is linearly present in the embedding; the unsupervised objective simply does
not optimize for it. This is non-circular: the probe never sees cluster
assignments, only the frozen embedding + held-out top-down labels.

Reuses the exact embedding cache + label source as bertopic_compare.py.
"""
import io, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.join(HERE, "..", "..", "paper_survey")
OUTDIR = os.path.join(PS, "_pool_audit")
EMB = os.path.join(OUTDIR, "_specter_emb.npy")
LAB = os.path.join(OUTDIR, "bertopic_labels.npy")
SRC = os.path.join(PS, "papers_llm_cleaned.json")
if not os.path.exists(SRC):
    SRC = os.path.join(PS, "papers_llm.json")

MEGA = 0            # c0 mega-cluster id
MIN_CLASS = 25      # ignore top-down categories with too few members in c0
SEED = 0            # fixed; Math/np randomness is fine here (offline script)

def main():
    Z = np.load(EMB).astype(np.float32)
    recs = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    cats = np.array([r["category"] for r in recs])
    lab = np.load(LAB)
    assert len(recs) == Z.shape[0] == lab.shape[0], (len(recs), Z.shape, lab.shape)

    mask = lab == MEGA
    Zc, yc = Z[mask], cats[mask]
    print(f"c{MEGA} mega-cluster: n={mask.sum()} / {len(lab)} ({mask.mean()*100:.1f}%)")

    # keep only sufficiently-populated top-down categories inside c0
    uniq, cnt = np.unique(yc, return_counts=True)
    keep = set(uniq[cnt >= MIN_CLASS])
    km = np.array([y in keep for y in yc])
    X, y = Zc[km], yc[km]
    print(f"kept categories (>= {MIN_CLASS} in c{MEGA}): {sorted(keep)}")
    print(f"probe set: n={len(y)}  classes={len(keep)}")
    for u in sorted(keep):
        print(f"  {u:40s} {int((y==u).sum())}")

    # chance baselines
    _, c2 = np.unique(y, return_counts=True)
    maj = c2.max() / c2.sum()
    print(f"\nmajority-class accuracy (chance): {maj*100:.1f}%")
    print(f"uniform random macro-F1 ~ {100/len(keep):.1f}%")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    yhat = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    acc = accuracy_score(y, yhat)
    f1m = f1_score(y, yhat, average="macro")
    f1w = f1_score(y, yhat, average="weighted")
    print(f"\n=== 5-fold CV supervised linear probe (within c{MEGA}) ===")
    print(f"accuracy   : {acc*100:.1f}%   (chance {maj*100:.1f}%)")
    print(f"macro-F1   : {f1m*100:.1f}%   (chance ~{100/len(keep):.1f}%)")
    print(f"weighted-F1: {f1w*100:.1f}%")
    print("\nper-class:")
    print(classification_report(y, yhat, digits=3, zero_division=0))

    labels_sorted = sorted(keep)
    cm = confusion_matrix(y, yhat, labels=labels_sorted)
    cmn = cm / cm.sum(1, keepdims=True)

    out = dict(
        mega_cluster=int(MEGA),
        n_mega=int(mask.sum()),
        frac_mega=float(mask.mean()),
        min_class=MIN_CLASS,
        classes=labels_sorted,
        n_probe=int(len(y)),
        chance_majority_acc=float(maj),
        chance_uniform_macrof1=float(1/len(keep)),
        cv_accuracy=float(acc),
        cv_macro_f1=float(f1m),
        cv_weighted_f1=float(f1w),
        per_class_f1={c: float(f1_score(y == c, yhat == c, zero_division=0))
                      for c in labels_sorted},
        confusion_row_normalized={c: dict(zip(labels_sorted, map(float, cmn[i])))
                                  for i, c in enumerate(labels_sorted)},
    )
    op = os.path.join(OUTDIR, "c0_separability_probe.json")
    json.dump(out, io.open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {op}")

if __name__ == "__main__":
    main()
