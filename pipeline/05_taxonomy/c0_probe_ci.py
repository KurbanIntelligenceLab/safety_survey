#!/usr/bin/env python3
"""Bootstrap CIs + explicit protocol for the within-c0 linear probe
(reviewer R2: probe needs train/test split, class balance, confidence intervals).

Same setup as c0_separability_probe.py (frozen SPECTER embeddings, 5-fold
stratified CV, class-balanced logistic regression). Adds:
  - class balance table (already implied by counts; reported explicitly)
  - nonparametric bootstrap 95% CIs (B=2000) on accuracy / macro-F1 / weighted-F1,
    resampling the out-of-fold (y, yhat) pairs
  - a held-out 70/30 stratified split as a second, simpler protocol whose point
    estimate should agree with the CV figure

Run: python pipeline/05_taxonomy/c0_probe_ci.py
"""
import io, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import f1_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.join(HERE, "..", "..", "paper_survey")
OUTDIR = os.path.join(PS, "_pool_audit")
EMB = os.path.join(OUTDIR, "_specter_emb.npy")
LAB = os.path.join(OUTDIR, "bertopic_labels.npy")
SRC = os.path.join(PS, "papers_llm_cleaned.json")
MEGA, MIN_CLASS, SEED, B = 0, 25, 0, 2000


def boot_ci(y, yhat, B, seed):
    rng = np.random.RandomState(seed)
    n = len(y); accs, f1ms, f1ws = [], [], []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        ys, ps = y[idx], yhat[idx]
        accs.append(accuracy_score(ys, ps))
        f1ms.append(f1_score(ys, ps, average="macro", zero_division=0))
        f1ws.append(f1_score(ys, ps, average="weighted", zero_division=0))
    pct = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return dict(accuracy_ci=pct(accs), macro_f1_ci=pct(f1ms), weighted_f1_ci=pct(f1ws))


def main():
    Z = np.load(EMB).astype(np.float32)
    recs = [p for p in json.load(io.open(SRC, encoding="utf-8")) if p.get("is_safety")]
    cats = np.array([r["category"] for r in recs])
    lab = np.load(LAB)
    mask = lab == MEGA
    Zc, yc = Z[mask], cats[mask]
    uniq, cnt = np.unique(yc, return_counts=True)
    keep = set(uniq[cnt >= MIN_CLASS])
    km = np.array([y in keep for y in yc])
    X, y = Zc[km], yc[km]
    print(f"probe set n={len(y)} classes={len(keep)}")

    # class balance
    u, c = np.unique(y, return_counts=True)
    bal = {str(a): int(b) for a, b in zip(u, c)}
    print("class balance:", bal, " (min/maj ratio %.2f)" % (c.min() / c.max()))

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))

    # 5-fold stratified CV out-of-fold predictions (same as the probe)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    yhat = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    acc = accuracy_score(y, yhat); f1m = f1_score(y, yhat, average="macro")
    f1w = f1_score(y, yhat, average="weighted")
    ci = boot_ci(y, yhat, B, SEED)
    print(f"\n5-fold CV: acc={acc*100:.1f}% macroF1={f1m*100:.1f}% weightedF1={f1w*100:.1f}%")
    print(f"  bootstrap95 (B={B}): acc {ci['accuracy_ci']} macroF1 {ci['macro_f1_ci']} weightedF1 {ci['weighted_f1_ci']}")

    # held-out 70/30 stratified split (second protocol)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    h_acc = accuracy_score(yte, yp); h_f1m = f1_score(yte, yp, average="macro")
    h_ci = boot_ci(yte, yp, B, SEED)
    print(f"holdout 70/30: acc={h_acc*100:.1f}% macroF1={h_f1m*100:.1f}%  "
          f"bootstrap95 macroF1 {h_ci['macro_f1_ci']}")

    out = dict(n_probe=int(len(y)), n_classes=len(keep), class_balance=bal,
               min_maj_ratio=float(c.min() / c.max()),
               cv_accuracy=float(acc), cv_macro_f1=float(f1m), cv_weighted_f1=float(f1w),
               cv_bootstrap95=ci, B=B,
               holdout_70_30=dict(accuracy=float(h_acc), macro_f1=float(h_f1m),
                                  bootstrap95=h_ci),
               chance_majority_acc=float(c.max() / c.sum()),
               chance_uniform_macro_f1=float(1 / len(keep)))
    op = os.path.join(OUTDIR, "c0_probe_ci.json")
    json.dump(out, io.open(op, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nwrote", op)


if __name__ == "__main__":
    main()
