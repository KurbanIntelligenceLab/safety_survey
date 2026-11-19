#!/usr/bin/env python3
"""Test whether merging confusable open-problem labels makes them recoverable.

op_label_probe.py found 12 labels with one-vs-rest AUC < 0.65 and, for each, the
nearest other label by centroid cosine. This script asks the follow-up question:
if we merge a weak label with its nearest neighbour, does the MERGED concept become
recoverable? If yes, the two labels were splitting one real open problem and should
be collapsed. If the union is no better than the parts, the weak label is genuinely
diffuse and merging would only hide that.

IMPORTANT CAVEAT, carried into the report: the merge pairs were CHOSEN by centroid
proximity and are RE-SCORED on the same embeddings, so the union is expected to look
somewhat better by construction (it also has more positives). Treat a union AUC that
merely edges past its parts as unconvincing; look for a decisive jump. The honest
test of a merge is whether the two labels read as one problem to a human, which this
script cannot settle -- it only flags which pairs are worth that read.

  sbatch pipeline/07_open_problems/op_merge_test.sbatch
"""
import io, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey", "_fulltext12987"))
OUTDIR = os.path.join(FT, "opcluster")
EMB = os.path.join(OUTDIR, "_op_emb.npy")
META = os.path.join(OUTDIR, "_op_meta.json")
PROBE = os.path.join(OUTDIR, "op_label_probe.json")
OUT = os.path.join(OUTDIR, "op_merge_test.json")

# Data-driven merge proposals: every weak label (AUC<0.65) paired with the nearest
# label op_label_probe.py named. Deepfakes forms a 3-way chain
# (unseen_generators -> arms_race -> adversarial_evasion) so it is tested as a triple.
MERGES = [
    ("Hallucination & Factuality", ["detection_to_prevention", "eval_bottleneck"]),
    ("Hallucination & Factuality", ["parametric_vs_contextual", "rag_verification"]),
    ("Watermarking & IP Protection", ["removal_robust_watermark", "insufficient_backfire"]),
    ("Toxicity & Content Safety", ["guard_calibration", "jailbreak_robustness"]),
    ("Deepfakes & Media Integrity", ["unseen_generators", "arms_race", "adversarial_evasion"]),
    ("Fairness & Bias", ["robust_to_manipulation", "without_sensitive_attrs"]),
    ("Safe Reinforcement Learning", ["sample_efficiency", "overhead_scalability"]),
    ("Out-of-Distribution & Uncertainty", ["uqood_compute_cost", "extend_uq_modality"]),
    ("Interpretability & Explainability", ["interp_as_infra", "faithfulness"]),
    ("Backdoor & Data Poisoning", ["defenses_ineffective", "stronger_attacks"]),
]


def ovr_auc(Z, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    y = y.astype(int)
    npos = int(y.sum())
    if npos < 8 or npos == len(y):
        return None
    k = int(min(5, npos))
    if k < 3:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    s = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(Z, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(Z[tr], y[tr])
        s[te] = clf.predict_proba(Z[te])[:, 1]
    return dict(n_pos=npos, auc=round(float(roc_auc_score(y, s)), 3))


def main():
    Z = np.load(EMB).astype(np.float32)
    meta = json.load(io.open(META, encoding="utf-8"))
    probe = json.load(io.open(PROBE, encoding="utf-8"))

    results = []
    for sf, group in MERGES:
        idx = np.array([i for i, m in enumerate(meta) if m["subfield"] == sf])
        Zs = Z[idx]
        parts = []
        for l in group:
            y = np.array([l in meta[i]["labels"] for i in idx])
            r = ovr_auc(Zs, y)
            prior = probe["subfields"][sf]["labels"].get(l, {})
            parts.append(dict(label=l, n_pos=int(y.sum()),
                              auc=r["auc"] if r else None,
                              auc_reported=prior.get("auc")))
        yu = np.array([any(l in meta[i]["labels"] for l in group) for i in idx])
        ru = ovr_auc(Zs, yu)
        part_aucs = [p["auc"] for p in parts if p["auc"] is not None]
        best_part = max(part_aucs) if part_aucs else None
        union_auc = ru["auc"] if ru else None
        gain = round(union_auc - best_part, 3) if (union_auc and best_part) else None
        verdict = ("merge helps" if gain is not None and gain >= 0.08 else
                   "marginal" if gain is not None and gain >= 0.03 else
                   "no gain")
        results.append(dict(subfield=sf, group=group, parts=parts,
                            union_n_pos=int(yu.sum()), union_auc=union_auc,
                            best_part_auc=best_part, gain=gain, verdict=verdict))
        print("%-34s %-52s parts=%s union=%.3f (n=%d) gain=%s -> %s"
              % (sf, "+".join(group),
                 [p["auc"] for p in parts], union_auc or 0, int(yu.sum()),
                 ("%+.3f" % gain) if gain is not None else "n/a", verdict), flush=True)

    out = dict(note="merge pairs selected by centroid proximity and rescored on the "
                    "same embeddings; expect optimistic bias, require a decisive gain",
               results=results,
               n_merge_helps=sum(1 for r in results if r["verdict"] == "merge helps"))
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=2)
    print("\nmerges that clearly help:", out["n_merge_helps"], "of", len(results))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
