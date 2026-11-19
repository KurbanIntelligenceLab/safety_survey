#!/usr/bin/env python3
"""Assign every paper its category FROM the BERTopic clustering (bottom-up labels).

Realizes the bottom-up pipeline: categories are the clusters, not a fixed list.
- Leaf clusters = tier-1 clusters (except the mega-cluster c1) + tier-2 sub-clusters.
- Each leaf maps to a named category (human-reviewed merge of fragments).
- A paper takes its leaf's category; HDBSCAN-noise papers are assigned to the
  nearest leaf centroid in the SPECTER space (so coverage is 100% from clustering).
- Verification & Miscellaneous never form a leaf, so they vanish as categories and
  their papers fall into whichever cluster they actually sit in.

Writes paper_survey/_census/corpus_clusterlabeled.json (category = cluster-derived).
"""
import io, json
import numpy as np

PS = "paper_survey"
OUT = f"{PS}/_census/corpus_clusterlabeled.json"

T1 = {0:"Adversarial Robustness",2:"Safe Reinforcement Learning",
 3:"Privacy & Data Protection",4:"Backdoor & Data Poisoning",5:"Adversarial Robustness",
 6:"Hallucination & Factuality",7:"Out-of-Distribution & Uncertainty",
 8:"Interpretability & Explainability",9:"Interpretability & Explainability",
 10:"Fairness & Bias",11:"Watermarking & IP Protection",12:"Toxicity & Content Safety",
 13:"Deepfakes & Media Integrity",14:"AI Governance & Policy",15:"Fairness & Bias",
 16:"Adversarial Robustness",17:"Out-of-Distribution & Uncertainty",
 18:"Out-of-Distribution & Uncertainty"}  # 1 = mega-cluster -> recurse (tier 2)
T2 = {0:"Watermarking & IP Protection",1:"Alignment & RLHF",2:"Privacy & Data Protection",
 3:"Adversarial Robustness",4:"Toxicity & Content Safety",5:"Fairness & Bias",
 6:"Agentic & Frontier-Model Safety",7:"Toxicity & Content Safety",
 8:"Interpretability & Explainability",9:"Hallucination & Factuality",
 10:"Hallucination & Factuality"}
MEGA = 1


def main():
    recs = json.load(io.open(f"{PS}/_census/cluster_corpus_census.json", encoding="utf-8"))
    Z = np.load(f"{PS}/_pool_audit/_specter_emb_census.npy").astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)          # cosine space
    lab1 = np.load(f"{PS}/_pool_audit/bertopic_labels_census.npy")
    lab2 = np.load(f"{PS}/_pool_audit/bertopic_tier2_blind_census_labels.npy")  # -2 outside mega
    N = len(recs)
    assert N == Z.shape[0] == lab1.shape[0] == lab2.shape[0]

    # leaf id per paper where it is DEFINITIVELY assigned (else None -> noise)
    leaf = np.full(N, None, dtype=object)
    for i in range(N):
        k1 = int(lab1[i])
        if k1 == MEGA:
            k2 = int(lab2[i])
            if k2 >= 0:
                leaf[i] = f"s{k2}"
        elif k1 >= 0:
            leaf[i] = f"c{k1}"
    # leaf -> category
    leaf_cat = {f"c{k}": v for k, v in T1.items()}
    leaf_cat.update({f"s{k}": v for k, v in T2.items()})
    # leaf centroids (normalized mean of members)
    leaves = sorted(leaf_cat)
    cent = {}
    for lf in leaves:
        idx = np.where(leaf == lf)[0]
        c = Z[idx].mean(0); cent[lf] = c / (np.linalg.norm(c) + 1e-9)
    C = np.stack([cent[lf] for lf in leaves])

    # assign noise papers to nearest leaf centroid
    noise = np.where(leaf == None)[0]
    sims = Z[noise] @ C.T
    nn = sims.argmax(1)
    for j, i in enumerate(noise):
        leaf[i] = leaves[nn[j]]

    from collections import Counter
    out = []
    for i, p in enumerate(recs):
        q = dict(p)
        q["cluster_leaf"] = leaf[i]
        q["category"] = leaf_cat[leaf[i]]
        q["from_noise"] = bool(i in set(noise.tolist()))
        out.append(q)
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    cnt = Counter(p["category"] for p in out)
    tot = len(out)
    print(f"labeled {tot} papers into {len(cnt)} cluster-derived categories "
          f"({len(noise)} were HDBSCAN-noise, reassigned to nearest cluster)\n")
    print("%-36s %6s %7s" % ("category", "count", "share"))
    for c, n in cnt.most_common():
        print("%-36s %6d %6.1f%%" % (c, n, 100 * n / tot))
    print(f"\nwrote {OUT}")
    # sanity: where did the old Verification papers go?
    vmoved = Counter(out[i]["category"] for i, p in enumerate(recs)
                     if p["category"] == "Verification & Formal Methods")
    print("\nold 'Verification' papers now land in:", dict(vmoved.most_common()))


if __name__ == "__main__":
    main()
