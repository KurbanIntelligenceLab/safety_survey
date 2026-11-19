#!/usr/bin/env python3
"""
Expanded taxonomy: synonyms + adjacent terms per category, plus an expanded
DBLP query list, addressing the review concern that the original 72 queries /
title-keywords miss the words a paper would actually use.

Two things live here:
  EXPANDED_TAXONOMY - per-category keyword sets (superset of the original),
                      used for (a) multi-label heuristic tagging and (b) the
                      Miscellaneous-rescue analysis. Final labels will come
                      from the LLM classifier on title+abstract; these keywords
                      are for candidate gathering and diagnostics only.
  EXPANDED_QUERIES  - DBLP/abstract search terms (superset of the original 72),
                      adding the synonyms reviewers flagged (honesty, sycophancy,
                      scalable oversight, conformal prediction, model editing,
                      concept erasure, prompt injection, agentic safety, ...).

classify_multi(text) returns ALL matching categories (order-independent), which
fixes the original first-match-wins bias (e.g. Verification being cannibalized
by Adversarial Robustness on shared "certified/provable/lipschitz" terms).
"""

# New keywords added beyond the original set are grouped after a `# +` marker
# in comments so the diff against scrape_ai_safety_papers.py is auditable.
EXPANDED_TAXONOMY = [
    ("Adversarial Robustness", [
        "adversarial attack", "adversarial example", "adversarial robustness",
        "adversarial perturbation", "adversarial training", "certified defense",
        "certified robustness", "adversarial patch", "evasion attack",
        "robust accuracy", "adversarial defense", "pgd attack",
        "auto-attack", "autoattack", "adversarial vulnerability",
        "adversarial image", "adversarial threat", "adversarial input",
        # +
        "transfer attack", "transferable adversarial", "black-box attack",
        "white-box attack", "query-based attack", "decision-based attack",
        "robust generalization", "robustness-accuracy trade", "perturbation budget",
        "l-infinity", "l_inf", "lp-bounded", "universal perturbation",
        "physical attack", "patch attack", "robustbench",
    ]),
    ("Backdoor & Data Poisoning", [
        "backdoor", "trojan", "data poisoning", "poisoning attack",
        "neural trojan", "clean-label attack", "trigger pattern",
        "supply chain attack",
        # +
        "poisoned", "backdoored", "trigger injection", "label flipping",
        "availability attack", "indiscriminate poisoning", "gradient matching",
        "training-time attack", "model hijack",
    ]),
    ("Fairness & Bias", [
        "fairness", " bias", "demographic parity", "equalized odds",
        "algorithmic fairness", "discrimination", "disparate impact",
        "group fairness", "individual fairness", "debiasing", "debias",
        "representational harm", "unfair",
        # +
        "fairness-aware", "subgroup", "protected attribute", "spurious correlation",
        "stereotype", "social bias", "gender bias", "racial bias",
        "allocative harm", "equity", "counterfactual fairness", "calibration fairness",
        "intersectional",
    ]),
    ("Interpretability & Explainability", [
        "interpretab", "explainab", "saliency", "feature attribution",
        "concept activation", "grad-cam", "gradcam", "shap ", " lime ",
        "attention visualization", "mechanistic interpret",
        "circuit analysis", "probing classifier", "xai",
        # +
        "sparse autoencoder", "superposition", "polysemantic", "monosemantic",
        "feature visualization", "activation patching", "causal tracing",
        "concept bottleneck", "concept-based explanation", "counterfactual explanation",
        "attribution method", "self-explanation", "rationale", "faithful explanation",
        "circuit discovery", "representation engineering", "probing",
    ]),
    ("Privacy & Data Protection", [
        "differential privacy", "federated learning", "membership inference",
        "model inversion", "data extraction", "privacy attack",
        "machine unlearning", "data deletion", "right to be forgotten",
        "privacy-preserving", "privacy preserving",
        # +
        "dp-sgd", "private training", "privacy budget", "gradient inversion",
        "gradient leakage", "data reconstruction", "memorization",
        "training data extraction", "concept erasure", "knowledge unlearning",
        "model editing", "knowledge editing", "secure aggregation",
        "homomorphic encryption", "private aggregation",
    ]),
    ("Out-of-Distribution & Uncertainty", [
        "out-of-distribution", "ood detection", "anomaly detection",
        "uncertainty estimation", "selective prediction", "calibration",
        "predictive uncertainty", "epistemic uncertainty",
        "distribution shift", "domain shift", "covariate shift",
        "open-set recognition", "misclassification detection",
        "distributional robustness",
        # +
        "ood generalization", "out-of-distribution generalization", "abstention",
        "conformal prediction", "selective classification", "reject option",
        "failure prediction", "novelty detection", "open-world", "open world",
        "confidence calibration", "aleatoric uncertainty", "deep ensemble",
        "outlier detection", "near-ood", "far-ood", "test-time adaptation",
    ]),
    ("Safe Reinforcement Learning", [
        "safe reinforcement", "safe rl", "constrained mdp",
        "safe exploration", "reward hacking", "reward misspecification",
        "safe policy", "constrained reinforcement", "safety constraint",
        # +
        "specification gaming", "cmdp", "lagrangian", "shielding",
        "safe control", "risk-sensitive rl", "safety filter rl", "safe policy optimization",
        "constraint violation", "safe multi-agent",
    ]),
    ("Alignment & RLHF", [
        "alignment", "rlhf", "reinforcement learning from human",
        "human feedback", "reward model", "preference learning",
        "value alignment", "corrigib", "ai alignment",
        "instruction tuning", "constitutional ai",
        # +
        "honesty", "truthfulness", "sycophancy", "deceptive alignment",
        "scalable oversight", "weak-to-strong", "weak to strong",
        "direct preference optimization", "dpo", "rlaif", "preference optimization",
        "reward modeling", "reward overoptimization", "ai feedback",
        "instruction following", "helpful and harmless", "refusal",
        "harmful fine-tuning", "safety fine-tuning", "safety alignment",
        "self-alignment", "debate", "process supervision", "value learning",
        "goal misgeneralization", "specification", "power-seeking", "scheming",
    ]),
    ("Deepfakes & Media Integrity", [
        "deepfake", "face forgery", "face swap", "face manipulation",
        "media forensic", "image forgery", "gan-generated",
        "synthetic media", "face generation detection",
        # +
        "ai-generated image detection", "diffusion-generated", "fake image detection",
        "forgery detection", "tampering detection", "provenance", "content credential",
        "synthetic image detection", "talking head", "lip sync", "audio deepfake",
        "voice cloning detection",
    ]),
    ("Hallucination & Factuality", [
        "hallucination", "factuality", "faithfulness", "factual consistency",
        "grounded generation", "knowledge grounding",
        # +
        "confabulation", "factual error", "factual accuracy", "factual precision",
        "attribution", "citation generation", "retrieval-augmented",
        "rag faithfulness", "object hallucination", "groundedness",
        "hallucination detection", "self-consistency", "factual knowledge",
        "fact verification", "fact-checking",
    ]),
    ("Toxicity & Content Safety", [
        "toxicity", "hate speech", "harmful content", "offensive language",
        "content moderation", "abusive language", "toxic ",
        "safety filter", "guardrail", "jailbreak",
        # +
        "prompt injection", "harmful instruction", "unsafe generation",
        "harmful output", "red team", "red-team", "safety guardrail",
        "content safety", "moderation", "harmful behavior", "harmful response",
        "safe decoding", "detoxification", "harm reduction", "abuse detection",
        "csam", "self-harm",
    ]),
    ("Verification & Formal Methods", [
        "formal verification", "neural network verification",
        "abstract interpretation", "satisfiability", "smt solver",
        "certified", "provable", "lipschitz", "specification",
        "reachability analysis",
        # +
        "bound propagation", "crown", "interval bound", "complete verification",
        "incomplete verification", "verified robustness", "neural network certification",
        "proof", "formal guarantee", "branch and bound", "milp verification",
    ]),
    ("Watermarking & IP Protection", [
        "watermark", "intellectual property", "model fingerprint",
        "ownership verification", "model stealing",
        "model extraction",
        # +
        "fingerprinting", "model ownership", "dataset ownership", "llm watermark",
        "text watermark", "image watermark", "generative watermark",
        "content provenance", "passport", "knockoff", "distillation attack",
    ]),
    ("AI Governance & Policy", [
        "governance", "regulation", "accountability", "audit",
        "ai policy", "responsible ai", "ethical ai", "ai ethics",
        "impact assessment", "societal impact",
        # +
        "ai act", "ai regulation", "algorithmic audit", "model card",
        "datasheet", "transparency report", "compliance", "risk assessment",
        "sociotechnical", "ai governance", "frontier ai", "responsible scaling",
        "deployment governance", "third-party audit", "incident",
    ]),
    # Candidate NEW category for review (currently most of these fall into Misc
    # or get mislabeled). Decide whether to promote it after the LLM pass.
    ("Agentic & Frontier-Model Safety", [
        "agent safety", "agentic safety", "llm agent", "autonomous agent safety",
        "tool use safety", "multi-agent safety", "dangerous capability",
        "capability evaluation", "dual-use", "biosecurity", "cyber misuse",
        "situational awareness", "autonomy evaluation", "frontier model evaluation",
        "model autonomy", "agent monitoring",
    ]),
    ("Miscellaneous Safety", []),
]


def classify_multi(text: str):
    """Return ALL matching categories (order-independent). Fixes first-match bias."""
    blob = f" {text} ".lower()
    hits = [cat for cat, kws in EXPANDED_TAXONOMY
            if kws and any(kw in blob for kw in kws)]
    return hits or ["Miscellaneous Safety"]


# Expanded DBLP / abstract search queries (superset of the original 72).
# Kept as a flat list; the collector dedups DBLP hits by key.
NEW_QUERIES = [
    # alignment / LLM-safety synonyms
    "sycophancy language model", "honesty language model", "truthfulness language model",
    "scalable oversight", "weak-to-strong generalization", "deceptive alignment",
    "direct preference optimization", "reward overoptimization", "RLAIF",
    "instruction following safety", "harmful fine-tuning", "safety fine-tuning",
    "refusal language model", "goal misgeneralization", "specification gaming",
    "process supervision reward",
    # OOD / uncertainty synonyms
    "conformal prediction", "selective classification", "abstention learning",
    "failure prediction neural network", "open-world recognition",
    "out-of-distribution generalization", "test-time adaptation", "deep ensemble uncertainty",
    # privacy / unlearning synonyms
    "concept erasure diffusion", "knowledge editing language model", "model editing",
    "training data memorization", "gradient inversion attack", "knowledge unlearning",
    # interpretability synonyms
    "sparse autoencoder interpretability", "mechanistic interpretability circuit",
    "activation patching", "concept bottleneck model", "representation engineering",
    "probing language model",
    # toxicity / jailbreak synonyms
    "prompt injection attack", "jailbreak defense", "red teaming language model",
    "detoxification language model", "guardrail language model", "harmful instruction",
    # hallucination synonyms
    "object hallucination vision language", "retrieval augmented generation faithfulness",
    "fact verification language model", "attribution language model",
    # deepfake / watermark synonyms
    "AI-generated image detection", "diffusion image detection", "text watermarking language model",
    "content provenance", "model fingerprinting",
    # governance / frontier
    "AI risk assessment", "frontier model safety", "dangerous capability evaluation",
    "model card", "algorithmic audit", "responsible scaling",
    # agentic
    "LLM agent safety", "autonomous agent safety", "tool use safety language model",
    "multi-agent safety",
]


if __name__ == "__main__":
    # quick self-report
    from scrape_ai_safety_papers import QUERIES as ORIG_Q, TAXONOMY as ORIG_T
    orig_kw = sum(len(k) for _, k in ORIG_T)
    new_kw = sum(len(k) for _, k in EXPANDED_TAXONOMY)
    print(f"categories: {len(ORIG_T)} -> {len(EXPANDED_TAXONOMY)} "
          f"(added: Agentic & Frontier-Model Safety)")
    print(f"classification keywords: {orig_kw} -> {new_kw} (+{new_kw - orig_kw})")
    print(f"queries: {len(ORIG_Q)} -> {len(ORIG_Q) + len(NEW_QUERIES)} "
          f"(+{len(NEW_QUERIES)} new)")
