#!/usr/bin/env python
"""Aggregate the gapmine per-subfield result files into a verdict on ruler
completeness: which subfields have a RECURRING open problem the ruler missed."""
import json, io, os

DIR = "paper_survey/_fulltext12987/gapmine"
SLUGS = {
    "AI_Governance_and_Policy": "AI Governance & Policy",
    "Adversarial_Robustness": "Adversarial Robustness",
    "Agentic_and_Frontier_Model_Safety": "Agentic & Frontier-Model Safety",
    "Alignment_and_RLHF": "Alignment & RLHF",
    "Backdoor_and_Data_Poisoning": "Backdoor & Data Poisoning",
    "Deepfakes_and_Media_Integrity": "Deepfakes & Media Integrity",
    "Fairness_and_Bias": "Fairness & Bias",
    "Hallucination_and_Factuality": "Hallucination & Factuality",
    "Interpretability_and_Explainability": "Interpretability & Explainability",
    "Out_of_Distribution_and_Uncertainty": "Out-of-Distribution & Uncertainty",
    "Privacy_and_Data_Protection": "Privacy & Data Protection",
    "Safe_Reinforcement_Learning": "Safe Reinforcement Learning",
    "Toxicity_and_Content_Safety": "Toxicity & Content Safety",
    "Watermarking_and_IP_Protection": "Watermarking & IP Protection",
}


def main():
    real = []   # (subfield, name, count, example)
    missing = []
    for slug, sub in SLUGS.items():
        rf = os.path.join(DIR, slug + ".result.json")
        if not os.path.exists(rf):
            missing.append(slug); continue
        obj = json.load(io.open(rf, encoding="utf-8"))
        for c in obj.get("candidates", []):
            if c.get("recurring") and not c.get("covered_by_existing"):
                real.append((sub, c["name"], c.get("approx_count", 0), c.get("example", "")))

    real.sort(key=lambda x: -x[2])
    print("=== RECURRING open problems the ruler MISSED (candidate new labels) ===\n")
    if not real:
        print("  none — every recurring theme is already covered by the ruler.")
    for sub, name, n, ex in real:
        print("  [%2d papers] %-32s  %s" % (n, sub[:32], name))
        if ex:
            print("              e.g. %s" % ex[:90])
    print("\n=== per-subfield verdicts ===")
    for slug, sub in SLUGS.items():
        rf = os.path.join(DIR, slug + ".result.json")
        if os.path.exists(rf):
            v = json.load(io.open(rf, encoding="utf-8")).get("verdict", "")
            print("  %-34s %s" % (sub[:34], v[:110]))
    if missing:
        print("\nMISSING result files:", missing)


if __name__ == "__main__":
    main()
