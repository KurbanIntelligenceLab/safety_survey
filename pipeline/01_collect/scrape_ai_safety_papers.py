#!/usr/bin/env python3
"""
Scrape AI-safety papers from top venues (2018-2026) via the DBLP API,
deduplicate, classify into a taxonomy, and write:
  - papers_all.csv          - every paper with metadata + taxonomy tag
  - taxonomy_summary.md     - human-readable taxonomy with counts and paper lists
  - papers_all.json         - machine-readable dump
"""

import argparse, csv, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# -- Configuration -----------------------------------------------------------

DBLP_SEARCH = "https://dblp.org/search/publ/api"
YEAR_MIN = 2018
YEAR_MAX = 2026

# DBLP venue keys for filtering
VENUE_KEYS = {
    "conf/cvpr":    "CVPR",
    "conf/iccv":    "ICCV",
    "conf/icml":    "ICML",
    "conf/iclr":    "ICLR",
    "conf/nips":    "NeurIPS",
    "conf/aaai":    "AAAI",
    "conf/aies":    "AIES",
    "conf/fat":     "FAccT",
    "conf/facct":   "FAccT",
    "conf/acl":     "ACL",
    "conf/emnlp":   "EMNLP",
    "conf/safeai":  "SafeAI",
}


def venue_from_dblp_key(key):
    """Extract venue name from a DBLP publication key like 'conf/cvpr/SmithJ21'."""
    key_lower = key.lower()
    for prefix, name in VENUE_KEYS.items():
        if prefix in key_lower:
            return name
    return None


# -- Taxonomy ----------------------------------------------------------------

TAXONOMY = [
    ("Adversarial Robustness", [
        "adversarial attack", "adversarial example", "adversarial robustness",
        "adversarial perturbation", "adversarial training", "certified defense",
        "certified robustness", "adversarial patch", "evasion attack",
        "robust accuracy", "adversarial defense", "pgd attack",
        "auto-attack", "autoattack", "adversarial vulnerability",
        "adversarial image", "adversarial threat", "adversarial input",
    ]),
    ("Backdoor & Data Poisoning", [
        "backdoor", "trojan", "data poisoning", "poisoning attack",
        "neural trojan", "clean-label attack", "trigger pattern",
        "supply chain attack",
    ]),
    ("Fairness & Bias", [
        "fairness", " bias", "demographic parity", "equalized odds",
        "algorithmic fairness", "discrimination", "disparate impact",
        "group fairness", "individual fairness", "debiasing", "debias",
        "representational harm", "unfair",
    ]),
    ("Interpretability & Explainability", [
        "interpretab", "explainab", "saliency", "feature attribution",
        "concept activation", "grad-cam", "gradcam", "shap ", " lime ",
        "attention visualization", "mechanistic interpret",
        "circuit analysis", "probing classifier", "xai",
    ]),
    ("Privacy & Data Protection", [
        "differential privacy", "federated learning", "membership inference",
        "model inversion", "data extraction", "privacy attack",
        "machine unlearning", "data deletion", "right to be forgotten",
        "privacy-preserving", "privacy preserving",
    ]),
    ("Out-of-Distribution & Uncertainty", [
        "out-of-distribution", "ood detection", "anomaly detection",
        "uncertainty estimation", "selective prediction", "calibration",
        "predictive uncertainty", "epistemic uncertainty",
        "distribution shift", "domain shift", "covariate shift",
        "open-set recognition", "misclassification detection",
        "distributional robustness",
    ]),
    ("Safe Reinforcement Learning", [
        "safe reinforcement", "safe rl", "constrained mdp",
        "safe exploration", "reward hacking", "reward misspecification",
        "safe policy", "constrained reinforcement", "safety constraint",
    ]),
    ("Alignment & RLHF", [
        "alignment", "rlhf", "reinforcement learning from human",
        "human feedback", "reward model", "preference learning",
        "value alignment", "corrigib", "ai alignment",
        "instruction tuning", "constitutional ai",
    ]),
    ("Deepfakes & Media Integrity", [
        "deepfake", "face forgery", "face swap", "face manipulation",
        "media forensic", "image forgery", "gan-generated",
        "synthetic media", "face generation detection",
    ]),
    ("Hallucination & Factuality", [
        "hallucination", "factuality", "faithfulness", "factual consistency",
        "grounded generation", "knowledge grounding",
    ]),
    ("Toxicity & Content Safety", [
        "toxicity", "hate speech", "harmful content", "offensive language",
        "content moderation", "abusive language", "toxic ",
        "safety filter", "guardrail", "jailbreak",
    ]),
    ("Verification & Formal Methods", [
        "formal verification", "neural network verification",
        "abstract interpretation", "satisfiability", "smt solver",
        "certified", "provable", "lipschitz", "specification",
        "reachability analysis",
    ]),
    ("Watermarking & IP Protection", [
        "watermark", "intellectual property", "model fingerprint",
        "ownership verification", "model stealing",
        "model extraction",
    ]),
    ("AI Governance & Policy", [
        "governance", "regulation", "accountability", "audit",
        "ai policy", "responsible ai", "ethical ai", "ai ethics",
        "impact assessment", "societal impact",
    ]),
    ("Miscellaneous Safety", []),
]


def classify(title: str) -> str:
    blob = f" {title} ".lower()
    for category, keywords in TAXONOMY:
        if keywords and any(kw in blob for kw in keywords):
            return category
    return "Miscellaneous Safety"


# -- Search queries ----------------------------------------------------------

QUERIES = [
    # broad
    "AI safety",
    "machine learning safety",
    "safe artificial intelligence",
    "trustworthy AI",
    "trustworthy machine learning",
    "robust machine learning",
    "responsible AI",
    # adversarial
    "adversarial robustness",
    "adversarial examples",
    "adversarial attacks deep learning",
    "certified robustness",
    "adversarial training",
    "adversarial patch",
    "physical adversarial",
    "robustness image classification",
    "adversarial defense",
    # backdoor
    "backdoor attack neural network",
    "data poisoning",
    "trojan attack neural network",
    # fairness
    "algorithmic fairness",
    "fairness machine learning",
    "bias mitigation deep learning",
    "debiasing neural network",
    "fair classification",
    # interpretability
    "interpretability deep learning",
    "explainability neural network",
    "explainable AI",
    "mechanistic interpretability",
    "feature attribution",
    "saliency map",
    # privacy
    "differential privacy deep learning",
    "federated learning privacy",
    "membership inference attack",
    "machine unlearning",
    "privacy preserving machine learning",
    # OOD / uncertainty
    "out-of-distribution detection",
    "uncertainty estimation neural network",
    "calibration deep learning",
    "selective prediction",
    "anomaly detection deep learning",
    "distribution shift",
    # safe RL
    "safe reinforcement learning",
    "constrained reinforcement learning",
    "reward hacking",
    # alignment
    "AI alignment",
    "RLHF",
    "reinforcement learning human feedback",
    "value alignment",
    # deepfakes
    "deepfake detection",
    "face forgery detection",
    # hallucination
    "hallucination language model",
    "factuality language model",
    # toxicity
    "toxicity detection",
    "hate speech detection",
    "content moderation AI",
    "jailbreak language model",
    # verification
    "neural network verification",
    "formal verification deep learning",
    # watermarking
    "watermarking neural network",
    "model extraction attack",
    "model stealing",
    # governance
    "AI ethics",
    "AI governance",
    "AI audit",
    # more specific
    "robustness certification",
    "adversarial transferability",
    "poisoning defense",
    "model robustness",
    "safety alignment",
    "red teaming language model",
    "prompt injection",
    "model safety",
]


# -- Helpers -----------------------------------------------------------------

def fetch_dblp(query, num_hits=1000, retries=3):
    """Fetch results from DBLP search API. Returns list of hit info dicts."""
    params = {
        "q": query,
        "h": num_hits,
        "format": "json",
    }
    for attempt in range(retries):
        try:
            r = requests.get(DBLP_SEARCH, params=params, timeout=30)
            if r.status_code == 429:
                wait = 2 ** attempt * 5
                sys.stdout.write(f"  DBLP rate-limited, waiting {wait}s ...\n")
                sys.stdout.flush()
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            result = data.get("result", {})
            hits = result.get("hits", {}).get("hit", [])
            return hits
        except (requests.RequestException, json.JSONDecodeError) as e:
            wait = 2 ** attempt * 3
            sys.stdout.write(f"  request error ({e}), retry in {wait}s ...\n")
            sys.stdout.flush()
            time.sleep(wait)
    return []


def parse_dblp_hit(hit):
    """Parse a DBLP hit into our normalized paper dict. Returns None if filtered."""
    info = hit.get("info", {})
    title = info.get("title", "").rstrip(".")
    if not title:
        return None

    year_str = info.get("year", "")
    try:
        year = int(year_str)
    except (ValueError, TypeError):
        return None
    if year < YEAR_MIN or year > YEAR_MAX:
        return None

    dblp_key = info.get("key", "")
    venue = venue_from_dblp_key(dblp_key)
    if venue is None:
        return None

    authors_raw = info.get("authors", {}).get("author", [])
    if isinstance(authors_raw, dict):
        authors_raw = [authors_raw]
    authors = [a.get("text", a) if isinstance(a, dict) else str(a)
               for a in authors_raw]

    url = info.get("ee", "") or info.get("url", "")
    if isinstance(url, list):
        url = url[0] if url else ""
    doi = info.get("doi", "")

    return {
        "title": title,
        "year": year,
        "venue": venue,
        "authors": authors,
        "url": url,
        "doi": doi,
        "dblp_key": dblp_key,
    }


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="paper_survey", help="output directory")
    parser.add_argument("--hits-per-query", type=int, default=1000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seen_keys = set()
    papers = []

    total_queries = len(QUERIES)
    for qi, query in enumerate(QUERIES, 1):
        sys.stdout.write(f"\n[{qi}/{total_queries}] query: {query!r}\n")
        sys.stdout.flush()

        hits = fetch_dblp(query, num_hits=args.hits_per_query)
        matched = 0
        for hit in hits:
            p = parse_dblp_hit(hit)
            if p is None:
                continue
            key = p["dblp_key"]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            papers.append(p)
            matched += 1

        sys.stdout.write(f"  {len(hits)} hits, {matched} new venue-matched papers\n")
        sys.stdout.flush()
        time.sleep(1.5)

    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write(f"Total unique papers collected: {len(papers)}\n")
    sys.stdout.flush()

    # -- Classify ------------------------------------------------------------
    for p in papers:
        p["category"] = classify(p["title"])

    papers.sort(key=lambda p: (p["category"], p.get("year", 0), p.get("title", "")))

    # -- Write CSV -----------------------------------------------------------
    csv_path = outdir / "papers_all.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "year", "venue", "title", "authors", "url", "doi"])
        for p in papers:
            authors_str = "; ".join(p.get("authors", []))
            w.writerow([
                p["category"],
                p.get("year", ""),
                p.get("venue", ""),
                p.get("title", ""),
                authors_str,
                p.get("url", ""),
                p.get("doi", ""),
            ])
    sys.stdout.write(f"Wrote {csv_path}  ({len(papers)} rows)\n")

    # -- Write JSON ----------------------------------------------------------
    json_path = outdir / "papers_all.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    sys.stdout.write(f"Wrote {json_path}\n")

    # -- Write taxonomy summary ----------------------------------------------
    by_cat = defaultdict(list)
    for p in papers:
        by_cat[p["category"]].append(p)

    md_path = outdir / "taxonomy_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AI Safety Paper Taxonomy (2018-2026)\n\n")
        f.write(f"**Total papers:** {len(papers)}  \n")
        f.write("**Venues:** CVPR, ICCV, ICML, ICLR, NeurIPS, AAAI, AIES, FAccT  \n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d')}  \n")
        f.write(f"**Source:** DBLP  \n\n")
        f.write("---\n\n")

        # Overview table
        f.write("## Overview\n\n")
        f.write("| Category | Count |\n|---|---:|\n")
        for cat_name, _ in TAXONOMY:
            count = len(by_cat.get(cat_name, []))
            if count:
                f.write(f"| {cat_name} | {count} |\n")
        f.write(f"| **Total** | **{len(papers)}** |\n\n")
        f.write("---\n\n")

        # Per-category listings
        for cat_name, _ in TAXONOMY:
            cat_papers = by_cat.get(cat_name, [])
            if not cat_papers:
                continue
            f.write(f"## {cat_name}  ({len(cat_papers)} papers)\n\n")
            cat_papers.sort(key=lambda p: (p.get("year", 0), p.get("title", "")))
            for p in cat_papers:
                year = p.get("year", "?")
                venue = p.get("venue", "?")
                title = p.get("title", "(untitled)")
                url = p.get("url", "")
                link = f"[{title}]({url})" if url else title
                authors_short = p.get("authors", [])
                if len(authors_short) > 3:
                    authors_str = f"{authors_short[0]} et al."
                else:
                    authors_str = ", ".join(authors_short)
                f.write(f"- **[{venue} {year}]** {link} -- {authors_str}\n")
            f.write("\n")

        # Year distribution
        f.write("---\n\n## Year Distribution\n\n")
        f.write("| Year | Count |\n|---|---:|\n")
        year_counts = defaultdict(int)
        for p in papers:
            year_counts[p.get("year", 0)] += 1
        for y in sorted(year_counts):
            f.write(f"| {y} | {year_counts[y]} |\n")
        f.write("\n")

        # Venue distribution
        f.write("## Venue Distribution\n\n")
        f.write("| Venue | Count |\n|---|---:|\n")
        venue_counts = defaultdict(int)
        for p in papers:
            venue_counts[p.get("venue", "?")] += 1
        for v, c in sorted(venue_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {v} | {c} |\n")
        f.write("\n")

    sys.stdout.write(f"Wrote {md_path}\n")
    sys.stdout.write("\nDone.\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
