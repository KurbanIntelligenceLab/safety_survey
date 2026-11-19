#!/usr/bin/env python3
"""
Merge the classified 2025 backfill (AIES + NeurIPS) into the LLM-classified
corpus. Reads the per-batch out_*.json produced by the classification workflow,
maps them onto the batch index + original candidate abstracts, dedups, and
appends to papers_llm.json -> papers_llm_2025augmented.json (non-destructive).

Key decision: the backfill papers are tagged source="pool" so they enter the
"recency-complete" pool-only temporal series that paper_stats.py / the manuscript
use for all per-year and growth claims. Their original provenance is preserved in
a "backfill" field, and the proceedings url is kept for dedup/audit.

  python3 pipeline/03_classify/merge_2025_backfill.py
"""
import io, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MERGE_DIR = ROOT / "paper_survey/_2025merge"
PROMPT_DIR = MERGE_DIR / "prompts"
INDEX = MERGE_DIR / "batch_index.json"
CORPUS = ROOT / "paper_survey/papers_llm.json"
OUT = ROOT / "paper_survey/papers_llm_2025augmented.json"
CAND_SOURCES = [
    ROOT / "paper_survey/_aies2025/aies2025_candidates.json",
    ROOT / "paper_survey/_neurips2025/neurips2025_candidates.json",
]
CATEGORIES = {
    "Adversarial Robustness", "Backdoor & Data Poisoning", "Fairness & Bias",
    "Interpretability & Explainability", "Privacy & Data Protection",
    "Out-of-Distribution & Uncertainty", "Safe Reinforcement Learning",
    "Alignment & RLHF", "Deepfakes & Media Integrity", "Hallucination & Factuality",
    "Toxicity & Content Safety", "Verification & Formal Methods",
    "Watermarking & IP Protection", "AI Governance & Policy",
    "Agentic & Frontier-Model Safety", "Miscellaneous Safety",
}


def norm_title(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (t or "").lower())).strip()


def L(p):
    return json.load(io.open(p, encoding="utf-8"))


def main():
    index = L(INDEX)  # str(id) -> {title,venue,year,dblp_key,doi,url,source}

    # Reconstruct the combined candidate list (same order prep used) for abstracts.
    combined = []
    for path in CAND_SOURCES:
        combined.extend(L(path))
    assert len(combined) == len(index), \
        f"index {len(index)} != candidates {len(combined)} (rebuild prep!)"

    # Load + validate every out batch.
    results = {}
    missing, badjson, n_files = [], [], 0
    for i in range((len(index) + 19) // 20):
        f = PROMPT_DIR / f"out_{i:03d}.json"
        if not f.exists():
            missing.append(i); continue
        try:
            obj = L(f)
        except Exception as e:
            badjson.append((i, str(e))); continue
        n_files += 1
        for r in obj.get("results", []):
            results[int(r["id"])] = r

    print(f"out files read: {n_files}  missing batches: {missing}  bad json: {badjson}")
    unclassified = [i for i in range(len(index)) if i not in results]
    print(f"classified ids: {len(results)} / {len(index)}  "
          f"unclassified ids: {len(unclassified)}")
    if missing or badjson or unclassified:
        print("WARNING: incomplete — re-run the listed batches before trusting the merge.")
        if "--force" not in sys.argv:
            sys.exit("Refusing to merge an incomplete classification (pass --force to override).")

    # Build labeled records in corpus schema.
    new_recs = []
    bad_cat = 0
    for i_str, meta in index.items():
        i = int(i_str)
        r = results.get(i)
        if not r:
            continue
        prim = r.get("primary")
        sec = r.get("secondary")
        if prim not in CATEGORIES:
            bad_cat += 1
            prim = "Miscellaneous Safety"
        if sec not in CATEGORIES:
            sec = None
        new_recs.append({
            "title": meta["title"],
            "abstract": combined[i].get("abstract"),
            "venue": meta["venue"],
            "year": meta["year"],
            "doi": meta.get("doi") or "",
            "source": "pool",                 # enter recency-complete series
            "backfill": meta.get("source"),   # aies2025 / neurips2025
            "url": meta.get("url"),
            "is_safety": bool(r["is_safety"]),
            "category": prim,
            "category_secondary": sec,
            "llm_confidence": r.get("confidence"),
        })
    if bad_cat:
        print(f"note: {bad_cat} out-of-vocab primary labels coerced to Miscellaneous Safety")

    # ---- dedup ----
    corpus = L(CORPUS)
    corp_titles = {norm_title(p.get("title")) for p in corpus}
    kept, drop_intra, drop_corpus = [], 0, 0
    seen_url, seen_doi, seen_title = set(), set(), set()
    for p in new_recs:
        u = (p.get("url") or "").strip().lower()
        d = (p.get("doi") or "").strip().lower()
        nt = norm_title(p["title"])
        if (u and u in seen_url) or (d and d in seen_doi) or (nt and nt in seen_title):
            drop_intra += 1; continue
        if nt in corp_titles:
            drop_corpus += 1; continue
        if u: seen_url.add(u)
        if d: seen_doi.add(d)
        if nt: seen_title.add(nt)
        kept.append(p)

    merged = corpus + kept
    for i, p in enumerate(merged):
        p["id"] = i
    json.dump(merged, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- report ----
    from collections import Counter
    ksafe = [p for p in kept if p["is_safety"]]
    print(f"\n=== MERGE SUMMARY ===")
    print(f"new classified records      : {len(new_recs)}")
    print(f"dropped (intra-set dup)      : {drop_intra}")
    print(f"dropped (already in corpus)  : {drop_corpus}")
    print(f"appended to corpus           : {len(kept)}  "
          f"(is_safety=true: {len(ksafe)}, dropped non-safety: {len(kept)-len(ksafe)})")
    print(f"corpus {len(corpus)} -> {len(merged)}")
    print(f"\nappended is_safety=true by venue: {dict(Counter(p['venue'] for p in ksafe))}")
    print(f"appended is_safety=true by backfill: {dict(Counter(p['backfill'] for p in ksafe))}")
    print(f"top categories (appended safety): "
          f"{Counter(p['category'] for p in ksafe).most_common(8)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
