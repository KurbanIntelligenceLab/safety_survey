#!/usr/bin/env python3
"""Score the FULL-corpus full-text re-classification against stored labels.

Reads reclass_full_out/batch_NNN.json (one {"results":[...]} per batch produced
by the workflow), merges with reclass_full_gold.json, and reports:
  - exact primary agreement + Cohen's kappa over re-run papers
  - primary-or-secondary agreement
  - re-flagged not-safety rate under full text
  - the CORPUS-WEIGHTED category distribution under full-text labels vs the
    stored (abstract) distribution, for both the re-run subset and the full
    6,827-paper corpus (papers without full text keep their abstract label).
  - a confusion summary of where mass moved between categories.

Writes reclass_full_result.md + reclass_full_merged.json.
"""
import io, json, os, glob
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
# Paths overridable via env (RC_OUTDIR = the _fulltext dir, RC_LAB = label file)
# so the same scorer serves the census corpus without touching the old run.
FT = Path(os.environ.get("RC_OUTDIR", ROOT / "paper_survey" / "_fulltext"))
LAB = Path(os.environ.get("RC_LAB", ROOT / "paper_survey" / "papers_llm_cleaned.json"))
OUTDIR = FT / "reclass_full_out"
GOLD = FT / "reclass_full_gold.json"


def cohen_kappa(a, b):
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a); cb = Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main():
    gold = json.load(io.open(GOLD, encoding="utf-8"))
    new = {}
    nb = 0
    missing_batches = []
    manifest = json.load(io.open(FT / "reclass_full_manifest.json", encoding="utf-8"))
    for bname in sorted(manifest["batches"]):
        f = OUTDIR / (bname + ".json")
        if not f.exists():
            missing_batches.append(bname)
            continue
        nb += 1
        obj = json.load(io.open(f, encoding="utf-8"))
        results = obj["results"] if isinstance(obj, dict) else obj
        for r in results:
            new[str(r["id"])] = r

    # align to gold (re-run papers only)
    ids = [i for i in gold if i in new]
    a = [gold[i]["category"] for i in ids]                 # stored (abstract)
    b = [new[i]["primary"] for i in ids]                   # full-text
    exact = sum(1 for x, y in zip(a, b) if x == y)

    # primary-or-secondary agreement
    pos = 0
    for i in ids:
        g = {gold[i]["category"], gold[i].get("secondary")}
        n = {new[i]["primary"], new[i].get("secondary")}
        if (g & n) - {None}:
            pos += 1

    reflag = sum(1 for i in ids if new[i].get("is_safety") is False)

    kappa = cohen_kappa(a, b)

    # ---- distributions -----------------------------------------------------
    # stored full-corpus distribution (all safety papers)
    papers = json.load(io.open(LAB, encoding="utf-8"))
    saf = [p for p in papers if p.get("is_safety")]
    stored_full = Counter(p["category"] for p in saf)

    # full-text-corrected full-corpus distribution:
    #   re-run papers -> new primary (only if still is_safety); others -> stored.
    ft_full = Counter()
    dropped = 0
    for p in saf:
        pid = str(p["id"])
        if pid in new:
            if new[pid].get("is_safety") is False:
                dropped += 1
                continue  # falls out of safety corpus under full text
            ft_full[new[pid]["primary"]] += 1
        else:
            ft_full[p["category"]] += 1

    # distribution restricted to the re-run subset (stored vs full-text)
    stored_sub = Counter(a)
    ft_sub = Counter(new[i]["primary"] for i in ids if new[i].get("is_safety") is not False)

    # movement: net change per category over re-run subset
    cats = sorted(set(stored_sub) | set(ft_sub))

    # ---- write report ------------------------------------------------------
    L = []
    w = L.append
    w("# Full-corpus full-text re-classification (2026-06-05)\n")
    w("Re-classifies EVERY safety paper with extracted full text (method or")
    w("conclusion) using the canonical instrument fed TITLE+ABSTRACT+METHOD+")
    w("CONCLUSION, and compares to the stored abstract-based labels. Unlike the")
    w("191-paper stratified check, this measures the CORPUS-WEIGHTED Table-1")
    w("distribution, not just per-category stability.\n")
    w("## Coverage")
    w(f"- Safety papers: {len(saf)}")
    w(f"- Re-classified (had full text): {len(new)}   batches scored: {nb}/{manifest['n_batches']}")
    if missing_batches:
        w(f"- MISSING batch outputs: {len(missing_batches)} -> {', '.join(missing_batches[:8])}"
          + (" ..." if len(missing_batches) > 8 else ""))
    w(f"- Not re-run (no full text, keep abstract label): {len(saf) - len(new)}\n")
    w("## Agreement (re-run subset, n=%d)" % len(ids))
    w(f"- Exact primary agreement: {exact}/{len(ids)} = {100*exact/len(ids):.1f}%")
    w(f"- Primary-or-secondary agreement: {pos}/{len(ids)} = {100*pos/len(ids):.1f}%")
    w(f"- Cohen's kappa (primary): {kappa:.3f}")
    w(f"- Re-flagged not-safety under full text: {reflag}/{len(ids)} = {100*reflag/len(ids):.1f}%\n")
    w("## Corpus-weighted distribution: stored (abstract) vs full-text-corrected")
    w("Full corpus = all %d safety papers; re-run papers take the full-text" % len(saf))
    w("primary (or drop if re-flagged not-safety), the rest keep abstract labels.")
    w(f"Papers dropped from safety corpus under full text: {dropped}")
    w(f"Full-text-corrected safety corpus size: {sum(ft_full.values())}\n")
    w("| Category | stored n | stored % | full-text n | full-text % | Δpp |")
    w("|---|---:|---:|---:|---:|---:|")
    tot_s = sum(stored_full.values()); tot_f = sum(ft_full.values())
    for c in sorted(set(stored_full) | set(ft_full), key=lambda x: -stored_full.get(x, 0)):
        ps = 100*stored_full.get(c, 0)/tot_s
        pf = 100*ft_full.get(c, 0)/tot_f
        w(f"| {c} | {stored_full.get(c,0)} | {ps:.1f} | {ft_full.get(c,0)} | {pf:.1f} | {pf-ps:+.1f} |")
    w("")
    w("## Movement within re-run subset (stored -> full-text primary)")
    w("| Category | stored n | full-text n | Δ |")
    w("|---|---:|---:|---:|")
    for c in sorted(cats, key=lambda x: -stored_sub.get(x, 0)):
        w(f"| {c} | {stored_sub.get(c,0)} | {ft_sub.get(c,0)} | {ft_sub.get(c,0)-stored_sub.get(c,0):+d} |")
    w("")

    rep = "\n".join(L)
    (FT / "reclass_full_result.md").write_text(rep, encoding="utf-8")

    merged = {i: {"stored": gold[i], "fulltext": new.get(i)} for i in gold}
    json.dump(merged, io.open(FT / "reclass_full_merged.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(rep)
    print("\nwrote", FT / "reclass_full_result.md")
    if missing_batches:
        print("WARNING: %d batches missing -> %s" % (len(missing_batches), missing_batches[:10]))


if __name__ == "__main__":
    main()
