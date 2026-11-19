#!/usr/bin/env python3
"""
Independent-validation comparison for the 250-paper sample.

Three annotators on the SAME papers:
  - stored  : original labels (Claude Sonnet 4.6)            -> _val_opus/_gold.json
  - opus    : independent blind re-annotation (Claude Opus)  -> _val_opus/out/batch_*.json
  - sonnet* : same-model blind re-run (self-consistency)     -> _pool_audit/_pool_independent.json
  - human   : optional, if you filled human_label_sheet.csv  -> _val_opus/human_label_sheet.csv

Reports cross-annotator agreement + Cohen's kappa so we can claim DIFFERENT-MODEL
agreement (opus vs stored), shown next to SAME-MODEL agreement (sonnet vs stored)
on identical papers. Writes _val_opus/validation_report.md, _val_opus_metrics.json,
_val_opus/threeway.csv.
"""
import csv, io, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from classify_llm import CATEGORIES

PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
OUT = os.path.join(PS, "_val_opus")
CATSET = set(CATEGORIES)
ADV = "Adversarial Robustness"


def load_gold():
    return json.load(io.open(os.path.join(OUT, "_gold.json"), encoding="utf-8"))


def merge_opus(index):
    """Merge out/batch_*.json -> {id_str: {primary,secondary,is_safety,...}}; check coverage."""
    merged, bad = {}, []
    for fn in sorted(index):
        path = os.path.join(OUT, "out", fn.replace(".txt", ".json"))
        if not os.path.exists(path):
            bad.append((fn, "missing")); continue
        try:
            obj = json.load(io.open(path, encoding="utf-8"))
            results = obj["results"] if isinstance(obj, dict) else obj
        except Exception as e:
            bad.append((fn, f"parse:{e}")); continue
        got = {r["id"] for r in results if isinstance(r, dict) and "id" in r}
        want = set(index[fn])
        if got != want:
            bad.append((fn, f"miss={sorted(want-got)} extra={sorted(got-want)}"))
        for r in results:
            if isinstance(r, dict) and "id" in r:
                merged[str(r["id"])] = r
    return merged, bad


def load_sonnet():
    p = os.path.join(PS, "_pool_audit", "_pool_independent.json")
    if not os.path.exists(p):
        return {}
    return {str(r["id"]): r for r in json.load(io.open(p, encoding="utf-8"))}


def load_human():
    p = os.path.join(OUT, "human_label_sheet.csv")
    out = {}
    if not os.path.exists(p):
        return out
    for row in csv.DictReader(io.open(p, encoding="utf-8")):
        prim = (row.get("human_primary") or "").strip()
        if not prim:
            continue
        sec = (row.get("human_secondary (or blank)") or "").strip() or None
        isf = (row.get("human_is_safety (yes/no)") or "").strip().lower()
        out[str(row["id"])] = {"primary": prim, "secondary": sec,
                               "is_safety": isf in ("yes", "y", "true", "1")}
    return out


def kappa_strict(ids, a, b):
    """a,b: dict id->{'primary':...}. Returns (strict_pct, kappa, n)."""
    ids = [i for i in ids if i in a and i in b and a[i].get("primary") and b[i].get("primary")]
    n = len(ids)
    if not n:
        return 0.0, 0.0, 0
    strict = sum(1 for i in ids if a[i]["primary"] == b[i]["primary"])
    po = strict / n
    ca = Counter(a[i]["primary"] for i in ids)
    cb = Counter(b[i]["primary"] for i in ids)
    cats = set(ca) | set(cb)
    pe = sum((ca[c]/n) * (cb[c]/n) for c in cats)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return round(100*po, 1), round(k, 3), n


def lenient_pct(ids, gold, other):
    """stored primary appears in other's primary|secondary."""
    ids = [i for i in ids if i in other and other[i].get("primary")]
    if not ids:
        return 0.0
    ok = sum(1 for i in ids if gold[i]["category"] in (other[i].get("primary"), other[i].get("secondary")))
    return round(100*ok/len(ids), 1)


def main():
    gold = load_gold()
    index = json.load(io.open(os.path.join(OUT, "batch_index.json"), encoding="utf-8"))
    opus, bad = merge_opus(index)
    json.dump([opus[k] for k in sorted(opus, key=lambda x: int(x))],
              io.open(os.path.join(OUT, "_opus.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    sonnet = load_sonnet()
    human = load_human()
    ids = sorted(gold, key=lambda x: int(x))

    missing = [i for i in ids if i not in opus]
    print(f"opus coverage: {len(ids)-len(missing)}/{len(ids)}  missing={missing}")
    if bad:
        print("BAD opus batches (re-run):")
        for fn, why in bad:
            print(f"  {fn}: {why}")

    # stored as annotator dict
    stored = {i: {"primary": gold[i]["category"], "secondary": gold[i].get("category_secondary"),
                  "is_safety": gold[i]["is_safety"]} for i in ids}

    # ---- pairwise primary agreement ----
    rows = []
    pairs = [("opus vs stored (DIFFERENT model)", stored, opus),
             ("sonnet-rerun vs stored (SAME model)", stored, sonnet)]
    if human:
        pairs += [("human vs stored", stored, human),
                  ("opus vs human", human, opus),
                  ("sonnet-rerun vs human", human, sonnet)]
    for name, a, b in pairs:
        s, k, n = kappa_strict(ids, a, b)
        rows.append((name, s, k, n))

    # opus is_safety still-kept; opus AdvRob recovery vs stored
    both = [i for i in ids if i in opus]
    opus_keeps = sum(1 for i in both if opus[i].get("is_safety"))
    opus_adv_recover = [i for i in both if opus[i].get("primary") == ADV and gold[i]["category"] != ADV]
    opus_len = lenient_pct(both, gold, opus)

    # per-category opus-vs-stored lenient confirmation
    by = defaultdict(lambda: [0, 0])
    for i in both:
        c = gold[i]["category"]
        by[c][1] += 1
        if c in (opus[i].get("primary"), opus[i].get("secondary")):
            by[c][0] += 1

    metrics = {
        "n_sample": len(ids),
        "opus_coverage": len(both),
        "opus_vs_stored_strict_pct": rows[0][1],
        "opus_vs_stored_kappa": rows[0][2],
        "opus_vs_stored_lenient_pct": opus_len,
        "sonnet_rerun_vs_stored_strict_pct": rows[1][1],
        "sonnet_rerun_vs_stored_kappa": rows[1][2],
        "opus_still_safety_pct": round(100*opus_keeps/len(both), 1) if both else 0,
        "opus_advrob_recoveries": len(opus_adv_recover),
        "human_labeled_n": len(human),
    }
    json.dump(metrics, io.open(os.path.join(PS, "_val_opus_metrics.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- per-paper three-way CSV ----
    with io.open(os.path.join(OUT, "threeway.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "venue", "year", "stored_primary", "stored_secondary",
                    "opus_primary", "opus_secondary", "opus_conf",
                    "sonnet_primary", "human_primary",
                    "opus_agrees_stored", "all_three_agree"])
        for i in ids:
            g = gold[i]; o = opus.get(i, {}); s = sonnet.get(i, {}); h = human.get(i, {})
            oa = (o.get("primary") == g["category"])
            allthree = oa and (s.get("primary") == g["category"]) and \
                       (not h or h.get("primary") == g["category"])
            w.writerow([i, g["title"], g.get("venue"), g.get("year"),
                        g["category"], g.get("category_secondary"),
                        o.get("primary"), o.get("secondary"), o.get("confidence"),
                        s.get("primary"), h.get("primary", ""),
                        oa, allthree])

    # ---- report ----
    with io.open(os.path.join(OUT, "validation_report.md"), "w", encoding="utf-8") as f:
        f.write("# Independent validation of the pool taxonomy labels (n=%d)\n\n" % len(ids))
        f.write("Stratified sample of the 2,572 pool safety papers (all 16 categories), "
                "re-annotated **blind** by an independent model (**Claude Opus**), compared "
                "to the stored labels (**Claude Sonnet 4.6**). The same-model blind re-run "
                "(Sonnet) on the identical papers is shown alongside as a reference. "
                "Abstract-grounded.\n\n")
        f.write("## Cross-annotator primary-category agreement\n\n")
        f.write("| comparison | strict agree | Cohen's kappa | n |\n|---|---|---|---|\n")
        for name, s, k, n in rows:
            f.write(f"| {name} | {s}% | {k} | {n} |\n")
        f.write("\n")
        f.write("**Reading it:** *opus vs stored* is the independent (different-model) "
                "validation. If it is close to *sonnet-rerun vs stored*, the labels are "
                "robust to the choice of annotator model, not just self-consistent.\n\n")
        f.write("## Opus vs stored — details\n\n")
        f.write(f"- Lenient agreement (stored primary in Opus primary|secondary): "
                f"**{opus_len}%**\n")
        f.write(f"- Opus still calls safety: **{metrics['opus_still_safety_pct']}%**\n")
        f.write(f"- Opus assigns AdvRob-primary where stored gave another primary: "
                f"**{len(opus_adv_recover)}** (independent echo of the AdvRob-undercount signal)\n\n")
        f.write("### Per-category confirmation (Opus lenient vs stored)\n\n")
        f.write("| stored category | confirmed/total | % |\n|---|---|---|\n")
        for c in sorted(by, key=lambda c: -by[c][1]):
            ok, tot = by[c]
            f.write(f"| {c} | {ok}/{tot} | {100*ok/tot:.0f}% |\n")
        if human:
            f.write(f"\n## Human labels folded in ({len(human)} labeled)\n\n")
            f.write("See the `human vs *` rows above; `threeway.csv` has per-paper "
                    "stored/opus/sonnet/human columns and an `all_three_agree` flag.\n")
        else:
            f.write("\n## Human gold (pending)\n\n")
            f.write("Fill `paper_survey/_val_opus/human_label_sheet.csv` (guide: "
                    "`TAXONOMY.md`) and re-run this script — human-vs-model rows and the "
                    "`all_three_agree` column populate automatically.\n")
        f.write("\n## Caveats\n\n")
        f.write("- Opus and Sonnet are the same vendor (Claude); errors may still correlate. "
                "Human labels are the true-gold check.\n")
        f.write("- Abstract-grounded (no full text). Single pass per model.\n")

    print(json.dumps(metrics, indent=2))
    print("\nwrote: _val_opus_metrics.json, _val_opus/validation_report.md, _val_opus/threeway.csv, _val_opus/_opus.json")


if __name__ == "__main__":
    main()
