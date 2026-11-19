#!/usr/bin/env python3
"""
Score a cross-family blind re-classification (out_<tag>/) against the stored
Sonnet labels (_gold.json), using the SAME metrics as validate_opus.py so the
number is directly comparable to the intra-Claude (Opus) check. Also folds in
the Opus run and the Sonnet self-rerun for a single side-by-side table.

Writes paper_survey/_val_<tag>_metrics.json and a short markdown report.
"""
import argparse, io, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from classify_llm import CATEGORIES  # noqa

PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
VAL = os.path.join(PS, "_val_opus")
ADV = "Adversarial Robustness"


def kappa_strict(ids, a, b):
    ids = [i for i in ids if i in a and i in b and a[i].get("primary") and b[i].get("primary")]
    n = len(ids)
    if not n:
        return 0.0, 0.0, 0
    strict = sum(1 for i in ids if a[i]["primary"] == b[i]["primary"])
    po = strict / n
    ca = Counter(a[i]["primary"] for i in ids)
    cb = Counter(b[i]["primary"] for i in ids)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(ca) | set(cb))
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return round(100 * po, 1), round(k, 3), n


def lenient_pct(ids, gold, other):
    ids = [i for i in ids if i in other and other[i].get("primary")]
    if not ids:
        return 0.0
    ok = sum(1 for i in ids if gold[i]["category"] in (other[i].get("primary"), other[i].get("secondary")))
    return round(100 * ok / len(ids), 1)


def merge_out(tag, index):
    outdir = os.path.join(VAL, f"out_{tag}")
    merged = {}
    for fn in sorted(index):
        path = os.path.join(outdir, fn.replace(".txt", ".json"))
        if not os.path.exists(path):
            continue
        obj = json.load(io.open(path, encoding="utf-8"))
        results = obj["results"] if isinstance(obj, dict) else obj
        for r in results:
            if isinstance(r, dict) and "id" in r:
                merged[str(r["id"])] = r
    return merged


def load_opus():
    p = os.path.join(VAL, "_opus.json")
    if not os.path.exists(p):
        return {}
    return {str(r["id"]): r for r in json.load(io.open(p, encoding="utf-8"))}


def load_sonnet():
    p = os.path.join(PS, "_pool_audit", "_pool_independent.json")
    if not os.path.exists(p):
        return {}
    return {str(r["id"]): r for r in json.load(io.open(p, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen")
    ap.add_argument("--model-label", default="cross-family model")
    args = ap.parse_args()

    gold = json.load(io.open(os.path.join(VAL, "_gold.json"), encoding="utf-8"))
    index = json.load(io.open(os.path.join(VAL, "batch_index.json"), encoding="utf-8"))
    ids = sorted(gold, key=lambda x: int(x))

    xfam = merge_out(args.tag, index)
    opus = load_opus()
    sonnet = load_sonnet()
    stored = {i: {"primary": gold[i]["category"],
                  "secondary": gold[i].get("category_secondary"),
                  "is_safety": gold[i]["is_safety"]} for i in ids}

    both = [i for i in ids if i in xfam and xfam[i].get("primary")]
    print(f"[{args.tag}] coverage {len(both)}/{len(ids)}")

    pairs = [
        (f"{args.model_label} vs stored (DIFFERENT FAMILY)", stored, xfam),
        ("Claude Opus vs stored (same family, diff tier)", stored, opus),
        ("Claude Sonnet re-run vs stored (same model)", stored, sonnet),
    ]
    rows = []
    for name, a, b in pairs:
        s, k, n = kappa_strict(ids, a, b)
        rows.append((name, s, k, n))

    xfam_keeps = sum(1 for i in both if xfam[i].get("is_safety"))
    xfam_safety_agree = sum(1 for i in both
                            if bool(xfam[i].get("is_safety")) == bool(gold[i]["is_safety"]))
    xfam_adv_recover = [i for i in both
                        if xfam[i].get("primary") == ADV and gold[i]["category"] != ADV]
    xfam_len = lenient_pct(both, gold, xfam)

    by = defaultdict(lambda: [0, 0])
    for i in both:
        c = gold[i]["category"]
        by[c][1] += 1
        if c in (xfam[i].get("primary"), xfam[i].get("secondary")):
            by[c][0] += 1

    metrics = {
        "tag": args.tag,
        "model_label": args.model_label,
        "n_sample": len(ids),
        "coverage": len(both),
        "xfam_vs_stored_strict_pct": rows[0][1],
        "xfam_vs_stored_kappa": rows[0][2],
        "xfam_vs_stored_lenient_pct": xfam_len,
        "xfam_is_safety_agreement_pct": round(100 * xfam_safety_agree / len(both), 1) if both else 0,
        "xfam_still_safety_pct": round(100 * xfam_keeps / len(both), 1) if both else 0,
        "xfam_advrob_recoveries": len(xfam_adv_recover),
        "opus_vs_stored_kappa": rows[1][2],
        "sonnet_rerun_vs_stored_kappa": rows[2][2],
    }
    out_metrics = os.path.join(PS, f"_val_{args.tag}_metrics.json")
    json.dump(metrics, io.open(out_metrics, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    report = os.path.join(VAL, f"crossfamily_{args.tag}_report.md")
    with io.open(report, "w", encoding="utf-8") as f:
        f.write(f"# Cross-family validation: {args.model_label} (n={len(both)})\n\n")
        f.write("Identical blind sample to the Opus check; stored labels = Claude Sonnet 4.6.\n\n")
        f.write("| comparison | strict agree | Cohen's kappa | n |\n|---|---|---|---|\n")
        for name, s, k, n in rows:
            f.write(f"| {name} | {s}% | {k} | {n} |\n")
        f.write(f"\n- Cross-family lenient (stored primary in model's primary|secondary): **{xfam_len}%**\n")
        f.write(f"- is_safety gate agreement: **{metrics['xfam_is_safety_agreement_pct']}%**\n")
        f.write(f"- AdvRob recoveries (independent echo of the undercount signal): **{len(xfam_adv_recover)}**\n\n")
        f.write("### Per-category lenient confirmation (cross-family vs stored)\n\n")
        f.write("| stored category | confirmed/total | % |\n|---|---|---|\n")
        for c in sorted(by, key=lambda c: -by[c][1]):
            ok, tot = by[c]
            f.write(f"| {c} | {ok}/{tot} | {100*ok/tot:.0f}% |\n")

    print(json.dumps(metrics, indent=2))
    print(f"\nwrote {out_metrics} and {report}")


if __name__ == "__main__":
    main()
