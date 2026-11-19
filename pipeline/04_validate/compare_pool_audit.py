#!/usr/bin/env python3
"""
Stage 3: compare the blind re-classification against the stored labels for the
2,572 pooled safety papers, mirroring pipeline/04_validate/validate_compare.py's metric
definitions, and emit the deliverables.

Reads : paper_survey/_pool_audit/_pool_gold.json
        paper_survey/_pool_audit/_pool_independent.json
Writes: paper_survey/_pool_audit/_pool_metrics.json
        paper_survey/_pool_audit/pool_summaries.csv
        paper_survey/_pool_audit/pool_summaries.md
        paper_survey/_pool_audit/pool_audit_report.md
"""
import csv, io, json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
OUT = os.path.join(PS, "_pool_audit")
ADV = "Adversarial Robustness"

gold = json.load(io.open(os.path.join(OUT, "_pool_gold.json"), encoding="utf-8"))
ind_raw = json.load(io.open(os.path.join(OUT, "_pool_independent.json"), encoding="utf-8"))
ind = {str(r["id"]): r for r in ind_raw}

ids = [i for i in gold if i in ind]
n = len(ids)

# ---- is_safety (all gold are stored-kept; measure how many fresh still keeps) --
still_safe = sum(1 for i in ids if ind[i].get("is_safety"))
now_dropped = [i for i in ids if not ind[i].get("is_safety")]

# ---- primary agreement on the both-safe subset ----
both = [i for i in ids if gold[i]["is_safety"] and ind[i].get("is_safety")]


def lenient_match(i):
    g = gold[i]["category"]
    return g in (ind[i].get("primary"), ind[i].get("secondary"))


strict = sum(1 for i in both if gold[i]["category"] == ind[i].get("primary"))
lenient = sum(1 for i in both if lenient_match(i))

cats = sorted({gold[i]["category"] for i in both} |
              {ind[i].get("primary") for i in both if ind[i].get("primary")})
po = strict / len(both) if both else 0.0
gc = Counter(gold[i]["category"] for i in both)
ic = Counter(ind[i].get("primary") for i in both)
pe = sum((gc[c] / len(both)) * (ic[c] / len(both)) for c in cats) if both else 0.0
kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

# ---- AdvRob demotion ----
demoted = [i for i in both if ind[i].get("primary") == ADV and gold[i]["category"] != ADV]
demoted_adv_sec = [i for i in demoted if gold[i].get("category_secondary") == ADV]

# ---- per-stored-category confirmation (lenient precision proxy) ----
by = defaultdict(lambda: [0, 0])
for i in both:
    by[gold[i]["category"]][1] += 1
    if lenient_match(i):
        by[gold[i]["category"]][0] += 1

# ---- disagreement pairs ----
dis = [i for i in both if gold[i]["category"] != ind[i].get("primary")]
pair_counts = Counter((gold[i]["category"], ind[i].get("primary")) for i in dis)

# ---- title-only slice ----
to_ids = [i for i in both if not gold[i]["has_abstract"]]
to_strict = sum(1 for i in to_ids if gold[i]["category"] == ind[i].get("primary"))

metrics = {
    "n_matched": n,
    "n_both_safe": len(both),
    "still_safety_pct": round(100 * still_safe / n, 1) if n else 0,
    "n_now_dropped": len(now_dropped),
    "primary_strict_pct": round(100 * po, 1),
    "primary_lenient_pct": round(100 * lenient / len(both), 1) if both else 0,
    "primary_kappa": round(kappa, 3),
    "advrob_demotions": len(demoted),
    "advrob_demotions_from_secondary": len(demoted_adv_sec),
    "n_primary_disagreements": len(dis),
    "title_only_n": len(to_ids),
    "title_only_strict_pct": round(100 * to_strict / len(to_ids), 1) if to_ids else 0,
}
json.dump(metrics, io.open(os.path.join(OUT, "_pool_metrics.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)


def agree_tag(i):
    if gold[i]["category"] == ind[i].get("primary"):
        return "strict"
    if lenient_match(i):
        return "lenient"
    return "disagree"


# ---- per-paper CSV ----
with io.open(os.path.join(OUT, "pool_summaries.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "title", "venue", "year", "has_abstract", "summary",
                "stored_primary", "stored_secondary", "fresh_primary",
                "fresh_secondary", "fresh_confidence", "primary_agree", "still_safety"])
    for i in sorted(ids, key=lambda i: int(i)):
        g, r = gold[i], ind[i]
        w.writerow([i, g["title"], g.get("venue"), g.get("year"),
                    g["has_abstract"], (r.get("summary") or "").replace("\n", " "),
                    g["category"], g.get("category_secondary"),
                    r.get("primary"), r.get("secondary"), r.get("confidence"),
                    agree_tag(i) if i in both else "now-dropped",
                    r.get("is_safety")])

# ---- per-paper markdown grouped by stored category ----
by_cat = defaultdict(list)
for i in ids:
    by_cat[gold[i]["category"]].append(i)
with io.open(os.path.join(OUT, "pool_summaries.md"), "w", encoding="utf-8") as f:
    f.write("# Pooled papers — per-paper summary & re-categorization\n\n")
    f.write(f"{n} pooled safety papers, grouped by **stored** primary category. "
            f"Each entry: summary (abstract-grounded) → stored vs. fresh label → agreement.\n\n")
    for c in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        f.write(f"## {c} ({len(by_cat[c])})\n\n")
        for i in sorted(by_cat[c], key=lambda i: int(i)):
            g, r = gold[i], ind[i]
            tag = agree_tag(i) if i in both else "NOW-DROPPED (fresh: not safety)"
            flag = "" if tag == "strict" else f"  **[{tag}]**"
            sec = f" / {r.get('secondary')}" if r.get("secondary") else ""
            gsec = f" / {g['category_secondary']}" if g.get("category_secondary") else ""
            abs_flag = "" if g["has_abstract"] else " _(title-only)_"
            f.write(f"- **[{i}] {g['title']}** ({g.get('venue')} {g.get('year')}){abs_flag}\n")
            f.write(f"  - {r.get('summary')}\n")
            f.write(f"  - stored: {g['category']}{gsec} → fresh: {r.get('primary')}{sec}{flag}\n")
        f.write("\n")

# ---- aggregate report ----
with io.open(os.path.join(OUT, "pool_audit_report.md"), "w", encoding="utf-8") as f:
    f.write("# Pooled-paper re-classification audit\n\n")
    f.write("Blind independent re-classification of every **pool-sourced safety paper** "
            "(`source=='pool' & is_safety`) vs. the stored labels, using the canonical "
            "taxonomy/prompt. Summaries and labels are **abstract-grounded** (the corpus "
            "stores abstracts, not full text).\n\n")
    f.write("## Headline\n\n")
    f.write(f"- Papers compared: **{n}** ({len(both)} still safety in both passes)\n")
    f.write(f"- Fresh pass still calls safety: **{metrics['still_safety_pct']}%** "
            f"({metrics['n_now_dropped']} would now be dropped)\n")
    f.write(f"- Primary strict agreement: **{metrics['primary_strict_pct']}%**; "
            f"lenient (stored primary in fresh primary|secondary): "
            f"**{metrics['primary_lenient_pct']}%**\n")
    f.write(f"- Cohen's kappa (primary, 16 cats): **{metrics['primary_kappa']}**\n")
    f.write(f"- AdvRob demotions (fresh AdvRob-primary, stored other): "
            f"**{metrics['advrob_demotions']}** "
            f"({metrics['advrob_demotions_from_secondary']} had AdvRob as stored secondary)\n")
    f.write(f"- Primary disagreements: **{metrics['n_primary_disagreements']}**\n\n")
    f.write("## Per-stored-category confirmation (lenient precision proxy)\n\n")
    f.write("| stored category | confirmed/total | % |\n|---|---|---|\n")
    for c in sorted(by, key=lambda c: -by[c][1]):
        ok, tot = by[c]
        f.write(f"| {c} | {ok}/{tot} | {100*ok/tot:.1f}% |\n")
    f.write("\n## Top disagreement flows (stored → fresh)\n\n")
    f.write("| stored primary | fresh primary | count |\n|---|---|---|\n")
    for (gp, fp), ct in pair_counts.most_common(25):
        f.write(f"| {gp} | {fp} | {ct} |\n")
    f.write("\n## Title-only subset (no abstract — lower-confidence)\n\n")
    f.write(f"- {metrics['title_only_n']} pooled papers are title-only; strict primary "
            f"agreement on them: **{metrics['title_only_strict_pct']}%** "
            f"(treat their labels/summaries as best-effort).\n\n")
    if now_dropped:
        f.write("## Papers the fresh pass would drop as NOT safety\n\n")
        for i in sorted(now_dropped, key=lambda i: int(i))[:60]:
            f.write(f"- [{i}] stored={gold[i]['category']} — {gold[i]['title']}\n")
    f.write("\n## Caveats\n\n")
    f.write("- Summaries reflect methods/conclusions **as stated in the abstract**, not full text.\n")
    f.write("- Single fresh annotator (one LLM pass); kappa is stored-vs-fresh, not human gold.\n")
    f.write(f"- Title-only papers ({metrics['title_only_n']}) classified from title alone.\n")

print(json.dumps(metrics, indent=2))
print("\nwrote: _pool_metrics.json, pool_summaries.csv, pool_summaries.md, pool_audit_report.md")
