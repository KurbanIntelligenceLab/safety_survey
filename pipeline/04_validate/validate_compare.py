#!/usr/bin/env python3
"""
Compare the independent (blind) re-classification against the stored labels to
quantify taxonomy reliability. Run AFTER the safety-label-validation workflow
returns; write its `labels` array to paper_survey/_val_independent.json first.

Reports:
  - is_safety agreement + safety-gate recall on the 40 known-dropped papers
  - primary-category raw agreement + Cohen's kappa (papers both call safety)
  - lenient agreement (independent primary OR secondary == stored primary)
  - the adversarial-robustness demotion signal
  - per-stored-category confirmation rate (a precision proxy)
  - a list of primary disagreements for eyeballing
"""
import io, json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.join(HERE, "..", "..", "paper_survey")
gold = json.load(io.open(os.path.join(PS, "_val_gold.json"), encoding="utf-8"))
ind_raw = json.load(io.open(os.path.join(PS, "_val_independent.json"), encoding="utf-8"))
ind = {str(r["id"]): r for r in ind_raw}

ids = [i for i in gold if i in ind]
print(f"matched {len(ids)} papers (gold={len(gold)}, independent={len(ind)})")

# ---- is_safety agreement ----
both_safe = sum(1 for i in ids if gold[i]["is_safety"] and ind[i]["is_safety"])
gsafe = sum(1 for i in ids if gold[i]["is_safety"])
agree_safety = sum(1 for i in ids if gold[i]["is_safety"] == ind[i]["is_safety"])
# recall of the safety gate: of the 40 stored-DROPPED, how many independent also drops
drop_ids = [i for i in ids if not gold[i]["is_safety"]]
indep_also_drops = sum(1 for i in drop_ids if not ind[i]["is_safety"])
# of stored-KEPT, how many independent also keeps
keep_ids = [i for i in ids if gold[i]["is_safety"]]
indep_also_keeps = sum(1 for i in keep_ids if ind[i]["is_safety"])

print("\n=== is_safety gate ===")
print(f"  overall agreement : {agree_safety}/{len(ids)} = {100*agree_safety/len(ids):.1f}%")
print(f"  stored-KEPT  also kept by independent : {indep_also_keeps}/{len(keep_ids)} = {100*indep_also_keeps/len(keep_ids):.1f}%")
print(f"  stored-DROP  also dropped by independent: {indep_also_drops}/{len(drop_ids)} = {100*indep_also_drops/len(drop_ids):.1f}%")
print(f"  -> {len(keep_ids)-indep_also_keeps} stored-kept now flagged NOT-safety; "
      f"{len(drop_ids)-indep_also_drops} stored-dropped now flagged safety (possible false negatives)")

# ---- primary-category agreement (papers BOTH call safety) ----
both = [i for i in ids if gold[i]["is_safety"] and ind[i]["is_safety"]]
strict = sum(1 for i in both if gold[i]["category"] == ind[i].get("primary"))
def lenient_match(i):
    g = gold[i]["category"]
    return g in (ind[i].get("primary"), ind[i].get("secondary"))
lenient = sum(1 for i in both if lenient_match(i))

# Cohen's kappa on primary over the 16 categories (both-safe subset)
cats = sorted({gold[i]["category"] for i in both} | {ind[i].get("primary") for i in both if ind[i].get("primary")})
po = strict / len(both)
gc = Counter(gold[i]["category"] for i in both)
ic = Counter(ind[i].get("primary") for i in both)
pe = sum((gc[c]/len(both)) * (ic[c]/len(both)) for c in cats)
kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

print("\n=== primary-category (both-safe subset, n=%d) ===" % len(both))
print(f"  strict agreement (primary==primary) : {strict}/{len(both)} = {100*po:.1f}%")
print(f"  lenient (stored primary in indep primary|secondary): {lenient}/{len(both)} = {100*lenient/len(both):.1f}%")
print(f"  Cohen's kappa (primary)             : {kappa:.3f}")

# ---- adversarial-robustness signal ----
ADV = "Adversarial Robustness"
# papers independent calls AdvRob-primary but stored gave a different primary (AdvRob demoted in stored)
demoted = [i for i in both if ind[i].get("primary") == ADV and gold[i]["category"] != ADV]
demoted_adv_secondary = [i for i in demoted if gold[i].get("category_secondary") == ADV]
print("\n=== adversarial-robustness demotion check ===")
print(f"  independent says AdvRob-primary but stored primary differs: {len(demoted)}")
print(f"     ...of which stored had AdvRob as secondary (demotion):   {len(demoted_adv_secondary)}")
if demoted:
    print("     stored primaries that borrowed AdvRob papers:",
          dict(Counter(gold[i]["category"] for i in demoted)))

# ---- per-stored-category confirmation (precision proxy) ----
print("\n=== per-stored-category confirmation (independent primary OR secondary == stored) ===")
by = defaultdict(lambda: [0, 0])
for i in both:
    by[gold[i]["category"]][1] += 1
    if lenient_match(i):
        by[gold[i]["category"]][0] += 1
for c in sorted(by, key=lambda c: -by[c][1]):
    ok, n = by[c]
    print(f"  {ok:3d}/{n:3d}  {100*ok/n:5.1f}%  {c}")

# ---- disagreement list ----
print("\n=== primary DISAGREEMENTS (stored vs independent), both-safe ===")
dis = [i for i in both if gold[i]["category"] != ind[i].get("primary")]
for i in sorted(dis, key=lambda i: gold[i]["category"]):
    star = "" if lenient_match(i) else "  <-- not even secondary match"
    print(f"  [{i}] stored={gold[i]['category']:34s} indep={ind[i].get('primary')}"
          f" (2nd={ind[i].get('secondary')}){star}")

out = {
    "n_matched": len(ids),
    "is_safety_agreement_pct": round(100*agree_safety/len(ids), 1),
    "kept_confirmed_pct": round(100*indep_also_keeps/len(keep_ids), 1),
    "dropped_confirmed_pct": round(100*indep_also_drops/len(drop_ids), 1),
    "primary_strict_pct": round(100*po, 1),
    "primary_lenient_pct": round(100*lenient/len(both), 1),
    "primary_kappa": round(kappa, 3),
    "advrob_demotions": len(demoted),
    "n_primary_disagreements": len(dis),
}
json.dump(out, io.open(os.path.join(PS, "_val_metrics.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nwrote paper_survey/_val_metrics.json")
print(json.dumps(out, indent=2))
