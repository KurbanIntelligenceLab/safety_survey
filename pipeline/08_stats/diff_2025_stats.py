#!/usr/bin/env python3
"""Before/after diff of every 2025-relevant number, old vs 2025-augmented stats."""
import io, json

def L(f): return json.load(io.open(f, encoding="utf-8"))
old = L("paper_survey/paper_stats.json")
new = L("paper_survey/paper_stats_2025augmented.json")

def table(title, oy, ny):
    yrs = sorted(set(oy) | set(ny), key=int)
    print(f"\n=== {title} ===")
    print(f"{'year':>6} {'OLD':>7} {'NEW':>7} {'delta':>7}")
    for y in yrs:
        o, n = oy.get(y, 0), ny.get(y, 0)
        mark = "  <-- 2025" if y == "2025" else ""
        print(f"{y:>6} {o:>7} {n:>7} {n-o:>+7}{mark}")

print(f"corpus all-classified: {old['n_all_classified']} -> {new['n_all_classified']}")
print(f"kept safety          : {old['n_kept_safety']} -> {new['n_kept_safety']}")
print(f"pool n               : {old['pool']['n']} -> {new['pool']['n']}")

table("POOL year_totals (recency-complete headline series)",
      old["pool"]["year_totals"], new["pool"]["year_totals"])

for cat in ["Fairness & Bias", "Adversarial Robustness", "Privacy & Data Protection",
            "Alignment & RLHF", "Out-of-Distribution & Uncertainty",
            "AI Governance & Policy", "Interpretability & Explainability"]:
    table(f"POOL {cat} by year",
          old["pool"]["category_x_year"].get(cat, {}),
          new["pool"]["category_x_year"].get(cat, {}))

# growth 2023->2025 (pool)
print("\n=== POOL growth 2023->2025 (pct) OLD -> NEW ===")
og, ng = old["pool"]["growth_2023_2025"], new["pool"]["growth_2023_2025"]
for c in sorted(set(og) | set(ng)):
    o, n = og.get(c, {}), ng.get(c, {})
    print(f"{c:>38}: 2025 {o.get('y2025')}->{n.get('y2025')}  pct {o.get('pct')}->{n.get('pct')}")
