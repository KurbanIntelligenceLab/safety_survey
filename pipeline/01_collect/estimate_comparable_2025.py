#!/usr/bin/env python3
"""
Estimate an apples-to-apples 2025 count: the original 2018-2024 pool was a DBLP
TITLE-phrase search (low recall); the 2025 backfill is a full-proceedings
title+abstract census (high recall). To compare like with like, re-filter the
2025 backfill safety papers down to those whose TITLE contains one of the
original DBLP query phrases, approximating what the original method would have
surfaced in 2025.
"""
import io, json, re, sys
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from scrape_ai_safety_papers import QUERIES

def L(f): return json.load(io.open(f, encoding="utf-8"))

# DBLP matches word-ish; approximate with phrase substring on normalized title.
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
PHRASES = [norm(q) for q in QUERIES]
# also split multiword queries: DBLP ANDs the words, so a title matches a query
# if it contains ALL words of that query (order-free) -> higher recall than phrase.
QWORDS = [set(norm(q).split()) for q in QUERIES]

def title_matches(title):
    nt = set(norm(title).split())
    ntext = norm(title)
    # phrase hit (strict) OR all-words hit (DBLP AND semantics, looser)
    if any(p in ntext for p in PHRASES): return True
    if any(w and w <= nt for w in QWORDS): return True
    return False

aug = L("paper_survey/papers_llm_2025augmented.json")
backfill_safety = [p for p in aug if p.get("backfill") and p.get("is_safety")]
print(f"backfill safety papers (full census): {len(backfill_safety)}")

matched = [p for p in backfill_safety if title_matches(p["title"])]
print(f"  of which title matches original DBLP queries: {len(matched)} "
      f"({100*len(matched)/len(backfill_safety):.0f}%)")

from collections import Counter
print("\nfull-census backfill safety by category:")
for c, k in Counter(p["category"] for p in backfill_safety).most_common():
    m = sum(1 for p in matched if p["category"] == c)
    print(f"  {c:>38}: census {k:>4}   DBLP-comparable {m:>4}")

# What the pool 2025 totals/fairness would be under each regime
old = L("paper_survey/paper_stats.json")
base_2025 = int(old["pool"]["year_totals"]["2025"])          # 873 (NeurIPS=0)
base_fair = int(old["pool"]["category_x_year"]["Fairness & Bias"]["2025"])  # 19
add_census = len(backfill_safety)
add_cmp = len(matched)
fair_census = sum(1 for p in backfill_safety if p["category"] == "Fairness & Bias")
fair_cmp = sum(1 for p in matched if p["category"] == "Fairness & Bias")

print("\n=== POOL 2025 TOTAL under three regimes ===")
print(f"  (A) original, NeurIPS=0      : {base_2025}")
print(f"  (B) full-census backfill     : {base_2025 + add_census}")
print(f"  (C) DBLP-comparable backfill : {base_2025 + add_cmp}")

print("\n=== POOL Fairness 2025 + 2023->2025 growth ===")
f2023 = int(old["pool"]["category_x_year"]["Fairness & Bias"]["2023"])  # 25
for tag, f2025 in [("A original", base_fair),
                   ("B census", base_fair + fair_census),
                   ("C DBLP-comparable", base_fair + fair_cmp)]:
    pct = round(100*(f2025 - f2023)/f2023, 1)
    print(f"  {tag:>20}: 2025={f2025:>4}  2023->2025 = {pct:+.1f}%")
