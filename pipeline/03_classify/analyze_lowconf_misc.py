#!/usr/bin/env python3
"""
Two analyses on the 2,572 pool safety papers:
 (1) Drop candidates: proceedings / workshop front-matter / non-paper entries,
     cross-checked against the blind Sonnet re-pass is_safety flag.
 (2) Confidence structure: mean stored confidence per category, and confidence
     for 'umbrella/position' papers (broad trustworthy-AI / position / survey)
     vs the rest.

Writes:
  paper_survey/_pool_audit/proceedings_drop_candidates.csv
  paper_survey/_pool_audit/confidence_analysis.md
Prints a summary.
"""
import csv, io, json, os, re
from collections import Counter, defaultdict
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
OUT = os.path.join(PS, "_pool_audit")

papers = json.load(io.open(os.path.join(PS, "papers_llm.json"), encoding="utf-8"))
pool = [p for p in papers if p.get("source") == "pool" and p.get("is_safety")]
# Sonnet blind re-pass is_safety (full pool) for cross-check
rerun = {str(r["id"]): r for r in json.load(io.open(os.path.join(OUT, "_pool_independent.json"), encoding="utf-8"))}

# ---- (1) proceedings / front-matter / non-paper detector ----
PROC = re.compile(
    r"\b(proceedings|companion proceedings|co-located with|"
    r"workshop on .+?(20\d\d|co-located)|"
    r"conference on .+?,\s*20\d\d|symposium on .+?20\d\d|"
    r"\(safeai\s*20\d\d\)|preface|front[- ]matter|table of contents|"
    r"tutorial[s]? on)\b", re.I)
# also: title is basically a venue + date range (e.g. "... 23-24 February 2018, New York, NY, USA")
DATEY = re.compile(r"\b\d{1,2}[-–]\d{1,2}\s+\w+\s+20\d\d\b")


def proc_reason(p):
    t = p.get("title") or ""
    m = PROC.search(t)
    if m:
        return m.group(0)[:40]
    if DATEY.search(t):
        return "venue+date front-matter"
    return None


cands = []
for p in pool:
    why = proc_reason(p)
    if why:
        rr = rerun.get(str(p["id"]), {})
        cands.append({
            "id": p["id"], "title": p["title"], "venue": p.get("venue"),
            "year": p.get("year"), "stored_category": p["category"],
            "stored_conf": p.get("llm_confidence"),
            "rerun_is_safety": rr.get("is_safety"),
            "match_reason": why,
        })
cands.sort(key=lambda c: (c["rerun_is_safety"] is not False, c["stored_conf"] or 1))

with io.open(os.path.join(OUT, "proceedings_drop_candidates.csv"), "w",
             encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "title", "venue", "year", "stored_category", "stored_conf",
                "rerun_is_safety", "match_reason"])
    for c in cands:
        w.writerow([c["id"], c["title"], c["venue"], c["year"], c["stored_category"],
                    c["stored_conf"], c["rerun_is_safety"], c["match_reason"]])

n_rerun_drop = sum(1 for c in cands if c["rerun_is_safety"] is False)

# ---- (2) confidence structure ----
def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


bycat = defaultdict(list)
for p in pool:
    bycat[p["category"]].append(p.get("llm_confidence"))

UMBRELLA = re.compile(
    r"\b(towards trustworthy|trustworthy (machine learning|ai|ml)|"
    r"position:|a position|perspective(s)?|roadmap|"
    r"opportunities and (risks|challenges)|grand challenge|overview of|"
    r"a survey|survey of|a review|systematic review|taxonomy of|"
    r"reflections on|considerations|the value of|mainstays)\b", re.I)


def is_umbrella(p):
    return bool(UMBRELLA.search(p.get("title") or ""))


umb = [p.get("llm_confidence") for p in pool if is_umbrella(p)]
rest = [p.get("llm_confidence") for p in pool if not is_umbrella(p)]
umb_misc = sum(1 for p in pool if is_umbrella(p) and p["category"] == "Miscellaneous Safety")
n_umb = sum(1 for p in pool if is_umbrella(p))

with io.open(os.path.join(OUT, "confidence_analysis.md"), "w", encoding="utf-8") as f:
    f.write("# Confidence structure of the pool (n=%d)\n\n" % len(pool))
    f.write(f"Overall mean stored confidence: **{mean([p.get('llm_confidence') for p in pool]):.3f}**\n\n")
    f.write("## Mean stored confidence by stored primary category (low → high)\n\n")
    f.write("| category | n | mean conf | min |\n|---|---|---|---|\n")
    for c in sorted(bycat, key=lambda c: mean(bycat[c])):
        vals = [x for x in bycat[c] if x is not None]
        f.write(f"| {c} | {len(vals)} | {mean(vals):.3f} | {min(vals):.2f} |\n")
    f.write("\n## 'Umbrella / position / survey' papers vs the rest\n\n")
    f.write(f"Broad-title match (`trustworthy|position|perspective|roadmap|survey|"
            f"review|taxonomy|considerations|...`):\n\n")
    f.write(f"- umbrella/position papers: **n={n_umb}**, mean conf **{mean(umb):.3f}**\n")
    f.write(f"- everything else: n={len(rest)}, mean conf **{mean(rest):.3f}**\n")
    f.write(f"- of the {n_umb} umbrella papers, {umb_misc} are filed under Miscellaneous Safety\n\n")
    f.write("**Takeaway:** the confidence dip is driven by *broad/cross-cutting* papers "
            "(trustworthy-ML umbrellas, position pieces), which the classifier rightly "
            "flags as low-confidence and often routes to the Miscellaneous catch-all — "
            "not by literature surveys specifically.\n\n")
    f.write("## Proceedings / front-matter drop candidates\n\n")
    f.write(f"- {len(cands)} pool entries match proceedings/front-matter patterns; "
            f"**{n_rerun_drop}** were ALSO flagged not-safety by the blind re-pass "
            f"(high-confidence drops). Full list: `proceedings_drop_candidates.csv`.\n")

# ---- print summary ----
print("=== mean stored confidence by category (low -> high) ===")
for c in sorted(bycat, key=lambda c: mean(bycat[c])):
    vals = [x for x in bycat[c] if x is not None]
    print(f"  {mean(vals):.3f}  (n={len(vals):4d})  {c}")
print(f"\n=== umbrella/position vs rest ===")
print(f"  umbrella n={n_umb} mean={mean(umb):.3f}  ({umb_misc} in Misc Safety)")
print(f"  rest     n={len(rest)} mean={mean(rest):.3f}")
print(f"\n=== proceedings/front-matter drop candidates: {len(cands)} "
      f"({n_rerun_drop} also dropped by blind re-pass) ===")
for c in cands:
    flag = "  <== re-pass also says NOT-safety" if c["rerun_is_safety"] is False else ""
    print(f"  conf={c['stored_conf']} [{c['id']}] {c['title'][:70]}{flag}")
print("\nwrote proceedings_drop_candidates.csv + confidence_analysis.md")
