#!/usr/bin/env python3
"""Deduplicate the census safety corpus before merging with the old corpus.

screen_census.py deduped the census against papers_llm_13cat.json by normalized
title only, WITHOUT html.unescape -- so census rows whose raw titles carried HTML
entities (&quot; &apos; &amp;) failed to match their clean old-corpus twins and
leaked through. This pass removes, from census_gated.json's is_safety survivors:
  (a) cross dups: any paper already in the old corpus by unescaped-title OR doi
      OR s2_id, and
  (b) within-census dups: repeated unescaped-title / doi / s2_id (keep first).
Writes the deduped survivors + a dropped-ids report. Non-destructive.

  python3 pipeline/01_collect/dedup_census.py
"""
import io, json, re, html
from collections import defaultdict

OLD = "paper_survey/papers_llm_13cat.json"
NEW = "paper_survey/_census/census_gated.json"
OUT = "paper_survey/_census/census_gated_dedup.json"
REPORT = "paper_survey/_census/dedup_report.json"


def L(p):
    return json.load(io.open(p, encoding="utf-8"))


def nt(t):
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(t or "").lower()).strip()


def nd(d):
    return ((d or "").lower().strip()) or None


def main():
    old = [p for p in L(OLD) if p.get("is_safety")]
    new = [p for p in L(NEW) if p.get("is_safety")]

    old_t = {nt(p.get("title")) for p in old if nt(p.get("title"))}
    old_d = {nd(p.get("doi")) for p in old if nd(p.get("doi"))}
    old_s = {p.get("s2_id") for p in old if p.get("s2_id")}

    kept, drop_cross, drop_within = [], [], []
    seen_t, seen_d, seen_s = set(), set(), set()
    for p in new:
        t, d, s = nt(p.get("title")), nd(p.get("doi")), p.get("s2_id")
        if (t and t in old_t) or (d and d in old_d) or (s and s in old_s):
            drop_cross.append(p["id"]); continue
        if (t and t in seen_t) or (d and d in seen_d) or (s and s in seen_s):
            drop_within.append(p["id"]); continue
        if t: seen_t.add(t)
        if d: seen_d.add(d)
        if s: seen_s.add(s)
        kept.append(p)

    json.dump(kept, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    rep = {"new_in": len(new), "kept": len(kept),
           "dropped_cross_old": sorted(drop_cross),
           "dropped_within_census": sorted(drop_within)}
    json.dump(rep, io.open(REPORT, "w", encoding="utf-8"), indent=1)

    print(f"new safety in : {len(new)}")
    print(f"dropped cross-old corpus : {len(drop_cross)}")
    print(f"dropped within-census    : {len(drop_within)}")
    print(f"clean census kept        : {len(kept)}")
    print(f"wrote {OUT}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
