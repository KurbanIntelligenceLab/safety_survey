#!/usr/bin/env python3
"""
Merge the two enriched pools (existing query-pool + snowball) into one stable-id
input for stage-4 LLM classification. Dedups by doi then title (snowball already
excluded papers present in the seed set, so overlap should be small). Assigns a
contiguous integer `id` used by classify_llm batching.

Writes paper_survey/classify_input.json with: id,title,abstract,venue,year,doi,source.
"""
import io, json, os

POOLS = [
    ("paper_survey/papers_enriched.json", "pool"),
    ("paper_survey/snowball_enriched.json", "snowball"),
]


def main():
    seen_doi, seen_title, merged = set(), set(), []
    for path, src in POOLS:
        if not os.path.exists(path):
            print(f"skip (missing): {path}"); continue
        recs = json.load(io.open(path, encoding="utf-8"))
        added = 0
        for p in recs:
            title = (p.get("title") or "").strip()
            doi = (p.get("doi") or "").lower()
            if not title:
                continue
            if doi and doi in seen_doi:
                continue
            if title.lower() in seen_title:
                continue
            if doi:
                seen_doi.add(doi)
            seen_title.add(title.lower())
            merged.append({
                "title": title,
                "abstract": p.get("abstract"),
                "venue": p.get("venue"),
                "year": p.get("year"),
                "doi": p.get("doi"),
                "source": src,
            })
            added += 1
        print(f"{src:8s} {path}: +{added} (total {len(merged)})")

    for i, p in enumerate(merged):
        p["id"] = i

    out = "paper_survey/classify_input.json"
    json.dump(merged, io.open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with_abs = sum(1 for p in merged if p.get("abstract"))
    print(f"\nmerged={len(merged)}  with_abstract={with_abs} "
          f"({100*with_abs/len(merged):.1f}%)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
