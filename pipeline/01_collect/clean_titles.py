#!/usr/bin/env python3
"""Unescape HTML entities baked into stored paper titles.

dedup_census.py only unescaped titles for *comparison*; the raw `title` strings
still carry entities (&quot; &apos; &amp; ...). Anything that renders a title for
a human (reference lists, LaTeX tables, figure captions) would print the literal
entity. This pass rewrites the `title` field in place via html.unescape, after
backing up each file. Idempotent: html.unescape is a no-op on already-clean text.

  python3 pipeline/01_collect/clean_titles.py
"""
import io, json, re, html, shutil

FILES = [
    "paper_survey/papers_llm_13cat.json",
    "paper_survey/_census/census_gated_dedup.json",
]
ENT = re.compile(r"&(?:amp|quot|apos|lt|gt|#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")


def L(p):
    return json.load(io.open(p, encoding="utf-8"))


def main():
    for f in FILES:
        data = L(f)
        changed = []
        for p in data:
            t = p.get("title")
            if not t:
                continue
            u = html.unescape(t)
            if u != t:
                changed.append((t, u))
                p["title"] = u
        if changed:
            shutil.copy2(f, f + ".bak_entities")
            json.dump(data, io.open(f, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        # sanity: nothing left that looks like an entity
        leftover = sum(1 for p in data if p.get("title") and ENT.search(p["title"]))
        print(f"{f}: cleaned {len(changed)} titles, {leftover} entity-like remaining"
              + (f"  (backup: {f}.bak_entities)" if changed else "  (no change)"))
        for old, neu in changed[:3]:
            print(f"    - {old!r}\n    + {neu!r}")


if __name__ == "__main__":
    main()
