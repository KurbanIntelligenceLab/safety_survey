#!/usr/bin/env python
"""Score the Layer-2 open-problem census -> per-subfield prevalence.

For each subfield: denominator = classified papers with usable closing text and
not off_topic. For each label: share = (# papers assigned that label) / denom.
Also reports the 'other' rate and the leading (most prevalent) open problem.

  python score_op_census.py            # writes op_prevalence.json + prints table
"""
import json, io, os, collections

PS = "paper_survey"
ROOT = os.path.join(PS, "_fulltext12987")
RESULTS = os.path.join(ROOT, "op_results.json")
LABELS = os.path.join(ROOT, "open_problem_labels.json")
OUT = os.path.join(ROOT, "op_prevalence.json")


def main():
    labels = json.load(io.open(LABELS, encoding="utf-8"))
    merged = json.load(io.open(RESULTS, encoding="utf-8"))

    per = collections.defaultdict(lambda: {"denom": 0, "off_topic": 0, "other": 0,
                                           "lab": collections.Counter()})
    seen = set()
    for batch in merged:
        sub = batch["subfield"]
        for r in batch["results"]:
            pid = r.get("id")
            if pid in seen:
                continue
            seen.add(pid)
            d = per[sub]
            if r.get("off_topic"):
                d["off_topic"] += 1
                continue
            d["denom"] += 1
            if r.get("other"):
                d["other"] += 1
            valid = set(labels.get(sub, {}))
            for L in set(r.get("labels") or []):
                if L in valid:
                    d["lab"][L] += 1

    table = {}
    print("%-32s %5s %5s  leading open problem (share)" % ("subfield", "N", "off"))
    for sub in sorted(per):
        d = per[sub]
        n = max(1, d["denom"])
        shares = {L: round(100 * c / n, 1) for L, c in d["lab"].items()}
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:1]
        topstr = ("%s %.1f%%" % (top[0][0], top[0][1])) if top else "-"
        table[sub] = {"denom": d["denom"], "off_topic": d["off_topic"],
                      "other_rate": round(100 * d["other"] / n, 1),
                      "shares": dict(sorted(shares.items(), key=lambda kv: -kv[1]))}
        print("%-32s %5d %5d  %s" % (sub[:32], d["denom"], d["off_topic"], topstr))

    json.dump(table, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tot = sum(d["denom"] for d in per.values())
    print("\ntotal classified (not off_topic): %d   wrote %s" % (tot, OUT))


if __name__ == "__main__":
    main()
