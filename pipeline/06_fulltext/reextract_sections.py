#!/usr/bin/env python
"""Re-extract Method/Conclusion sections from ALREADY-CACHED PDFs.

The original fetch_fulltext.py downloaded PDFs fine but used a narrow heading
regex, so 722 PDFs got no conclusion and 1,051 got no method even though the
files are on disk. This pass re-runs a broadened, full-line-anchored heading
matcher (plus sorted reading order for two-column layouts) over the cached PDFs
for records still missing a section. It NEVER re-downloads and NEVER overwrites
a section that was already extracted -- it only fills gaps.

Usage:
  reextract_sections.py --selftest   # run regex unit tests, no I/O
  reextract_sections.py --dry        # report how many gaps would be filled
  reextract_sections.py              # update section JSONs + coverage in place
"""
import json, io, os, glob, re, sys, argparse
from collections import Counter
import fitz  # PyMuPDF

SEC = "paper_survey/_fulltext/sections"
PDF = "paper_survey/_fulltext/pdf"
REPORT = "paper_survey/_fulltext/coverage.json"

NUM = r"(\d{1,2}(\.\d+)*\.?\s+)?"          # optional section number: "5.", "6.2 "
TAIL = r"[a-z &,/'-]*"                      # trailing words (re.I -> also A-Z)
END = r"[:.\s]*$"

# Full-line conclusion/closing-section headings. Anchored start+end so inline
# prose ("Finally, we show...", "concludes the proof", "Remark 2") can't match.
CONC = re.compile(
    r"^\s*" + NUM + r"("
    r"conclusion(s)?" + TAIL +            # Conclusion, Conclusions and Future Work
    r"|concluding remarks?" + TAIL +
    r"|discussion(s)?" + TAIL +           # Discussion, Discussion & Conclusion, ...
    r"|summary( and (conclusion|discussion)s?)?" +  # Summary, Summary and Conclusion
    r"|(final|closing) remarks?" + TAIL +
    r"|future (work|directions?)" + TAIL +
    r"|broader impacts?( and (limitation|conclusion)s?)?" +
    r"|limitations? and (conclusion|future work|discussion)s?" +
    r")" + END, re.I)

# Full-line method/approach headings. Bare "Model" allowed only as a standalone
# heading word (optionally numbered) -- table headers like "model: mlp ..." and
# "model / attack" are rejected by the end anchor.
METH = re.compile(
    r"^\s*" + NUM + r"("
    r"method(s|ology)?" + TAIL +
    r"|approach(es)?" + TAIL +
    r"|(proposed|our) (method|approach|model|framework|technique|architecture|design)" + TAIL +
    r"|(model|system|network)? ?architecture" + TAIL +
    r"|preliminar(y|ies)" + TAIL +
    r"|problem (formulation|setup|definition|statement)" + TAIL +
    r"|(experimental |our )?(setup|formulation|design)" + TAIL +
    r"|algorithms?" +                      # "Algorithm", "Algorithms" (not "Algorithm 1 ...")
    r"|model(s)?" +                         # bare "Model"/"Models" heading only
    r"|our contributions?" +
    r")" + END, re.I)


def extract(txt):
    lines = [ln.strip() for ln in txt.split("\n")]
    def grab(startre):
        for i, ln in enumerate(lines):
            if len(ln) < 55 and startre.match(ln):
                buf = []
                for ln2 in lines[i + 1:i + 220]:
                    # stop at the next short heading-ish line once we have content
                    if len(buf) > 8 and ln2 and len(ln2) < 55 and re.match(
                            r"^\s*(\d{1,2}(\.\d+)*\.?\s+)?[A-Z][A-Za-z ]{2,40}[:.]?\s*$", ln2):
                        break
                    if ln2:
                        buf.append(ln2)
                if len(" ".join(buf)) > 120:   # require a real paragraph, not a stray match
                    return " ".join(buf).strip()
        return ""
    return grab(METH), grab(CONC)


def selftest():
    catch_c = ["Discussion", "6. Discussion", "5. Discussion and Future Work",
               "Discussion & Conclusion", "Discussion and Conclusions", "Summary",
               "Summary:", "Discussion, Limitations, and Future Work",
               "Discussion, Broader Impacts, and Limitations", "Future Directions",
               "7. Discussion and Concluding Remarks", "Conclusion", "Conclusions",
               "Discussion and broader outlook", "Final Remarks", "Closing Remarks"]
    rej_c = ["Final answer: \\boxed{19}.", "concludes the proof.", "Remark 2. d",
             "Finally, we study the generalization", "discuss the causes of the great depression",
             "Summary of baseline performances", "discussion of performance of (bose and",
             "summarized by the average of the losses:", "remark. as long as"]
    catch_m = ["Model", "Models", "Preliminaries", "Algorithm", "Architecture",
               "Our Contributions", "3. Method", "Methodology", "Proposed Approach",
               "Model Architecture", "Problem Formulation", "Experimental Setup"]
    rej_m = ["model: mlp   dataset: mnist", "model / attack", "Algorithm 1 GOOSE",
             "models under different training settings", "Proposition 1 (chain rule)",
             "model was in fact trained on", "Model type for each case"]
    ok = True
    for label, pat, items, want in [("CONC+", CONC, catch_c, True), ("CONC-", CONC, rej_c, False),
                                    ("METH+", METH, catch_m, True), ("METH-", METH, rej_m, False)]:
        for s in items:
            got = bool(pat.match(s.strip()))
            if got != want:
                ok = False
                print("  FAIL [%s] want=%s got=%s  %r" % (label, want, got, s))
    print("SELFTEST", "PASS" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not selftest():
        print("Refusing to run: selftest failed."); sys.exit(1)

    files = glob.glob(SEC + "/*.json")
    recs = {f: json.load(io.open(f, encoding="utf-8")) for f in files}
    targets = [(f, r) for f, r in recs.items()
               if r.get("ok") and (not r.get("has_conclusion") or not r.get("has_method"))]
    print("scanning %d cached PDFs with a missing section..." % len(targets))

    filled_c = filled_m = errs = 0
    for k, (f, r) in enumerate(targets):
        pdf = os.path.join(PDF, "%s.pdf" % r["id"])
        if not os.path.exists(pdf):
            continue
        try:
            doc = fitz.open(pdf)
            txt = "\n".join(pg.get_text(sort=True) for pg in doc)  # sorted reading order
            doc.close()
        except Exception:
            errs += 1
            continue
        meth, conc = extract(txt)
        changed = False
        if conc and not r.get("has_conclusion"):
            r["has_conclusion"] = True; r["conclusion"] = conc; filled_c += 1; changed = True
        if meth and not r.get("has_method"):
            r["has_method"] = True; r["method"] = meth; filled_m += 1; changed = True
        if changed and not args.dry:
            r["reextract"] = True
            json.dump(r, io.open(f, "w", encoding="utf-8"), ensure_ascii=False)
        if (k + 1) % 200 == 0:
            print("  [%d/%d] +conc=%d +meth=%d" % (k + 1, len(targets), filled_c, filled_m))

    print("\n== re-extract %s ==" % ("DRY" if args.dry else "APPLIED"))
    print("newly filled conclusion: %d" % filled_c)
    print("newly filled method:     %d" % filled_m)
    print("parse errors:            %d" % errs)

    # recompute coverage over the full set
    allrecs = list(recs.values())
    n = len(allrecs); got = sum(1 for r in allrecs if r.get("ok"))
    meth = sum(1 for r in allrecs if r.get("has_method"))
    conc = sum(1 for r in allrecs if r.get("has_conclusion"))
    both = sum(1 for r in allrecs if r.get("has_method") and r.get("has_conclusion"))
    print("\nNEW coverage: pdf=%d/%d  method=%d (%.0f%%)  conclusion=%d (%.0f%%)  both=%d (%.0f%%)"
          % (got, n, meth, 100*meth/got, conc, 100*conc/got, both, 100*both/got))
    if not args.dry:
        rep = json.load(io.open(REPORT, encoding="utf-8"))
        rep.update(has_method=meth, has_conclusion=conc, has_both=both, reextracted=True)
        json.dump(rep, io.open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
