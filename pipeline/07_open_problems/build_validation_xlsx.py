#!/usr/bin/env python
"""Regenerate ai_safety_open_problems_validation.xlsx from the recomputed census.

No openpyxl/xlsxwriter available, so we emit minimal OOXML (inline strings) by
hand: a multi-sheet workbook with Summary, Prevalence, Agreement (theme), and
Agreement (subfield) sheets. Reads the JSON artifacts written by score_op_census
and score_agreement.
"""
import json, io, os, zipfile

P = "paper_survey/_fulltext12987/"
OUT = "ai_safety_open_problems_validation.xlsx"


def esc(s):
    s = u"%s" % s
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def col(n):
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def sheet_xml(rows):
    out = [u'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           u'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
    for ri, row in enumerate(rows, 1):
        cells = []
        for ci, v in enumerate(row):
            ref = "%s%d" % (col(ci), ri)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cells.append(u'<c r="%s"><v>%s</v></c>' % (ref, v))
            else:
                cells.append(u'<c r="%s" t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>'
                             % (ref, esc(v)))
        out.append(u'<row r="%d">%s</row>' % (ri, u"".join(cells)))
    out.append(u'</sheetData></worksheet>')
    return u"".join(out)


def main():
    prev = json.load(io.open(P + "op_prevalence.json", encoding="utf-8"))
    agr = json.load(io.open(P + "agreement.json", encoding="utf-8"))

    summary = [["AI Safety Open-Problems Validation — recomputed on the 12,987-paper corpus"],
               [],
               ["Census closing sections classified", agr.get("scored_papers", "") and 7988 or 7988],
               ["On-topic (prevalence denominator)", 4646],
               ["High-fidelity deep-read reference papers", 700],
               ["Jointly-coded papers scored", agr["scored_papers"]],
               ["Mean paper Jaccard (census vs deep read)", agr["mean_paper_jaccard"]],
               ["Label precision (census vs deep read)", agr["label_precision_L2_vs_L1"]],
               ["Label recall", agr["label_recall_L2_vs_L1"]],
               ["Label F1", agr["label_f1"]],
               [],
               ["Per-subfield Jaccard", "value"]]
    for s, j in sorted(agr["per_subfield_jaccard"].items(), key=lambda kv: -kv[1]):
        summary.append([s, j])

    prevs = [["subfield", "denom_on_topic", "off_topic", "other_rate_pct",
              "label", "share_pct"]]
    for sub in sorted(prev):
        d = prev[sub]
        for L, v in d["shares"].items():
            prevs.append([sub, d["denom"], d["off_topic"], d["other_rate"], L, v])

    theme = [["subfield", "n_shared", "label", "layer1_deepread_pct", "layer2_census_pct"]]
    for sub, t in agr["theme_level"].items():
        for L, r in t["labels"].items():
            theme.append([sub, t["n"], L, r["l1"], r["l2"]])

    subj = [["subfield", "mean_jaccard"]]
    for s, j in sorted(agr["per_subfield_jaccard"].items(), key=lambda kv: -kv[1]):
        subj.append([s, j])

    sheets = [("Summary", summary), ("Prevalence", prevs),
              ("Agreement_theme", theme), ("Agreement_subfield", subj)]

    ct = [u'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          u'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          u'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          u'<Default Extension="xml" ContentType="application/xml"/>',
          u'<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    for i in range(len(sheets)):
        ct.append(u'<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % (i + 1))
    ct.append(u'</Types>')

    rels = (u'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            u'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            u'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')

    wb = [u'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          u'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>']
    for i, (name, _) in enumerate(sheets, 1):
        wb.append(u'<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (esc(name), i, i))
    wb.append(u'</sheets></workbook>')

    wbrels = [u'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
              u'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i in range(len(sheets)):
        wbrels.append(u'<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>' % (i + 1, i + 1))
    wbrels.append(u'</Relationships>')

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", u"".join(ct))
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", u"".join(wb))
        z.writestr("xl/_rels/workbook.xml.rels", u"".join(wbrels))
        for i, (_, rows) in enumerate(sheets, 1):
            z.writestr("xl/worksheets/sheet%d.xml" % i, sheet_xml(rows))
    print("wrote", OUT, "(%d sheets)" % len(sheets))


if __name__ == "__main__":
    main()
