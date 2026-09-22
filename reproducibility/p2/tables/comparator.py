#!/usr/bin/env python3
"""Value comparator at manuscript displayed precision.

Compares the regenerated tables of one appendix against the manuscript values
extracted into ``expected.json``.  A cell is compared numerically when both sides
contain the same number of numeric tokens (tolerance 5e-4 relative, 1e-12
absolute) and as normalised text otherwise.  ``None`` cells are reported as
explicit gaps.
"""
import json
import os
import re

NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
DASHES = {"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2264": "", "\u2026": "..."}


def _prep(s):
    if s is None:
        return None
    for a, b in DASHES.items():
        s = s.replace(a, b)
    s = s.replace("`", "").replace("*", "")
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)   # 18 320 -> 18320
    return s


def _nums(s):
    return [float(x) for x in NUM.findall(s)]


def _text(s):
    t = NUM.sub(" ", s)
    t = re.sub(r"[^A-Za-z\u00b7\u207b\u00b9%/]+", " ", t)
    return " ".join(t.lower().split())


def _close(a, b):
    if a == b:
        return True
    if not (abs(a) < float("inf") and abs(b) < float("inf")):
        return False
    # Relative-only tolerance (tiny absolute floor for exact-zero handling).  A
    # 1e-12 absolute floor would mask mutations of the small residual cells
    # (1e-13..1e-17) and make the comparator blind to real source changes.
    return abs(a - b) <= 5e-4 * max(abs(a), abs(b)) + 1e-300


def compare_cell(exp, got):
    if got is None:
        return "gap"
    e, g = _prep(exp), _prep(got)
    en, gn = _nums(e), _nums(g)
    if en and gn and len(en) == len(gn) and all(_close(a, b) for a, b in zip(en, gn)):
        if _text(e) == _text(g):
            return "ok"
        # numbers agree; text differs (units/labels) -> still a mismatch
        return "text"
    if _text(e) == _text(g) and not en and not gn:
        return "ok"
    return "value"


def run(tid, tables, gaps, table_dir, declared=None):
    with open(os.path.join(table_dir, "expected.json"), encoding="utf-8") as f:
        exp = json.load(f)["manuscript_tables"]
    declared = {(ti, ri, ci) for ti, ri, ci, _r in (declared or [])}
    mismatches = []
    checked = 0
    gap_cells = 0
    for ti, exp_table in enumerate(exp):
        got_table = tables[ti] if ti < len(tables) else []
        exp_rows = exp_table[1:]  # drop header
        if len(exp_rows) != len(got_table):
            mismatches.append({"table": ti, "row": "*",
                               "detail": "row count expected=%d got=%d"
                               % (len(exp_rows), len(got_table))})
        for ri, er in enumerate(exp_rows):
            gr = got_table[ri] if ri < len(got_table) else []
            for ci, ecell in enumerate(er):
                if ci >= len(gr):
                    mismatches.append({"table": ti, "row": ri, "col": ci,
                                       "detail": "missing cell"})
                    continue
                if (ti, ri, ci) in declared:
                    gap_cells += 1
                    checked += 1
                    continue
                status = compare_cell(ecell, gr[ci])
                if status == "gap":
                    gap_cells += 1
                elif status != "ok":
                    mismatches.append({"table": ti, "row": ri, "col": ci,
                                       "expected": ecell, "got": gr[ci], "kind": status})
                checked += 1
    os.makedirs(os.path.join(table_dir, "output"), exist_ok=True)
    result = {"appendix": tid, "checked_cells": checked, "gap_cells": gap_cells,
              "mismatches": mismatches, "gaps": gaps,
              "ok": (not mismatches)}
    with open(os.path.join(table_dir, "output", "comparison.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write("\n")
    # write the regenerated table as CSV
    import csv
    with open(os.path.join(table_dir, "output", "table_%s.csv" % tid), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for table in tables:
            for row in table:
                w.writerow(["" if c is None else c for c in row])
            w.writerow([])
    print("[%s] checked=%d gaps=%d mismatches=%d" % (tid, checked, gap_cells, len(mismatches)))
    for m in mismatches[:6]:
        print("   mismatch:", m)
    for g in gaps:
        print("   gap:", g)
    return result
