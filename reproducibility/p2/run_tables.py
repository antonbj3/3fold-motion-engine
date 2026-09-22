#!/usr/bin/env python3
"""Run every table regeneration and write tables/report.json."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tables"))
import tablelib  # noqa: E402
import comparator  # noqa: E402


def main():
    report = {}
    n_ok = 0
    for tid in tablelib.TABLES:
        try:
            out = tablelib.build(tid)
            res = comparator.run(tid, out["tables"], out["gaps"],
                                 os.path.join(HERE, "tables", "table_%s" % tid),
                                 out.get("declared_gaps"))
            report[tid] = {"ok": res["ok"], "checked": res["checked_cells"],
                           "gaps": res["gap_cells"], "mismatches": len(res["mismatches"]),
                           "notes": res["gaps"]}
            if res["ok"]:
                n_ok += 1
        except Exception as exc:  # noqa: BLE001
            report[tid] = {"ok": False, "error": repr(exc)}
    with open(os.path.join(HERE, "tables", "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")
    print("tables ok: %d/%d" % (n_ok, len(tablelib.TABLES)))
    return 0 if n_ok == len(tablelib.TABLES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
