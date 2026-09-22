#!/usr/bin/env python3
"""Print and export the 15 SI refresh cells from measured public rows."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "refresh_contrast.json").read_text())
rows = [r for r in data["rows"] if r["unit"] == "SI"]
assert len(data["rows"]) == 63 and len(rows) == 15
assert {(r["index"], r["arm"]) for r in rows} == {
    (i, arm) for i in (0, 1, 2, 9, 12)
    for arm in ("per_iter", "block", "norm_block")}
rows.sort(key=lambda r: (r["index"], ("per_iter", "block", "norm_block").index(r["arm"])))
cols = ("index", "scene", "unit", "arm", "iterations", "converged", "cap", "n_chol")
with (HERE / "refresh_schedule_si.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows({key: row[key] for key in cols} for row in rows)
for row in rows:
    print(" | ".join(str(row[key]) for key in cols))
