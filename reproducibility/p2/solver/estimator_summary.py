#!/usr/bin/env python3
"""residual study report: aggregate the series, chunk, residual-case and FD tables.

Reads only files under build/residual study and writes series_summary.csv,
accepted_residual_pass.csv and summary.json.  No GPU, no warp import.
"""
import csv
import glob
import json
import os

import estimator_common as C


def read_records():
    recs = {}
    for f in sorted(glob.glob(os.path.join(C.HERE, "out", "*", "record.json"))):
        d = json.load(open(f))
        recs[d["run_id"]] = d
    return recs


def series_table(recs):
    rows = []
    for rid in sorted(recs):
        if not rid.startswith("series_"):
            continue
        d = recs[rid]
        h = d["hist"]
        rows.append(dict(
            run=rid, acc=d["acc"], gate=d["gate"], sim=d["sim"], B=d["B"],
            iters=d["iters"], nit=d["nit"], res_tol=d["res_tol"], wall_s=round(d["wall_s"], 2),
            theta_sha256=d["theta_sha256"],
            m_p50=d["rel"]["m"]["p50"], m_p95=d["rel"]["m"]["p95"], m_max=d["rel"]["m"]["max"],
            I_p50=d["rel"]["I_zz"]["p50"], I_p95=d["rel"]["I_zz"]["p95"],
            I_max=d["rel"]["I_zz"]["max"],
            mu_p50=d["rel"]["mu"]["p50"], mu_p95=d["rel"]["mu"]["p95"],
            mu_max=d["rel"]["mu"]["max"],
            accepted_last=h[-1]["accepted"],
            res_acc_max_last=h[-1]["res_acc_max"],
            gate_pass_last=h[-1]["gate_pass"],
            res_lastcand_max_last=h[-1]["res_lastcand_max"],
            cost_last=h[-1]["cost_sum"],
        ))
    with open(os.path.join(C.HERE, "series_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def residual_pass_table(recs):
    rows = []
    for rid in sorted(recs):
        if not rid.startswith("series_"):
            continue
        d = recs[rid]
        for h in d["hist"]:
            rows.append(dict(run=rid, it=h["it"], accepted=h["accepted"], B=d["B"],
                             gate=d["gate"], res_acc_max=h["res_acc_max"],
                             res_acc_p95=h["res_acc_p95"], gate_pass=h["gate_pass"],
                             res_lastcand_max=h["res_lastcand_max"],
                             cost_sum=h["cost_sum"], ms_total=h["ms_total"]))
    with open(os.path.join(C.HERE, "accepted_residual_pass.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def chunk_table():
    rows = []
    for f in sorted(glob.glob(os.path.join(C.HERE, "chunk_accepted_*.json"))):
        d = json.load(open(f))
        rows.append(dict(file=os.path.basename(f), F=d["F"], B=d["B"],
                         ref_chunk=d["ref_chunk"], ref_digest=d["ref_digest"],
                         theta_sha256=d["theta_sha256"], obs_sha256=d["obs_sha256"],
                         all_bit_identical=all(v == d["ref_digest"]
                                               for v in d["digests"].values()),
                         n_calls=d["n_calls"]))
    return rows


def residual_case_table():
    rows = []
    for f in sorted(glob.glob(os.path.join(C.HERE, "accepted_cases_*.csv"))):
        for r in csv.DictReader(open(f)):
            rows.append(r)
    return rows


def combine_deliverables():
    """Concatenate the per-run matched-application logs and the per-run accepted-case
    selections into the two top-level deliverables named by the brief."""
    mrows, mfields = [], None
    for f in sorted(glob.glob(os.path.join(C.HERE, "out", "series_*",
                                           "matched_application.csv"))):
        r = list(csv.DictReader(open(f)))
        if r:
            mfields = list(r[0].keys())
            mrows.extend(r)
    with open(os.path.join(C.HERE, "matched_application.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfields)
        w.writeheader()
        for r in mrows:
            w.writerow(r)
    crows, cfields = [], None
    for f in sorted(glob.glob(os.path.join(C.HERE, "accepted_cases_*.csv"))):
        r = list(csv.DictReader(open(f)))
        if r:
            cfields = list(r[0].keys())
            crows.extend(r)
    with open(os.path.join(C.HERE, "accepted_cases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cfields)
        w.writeheader()
        for r in crows:
            w.writerow(r)
    return len(mrows), len(crows)


def main():
    recs = read_records()
    s = series_table(recs)
    rp = residual_pass_table(recs)
    ch = chunk_table()
    rc = residual_case_table()
    nma, nac = combine_deliverables()
    summary = dict(n_runs=len(recs), series=s, chunk=ch,
                   residual_cases=len(rc),
                   matched_application_rows=nma, accepted_cases_rows=nac,
                   residual_reproduced=sum(r["reproduced_at_budget"] == "True" for r in rc),
                   residual_pass_after=sum(r["pass_after_recompute"] == "True" for r in rc))
    json.dump(summary, open(os.path.join(C.HERE, "summary.json"), "w"), indent=1)
    print("series runs:", len(s), "matched rows:", nma, "accepted-case rows:", nac)
    for r in s:
        print(f"  {r['run']}: gate={r['gate']} theta={r['theta_sha256'][:16]} "
              f"m_p95={r['m_p95']:.3e} I_p95={r['I_p95']:.3e} mu_p95={r['mu_p95']:.3e} "
              f"res_max_last={r['res_acc_max_last']:.3e} gate_pass={r['gate_pass_last']}")
    print("chunk files:", len(ch), "all_bit_identical:",
          all(r["all_bit_identical"] for r in ch))
    print("residual cases:", len(rc), "reproduced:",
          summary["residual_reproduced"], "pass after recompute:",
          summary["residual_pass_after"])


if __name__ == "__main__":
    main()
