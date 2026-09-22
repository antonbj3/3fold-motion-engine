#!/usr/bin/env python3
"""run task 3: chunk-size sweep on an ACTUAL accepted theta trajectory.

The observation set (`sp`), the accepted parameter array `theta` (reconstructed from a
run's saved `accepted_states.csv` per-iteration log, or read from a theta.npz) and the
window are built ONCE and reused for every chunk size.  Only the solver chunk (batch)
size is varied.  For each chunk we run the common state-validation pass and compare the
forward impulse, per-env residual, cost, Hessian, gradient and the sha256 digest to the
single-chunk reference: correctness (max abs) and bit identity (full sha256) are separate
columns.  The chunk list includes sizes with an incomplete final chunk.

Run twice in two separate processes with different --out to establish process
reproducibility.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

import estimator_common as C
import talos_ident_gn as GN
import estimator_state as ST
import estimator_gpu as G

KEYS = ("cost", "lam", "res_env", "Hs", "gs")


def read_theta_csv(path, it, B):
    theta = np.full((B, 3), np.nan)
    seen = np.zeros(B, bool)
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["it"]) != it:
                continue
            e = int(r["env"])
            theta[e] = (float(r["m"]), float(r["I_zz"]), float(r["mu"]))
            seen[e] = True
    return theta, int(seen.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--nB", type=int, default=None)
    ap.add_argument("--env-off", type=int, default=None)
    ap.add_argument("--chunks", default="")
    ap.add_argument("--nit", type=int, default=None)
    ap.add_argument("--cg", type=int, default=None)
    ap.add_argument("--grad-cg", type=int, default=None)
    ap.add_argument("--source-run", default=None,
                    help="run_id under out/ whose accepted_states.csv holds the theta")
    ap.add_argument("--source-it", type=int, default=0)
    ap.add_argument("--theta-npz", default=None)
    ap.add_argument("--out", default="chunk_accepted.csv")
    a = ap.parse_args()

    if a.source_run:
        rec = json.load(open(os.path.join(C.HERE, "out", a.source_run, "record.json")))
        for k in ("sim", "sigma", "window", "seed", "nB", "env_off", "nit", "cg",
                  "grad_cg"):
            if getattr(a, k) is None:
                setattr(a, k, rec[k])
        if a.sim is None:
            a.sim = "identification_sim_B512.npz"
    else:
        a.sim = a.sim or "identification_sim_B512.npz"
        a.sigma = 1.0 if a.sigma is None else a.sigma
        a.window = 50 if a.window is None else a.window
        a.seed = 4242 if a.seed is None else a.seed
        a.nB = 0 if a.nB is None else a.nB
        a.env_off = 0 if a.env_off is None else a.env_off
        a.nit = 300 if a.nit is None else a.nit
        a.cg = 4 if a.cg is None else a.cg
        a.grad_cg = 60 if a.grad_cg is None else a.grad_cg

    if a.source_run:
        csv_path = os.path.join(C.HERE, "out", a.source_run, "accepted_states.csv")
        d = np.load(os.path.join(C.HERE, a.sim))
        d = {k: d[k] for k in d.files}
        if a.nB and a.nB < d["m_true"].shape[0]:
            d = C.subset_env(d, a.nB, a.env_off)
        sp = GN.static_parts(d, a.sigma, a.window, a.seed)
        theta, nseen = read_theta_csv(csv_path, a.source_it, sp["B"])
        if nseen != sp["B"]:
            raise SystemExit(f"theta CSV has {nseen}/{sp['B']} envs for it={a.source_it}")
    else:
        d = np.load(os.path.join(C.HERE, a.sim))
        d = {k: d[k] for k in d.files}
        if a.nB and a.nB < d["m_true"].shape[0]:
            d = C.subset_env(d, a.nB, a.env_off)
        sp = GN.static_parts(d, a.sigma, a.window, a.seed)
        theta = np.load(os.path.join(C.HERE, a.theta_npz))["theta"]

    F = sp["F"]
    obs_digest = C.sp_digest(sp)
    chunks = ([int(x) for x in a.chunks.split(",") if x] if a.chunks
              else [F, 2048, 1024, 512, 320])
    chunks = sorted({min(c, F) for c in chunks}, reverse=True)
    ref_chunk = chunks[0]

    # provenance pre before the first launch (no warp import has happened in this
    # process beyond module import of estimator_state -> batch_adjoint_gpu; still, record and hash)
    budget = dict(n_iter=a.nit, cg=a.cg, grad_cg=a.grad_cg, refresh_every=4, chunk=ref_chunk,
                  window=a.window, sigma=a.sigma, seed=a.seed, nB=a.nB,
                  env_off=a.env_off, source_run=a.source_run, source_it=a.source_it)
    C.capture_pre(a.out.replace(".csv", ""),
                  inputs={"sim": os.path.join(C.HERE, a.sim),
                          "sim_sha256": C.sha256_file(os.path.join(C.HERE, a.sim))},
                  budget=budget, extra=dict(theta_sha256=C.theta_sha256(theta),
                                            obs_sha256=obs_digest))

    recs = {}
    for ch in chunks:
        gpu = G.make_gpu(ch, a.nit, a.cg, a.grad_cg)
        st = ST.state_eval(sp, gpu, theta)
        recs[ch] = dict(chunk=ch, nchunks=int(np.ceil(F / ch)),
                        last_chunk=F - (int(np.ceil(F / ch)) - 1) * ch,
                        cost_arr=st["cost"], lam_arr=st["lam"], res_arr=st["res_env"],
                        Hs_arr=st["Hs"], gs_arr=st["gs"], digest=st["digest"],
                        res_max=st["res_max"], cost_sum=float(st["cost"].sum()),
                        ms_total=st["ms_total"], n_calls=st["budget"]["n_calls"],
                        arr_dig={k: C.arr_sha256(v) for k, v in
                                 zip(KEYS, (st["cost"], st["lam"], st["res_env"],
                                            st["Hs"], st["gs"]))})
    ref = recs[ref_chunk]

    def maxabs(a, b):
        return float(np.abs(np.asarray(a) - np.asarray(b)).max())

    hdr = ["chunk", "nchunks", "last_chunk", "cost_sum", "res_max", "n_calls",
           "cost_maxabs", "lam_maxabs", "res_maxabs", "Hs_maxabs", "gs_maxabs",
           "cost_bit", "lam_bit", "res_bit", "Hs_bit", "gs_bit", "digest"]
    lines = [",".join(hdr)]
    for ch in chunks:
        r = recs[ch]
        row = [ch, r["nchunks"], r["last_chunk"], repr(r["cost_sum"]), repr(r["res_max"]),
               r["n_calls"],
               repr(maxabs(r["cost_arr"], ref["cost_arr"])),
               repr(maxabs(r["lam_arr"], ref["lam_arr"])),
               repr(maxabs(r["res_arr"], ref["res_arr"])),
               repr(maxabs(r["Hs_arr"], ref["Hs_arr"])),
               repr(maxabs(r["gs_arr"], ref["gs_arr"])),
               int(r["arr_dig"]["cost"] == ref["arr_dig"]["cost"]),
               int(r["arr_dig"]["lam"] == ref["arr_dig"]["lam"]),
               int(r["arr_dig"]["res_env"] == ref["arr_dig"]["res_env"]),
               int(r["arr_dig"]["Hs"] == ref["arr_dig"]["Hs"]),
               int(r["arr_dig"]["gs"] == ref["arr_dig"]["gs"]),
               r["digest"]]
        lines.append(",".join(str(x) for x in row))
    open(os.path.join(C.HERE, a.out), "w").write("\n".join(lines) + "\n")
    json.dump(dict(F=F, B=sp["B"], chunks=chunks, ref_chunk=ref_chunk,
                   theta_sha256=C.theta_sha256(theta), obs_sha256=obs_digest,
                   ref_digest=ref["digest"],
                   digests={str(k): recs[k]["digest"] for k in chunks},
                   arr_digests={str(k): recs[k]["arr_dig"] for k in chunks},
                   n_calls={str(k): recs[k]["n_calls"] for k in chunks},
                   ms_total={str(k): recs[k]["ms_total"] for k in chunks}),
              open(os.path.join(C.HERE, a.out.replace(".csv", ".json")), "w"), indent=1)
    print(f"-> {a.out} ref_chunk {ref_chunk} ref_digest {ref['digest']}")


if __name__ == "__main__":
    main()
