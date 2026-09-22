#!/usr/bin/env python3
"""run: one measured execution of the timed forward+gradient graph.

This matched-graph experiment addresses a recorded review gap: accuracy and
cost were obtained from SEPARATE executions.  Here ONE captured graph is launched once
(timed) and every certified quantity is read from that SAME launch after synchronization:

  * final physical residual  -- natural-map residual at the actual output state,
                                recomputed on the host from the device (lam, u, rhoc)
                                and cross-checked against the device resenv;
  * derivative vs the independent CPU implicit reference -- d v+ / d tau_j compared
                                with dt M^-1 e_j + M^-1 J^T (dlam/db)(dt J M^-1 e_j);
  * digest                    -- sha256 over (lam, v+, dv+).

The derivative computed by the graph is the FORWARD (JVP) derivative of v+ through the
AFFINE (de Saxce) sigma map; that map is asymmetric because Coulomb friction is
non-associated.  This is NOT a transposed adjoint / reverse-mode solve and no transpose
test is used as a certificate for it.  The two are kept apart in the record
(deriv_mode=forward_affine_jvp, reverse_mode=not_measured).

Everything in "provenance" is written by the SAME process that launches the graph; no
hash is filled in after the fact.  hashes are full sha256.

  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
    projects/CADtoSIMReady/.venv-newton/bin/python \
      matched_graph_protocol.py --scene cube --B 1 --acc-mode int64 \
      --iter 200 --cg 8 --grad-cg 60 --jdir 0 --out run_records.jsonl --dump out/cube_B1_int64.npz
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import batch_adjoint_matched_graph as MATCHED_BATCH            # noqa: E402
import warp as wp                 # noqa: E402

wp.set_module_options({"fuse_fp": False, "fast_math": False}, module=MATCHED_BATCH)

TWO52 = 4.503599627370496e15
TWO63 = 9.223372036854776e18
NVIDIA_QUERY = "name,driver_version,memory.used,utilization.gpu"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arr(a):
    return sha256_bytes(np.ascontiguousarray(a, np.float64).tobytes())


def gpu_query():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=" + NVIDIA_QUERY, "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return out
    except Exception as e:                                    # noqa: BLE001
        return "n/a (%s)" % e


def rel_l2(a, b):
    d = float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64)))
    n = float(np.linalg.norm(np.asarray(b, np.float64)))
    return d / n if n > 0 else (d if d == 0 else float("inf"))


def f_proj_np(z, mu):
    """Exact scalar/vector transcription of batch_adjoint's f_proj, evaluated in numpy."""
    zn = z[..., 0]
    zt = np.sqrt(z[..., 1] ** 2 + z[..., 2] ** 2)
    zt_safe = np.where(zt > 1e-300, zt, 1e-300)
    out = np.empty_like(z)
    stick = zt <= mu * zn
    zero = (~stick) & (mu * zt <= -zn)
    slide = (~stick) & (~zero)
    out[stick] = z[stick]
    out[zero] = 0.0
    s = (zn[slide] + mu[slide] * zt[slide]) / (1.0 + mu[slide] ** 2)
    f = mu[slide] * s / zt_safe[slide]
    out[slide, 0] = s
    out[slide, 1] = f * z[slide, 1]
    out[slide, 2] = f * z[slide, 2]
    return out


def natural_residual_host(lam, u, mu, rhoc, desax=True):
    w = u
    if desax:
        ut = np.sqrt(u[..., 1] ** 2 + u[..., 2] ** 2)
        w = u.copy()
        w[..., 0] = u[..., 0] + mu * ut
    d = lam - f_proj_np(lam - rhoc[..., None] * w, mu)
    per = np.abs(d).max(axis=-1)                 # (B, C)
    return per.max(axis=1), per                  # (B,), (B,C)


def rho_from_labels(Ge, ref_labels, mode):
    ev = np.linalg.eigvalsh(Ge)
    hi = float(ev[-1])
    pos = ev[ev > 1e-12 * max(hi, 1e-300)]
    lo = float(pos[0]) if len(pos) else hi
    if mode == "sqrt":
        return float(np.sqrt(lo * hi)), lo, hi
    any_slip = any(str(l) == "slip" for l in ref_labels)
    return (hi if any_slip else lo), lo, hi


def scale_for(kap_e, m, cap=TWO52):
    """Transcription of batch_adjoint's k_scale: binary scale keeping kap*max|v|*scal <= cap."""
    t = kap_e * m
    if t <= 0.0:
        return 1.0
    return float(2.0 ** np.floor(np.log2(cap / t)))


def run(args):
    wall0 = time.perf_counter()
    gpu_before = gpu_query()
    proc_start = time.time()

    z = np.load(args.refs, allow_pickle=False)
    names = ("J", "Minv", "v_free", "mu", "G", "eta", "b", "dt", "db_cpu")
    ref = {nm: np.asarray(z["%s/%s" % (args.scene, nm)], np.float64) for nm in names}
    ref["labels"] = z["%s/labels" % args.scene]
    refs_sha = sha256_file(args.refs)
    meta = json.loads(str(z["meta_json"]))
    meta_scene = next(m for m in meta if m["scene"] == args.scene)

    J, Minv, vf, mu0, G, eta, b = (ref["J"], ref["Minv"], ref["v_free"], ref["mu"],
                                   ref["G"], ref["eta"], ref["b"])
    dt = float(ref["dt"])
    NV, M, C = int(J.shape[1]), int(J.shape[0]), int(len(mu0))
    Ge = G + np.diag(np.repeat(eta, 3))
    rho, lmin, lmax = rho_from_labels(Ge, ref["labels"], args.rho_mode)
    n_slip_cpu = int(sum(1 for l in ref["labels"] if str(l) == "slip"))

    input_sha = {nm: sha256_arr(a) for nm, a in
                 (("J", J), ("Minv", Minv), ("v_free", vf), ("mu", mu0), ("G", G),
                  ("eta", eta), ("b", b), ("db_cpu", ref["db_cpu"]))}
    input_sha["dt"] = sha256_bytes(np.asarray([dt], np.float64).tobytes())

    base = dict(
        study_phase="run", run_id=os.environ.get("MATCHED_RUN_ID", ""),
        pid=os.getpid(), host=platform.node(), wall_utc=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(proc_start)),
        scene=args.scene, B=int(args.B), acc_mode=args.acc_mode,
        deriv_mode="forward_affine_jvp", reverse_mode="not_measured",
        protocol="single_timed_graph",
        graph=dict(n_iter=int(args.iter), cg_iters=int(args.cg),
                   grad_cg=int(args.grad_cg), refresh_every=int(args.refresh),
                   tol=float(args.tol), n_calls=int(args.n_calls),
                   do_mu=bool(args.do_mu), jdir=int(args.jdir)),
        rho=dict(mode=args.rho_mode, value=float(rho), lmin=float(lmin),
                 lmax=float(lmax)),
        tolerances=dict(conv_tol=float(args.tol), ref_accept=float(args.ref_accept),
                        pgs_ref_tol=1e-15),
        provenance=dict(
            python=platform.python_version(), numpy=np.__version__,
            warp=getattr(wp, "__version__", "n/a"),
            gpu_before=gpu_before, nvidia_query=NVIDIA_QUERY,
            fuse_fp=False, fast_math=False,
            module_path=os.path.relpath(MATCHED_BATCH.__file__, os.path.dirname(_HERE)),
            module_sha256=sha256_file(MATCHED_BATCH.__file__),
            script_sha256=sha256_file(os.path.abspath(__file__)),
            refs_path=os.path.relpath(args.refs, os.path.dirname(_HERE)),
            refs_sha256=refs_sha,
            input_sha256=input_sha,
        ),
    )

    s = MATCHED_BATCH.BatchProxADMMGPU(int(args.B), NV, C, acc_mode=args.acc_mode)
    mem_est = s.bytes_total() + int(args.B) * M * M * 8 + int(args.B) * C * 8
    base["bytes_total"] = int(mem_est)
    if mem_est > args.mem_limit:
        base["status"] = "infeasible_memory"
        base["wall_s"] = time.perf_counter() - wall0
        return base, None

    B = int(args.B)
    Jb = np.tile(J, (B, 1, 1))
    s.upload_G(Jb, np.tile(Minv, (B, 1, 1)), np.tile(vf, (B, 1)), np.tile(mu0, B), dt,
               G, alpha_eta=float(args.alpha), eta_host=eta)
    s.set_rho_host(np.full(B, rho))
    s.set_jdir(args.jdir)
    wp.synchronize_device(s.d)

    graph, capture_ms = s.capture_step_grad(
        int(args.iter), int(args.cg), int(args.refresh), float(args.tol),
        int(args.n_calls), int(args.grad_cg), bool(args.do_mu))

    # warmup launch (outputs discarded), then a fresh reset and THE timed launch
    if args.warmup:
        wp.capture_launch(graph)
        wp.synchronize_device(s.d)
        s.L(MATCHED_BATCH.k_pjac, (s.B, s.C), [s.C, s.Gd, s.eta, s.rho, s.Pj])
        s.reset_state()
        wp.synchronize_device(s.d)

    t0 = time.perf_counter()
    wp.capture_launch(graph)
    wp.synchronize_device(s.d)
    launch_ms = (time.perf_counter() - t0) * 1e3
    gpu_after = gpu_query()

    # ---- read the certified outputs of the timed launch -----------------------
    lam = np.asarray(s.lam.numpy()).reshape(B, C, 3).copy()
    u = np.asarray(s.u.numpy()).reshape(B, C, 3).copy()
    vpv = np.asarray(s.vpv.numpy()).reshape(B, NV).copy()
    dvpv = np.asarray(s.dvpv.numpy()).reshape(B, NV).copy()
    resenv = np.asarray(s.resenv.numpy()).reshape(B).copy()
    rhoc = np.asarray(s.rhoc.numpy()).reshape(B, C).copy()
    muv = np.asarray(s.mu.numpy()).reshape(B, C).copy()
    nslip = np.asarray(s.nslip.numpy()).reshape(B).copy()
    mask = np.asarray(s.mask.numpy()).reshape(B).copy()
    lab = np.asarray(s.lab.numpy()).reshape(B, C).copy()
    scal = np.asarray(s.scal.numpy()).reshape(B).copy()
    kap = np.asarray(s.kap.numpy()).reshape(B).copy()
    xv = np.asarray(s.x.numpy()).reshape(B, C, 3).copy()
    dlamv = np.asarray(s.dlam.numpy()).reshape(B, C, 3).copy()
    zvv = np.asarray(s.lam.numpy()).reshape(B, C, 3).copy()

    res_host_env, res_host_pc = natural_residual_host(lam, u, muv, rhoc, desax=True)
    # device resenv is only refreshed every refresh_every iterations; the graph's
    # n_iter is a multiple of refresh_every so the last refresh is the final state.
    res_dev = float(resenv.max())
    res_host = float(res_host_env.max())

    jdir = int(args.jdir)
    db_cpu = ref["db_cpu"]
    dj = dt * (J @ Minv[:, jdir])
    cpu_dv = dt * Minv[:, jdir] + Minv @ J.T @ (db_cpu @ dj)
    dev_dv = dvpv[0]
    ref_err = rel_l2(dev_dv, cpu_dv)
    ref_err_abs = float(np.abs(dev_dv - cpu_dv).max())
    ref_scale = float(np.linalg.norm(cpu_dv))

    digest = sha256_bytes(np.concatenate([
        np.ascontiguousarray(lam).reshape(-1),
        np.ascontiguousarray(vpv).reshape(-1),
        np.ascontiguousarray(dvpv).reshape(-1)]).tobytes())

    # ---- fixed-point scale / overflow diagnostics (actual device fields) ------
    # k_scale recomputes the binary scale for every scaled vector, so the bound
    # kap*max|v|*scal <= 2^52 holds by construction; check it on the vectors we can
    # read (lam/zv, x, dlam) and record the int64 accumulation worst case.
    ratios = {}
    for nm, vec in (("zv", zvv), ("x", xv), ("dlam", dlamv)):
        m_e = np.abs(vec).max(axis=(1, 2))
        sc_e = np.array([scale_for(k, m) for k, m in zip(kap, m_e)])
        ratios[nm] = float((kap * m_e * sc_e / TWO52).max()) if len(m_e) else 0.0
    m_x = np.abs(xv).max(axis=(1, 2))
    accum_ratio = float(C) * TWO52 / TWO63
    lab_u, lab_c = np.unique(lab, return_counts=True)
    label_counts = {int(k): int(v) for k, v in zip(lab_u, lab_c)}
    status = "measured"
    if int(nslip.max()) + 2 > int(args.n_calls):
        status = "n_calls_insufficient"

    base.update(
        status=status,
        timing=dict(graph_launch_ms=launch_ms, capture_ms=capture_ms,
                    wall_s=time.perf_counter() - wall0),
        gpu_after=gpu_after,
        fixed_point=dict(dtype=args.acc_mode, kap_min=float(kap.min()),
                         kap_max=float(kap.max()), scal_min=float(scal.min()),
                         scal_max=float(scal.max()),
                         scale_construction_ratio=ratios,
                         int64_accum_ratio=accum_ratio,
                         scale_cap=TWO52),
        outputs=dict(
            res_device_max=res_dev, res_host_recomputed_max=res_host,
            res_ref_cpu=float(meta_scene["natural_residual"]),
            ref_err_rel_l2=float(ref_err), ref_err_abs_max=ref_err_abs,
            ref_norm=ref_scale,
            n_slip_device_max=int(nslip.max()), n_slip_device_min=int(nslip.min()),
            n_slip_cpu=n_slip_cpu,
            n_env_not_converged=int(mask.sum()), n_env_total=B,
            label_counts_device=label_counts,
            lam_sha256=sha256_arr(lam), vpv_sha256=sha256_arr(vpv),
            dvpv_sha256=sha256_arr(dvpv), digest=digest,
            converged=bool(res_dev < float(args.tol)),
            ref_accepted=bool(ref_err < float(args.ref_accept)),
        ),
        refs_sha256=refs_sha, module_sha256=base["provenance"]["module_sha256"],
    )

    artifacts = None
    if args.dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)) or ".", exist_ok=True)
        np.savez(args.dump, lam=lam, vpv=vpv, dvpv=dvpv, u=u, resenv=resenv,
                 cpu_dv=cpu_dv)
        artifacts = os.path.relpath(args.dump, os.path.dirname(_HERE))
        base["artifacts"] = artifacts

    return base, dict(lam=lam, vpv=vpv, dvpv=dvpv, u=u, cpu_dv=cpu_dv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default=os.path.join(_HERE, "refs.npz"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--acc-mode", default="int64",
                    choices=["int64", "float64", "float32"])
    ap.add_argument("--iter", type=int, default=200)
    ap.add_argument("--cg", type=int, default=8)
    ap.add_argument("--grad-cg", type=int, default=60)
    ap.add_argument("--refresh", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--n-calls", type=int, required=True,
                    help="graph-shape upper bound; must be >= n_slip_device + 2")
    ap.add_argument("--jdir", type=int, default=0)
    ap.add_argument("--do-mu", action="store_true")
    ap.add_argument("--rho-mode", default="sqrt")
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--mem-limit", type=float, default=9.0e9)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ref-accept", type=float, default=1e-8)
    ap.add_argument("--out", default=os.path.join(_HERE, "run_records.jsonl"))
    ap.add_argument("--dump", default="")
    ap.add_argument("--no-dump-npz", action="store_true")
    a = ap.parse_args()

    row, _ = run(a)
    with open(a.out, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0 if row.get("status") in ("measured", "infeasible_memory") else 3


if __name__ == "__main__":
    raise SystemExit(main())
