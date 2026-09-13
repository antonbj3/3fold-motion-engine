#!/usr/bin/env python3
"""Engine metrics harness for the contact backends (M1, M4, M5, M6).

Pre-registered metric set, engine agnostic over the ContactEngine Protocol. Scenes are built in
code (no external input, no RNG in the layout).

  M1 stack_load_K4  per-body net contact force = Mg across a K=4 stack (load propagation).
                    worst-body relative error, threshold < 0.10.
  Sampling (2026-09-12): every force residual is the MEAN of |residual| over the last 20 % of the
                    steps, not the final sample, and any row read off a collapsed stack (final
                    spacing below 0.8 x the 0.205 m rest pitch, or body-body penetration past the
                    corner radius R) is reported INVALID and cannot pass. Both come from the
                    measurements in reports/m4_soft_diagnostic.json. Names and thresholds unchanged.
  M4 iters_to_tol   minimum vel_iters for a resting-force residual < 0.10 on both bodies of a
                    2-stack, swept over mass ratio (conditioning is the impulse-solver analogue of
                    penalty stiffness). Lower is better; None = not reached in the sweep. The final
                    spacing of the two bodies is reported alongside: a spacing far below the rest
                    pitch means the scene itself collapsed, and the residual on it is not a
                    meaningful anchor for either engine.
  M5 stack8_pen     max penetration in a K=8 stack normalised by the corner radius R, plus the
                    secular-vs-bounded drift class. threshold pen/R < 0.5 AND bounded.
  M6 determinism    run-to-run spread of the final centres of mass over repeats of one scene.
                    threshold eps < 1e-2 m.

Engines: the CPU SplitImpulseEngine with soft=False (fixed Baumgarte position pass) and with
soft=True (TGS soft-step), and a GPU backend when a CUDA device is present (skipped otherwise):
--gpu-engine jacobi (RelaxedJacobiContactEngine, default), colored (GraphColoredContactEngine),
manifold (the coloured engine with manifold_reduce=True), graph (the manifold engine with the solve loop
captured as a CUDA graph), both of the first two, or all four.

All force readings use substeps=1: contact_forces() divides the impulse accumulated over every
substep by dt/substeps, so it reads substeps-times too high when substeps > 1.

  python scripts/engine_metrics.py [--quick] [--cpu]
      [--gpu-engine jacobi|colored|manifold|graph|chunks|both|all]
                                  [--gpu-only]
                                                      # -> reports/engine_metrics.json
    --cpu  CPU rows only, plus the labelled soft_hertz=60 M4 row under res["labelled_rows"].
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motion_engine.contact_engine import G, R, SplitImpulseEngine  # noqa: E402

DT = 1 / 240
PITCH = 0.205
DIMS = (0.3, 0.3, 0.2)


# ───────────────────────── engine factories ─────────────────────────
def cpu_v0(**kw):
    return SplitImpulseEngine(mu=0.5, soft=False, **kw)


def cpu_soft(**kw):
    return SplitImpulseEngine(mu=0.5, soft=True, **kw)


def cpu_soft_hz60(**kw):
    """Third labelled CPU row: the soft path at the stiffness the M4 diagnostic identified as the
    cheapest candidate (60 Hz is the 0.125/h cap at the default 4 substeps). The engine default is
    NOT changed: this is a harness-side factory argument."""
    return SplitImpulseEngine(mu=0.5, soft=True, soft_hertz=60.0, **kw)


def _gpu_factory(which="jacobi"):
    """Returns a factory for a GPU backend, or None when no CUDA device is available.

    which = "jacobi"  -> contact_engine_gpu.RelaxedJacobiContactEngine (relaxed Jacobi, atomics)
    which = "colored" -> contact_engine_gpu_colored.GraphColoredContactEngine (Gauss-Seidel over colours)
    which = "manifold" -> the same engine with manifold_reduce=True (each body pair reduced to at most 4
                          E-optimal contact points before the colouring)
    which = "graph" -> the manifold engine with graph_capture=True (the solve loop replayed as a CUDA graph)
    which = "chunks" -> the captured manifold engine with pair_chunks=True (the body-pair graph is coloured
                        and a pair's contacts are solved as one sequential chunk in one thread)
    All take vit/pit/mu, so the harness passes the same keywords to any of them."""
    try:
        import warp as wp
        wp.init()
        if not any(d.is_cuda for d in wp.get_devices()):
            return None
        if which in ("colored", "manifold", "graph", "chunks"):
            from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine as Cls
        else:
            from motion_engine.contact_engine_gpu import RelaxedJacobiContactEngine as Cls
    except Exception:
        return None
    extra = ({"manifold_reduce": True} if which == "manifold" else
             {"manifold_reduce": True, "graph_capture": True} if which == "graph" else
             {"manifold_reduce": True, "graph_capture": True, "pair_chunks": True} if which == "chunks"
             else {})

    def mk(**kw):
        return Cls(dims=DIMS, mu=0.5, **dict(extra, **kw))
    return mk


def _add(e, tag, pos, density=700.):
    if tag == "gpu":
        return e.add_body(pos, density=density)
    return e.add_body(*DIMS, pos, density=density)


def _masses(e, tag):
    return e._M if tag == "gpu" else np.array([b.M for b in e.B])


# ──────────────── validity gate and tail sampling (M4 diagnostic 2026-09-12) ────────────────
# The M4 diagnostic (scripts/m4_soft_diagnostic.py, reports/m4_soft_diagnostic.json) measured two
# defects of the force metrics as they were sampled before:
#   - the residual was read from the FINAL step only, and at mass ratio 1000 that single sample sits
#     in an unsettled oscillation (the same run visits 0.009, 2.582, 4.269 and 1.408 at steps 20,
#     130, 150 and 200), so the reported number was sampling noise, not a converged error;
#   - every configuration that reached the tolerance did so on a COLLAPSED stack (final spacing
#     0.000-0.050 m against a 0.205 m rest pitch), i.e. the metric rewarded interpenetration.
# Sampling is therefore the mean of |residual| over the last TAIL_FRAC of the steps, and a row whose
# final spacing falls below SPACING_FLOOR_FRAC x the rest pitch, or whose body-body penetration
# exceeds the corner radius R, is reported INVALID rather than passing. Metric names and thresholds
# are unchanged.
TAIL_FRAC = 0.2                 # residual = mean |residual| over the last 20 % of the steps
SPACING_FLOOR_FRAC = 0.8        # below 0.8 x rest pitch the stack has collapsed: INVALID


def _min_spacing(e):
    z = np.sort(e.get_state().xc[:, 2])
    return float(np.min(np.diff(z))) if len(z) > 1 else None


def _body_body_pen(e, tag):
    """Max body-body penetration in the final contact set, or None when the backend exposes no
    collector (GPU). Read-only: _collect() rebuilds the contact list, it does not advance state."""
    if tag == "gpu" or not hasattr(e, "_collect"):
        return None
    pens = [float(c[5]) for c in e._collect() if c[1] is not None]
    return max(pens) if pens else 0.0


def _validity(spacing, pen_bb):
    """A geometry gate on the scene the residual was read from. Collapsed geometry cannot pass."""
    floor = SPACING_FLOOR_FRAC * PITCH
    why = []
    if spacing is not None and spacing < floor:
        why.append(f"spacing {spacing:.3f} m < {floor:.3f} m (0.8 x rest pitch {PITCH} m)")
    if pen_bb is not None and pen_bb > R:
        why.append(f"body-body penetration {pen_bb:.3f} m > R = {R} m")
    return {"valid": not why, "invalid_reasons": why,
            "spacing_floor_m": floor, "pen_ceiling_m": R}


def _tail_start(steps):
    return max(0, steps - max(1, int(round(TAIL_FRAC * steps))))


# ───────────────────────── M1 load propagation ─────────────────────────
def m1_stack_load(mk, tag, K=4, steps=400):
    """Per-body net contact force = Mg across a K-stack. The residual is the mean |relative error|
    over the last 20 % of the steps (not the final sample), and the row is INVALID when the stack
    has collapsed."""
    e = mk()
    for k in range(K):
        _add(e, tag, [0, 0, 0.101 + k * PITCH])
    Mg = None
    tail = []
    t0 = _tail_start(steps)
    for k in range(steps):
        e.step(DT, substeps=1)
        if k >= t0:
            if Mg is None:
                Mg = _masses(e, tag) * G
            tail.append(np.abs(e.contact_forces()[:, 2] - Mg) / Mg)
    tail = np.array(tail)
    relerr = tail.mean(axis=0)                    # per body, averaged over the tail
    relerr_final = tail[-1]
    worst = float(np.max(relerr))
    spacing = _min_spacing(e)
    pen_bb = _body_body_pen(e, tag)
    val = _validity(spacing, pen_bb)
    return {"anchor": f"K={K}-stack per-body net contact = Mg (load propagation)",
            "value_worst_relerr": worst, "per_body_relerr": [float(x) for x in relerr],
            "value_worst_relerr_final_sample": float(np.max(relerr_final)),
            "tail_frac": TAIL_FRAC, "tail_steps": int(len(tail)),
            "min_spacing_m": spacing, "max_pen_body_body_m": pen_bb, "rest_pitch_m": PITCH,
            "threshold": 0.10, "pass": bool(worst < 0.10 and val["valid"]),
            "status": ("INVALID" if not val["valid"] else ("PASS" if worst < 0.10 else "FAIL")),
            **val, "units": "dimensionless"}


# ───────────────────────── M4 iterations to tolerance ─────────────────────────
def _residual_2stack(mk, tag, vit, mass_ratio, steps):
    """Scene unchanged. Returns (mean |residual| over the last 20 % of steps, final spacing,
    max body-body penetration, final-sample residual)."""
    kw = {"vit": vit} if tag == "gpu" else {"vel_iters": vit, "pos_iters": 6}
    e = mk(**kw)
    _add(e, tag, [0, 0, 0.101], density=700.)
    _add(e, tag, [0, 0, 0.306], density=700. * mass_ratio)
    M = None
    tail = []
    t0 = _tail_start(steps)
    for k in range(steps):
        e.step(DT, substeps=1)
        if k >= t0:
            if M is None:
                M = _masses(e, tag)
            tail.append(float(np.max(np.abs(e.contact_forces()[:, 2] - M * G) / (M * G))))
    z = np.sort(e.get_state().xc[:, 2])
    return (float(np.mean(tail)), float(z[1] - z[0]), _body_body_pen(e, tag), float(tail[-1]))


def m4_iters_to_tol(mk, tag, mass_ratio, vits, steps, iter_independent=False):
    """Sweeps vel_iters for the residual tolerance. The residual is the mean |residual| over the
    last 20 % of the steps, because the final sample is one draw from an unsettled oscillation at
    high mass ratio. A row whose stack has collapsed (final spacing below 0.8 x the rest pitch, or
    body-body penetration past R) is reported INVALID and cannot reach the tolerance.

    The soft path applies exactly one biased and one relax sweep per substep by construction, so it
    is vel_iters independent; that is verified here (two ends of the sweep must agree to the last
    bit) rather than assumed, and the sweep is then reported as a single residual."""
    tol = 0.10
    residuals = {}
    if iter_independent:
        lo, hi = vits[0], vits[-1]
        r_lo, spacing, pen_bb, final_lo = _residual_2stack(mk, tag, lo, mass_ratio, steps)
        r_hi, _, _, _ = _residual_2stack(mk, tag, hi, mass_ratio, steps)
        verified = bool(r_lo == r_hi)
        residuals = {str(v): r_lo for v in vits}
        val = _validity(spacing, pen_bb)
        found = (vits[0] if (r_lo < tol and val["valid"]) else None)
        return {"mass_ratio": mass_ratio, "iters_to_tol_PRIMARY": found, "tol": tol,
                "residual_PRIMARY": r_lo, "residual_final_sample": final_lo,
                "tail_frac": TAIL_FRAC, "steps": steps,
                "residual_vs_vit": residuals, "vel_iters_independent": verified,
                "independence_check": {"vel_iters": [lo, hi], "residual": [r_lo, r_hi]},
                "final_spacing_m": spacing, "max_pen_body_body_m": pen_bb, "rest_pitch_m": PITCH,
                "status": ("INVALID" if not val["valid"] else
                           ("REACHED" if found is not None else "NOT REACHED")),
                **val, "units": "iters"}
    found = None
    spacing = pen_bb = final = None
    best = None
    for vit in vits:
        res, spacing, pen_bb, final = _residual_2stack(mk, tag, vit, mass_ratio, steps)
        residuals[str(vit)] = res
        best = res if best is None else min(best, res)
        if found is None and res < tol and _validity(spacing, pen_bb)["valid"]:
            found = vit
    val = _validity(spacing, pen_bb)
    return {"mass_ratio": mass_ratio, "iters_to_tol_PRIMARY": found, "tol": tol,
            "residual_PRIMARY": best, "residual_final_sample": final,
            "tail_frac": TAIL_FRAC, "steps": steps,
            "residual_vs_vit": residuals, "final_spacing_m": spacing,
            "max_pen_body_body_m": pen_bb, "rest_pitch_m": PITCH,
            "status": ("INVALID" if not val["valid"] else
                       ("REACHED" if found is not None else "NOT REACHED")),
            **val, "units": "iters"}


# ───────────────────────── M5 constraint violation ─────────────────────────
def m5_penetration(mk, tag, K=8, steps=600):
    """Penetration = rest pitch minus the spacing of adjacent bodies, normalised by R. The drift
    class comes from the slope over the second half of the run: secular when the extrapolated
    change exceeds a tenth of R over that half."""
    e = mk()
    for k in range(K):
        _add(e, tag, [0, 0, 0.101 + k * PITCH])
    pens = []
    for _ in range(steps):
        e.step(DT, substeps=1)
        z = np.sort(e.get_state().xc[:, 2])
        pens.append(float(np.max(PITCH - np.diff(z))))
    pens = np.array(pens)
    h = len(pens) // 2
    slope = float(np.polyfit(np.arange(h), pens[h:], 1)[0])
    drift = "secular" if abs(slope) * h > 0.1 * R else "bounded"
    maxpen_norm = float(pens.max() / R)
    spacing = _min_spacing(e)
    pen_bb = _body_body_pen(e, tag)
    val = _validity(spacing, pen_bb)          # a collapsed K-stack cannot pass either
    ok = bool(maxpen_norm < 0.5 and drift == "bounded")
    return {"max_pen_over_R": maxpen_norm, "max_pen_m": float(pens.max()), "drift_class": drift,
            "drift_slope_m_per_step": slope, "threshold_pen_over_R": 0.5,
            "min_spacing_m": spacing, "max_pen_body_body_m": pen_bb, "rest_pitch_m": PITCH,
            "pass": bool(ok and val["valid"]),
            "status": ("INVALID" if not val["valid"] else ("PASS" if ok else "FAIL")),
            **val, "units": "dimensionless (pen/R)"}


# ───────────────────────── M6 determinism ─────────────────────────
def m6_determinism(mk, tag, repeats=4, K=4, steps=100):
    finals = []
    for _ in range(repeats):
        e = mk()
        for k in range(K):
            _add(e, tag, [0, 0, 0.101 + k * PITCH])
        for _ in range(steps):
            e.step(DT, substeps=1)
        finals.append(e.get_state().xc.copy())
    finals = np.array(finals)
    spread = float(np.max(np.max(finals, axis=0) - np.min(finals, axis=0)))
    return {"eps_nondet_max_dxc_m": spread, "repeats": repeats, "steps": steps,
            "bit_deterministic": bool(spread == 0.0),
            "class": "bit-deterministic" if spread == 0 else "statistical (atomic order)",
            "threshold": 1e-2, "pass": bool(spread < 1e-2), "units": "m"}


# ───────────────────────── driver ─────────────────────────
def engine_hash():
    h = hashlib.sha256()
    for name in ("contact_engine.py", "soft_step.py", "contact_engine_gpu.py",
                 "contact_engine_gpu_colored.py"):
        p = ROOT / "src" / "motion_engine" / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run_engine(mk, tag, quick, iter_independent=False, m4_only=False):
    vits = [2, 10] if quick else [2, 4, 6, 8, 10, 20, 40]
    m5_steps = 120 if quick else 600
    m1_steps = 100 if quick else 400
    m4_steps = 60 if quick else 200
    ratios = [1.0] if quick else [1.0, 1000.0]
    out = {"M4_iters_to_tol": [m4_iters_to_tol(mk, tag, r, vits, m4_steps, iter_independent)
                               for r in ratios]}
    if not m4_only:                 # the labelled soft_hertz=60 row is an M4 diagnostic row only
        out["M1_stack_load_K4"] = m1_stack_load(mk, tag, 4, m1_steps)
        out["M5_stack8_pen"] = m5_penetration(mk, tag, 8, m5_steps)
        out["M6_determinism"] = m6_determinism(mk, tag, 3 if quick else 4, 4, 60 if quick else 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short scenes, for the test wrapper")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU rows only (no GPU probe) plus the labelled soft_hertz=60 M4 row")
    ap.add_argument("--gpu-engine",
                    choices=["jacobi", "colored", "manifold", "graph", "chunks", "both", "all"],
                    default="jacobi",
                    help="which GPU backend(s) to measure: relaxed Jacobi, graph-coloured, graph-coloured "
                         "with the manifold reduction, the same with the solve loop captured as a CUDA "
                         "graph, the same with the body-pair chunk solve, both of the first two, or all "
                         "five")
    ap.add_argument("--gpu-only", action="store_true", help="skip the CPU rows")
    args = ap.parse_args()

    engines = [] if args.gpu_only else [("cpu_v0", cpu_v0, "cpu", False), ("cpu_soft", cpu_soft, "cpu", True)]
    rowname = {"jacobi": "gpu", "colored": "gpu_colored", "manifold": "gpu_manifold",
               "graph": "gpu_graph", "chunks": "gpu_chunks"}
    want = [] if args.cpu else ([("gpu", "jacobi"), ("gpu_colored", "colored")]
                                if args.gpu_engine == "both" else
                                [(rowname[w], w) for w in ("jacobi", "colored", "manifold", "graph",
                                                          "chunks")]
                                if args.gpu_engine == "all" else
                                [(rowname[args.gpu_engine], args.gpu_engine)])
    gpu_mk = None
    for name, which in want:
        mk = _gpu_factory(which)
        if mk is not None:
            engines.append((name, mk, "gpu", False))
            gpu_mk = mk

    res = {"schema": "engine_metrics_v1", "quick": args.quick, "engine_content_hash": engine_hash(),
           "skipped": [] if gpu_mk is not None else ["gpu (no CUDA device)"], "engines": {}}
    t0 = time.time()
    for name, mk, tag, ind in engines:
        te = time.time()
        res["engines"][name] = run_engine(mk, tag, args.quick, ind)
        res["engines"][name]["wall_s"] = round(time.time() - te, 1)
    # Labelled diagnostic row, kept out of res["engines"] so the report shape is unchanged: the soft
    # path at soft_hertz=60 (M4 only; the harness passes the factory argument, no default is changed).
    if args.cpu:
        te = time.time()
        res["labelled_rows"] = {"cpu_soft_hz60": run_engine(cpu_soft_hz60, "cpu", args.quick,
                                                            iter_independent=True, m4_only=True)}
        res["labelled_rows"]["cpu_soft_hz60"]["wall_s"] = round(time.time() - te, 1)
    res["wall_s"] = round(time.time() - t0, 1)

    rows = []
    allrows = list(res["engines"].items()) + list(res.get("labelled_rows", {}).items())
    for name, m in allrows:
        m4 = ", ".join(f"{int(r['mass_ratio'])}:{r['iters_to_tol_PRIMARY']} {r['status']}"
                       for r in m["M4_iters_to_tol"])
        m4res = ", ".join(f"{int(r['mass_ratio'])}:{r['residual_PRIMARY']:.3f}/{r['final_spacing_m']:.3f}"
                          for r in m["M4_iters_to_tol"])
        m1 = m.get("M1_stack_load_K4")
        m5 = m.get("M5_stack8_pen")
        m6 = m.get("M6_determinism")
        rows.append((name,
                     "-" if m1 is None else f"{m1['value_worst_relerr']:.3f} {m1['status']}",
                     m4, m4res,
                     "-" if m5 is None else
                     f"{m5['max_pen_over_R']:.3f} {m5['drift_class']} {m5['status']}",
                     "-" if m6 is None else
                     f"{m6['eps_nondet_max_dxc_m']:.1e} {'PASS' if m6['pass'] else 'FAIL'}"))
    hdr = ("engine", "M1 stack_load_K4 (<0.10)", "M4 iters-to-tol (ratio:iters)",
           "M4 mean residual / spacing m", "M5 stack8 pen/R (<0.5)", "M6 eps_nondet (<1e-2)")
    n = len(hdr)
    w = [max(len(hdr[i]), *(len(r[i]) for r in rows)) for i in range(n)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * w[i] for i in range(n)))
    for r in rows:
        print("  ".join(r[i].ljust(w[i]) for i in range(n)))
    for s in res["skipped"]:
        print(f"skipped: {s}")
    print(f"wall {res['wall_s']} s")

    outp = ROOT / "reports" / "engine_metrics.json"
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    print(f"-> {outp.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
