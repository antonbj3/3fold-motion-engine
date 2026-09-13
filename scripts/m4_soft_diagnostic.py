#!/usr/bin/env python3
"""M4 soft-path diagnostic: why the 2-stack resting-force residual sits at 1.408 at mass ratio 1000.

Instruments the opt-in TGS soft-step path of SplitImpulseEngine on the M4 scene of
scripts/engine_metrics.py (scene reused verbatim through engine_metrics._residual_2stack's
definition: two 0.3x0.3x0.2 bodies at z=0.101 and z=0.306, density 700 and 700*ratio, dt=1/240,
substeps=1 at the call site).

Per run it reports: residual per body vs step (every 10 steps), final spacing and per-contact
penetration, the fraction of (contact, substep) evaluations where the CONTACT_SPEED bias clamp
binds, the cross-step warm-start key hit rate, and the effective hertz after contact_hertz()
clamping with the resulting mass_scale / impulse_scale / bias_rate.

Then it sweeps one knob at a time from the defaults (soft_substeps, soft_hertz,
CONTACT_DAMPING_RATIO, steps) plus mass-ratio 1 and 10 controls, and writes
reports/m4_soft_diagnostic.json.

Nothing in src/ is modified: CONTACT_DAMPING_RATIO is rebound on the contact_engine module object
inside this process only, and bias_velocity is wrapped by a counting proxy that calls the real one.

  python scripts/m4_soft_diagnostic.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import engine_metrics as em                                   # noqa: E402  (scene + constants, verbatim)
from motion_engine import contact_engine as ce                # noqa: E402
from motion_engine import soft_step as ss                     # noqa: E402

DT = em.DT
G = ce.G
R = ce.R
PITCH = em.PITCH
TOL = 0.10
N_DEFAULT = 200                                               # the non-quick m4_steps of engine_metrics
REAL_BIAS = ss.bias_velocity


class ClampCounter:
    """Counting proxy around soft_step.bias_velocity (the real function still does the math)."""

    def __init__(self):
        self.total = 0
        self.biased = 0
        self.bound = 0
        self.deepest = 0.0

    def __call__(self, separation, soft, use_bias, inv_h, contact_speed=ss.CONTACT_SPEED):
        self.total += 1
        if separation < 0.0 and use_bias:
            self.biased += 1
            raw = soft.mass_scale * soft.bias_rate * separation
            if raw < -contact_speed:
                self.bound += 1
            self.deepest = min(self.deepest, separation)
        return REAL_BIAS(separation, soft, use_bias, inv_h, contact_speed)


def run_case(mass_ratio=1000.0, steps=N_DEFAULT, soft=True, soft_substeps=4, soft_hertz=30.0,
             damping_ratio=10.0, vel_iters=2, trace_every=10):
    """One M4 run with instrumentation. Returns a dict; does not mutate src/."""
    counter = ClampCounter()
    old_bias, old_zeta = ce.bias_velocity, ce.CONTACT_DAMPING_RATIO
    ce.bias_velocity = counter
    ce.CONTACT_DAMPING_RATIO = damping_ratio
    try:
        e = ce.SplitImpulseEngine(mu=0.5, soft=soft, vel_iters=vel_iters, pos_iters=6,
                                  soft_substeps=soft_substeps, soft_hertz=soft_hertz)
        em._add(e, "cpu", [0, 0, 0.101], density=700.)
        em._add(e, "cpu", [0, 0, 0.306], density=700. * mass_ratio)
        M = np.array([b.M for b in e.B])
        trace, hits, prev_keys = [], [], None
        t0 = time.time()
        for k in range(steps):
            e.step(DT, substeps=1)
            keys = set(e._warm) if soft else set()
            if soft and prev_keys is not None:
                hits.append(len(keys & prev_keys) / max(len(keys), 1))
            prev_keys = keys
            if (k + 1) % trace_every == 0 or k == 0:
                F = e.contact_forces()
                rel = np.abs(F[:, 2] - M * G) / (M * G)
                trace.append({"step": k + 1, "res_bottom": float(rel[0]), "res_top": float(rel[1]),
                              "Fz": [float(x) for x in F[:, 2]]})
        wall = time.time() - t0
        F = e.contact_forces()
        rel = np.abs(F[:, 2] - M * G) / (M * G)
        z = np.sort(e.get_state().xc[:, 2])
        C = e._collect()
        pens = sorted((float(c[5]) for c in C), reverse=True)
        bb = [float(c[5]) for c in C if c[1] is not None]
        gnd = [float(c[5]) for c in C if c[1] is None]
        sdt = DT / max(1, soft_substeps) if soft else DT
        inv_h = 1.0 / sdt
        ch = ss.contact_hertz(inv_h, soft_hertz)
        sft = ss.make_soft(ch, damping_ratio, sdt)
        stat = ss.make_soft(2.0 * ch, 0.5 * damping_ratio, sdt)
        return {
            "mass_ratio": mass_ratio, "steps": steps, "soft": soft, "soft_substeps": soft_substeps,
            "soft_hertz": soft_hertz, "damping_ratio": damping_ratio, "vel_iters": vel_iters,
            "residual_bottom": float(rel[0]), "residual_top": float(rel[1]),
            "residual_max": float(np.max(rel)), "reached_tol": bool(np.max(rel) < TOL),
            "Fz_over_Mg": [float(x) for x in F[:, 2] / (M * G)], "masses_kg": [float(x) for x in M],
            "final_spacing_m": float(z[1] - z[0]), "rest_pitch_m": PITCH,
            "n_contacts_final": len(C), "max_pen_m": (pens[0] if pens else 0.0),
            "max_pen_body_body_m": (max(bb) if bb else 0.0), "max_pen_ground_m": (max(gnd) if gnd else 0.0),
            "clamp_bind_frac_of_biased": (counter.bound / counter.biased if counter.biased else 0.0),
            "clamp_bind_frac_of_all": (counter.bound / counter.total if counter.total else 0.0),
            "bias_evals": counter.total, "biased_evals": counter.biased, "clamp_bound": counter.bound,
            "deepest_separation_m": counter.deepest,
            "warm_key_hit_rate": (float(np.mean(hits)) if hits else None),
            "warm_key_hit_min": (float(np.min(hits)) if hits else None),
            "effective_hertz": float(ch), "hertz_clamped": bool(ch < soft_hertz),
            "soft_dynamic": {"bias_rate": sft.bias_rate, "mass_scale": sft.mass_scale,
                             "impulse_scale": sft.impulse_scale},
            "soft_static": {"bias_rate": stat.bias_rate, "mass_scale": stat.mass_scale,
                            "impulse_scale": stat.impulse_scale},
            "trace": trace, "wall_s": wall,
        }
    finally:
        ce.bias_velocity, ce.CONTACT_DAMPING_RATIO = old_bias, old_zeta


def main():
    out = {"scene": "M4 2-stack (engine_metrics scene, DT=1/240, substeps=1 at call site)",
           "tol": TOL, "defaults": {"soft_substeps": 4, "soft_hertz": 30.0,
                                    "CONTACT_DAMPING_RATIO": 10.0, "steps": N_DEFAULT,
                                    "vel_iters": 2},
           "relax_pass_count": "NOT EXPOSED: _soft_solve hardcodes solve(False, 1); varying it needs an "
                               "edit to contact_engine.py, which this diagnostic does not make",
           "runs": []}
    rows = []

    def add(knob, value, **kw):
        r = run_case(**kw)
        r["knob"], r["knob_value"] = knob, value
        out["runs"].append(r)
        rows.append(r)
        print(f"  {knob:>22} = {str(value):<8} res {r['residual_bottom']:.3f}/{r['residual_top']:.3f}  "
              f"spacing {r['final_spacing_m']:.3f}  clamp {r['clamp_bind_frac_of_biased']*100:.1f}%  "
              f"warm {r['warm_key_hit_rate'] if r['warm_key_hit_rate'] is None else round(r['warm_key_hit_rate'],3)}  "
              f"{'Y' if r['reached_tol'] else 'N'}  [{r['wall_s']:.1f}s]")

    print("baseline (soft, ratio 1000, defaults)")
    add("baseline", "defaults")
    base = rows[0]
    print("  residual vs step (every 10):")
    for t in base["trace"]:
        print(f"    step {t['step']:>4}  bottom {t['res_bottom']:.3f}  top {t['res_top']:.3f}  "
              f"Fz {t['Fz'][0]:.1f} / {t['Fz'][1]:.1f} N")
    print(f"  effective hertz {base['effective_hertz']} (clamped={base['hertz_clamped']}), "
          f"mass_scale {base['soft_dynamic']['mass_scale']:.4f} impulse_scale "
          f"{base['soft_dynamic']['impulse_scale']:.4f} bias_rate {base['soft_dynamic']['bias_rate']:.2f}")
    print(f"  penetration: max {base['max_pen_m']:.4f} m (body-body {base['max_pen_body_body_m']:.4f}, "
          f"ground {base['max_pen_ground_m']:.4f}) over {base['n_contacts_final']} contacts")

    print("soft=False control (ratio 1000)")
    add("soft_off", "ratio 1000", soft=False)

    print("soft_substeps")
    for v in (4, 8, 16, 32):
        add("soft_substeps", v, soft_substeps=v)
    print("soft_hertz")
    for v in (15, 30, 60, 120):
        add("soft_hertz", v, soft_hertz=float(v))
    print("CONTACT_DAMPING_RATIO")
    for v in (2, 5, 10, 20):
        add("CONTACT_DAMPING_RATIO", v, damping_ratio=float(v))
    print("steps")
    for v in (N_DEFAULT, 2 * N_DEFAULT, 4 * N_DEFAULT):
        add("steps", v, steps=v)
    print("mass ratio controls")
    for v in (1.0, 10.0):
        add("mass_ratio", v, mass_ratio=v)

    print("\n| knob | value | residual bottom | residual top | spacing (m) | clamp-bind frac | reached tol |")
    print("|---|---|---|---|---|---|---|")
    lines = []
    for r in rows:
        line = (f"| {r['knob']} | {r['knob_value']} | {r['residual_bottom']:.3f} | {r['residual_top']:.3f} "
                f"| {r['final_spacing_m']:.3f} | {r['clamp_bind_frac_of_biased']:.3f} "
                f"| {'Y' if r['reached_tol'] else 'N'} |")
        print(line)
        lines.append(line)
    out["table_markdown"] = lines
    p = ROOT / "reports" / "m4_soft_diagnostic.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
