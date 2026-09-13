#!/usr/bin/env python
"""HUM-SEQ-PLANNER v0 -- graf-nod HUM-SEQ-PLANNER (assembly-precedensgraf -> steg-
sekvens -> per-steg motion-mal). CPU only, no GPU or Newton (a user may be streaming
fotboll narmaste timmarna, burken-latt-order 2026-07-19 20:18: inga tunga
compute-jobb). Ren numpy-graf/geometri, <1s korning.

(1) Precedensgraf -> steg-sekvens: Kahn topologisk sortering. En INFEASIBLE
    (cykel) graf ger explicit fel, inte en tyst partiell ordning.
(2) Per-steg motion-mal, TVA cert-grindar:
    (a) STATISK KRAFT-FEASIBILITET: enkel lever-arm-modell tau_req = F_required *
        reach_m mot robotens FORAN VERIFIERADE TAU_MAX (samma konstanter som
        scripts/newton_stack.py:207-227, ateranvanda har inte gissade -- t.ex.
        franka TAU_MAX=[150,150,150,28,28,28]). safety_factor=2.0 (samma disciplin
        som grasp_*.py-kedjans safety_factor).
    (b) ATKOMST-KON-CLEARANCE: varje stegs approach-kon (riktning + halvvinkel)
        far INTE peka in i en REDAN-PLACERAD dels ockluderande sfar (enkel
        dot-product-vs-cos(halvvinkel)-check mot alla tidigare stegs
        placerings-punkter inom ett clearance-radie).

(2c) v1-TILLAGG (gap #2 i v0, tidsbudgeterad tick): clearance_ok ovan ar en
    RENT PUNKT-vs-KON-check pa target_pos -- den ignorerar den narmande DELENS
    EGEN GEOMETRI (en stor del kan kollidera langre upp pa infartsvagen aven
    om dess referenspunkt-vs-kon-test godkanner malet). tube_clearance_ok()
    lagger till en SVEPT-CYLINDER-check (radie tool_radius_m, valfritt per-steg-
    falt, default 0.0=av/v0-beteende) langs HELA raksegmentet fran approach-
    start till target_pos, inte bara nara sjalva malet. Strikt ADDITIVT: steg
    utan tool_radius_m paverkas inte (se FALSIFIERING 4 + dess kontroll).

(3) FALSIFIERINGAR x3 (v0) + x2 (v1-tub, se ovan) (kor i selftest):
    - SHUFFLAD ordning (slumpad, bryter en verklig beroende-kant) SKA flaggas
      infeasible av validate_order() -- annars ar topo-checken vakuos.
    - OVERKRAFT-steg (F_required 100x over TAU_MAX/reach) SKA FAILA
      kraft-grinden -- annars ar grinden fail-open.
    - CLEARANCE-brytande kon (riktad rakt mot en tidigare placerad dels
      position, inom clearance-radien) SKA FAILA clearance-grinden.

Kor: python3 scripts/hum_seq_planner_v0.py --selftest
"""
from __future__ import annotations

import sys
import numpy as np

# aterbrukade, redan-verifierade robot-TAU_MAX-profiler (scripts/newton_stack.py:207-227)
ROBOT_TAU_MAX = {
    "franka": np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0]),
    "abb_irb2400": np.array([400.0, 800.0, 400.0, 60.0, 40.0, 28.0]),
    "fanuc_m10ia": np.array([250.0, 300.0, 200.0, 40.0, 40.0, 28.0]),
    "motoman_gp12": np.array([250.0, 300.0, 200.0, 40.0, 40.0, 28.0]),
}
SAFETY_FACTOR = 2.0
CLEARANCE_RADIUS_M = 0.05


def topo_order(steps):
    """Kahn's algorithm. steps: {name: {"depends_on": [names]}}. Returns (order, ok)."""
    indeg = {n: 0 for n in steps}
    adj = {n: [] for n in steps}
    for n, s in steps.items():
        for d in s["depends_on"]:
            if d not in steps:
                raise ValueError(f"okand beroende '{d}' for steg '{n}'")
            adj[d].append(n)
            indeg[n] += 1
    q = [n for n in steps if indeg[n] == 0]
    order = []
    while q:
        n = q.pop(0)
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    ok = len(order) == len(steps)  # < len => cykel, nagra noder nadde aldrig indeg=0
    return order, ok


def validate_order(steps, proposed_order):
    """Explicit ordningsvalidering: varje steg maste komma EFTER alla sina
    depends_on i proposed_order. Returns (valid, violations)."""
    pos = {n: i for i, n in enumerate(proposed_order)}
    violations = []
    for n, s in steps.items():
        if n not in pos:
            violations.append((n, "saknas i proposed_order"))
            continue
        for d in s["depends_on"]:
            if d not in pos or pos[d] >= pos[n]:
                violations.append((n, f"beroende '{d}' kommer inte fore"))
    return (len(violations) == 0), violations


def force_feasible(step_name, F_required_N, reach_m, robot, safety_factor=SAFETY_FACTOR):
    """Statisk lever-arm-check: tau_req = F*reach mot robotens TAU_MAX[0] (varsta-
    ledens grans, den ledens som barr storst del av en utstrackt-arm-momentet)."""
    tau_max = ROBOT_TAU_MAX[robot]
    tau_req = F_required_N * reach_m
    tau_avail = float(tau_max[0]) / safety_factor
    return bool(tau_req <= tau_avail), tau_req, tau_avail


def clearance_ok(approach_dir, approach_half_angle_rad, target_pos, placed_positions,
                  clearance_radius=CLEARANCE_RADIUS_M):
    """approach_dir/target_pos ar np.array(3,). placed_positions: lista av np.array(3,)
    -- ENDAST icke-beroende (icke-mat-mot) tidigare placerade delar; call-site
    (plan()) utesluter redan sina egna depends_on-punkter INNAN anropet, eftersom
    att montera EN del PA sin beroende-del per definition kraver att narma sig den
    (det ar malet, inte en fara) -- bara OBEROENDE grannars punkter ar en genuin
    kollisionsrisk. FAIL om nagon (kvarvarande) placerad punkt ligger INOM
    clearance_radius OCH innanfor approach-konens halvvinkel fran target_pos, dvs
    kommande verktyg skulle svepa genom en redan-placerad, ORELATERAD del."""
    ad = approach_dir / (np.linalg.norm(approach_dir) + 1e-12)
    violations = []
    for p in placed_positions:
        v = p - target_pos
        dist = np.linalg.norm(v)
        if dist < 1e-9 or dist > clearance_radius:
            continue
        cos_ang = float(np.dot(v, ad) / dist)
        if cos_ang > np.cos(approach_half_angle_rad):
            violations.append((p.tolist(), dist, cos_ang))
    return (len(violations) == 0), violations


def _point_segment_dist(p, a, b):
    """Kortaste avstand fran punkt p till linjesegmentet [a,b] (np.array(3,) vardera)."""
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-12
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def tube_clearance_ok(approach_dir, approach_dist_m, target_pos, placed_positions, tool_radius_m):
    """v1-tillagg (gap #2 i v0): clearance_ok ovan ar en RENT PUNKT-vs-KON-check pa
    target_pos -- den missar en STOR narmande DEL/verktyg som sveper genom ett
    hinder LANGRE UPP pa infartsvagen (utanfor clearance_radius fran malet, men
    fortfarande pa den raka linjen roboten faktiskt for delen langs). Denna check
    modellerar den narmande delen som en SVEPT CYLINDER (radie tool_radius_m) langs
    hela raksegmentet fran approach-startpunkten (target_pos - approach_dir *
    approach_dist_m) till target_pos, och FAILAR om nagon (icke-beroende) placerad
    punkt ligger INOM tool_radius_m fran nagon punkt pa det segmentet -- inte bara
    nara sjalva malet. tool_radius_m<=0 (fallback, ingen deklarerad geometri) ->
    trivialt OK (degenererar till v0:s beteende, ingen regression for befintliga
    scenarion utan det nya faltet)."""
    if tool_radius_m is None or tool_radius_m <= 0:
        return True, []
    ad = approach_dir / (np.linalg.norm(approach_dir) + 1e-12)
    start = target_pos - ad * approach_dist_m
    violations = []
    for p in placed_positions:
        d = _point_segment_dist(p, start, target_pos)
        if d < tool_radius_m:
            violations.append((p.tolist(), d))
    return (len(violations) == 0), violations


def plan(steps, robot="franka"):
    """Full pipeline: topo-sort -> per-steg kraft+clearance-grindar langs den
    ordningen. Returns dict med order, per-steg-verdicts, overall PASS."""
    order, topo_ok = topo_order(steps)
    if not topo_ok:
        return dict(topo_ok=False, order=order, overall_pass=False, reason="cykel i precedensgrafen")

    valid, violations = validate_order(steps, order)
    placed = {}  # name -> pos, sa vi kan uteslutna direkta beroenden per steg
    per_step = []
    all_pass = valid
    for n in order:
        s = steps[n]
        f_ok, tau_req, tau_avail = force_feasible(n, s["F_required_N"], s["reach_m"], robot)
        dep_set = set(s["depends_on"])
        obstacle_positions = [p for name, p in placed.items() if name not in dep_set]
        c_ok, c_violations = clearance_ok(np.array(s["approach_dir"]), s["approach_half_angle_rad"],
                                           np.array(s["target_pos"]), obstacle_positions)
        # v1: svept-tub-check langs HELA infartsvagen (gap #2) -- tool_radius_m/
        # approach_dist_m ar OPTIONELLA per-steg-falt (default 0.0/reach_m) sa
        # befintliga scenarion utan dem far ovillkorligt tube_ok=True (ingen
        # regression), men ett steg som DEKLARERAR sin verktygs-/delradie far en
        # genuint starkare check an punkt-vs-kon.
        tool_radius_m = s.get("tool_radius_m", 0.0)
        approach_dist_m = s.get("approach_dist_m", s["reach_m"])
        t_ok, t_violations = tube_clearance_ok(np.array(s["approach_dir"]), approach_dist_m,
                                                np.array(s["target_pos"]), obstacle_positions,
                                                tool_radius_m)
        c_ok_combined = c_ok and t_ok
        step_pass = f_ok and c_ok_combined
        all_pass = all_pass and step_pass
        per_step.append(dict(step=n, force_ok=f_ok, tau_req_Nm=tau_req, tau_avail_Nm=tau_avail,
                              clearance_ok=c_ok_combined, clearance_violations=c_violations,
                              cone_ok=c_ok, tube_ok=t_ok, tube_violations=t_violations))
        placed[n] = np.array(s["target_pos"])
    return dict(topo_ok=True, order=order, order_valid=valid, order_violations=violations,
                per_step=per_step, overall_pass=all_pass)


# ---- realistiskt 5-stegs scenario, tyngre robot (abb_irb2400) ----
# SEKVENSEN ar illustrativ/syntetisk (ingen riktig CAD-monteringsplan bakom), men
# TAU_MAX ar den riktiga abb_irb2400-profilen (scripts/newton_stack.py:212-213,
# ateranvand har inte gissad) -- deklarerat explicit, inte dolt. Visar planeraren
# pa en tyngre 5-stegs kedja (bas->vaxellada->motorfaste->2 fasta bultar) med
# genuint varierande krav-storlek (F_required 12-180N) sa att TAU_MAX-marginalen
# faktiskt provas (irb2400:s j1-grans 400/2=200Nm safety_factor-justerad).
REALISTIC_5STEP_ABB = {
    "base_frame": dict(depends_on=[], F_required_N=180.0, reach_m=0.5,
                        approach_dir=[0, 0, -1], approach_half_angle_rad=0.3,
                        target_pos=[0.6, 0.0, 0.10]),
    "gearbox": dict(depends_on=["base_frame"], F_required_N=90.0, reach_m=0.45,
                     approach_dir=[0, 0, -1], approach_half_angle_rad=0.2,
                     target_pos=[0.6, 0.0, 0.22]),
    "motor_mount": dict(depends_on=["gearbox"], F_required_N=60.0, reach_m=0.45,
                         approach_dir=[0, 1, 0], approach_half_angle_rad=0.2,
                         target_pos=[0.6, 0.08, 0.28]),
    "bolt_a": dict(depends_on=["motor_mount"], F_required_N=12.0, reach_m=0.42,
                    approach_dir=[0, 0, -1], approach_half_angle_rad=0.15,
                    target_pos=[0.55, 0.06, 0.28], tool_radius_m=0.008),
    "bolt_b": dict(depends_on=["motor_mount"], F_required_N=12.0, reach_m=0.42,
                    approach_dir=[0, 0, -1], approach_half_angle_rad=0.15,
                    target_pos=[0.65, 0.06, 0.28], tool_radius_m=0.008),
}


# ---- referens-scenario: 3-stegs assembly (bas -> spindel -> lock), franka ----
DEMO_STEPS = {
    "base_plate": dict(depends_on=[], F_required_N=20.0, reach_m=0.3,
                        approach_dir=[0, 0, -1], approach_half_angle_rad=0.3,
                        target_pos=[0.4, 0.0, 0.1]),
    "spindle": dict(depends_on=["base_plate"], F_required_N=15.0, reach_m=0.35,
                     approach_dir=[0, 0, -1], approach_half_angle_rad=0.2,
                     target_pos=[0.4, 0.0, 0.15]),
    "cover": dict(depends_on=["spindle"], F_required_N=10.0, reach_m=0.35,
                   approach_dir=[0, 1, 0], approach_half_angle_rad=0.25,
                   target_pos=[0.4, 0.05, 0.2]),
}


def selftest():
    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))

    # (1) demo-scenariot ska FEASIBLE-planera rent
    res = plan(DEMO_STEPS, robot="franka")
    chk("demo: topo_ok", res["topo_ok"])
    chk("demo: order_valid", res.get("order_valid"))
    chk("demo: overall_pass", res.get("overall_pass"))
    chk("demo: 3 steg i ordningen", len(res["order"]) == 3)

    # realistiskt 5-stegs scenario, tyngre robot (abb_irb2400) -- syntetisk
    # sekvens + RIKTIGA TAU_MAX, genuint varierande krav (5-90Nm av 200 tillgangligt)
    res5 = plan(REALISTIC_5STEP_ABB, robot="abb_irb2400")
    chk("realistic5: topo_ok + overall_pass", res5["topo_ok"] and res5["overall_pass"])
    chk("realistic5: 5 steg, kraftmarginal genuint varierar",
        len(res5["order"]) == 5 and
        len({round(s["tau_req_Nm"], 1) for s in res5["per_step"]}) >= 3)

    # FALSIFIERING 1: shufflad ordning (broten beroende-kant) SKA flaggas infeasible
    bad_order = ["cover", "base_plate", "spindle"]  # cover fore sina beroenden
    valid, violations = validate_order(DEMO_STEPS, bad_order)
    chk("falsify-shuffle: broten ordning DETEKTERAD", (not valid) and len(violations) > 0)

    # FALSIFIERING 2: overkraft-steg SKA faila kraft-grinden
    overforce_steps = dict(DEMO_STEPS)
    overforce_steps = {**DEMO_STEPS, "spindle": dict(DEMO_STEPS["spindle"], F_required_N=5000.0)}
    res2 = plan(overforce_steps, robot="franka")
    spindle_step = next(s for s in res2["per_step"] if s["step"] == "spindle")
    chk("falsify-overforce: kraft-grind FAILAR", (not spindle_step["force_ok"]) and (not res2["overall_pass"]))

    # FALSIFIERING 3: clearance-brytande kon mot en OBEROENDE (icke-beroende) redan
    # placerad del -- "bracket" delar INGEN depends_on-relation med spindle, sa dess
    # punkt ar en genuin ORELATERAD hindersrisk, inte ett avsiktligt mat-mal
    clearance_break_steps = {**DEMO_STEPS,
        "bracket": dict(depends_on=[], F_required_N=5.0, reach_m=0.2,
                         approach_dir=[0, 0, -1], approach_half_angle_rad=0.2,
                         target_pos=[0.4, 0.0, 0.12]),  # 0.03m fran spindlens mal, INOM konen
    }
    res3 = plan(clearance_break_steps, robot="franka")
    spindle_step3 = next(s for s in res3["per_step"] if s["step"] == "spindle")
    chk("falsify-clearance: clearance-grind FAILAR mot oberoende hinder",
        (not spindle_step3["clearance_ok"]) and (not res3["overall_pass"]))
    # kontroll: SAMMA scenario men bracket ANGES som spindlens beroende (mat-mal,
    # inte hinder) SKA INTE faila -- bevisar att uteslutningen bara galler avsedda mal
    clearance_dep_steps = {**clearance_break_steps,
        "spindle": dict(DEMO_STEPS["spindle"], depends_on=["base_plate", "bracket"])}
    res3b = plan(clearance_dep_steps, robot="franka")
    spindle_step3b = next(s for s in res3b["per_step"] if s["step"] == "spindle")
    chk("kontroll: samma punkt som DEKLARERAT mal INTE clearance-blockerad",
        spindle_step3b["clearance_ok"])

    # FALSIFIERING 4 (v1, gap #2): en obstacle-punkt LANGT fran malet (0.15m,
    # UTANFOR clearance_radius 0.05 -- v0:s punkt-vs-kon-check missar den helt,
    # cone_ok forblir True) men LIGGANDE PA sjalva infartslinjen, langre upp.
    # Ett steg som DEKLARERAR sin egen del-/verktygsradie (tool_radius_m=0.03)
    # SKA fangas av den nya svept-tub-checken -- annars ar v1-tillagget vakuost.
    tube_break_steps = {**DEMO_STEPS,
        "spindle": dict(DEMO_STEPS["spindle"], tool_radius_m=0.03),
        "strut": dict(depends_on=[], F_required_N=5.0, reach_m=0.2,
                       approach_dir=[0, 0, -1], approach_half_angle_rad=0.2,
                       target_pos=[0.4, 0.0, 0.30]),  # pa spindelns infartslinje, 0.15m bortom malet
    }
    res4 = plan(tube_break_steps, robot="franka")
    spindle_step4 = next(s for s in res4["per_step"] if s["step"] == "spindle")
    chk("falsify-tube: kon-checken MISSAR (cone_ok True) men tub-checken FANGAR",
        spindle_step4["cone_ok"] and (not spindle_step4["tube_ok"])
        and (not spindle_step4["clearance_ok"]) and (not res4["overall_pass"]))

    # kontroll: SAMMA geometri men UTAN deklarerad tool_radius_m (default 0.0,
    # v0:s ursprungliga fallt-lista) SKA INTE flaggas -- bevisar att v1-tillagget
    # ar strikt ADDITIVT (ingen regression pa steg som inte deklarerar geometrin)
    tube_nodecl_steps = {**tube_break_steps,
        "spindle": DEMO_STEPS["spindle"]}  # ingen tool_radius_m
    res4b = plan(tube_nodecl_steps, robot="franka")
    spindle_step4b = next(s for s in res4b["per_step"] if s["step"] == "spindle")
    chk("kontroll: utan tool_radius_m ar samma geometri INTE tub-blockerad",
        spindle_step4b["tube_ok"] and spindle_step4b["clearance_ok"])

    # metric_audit: cykel-graf ska ge topo_ok=False (inte tyst partiell ordning)
    cyclic = {"a": dict(depends_on=["b"], F_required_N=1, reach_m=0.1, approach_dir=[0,0,-1],
                          approach_half_angle_rad=0.1, target_pos=[0,0,0]),
              "b": dict(depends_on=["a"], F_required_N=1, reach_m=0.1, approach_dir=[0,0,-1],
                          approach_half_angle_rad=0.1, target_pos=[1,0,0])}
    res_cyc = plan(cyclic, robot="franka")
    chk("audit: cyklisk graf -> topo_ok=False", not res_cyc["topo_ok"])

    ok = all(c for _, c in checks) and len(checks) > 0
    for n, c in checks:
        print(f"  {'✓' if c else '✗'} {n}")
    print(f"\nHUM-SEQ-PLANNER v0 selftest ({len(checks)} fall): {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    import json
    print(json.dumps(plan(DEMO_STEPS, robot="franka"), indent=2, default=str))
