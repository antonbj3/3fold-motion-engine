#!/usr/bin/env python3
"""Oracle-fidelity cap: downstream trust is capped by a twin's MEASURED fidelity.

A twin (simulator) is both the data generator and the reference, so a downstream world model or planner
must not claim higher trust than the twin's measured real-data fidelity. This module reads the fidelity
reports under reports/ and exposes each twin's measured fidelity as a consumable cap: trust <= measured
fidelity tier. A nominal, never-real-validated twin gets the cap 'unvalidated' regardless of what is
claimed downstream.

Consumption surfaces: (a) import wm_trust_cap()/cap_table(); (b) read reports/oracle_fidelity_cap_table.json.

  python -u scripts/twin_fidelity_cap.py   (pure python; reads reports/)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = ["unvalidated", "low", "medium", "high"]   # increasing trust
# numeric ceiling per tier (world-model trust <= this) = the measured fidelity level
TIER_CEILING = {"unvalidated": 0.0, "low": 0.60, "medium": 0.80, "high": 0.95}


def tier_of(r2, has_real):
    """Measured R2 + real-validation status -> trust tier. No real validation -> 'unvalidated'."""
    if not has_real or r2 is None:
        return "unvalidated"
    if r2 >= 0.93:
        return "high"
    if r2 >= 0.80:
        return "medium"
    if r2 >= 0.60:
        return "low"
    return "unvalidated"


def cap(claimed_tier, twin_tier):
    """Oracle-fidelity propagation: downstream trust is capped at the twin's fidelity tier (min)."""
    return TIERS[min(TIERS.index(claimed_tier), TIERS.index(twin_tier))]


# ───────────────────────── consumption surface ─────────────────────────
# Two ways to consume: (a) import wm_trust_cap()/cap_table(); (b) read reports/oracle_fidelity_cap_table.json
# (language-agnostic). An unknown or nominal twin -> ('unvalidated', 0.0): it cannot carry validated trust.

def cap_table():
    """Keyed lookup {('twin','regime'): {tier, ceiling, measured_r2, has_real}} for measured twins plus the nominal default.
    Contract: world-model trust(regime, twin) <= ceiling; a missing key falls back to the nominal default (0.0)."""
    recs, n_fleet = build_records()
    table = {}
    for r in recs:
        ceil = TIER_CEILING[r["tier"]] if r["measured_r2"] is None else min(float(r["measured_r2"]), TIER_CEILING[r["tier"]])
        table[f"{r['twin']}::{r['regime']}"] = dict(twin=r["twin"], regime=r["regime"], tier=r["tier"],
                                                    ceiling=round(ceil, 4), measured_r2=r["measured_r2"], has_real=r["has_real_validation"])
    # best per twin (a consumer that only has the twin id, not the regime, takes that twin's highest validated entry)
    by_twin = {}
    for r in recs:
        cur = by_twin.get(r["twin"])
        if cur is None or TIERS.index(r["tier"]) > TIERS.index(cur["tier"]):
            by_twin[r["twin"]] = dict(twin=r["twin"], tier=r["tier"], ceiling=TIER_CEILING[r["tier"]],
                                      best_regime=r["regime"], measured_r2=r["measured_r2"])
    return dict(by_twin_regime=table, by_twin=by_twin,
                nominal_default=dict(tier="unvalidated", ceiling=0.0,
                                     note="no real validation: the world model cannot claim validated trust"),
                fleet_nominal_count=n_fleet)


def wm_trust_cap(claimed_trust, twin, regime=None):
    """Cap a downstream world-model trust (0..1 or a tier string) at the oracle's measured fidelity.
    Unknown twin/regime -> nominal default (ceiling 0.0): an unvalidated twin cannot carry validated trust."""
    t = cap_table()
    rec = t["by_twin_regime"].get(f"{twin}::{regime}") if regime else t["by_twin"].get(twin)
    ceil_tier = rec["tier"] if rec else t["nominal_default"]["tier"]
    ceil_num = rec["ceiling"] if rec else t["nominal_default"]["ceiling"]
    if isinstance(claimed_trust, str):                       # tier string -> tier minimum
        return cap(claimed_trust, ceil_tier)
    return round(min(float(claimed_trust), ceil_num), 4)     # numeric -> min(claim, measured fidelity)


# (twin, regime, report file, R2 key, real-validated?) — the measured real-data fidelity reports
SOURCES = [
    ("franka_panda", "dynamics-rnea-heldout", "panda_twin_rnea_fidelity.json", "mean_r2_with_friction", True),
    ("franka_panda", "cross-instance", "panda_twin_cross_instance.json", "median_r2_full", True),
    ("franka_panda", "residual-ceiling", "reality_residual_ceiling.json", "approx_r2", True),
    ("ur10e", "electrical-power", "ur_current_twin_realitydoor.json", "power_r2", True),
    ("melfa_rv4frl", "dynamics-sysid-heldout-file", "melfa_dynamic_id.json", "heldout_r2_median", True),  # third vendor (held-out FILES, not a held-out instance)
]


def build_records():
    recs = []
    for twin, regime, fil, key, has_real in SOURCES:
        p = ROOT / "reports" / fil
        if not p.exists():
            continue
        d = json.loads(p.read_text()); r2 = d.get(key)
        recs.append(dict(twin=twin, regime=regime, measured_r2=r2, has_real_validation=has_real,
                         tier=tier_of(r2, has_real), source=fil))
    # nominal twins (no real validation) from the catalog -> unvalidated
    fc = ROOT / "reports" / "fleet_feasibility_catalog.json"
    n_fleet = 0
    if fc.exists():
        n_fleet = json.loads(fc.read_text()).get("n_loaded", 0)
    return recs, n_fleet


def main():
    recs, n_fleet = build_records()
    print(f"twin-fidelity-cap — oracle-fidelity propagation; {len(recs)} measured twins + {n_fleet} nominal ones")
    for r in recs:
        print(f"  {r['twin']:14s} {r['regime']:22s} measured R2 {r['measured_r2']} -> tier '{r['tier']}' (real={r['has_real_validation']}, {r['source']})")

    g1 = len(recs) >= 3 and all(r["measured_r2"] is not None for r in recs)   # read the actual reports

    # (2) the cap binds: claiming 'high' on different twins is capped to their tier
    panda_xi = next(r for r in recs if r["regime"] == "cross-instance")       # R²0.95 → high
    ur_el = next(r for r in recs if r["twin"] == "ur10e")                     # R²0.80 → medium
    cap_panda = cap("high", panda_xi["tier"]); cap_ur = cap("high", ur_el["tier"])
    cap_fleet = cap("high", "unvalidated")                                    # nominell fleet-twin
    g2 = cap_panda == "high" and cap_ur == "medium" and cap_fleet == "unvalidated"
    print(f"  (2) CAP BINDER: claim 'high' → Panda-xinst '{cap_panda}' (R²{panda_xi['measured_r2']}), ur10e-el '{cap_ur}' (R²{ur_el['measured_r2']}), fleet-nominell '{cap_fleet}' → {g2}")

    # (3) monotone: higher measured fidelity -> at least as high a tier
    order_ok = TIERS.index(panda_xi["tier"]) >= TIERS.index(ur_el["tier"]) >= TIERS.index("unvalidated")
    g3 = order_ok and panda_xi["measured_r2"] > ur_el["measured_r2"]
    print(f"  (3) MONOTON: Panda-xinst R²{panda_xi['measured_r2']}≥ur10e R²{ur_el['measured_r2']} → tier '{panda_xi['tier']}'≥'{ur_el['tier']}'≥'unvalidated' = {g3}")

    # (4) provenance-honest: a nominal twin is 'unvalidated' whatever is claimed
    g4 = cap("high", tier_of(0.99, has_real=False)) == "unvalidated"          # even R2 0.99 without real validation -> unvalidated
    print(f"  (4) provenance-honest: a nominal twin (even at high simulated R2) without real validation -> cap '{cap('high', tier_of(0.99, False))}' = {g4}")

    # (5) consumption surface: keyed cap table + numeric ceiling; nominal or unknown -> 0.0
    table = cap_table()
    num_demo = wm_trust_cap(0.99, "ur10e", "electrical-power")               # claim 0.99 -> capped at the measured UR10e value
    num_nominal = wm_trust_cap(0.99, "nonexistent_fleet_robot")              # unknown twin -> 0.0 (cannot carry validated trust)
    g5 = (len(table["by_twin_regime"]) >= 3 and table["nominal_default"]["ceiling"] == 0.0
          and abs(num_demo - 0.80) < 1e-6 and num_nominal == 0.0)
    print(f"  (5) consumption surface: {len(table['by_twin_regime'])} keyed caps; numeric ceiling demo "
          f"wm_trust_cap(0.99,ur10e)={num_demo} (<= measured 0.80), unknown twin={num_nominal} = {g5}")

    (ROOT / "reports" / "oracle_fidelity_cap_table.json").write_text(
        json.dumps(dict(role="oracle-fidelity cap table — consumed by session2 product_margin (P4.2)",
                        contract="WM-trust(regime,twin) <= ceiling; missing key -> nominal_default (0.0, fantom-vakt)",
                        consume_via=["import twin_fidelity_cap.wm_trust_cap / cap_table",
                                     "or read this JSON (by_twin_regime / by_twin / nominal_default)"],
                        tier_ceiling=TIER_CEILING, **table), ensure_ascii=False, indent=1))

    ok = g1 and g2 and g3 and g4 and g5
    print(f"\nVERDICT: twin-fidelity-cap = {'VALIDATED' if ok else 'NOT VALIDATED'}. "
          + (f"Each twin's measured real-data fidelity (read from the reports, not hardcoded) becomes a consumable "
             "trust cap. The cap binds: a downstream world model or planner cannot claim trust above the twin's "
             f"measured fidelity, and {n_fleet} nominal twins without real validation are capped at 'unvalidated' "
             "whatever their simulated R2. Consumption surface: keyed cap table "
             "(reports/oracle_fidelity_cap_table.json) plus importable wm_trust_cap()/cap_table() with a numeric "
             "ceiling equal to the measured fidelity. " if ok else
             f"Not validated (reads reports {g1}, cap binds {g2}, monotone {g3}, provenance-honest {g4}, consumption surface {g5}). ")
          + "Scope: this is provenance infrastructure, not new physics; the fidelity values come from the measured "
          "real-data reports, and catalog robots without real data are nominal.")

    out = dict(role="oracle-fidelity-propagation PRODUCER (session1 half)", measured_twins=recs, fleet_nominal_count=n_fleet,
               tiers=TIERS, tier_ceiling=TIER_CEILING, cap_binds=bool(g2), monotone=bool(g3), provenance_honest=bool(g4),
               consumption_surface=bool(g5), validated=bool(ok),
               consumer="session2 product_margin (P4.2) wires WM-trust ≤ this cap",
               consume_via=["import twin_fidelity_cap.wm_trust_cap(claimed_trust, twin, regime=None) / cap_table()",
                            "or read reports/oracle_fidelity_cap_table.json (by_twin_regime / by_twin / nominal_default)"],
               principle="WM-trust ≤ oracle-fidelity ≤ real-validation")
    (ROOT / "reports" / "twin_fidelity_cap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("  wrote reports/twin_fidelity_cap.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
