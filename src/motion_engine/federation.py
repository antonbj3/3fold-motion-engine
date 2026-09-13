#!/usr/bin/env python3
"""Federation: a router over specialised physics engines behind one data-oriented contract.

Not a monolithic engine but a federation of members, each sharp in its own regime, behind a common
contract, with a router that picks the member per scene. The empirical finding behind it: specialisation
beats generality (the specialised contact engine outperformed a general engine by a large factor on
massively parallel rigid scenes), and the regimes genuinely differ (granular vs rigid contact vs
articulated vs curved geometry vs deep coupling). A general engine wrapper covers the tail so not
everything has to be specialised.

The consumer describes the scene as data (SceneSpec) and the router returns the best-suited member and
why. Two members are wired to runnable backends (relaxed_jacobi_gpu, sdf_contact_core); the others raise
NotImplementedError by design until they are wired.

  python -u -m motion_engine.federation        # selftest: routes canonical regimes to the right specialist
"""
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable


# ───────────────────────── scen-beskrivning (router-input, ren DATA) ─────────────────────────
@dataclass
class SceneSpec:
    n_bodies: int = 1
    shapes: tuple = ("box",)             # 'sphere','box','cylinder','capsule','mesh','particle'
    granular: bool = False               # many small, near-identical particles (sand/powder/media)?
    articulated: bool = False            # kinematiska kedjor / leder (robot-arm)?
    coupling_depth: int = 1              # estimated K (stack depth / contact-graph depth)
    curved_geometry: bool = False        # needs smooth SDF normals (correct rolling)?
    needs_general: bool = False          # exotic features (sensors/actuators/arbitrary constraints) -> general engine
    many_envs: bool = False              # massiv parallell (RL many-env / DR-svep)?


# ───────────────────────── medlems-kontrakt ─────────────────────────
@runtime_checkable
class Member(Protocol):
    name: str
    provenance: str                      # where and how it was validated (script + measured result)
    def suitability(self, s: SceneSpec) -> float: ...   # 0..1, how well the member fits the scene


@dataclass
class _Member:
    name: str
    provenance: str
    _score: object                       # callable(SceneSpec)->float
    factory: object = None               # callable(**kw) -> ContactEngine, or None when no runnable backend is wired
    def suitability(self, s: SceneSpec) -> float:
        return float(self._score(s))
    def is_wired(self) -> bool:
        return self.factory is not None
    def build(self, **kw):
        if self.factory is None:
            raise NotImplementedError(f"member '{self.name}' has no runnable backend wired yet")
        return self.factory(**kw)


def _make_relaxed_jacobi_gpu(**kw):
    from motion_engine.contact_engine_gpu import RelaxedJacobiContactEngine  # lazy: warp is only needed when building
    return RelaxedJacobiContactEngine(**kw)


def _make_sdf_contact_core(**kw):
    from motion_engine.contact_engine_sdf import SDFContactEngine
    return SDFContactEngine(**kw)


# ───────────────────────── registered members (routing rules from the measured findings) ─────────────────────────
def _specialized_rigid_gpu(s: SceneSpec) -> float:
    # One thread per body, accumulated split impulse, ground/shallow contact. Wins on massively parallel rigid scenes.
    if s.granular or s.articulated or s.needs_general:
        return 0.0
    sc = 0.6
    if s.many_envs or s.n_bodies >= 64: sc += 0.3          # dess starka regim (embarrassingly-parallell)
    if s.coupling_depth <= 2: sc += 0.1                    # shallow coupling (no deep stacks in the same body thread)
    if s.curved_geometry: sc -= 0.2                        # the SDF member is needed for smooth normals
    return min(sc, 0.95)

def _relaxed_jacobi_gpu(s: SceneSpec) -> float:
    # Atomic relaxed Jacobi, body-body. Stable under deep coupling (a K=8 tower) where naive Jacobi diverges.
    if s.granular or s.articulated or s.needs_general:
        return 0.0
    sc = 0.4
    if s.n_bodies >= 2 and s.coupling_depth >= 2: sc += 0.4   # body-body-kontakt/staplar = dess regim
    if s.coupling_depth >= 6: sc += 0.1                       # deep coupling
    return min(sc, 0.9)

def _granular_dem_gpu(s: SceneSpec) -> float:
    # Per-particle penalty DEM + hash grid. 200k particles at 6.6x real time; general engines are not built for this.
    return 0.95 if s.granular else 0.0

def _sdf_contact_core(s: SceneSpec) -> float:
    # Analytic SDF contact (smooth normals) -> geometry-correct (a cylinder rolls at exactly 2/3 g sin theta).
    if s.granular or s.articulated or s.needs_general:
        return 0.0
    sc = 0.5
    if s.curved_geometry: sc += 0.4                        # its whole point: smooth normals for rolling
    if any(sh in ("sphere", "cylinder", "capsule") for sh in s.shapes): sc += 0.1
    if all(sh == "box" for sh in s.shapes): sc -= 0.3      # boxes do not need smooth SDF (corner contact suffices)
    return min(max(sc, 0.0), 0.92)

def _articulated_gpu(s: SceneSpec) -> float:
    # Reduced-coordinate articulated dynamics (one thread per environment). Feasibility shown on a double pendulum.
    return 0.7 if s.articulated and not s.needs_general else 0.0

def _mujoco_general(s: SceneSpec) -> float:
    # General mature fallback, covering the tail: exotic features, a single complex scene, anything not specialised.
    sc = 0.35                                              # always a baseline, so coverage is guaranteed
    if s.needs_general: sc = 0.9                           # features not specialised here
    if s.articulated and s.n_bodies > 1: sc += 0.1         # general articulated + contact
    return min(sc, 0.9)

REGISTRY: List[_Member] = [
    _Member("specialized_rigid_gpu", "specialised rigid GPU — measured 80-87x against a general GPU engine, Coulomb law verified", _specialized_rigid_gpu),
    _Member("relaxed_jacobi_gpu", "contact_engine_gpu — K=8 tower stable, contact force = Mg at 0%, set_kinematic and determinism checks pass (wired)", _relaxed_jacobi_gpu, factory=_make_relaxed_jacobi_gpu),
    _Member("granular_dem_gpu", "granular DEM GPU — 200k particles at 6.6x real time, hash-grid broad phase", _granular_dem_gpu),
    _Member("sdf_contact_core", "contact_engine_sdf — cylinder 2/3 and sphere 5/7 g sin theta at 0% error through the contract, box sticks (wired)", _sdf_contact_core, factory=_make_sdf_contact_core),
    _Member("articulated_gpu", "articulated GPU — double pendulum at 282x (feasibility only; 7-DOF with contact not built)", _articulated_gpu),
    _Member("mujoco_general", "general engine wrap — cross-checked and agrees on stick/slide and Fn = Mg cos theta", _mujoco_general),
]


# ───────────────────────── router + federation ─────────────────────────
@dataclass
class Selection:
    member: _Member
    score: float
    runner_up: Optional[str]
    rationale: str


class Federation:
    """Holds the members and the router. engine_for(scene) -> the best-suited specialist and why."""
    def __init__(self, members: Optional[List[_Member]] = None):
        self.members = members if members is not None else list(REGISTRY)

    def route(self, s: SceneSpec) -> Selection:
        scored = sorted(((m.suitability(s), m) for m in self.members), key=lambda x: -x[0])
        (best_sc, best), = scored[:1]
        runner = scored[1][1].name if len(scored) > 1 and scored[1][0] > 0 else None
        why = f"regim→{best.name} (score {best_sc:.2f})"
        return Selection(member=best, score=best_sc, runner_up=runner, rationale=why)

    def build_engine(self, s: SceneSpec, **kw):
        """Route the scene and construct the selected backend when it is wired. Returns (engine, Selection).
        Raises NotImplementedError when the selected member has no runnable backend yet."""
        sel = self.route(s)
        return sel.member.build(**kw), sel


# ───────────────────────── selftest (routes canonical regimes to the right specialist) ─────────────────────────
def _selftest():
    fed = Federation()
    cases = [
        ("RL many environments (4096 simple boxes)", SceneSpec(n_bodies=4096, shapes=("box",), many_envs=True, coupling_depth=1), "specialized_rigid_gpu"),
        ("deep stack K=8 (body-body)", SceneSpec(n_bodies=8, shapes=("box",), coupling_depth=8), "relaxed_jacobi_gpu"),
        ("granular media (200k particles)", SceneSpec(n_bodies=200000, shapes=("particle",), granular=True), "granular_dem_gpu"),
        ("rolling cylinders (curved geometry)", SceneSpec(n_bodies=32, shapes=("cylinder",), curved_geometry=True), "sdf_contact_core"),
        ("robot arm (articulated)", SceneSpec(n_bodies=1, shapes=("capsule",), articulated=True), "articulated_gpu"),
        ("exotic sensors/actuators", SceneSpec(n_bodies=3, shapes=("mesh",), needs_general=True), "mujoco_general"),
    ]
    ok = True
    print("federation — the router picks the right specialist per regime:")
    for desc, scene, expect in cases:
        sel = fed.route(scene)
        hit = sel.member.name == expect
        ok = ok and hit
        print(f"  [{'ok' if hit else 'FAIL'}] {desc:<38} -> {sel.member.name:<22} (score {sel.score:.2f}, expected {expect})")
    print(f"  -> {'federation routes every canonical regime correctly' if ok else 'FAIL: routing rule wrong'}")
    # wiring status: which members have a runnable backend (not just a registry entry)?
    wired = [m.name for m in fed.members if m.is_wired()]
    print(f"  wired backends (runnable): {wired if wired else '(none)'}")
    need = {"relaxed_jacobi_gpu", "sdf_contact_core"}                  # the wired backends
    got = {m.name for m in fed.members if m.is_wired()}
    w_ok = need.issubset(got)
    print(f"  wired backends include {need}: {'ok' if w_ok else 'FAIL, missing: ' + str(need - got)}")
    ok = ok and w_ok
    print("  members (provenance):")
    for m in fed.members:
        print(f"    - {m.name:<22} {'[wired]' if m.is_wired() else '[registry only]'} — {m.provenance}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
