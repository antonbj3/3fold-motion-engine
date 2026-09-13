#!/usr/bin/env python3
"""Motion stack: one importable class from URDF to a checked joint trajectory.

    from motion_engine.motion_stack import MotionStack
    ms = MotionStack(urdf, pkg, ee_frame="tool0", obstacle_box=[...], box_pose=[...])
    plan = ms.plan_cartesian(target_SE3, q_start)   # -> MotionPlan(trajectory_q, total_time_s, ...)

Pipeline: damped-least-squares IK (Cartesian goal) -> exact-mesh RRT-connect on coal geometry ->
C2 smoothing with collision re-validation -> velocity/effort time parametrisation -> RNEA effort
feasibility (rigid, plus identified friction when a twin contract is present) -> plan_confidence =
MIN(twin fidelity, effort provenance, collision) i.e. capped by the weakest validated link.
Also plan_line() (straight-line Cartesian, orientation-constrained) and servo_step()/servo()
(closed-loop reactive control with an SDF-gradient control-barrier safety filter when a WorldModel
is given).

  python -u -m motion_engine.selftest_motion_stack   (requires pinocchio + hppfcl/coal + scipy)
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin
import hppfcl

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from movement_ik import ik_multistart, ik_solve, fk                        # damped-LS IK (multi-start + warm-start)
from fleet_agnostic_planner import in_collision, edge_free, rrt_connect    # exact-mesh coal RRT
from smooth_safe_trajectory import smooth_safe                             # smoothing + collision re-validation
from fleet_waypoint_blending import time_param, blend                      # time parametrisation + spline blend
from twin_fidelity_cap import wm_trust_cap, TIERS                          # oracle-fidelity cap for plan confidence

TARGET_RATIO = 0.8


def reality_door_friction(twin_name, nv):
    """Identified real-data friction (torque-space Fc/Fv) from the twin contract for `twin_name`, if present.
    -> (Fc, Fv) of length nv (joint-mapped, 0 for joints the contract does not cover) or None.

    Effort feasibility then uses identified friction where real data exists instead of rigid-body only,
    which underestimates load. Only torque-space contracts apply; current-space data is not directly additive."""
    if not twin_name:
        return None
    p = _ROOT / "reports" / f"robot_twin_contract_{twin_name}.json"
    if not p.exists():
        return None
    try:
        fm = json.loads(p.read_text()).get("friction_model")
        if not fm:
            return None
        Fc = np.zeros(nv); Fv = np.zeros(nv)
        for e in fm:
            j = int(e.get("joint", 0)) - 1                     # kontrakt = 1-indexerat
            if 0 <= j < nv:
                Fc[j] = float(e.get("Fc", 0.0)); Fv[j] = float(e.get("Fv", 0.0))
        return Fc, Fv
    except Exception:
        return None


@dataclass
class MotionPlan:
    ok: bool
    stage: str = "complete"
    trajectory_q: Optional[np.ndarray] = None       # (N, nj) smoothed, collision-checked joint path
    waypoints: int = 0
    total_time_s: float = 0.0
    effort_feasible: object = None                   # bool el. "N/A" (effort-provenans)
    collision_free: bool = False
    plan_confidence: str = "unvalidated"             # MIN(twin, effort, collision) tier (kvalitativ)
    plan_sigma: Optional[dict] = None                # per-joint effort sigma + sigma-conservative peak ratio; None if no learned twin
    capped_by: str = ""
    ik_pos_err_mm: float = 0.0
    provenance: dict = field(default_factory=dict)

    def _td(self):
        if self.trajectory_q is None or not self.ok:
            raise ValueError(f"no executable trajectory (ok={self.ok}, stage={self.stage})")
        N = len(self.trajectory_q)
        return np.linspace(0.0, self.total_time_s or (0.05 * N), N), self.trajectory_q

    def to_csv(self, path):
        """Export to a joint-trajectory CSV (t,q0..qN). Times are uniform over total_time_s, which already
        respects the velocity/effort limits applied by time_param."""
        t, q = self._td()
        arr = np.hstack([t.reshape(-1, 1), q])
        hdr = "t," + ",".join(f"q{i}" for i in range(q.shape[1]))   # '#'-commented so np.loadtxt skips it
        np.savetxt(str(path), arr, delimiter=",", header=hdr)
        return str(path)

    def to_ros_jointtrajectory(self, joint_names=None):
        """-> ROS trajectory_msgs/JointTrajectory as a JSON-serialisable dict."""
        t, q = self._td()
        names = joint_names or [f"joint_{i+1}" for i in range(q.shape[1])]
        return {"joint_names": names, "points": [
            {"positions": [float(x) for x in q[i]], "time_from_start": float(t[i])} for i in range(len(q))]}


def _eff_tier(model, nj):
    eff = np.asarray(model.effortLimit[:nj])
    if not ((eff > 0).all() and np.isfinite(eff).all()):
        return "unvalidated", "missing"
    cv = float(eff.std() / (eff.mean() + 1e-12))
    return ("low", "placeholder") if cv < 0.02 else ("high", "real")


def _farthest_frame(indices, d):
    """Index of the frame farthest from base by |translation| (FK data `d` at pin.neutral()).
    Frames with a non-finite placement are excluded explicitly and an error is raised if every candidate
    is degenerate, so a corrupted URDF fails loudly instead of silently mis-identifying the tip."""
    finite = [i for i in indices if np.all(np.isfinite(d.oMf[i].translation))]
    if not finite:
        raise ValueError("_farthest_frame: no candidate frame has a finite placement -- URDF/model data "
                         "is degenerate")
    return max(finite, key=lambda i: float(np.linalg.norm(d.oMf[i].translation)))


class MotionStack:
    """Build the scene (URDF + coal) once; call plan_cartesian per goal."""

    @staticmethod
    def _obstacle_geom(spec, idx):
        """Build one hppfcl obstacle geometry from a spec {type, dims, pose, rot?}.
        type: box/sphere/cylinder/capsule/mesh. A mesh is an exact triangle mesh (BVHModel) or a CAD file
        loaded through MeshLoader, wrapped in pin.GeometryObject like the primitives, so the existing
        pin.computeCollisions path handles mesh-vs-robot-mesh exactly."""
        t = spec.get("type", "box"); dims = spec.get("dims", [])
        pose = np.asarray(spec.get("pose", [0.0, 0.0, 0.0]), float)
        R = np.asarray(spec["rot"], float) if "rot" in spec else np.eye(3)
        d = dims if hasattr(dims, "__len__") else [dims]
        if t == "box":
            shape = hppfcl.Box(float(d[0]), float(d[1]), float(d[2]))
        elif t == "sphere":
            shape = hppfcl.Sphere(float(d[0]))
        elif t == "cylinder":
            shape = hppfcl.Cylinder(float(d[0]), float(d[1]))      # radius, length
        elif t == "capsule":
            shape = hppfcl.Capsule(float(d[0]), float(d[1]))
        elif t == "mesh":
            if "file" in spec:                                     # CAD-mesh-fil (STL/OBJ)
                shape = hppfcl.MeshLoader().load(str(spec["file"]), np.ones(3) * float(spec.get("scale", 1.0)))
            else:                                                  # vertex+face arrays -> exact BVHModel
                V = np.asarray(spec["verts"], float); F = np.asarray(spec["faces"], np.int32)
                vv = hppfcl.StdVec_Vec3s()
                for p in V:
                    vv.append(np.asarray(p, float))
                tt = hppfcl.StdVec_Triangle()
                for a, b, c in F:
                    tt.append(hppfcl.Triangle(int(a), int(b), int(c)))
                shape = hppfcl.BVHModelOBBRSS(); shape.beginModel(len(F), len(V)); shape.addSubModel(vv, tt); shape.endModel()
        else:
            raise ValueError(f"unknown obstacle type '{t}' (box/sphere/cylinder/capsule/mesh)")
        return pin.GeometryObject(f"obstacle_{idx}_{t}", 0, pin.SE3(R, pose), shape)

    @staticmethod
    def _fps(P, k):
        """Farthest-point sampling -> k spatially uniform points (coverage-driven, not file order)."""
        sel = [0]; d2 = np.sum((P - P[0]) ** 2, axis=1)
        for _ in range(k - 1):
            j = int(d2.argmax()); sel.append(j); d2 = np.minimum(d2, np.sum((P - P[j]) ** 2, axis=1))
        return P[sel]

    @staticmethod
    def _surface_points(g, n):
        """Robot surface points = mesh vertices + triangle centroids. Vertices alone miss flat face contacts
        (measured ~33% false-free against a grazing/thin obstacle in the SDF path; centroids + FPS give ~8%).
        Farthest-point-sampled down to n."""
        V = np.asarray(g.vertices(), float)
        nt = int(g.num_tris)                                   # raises for primitive geometry (no mesh) -> handled by the caller
        idx = range(nt) if nt <= 2000 else np.linspace(0, nt - 1, 2000).astype(int)   # cap -> bounded init cost
        T = np.array([[int(g.tri_indices(t)[j]) for j in range(3)] for t in idx])
        P = np.vstack([V, V[T].mean(1)]) if len(T) else V      # vertices + centroider
        return MotionStack._fps(P, n) if len(P) > n else P

    def __init__(self, urdf, pkg=None, ee_frame="tool0", obstacle_box=None, box_pose=None, obstacles=None,
                 twin_name=None, friction=None, payload=None, accel_limit=None, jerk_limit=None,
                 lock_joints=None, dynamics_twin=None, world_model=None, self_collision=True, seed=0,
                 n_robot_points=128):
        model = pin.buildModelFromUrdf(str(urdf))
        if pkg is None:
            pkgs = [str(Path(urdf).resolve().parent)]
        elif isinstance(pkg, (list, tuple)):          # meshes spread over several vendor package roots
            pkgs = [str(p) for p in pkg]
        else:
            pkgs = [str(pkg)]
        gm = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.COLLISION, package_dirs=pkgs)
        # CONTINUOUS (unbounded) joints (RUBZ/RUBY) give nq != nv (angle as a cos/sin pair) while the Euclidean
        # planning stack (RRT/spline/time-parametrisation) assumes nq == nv. Such joints are either passive parallel
        # links (lever/piston/cylinder/balancer on ABB/FANUC, a closed chain modelled as an open tree) or a free
        # terminal roll (UR wrist_3). Lock them via buildReducedModel -> a clean nq == nv model.
        cont = [model.names[j] for j in range(model.njoints) if "UB" in model.joints[j].shortname()]
        extra = [n for n in (lock_joints or []) if model.existJointName(n)]   # user-locked (e.g. fingers, to match twin dof)
        # Dual-arm isolation: lock the other arm and the torso so planning moves only the target arm
        # (single-arm models lock nothing).
        self.isolated_arm, self._iso_tip, iso_locks = self._arm_isolation_locks(model, ee_frame)
        self.locked_joints = list(dict.fromkeys(cont + extra + iso_locks))
        if self.locked_joints:
            lock_ids = [model.getJointId(n) for n in self.locked_joints]
            model, gms = pin.buildReducedModel(model, [gm], lock_ids, pin.neutral(model))
            gm = gms[0]
        self.model = model; self.gm = gm
        self.data = self.model.createData()
        nrobot = self.gm.ngeoms
        self.disabled_self_pairs = []
        if self_collision:
            pj = [g.parentJoint for g in self.gm.geometryObjects]
            cand = [(i, j) for i in range(nrobot) for j in range(i + 1, nrobot) if abs(pj[i] - pj[j]) >= 2]
            for (i, j) in cand:
                self.gm.addCollisionPair(pin.CollisionPair(i, j))
            # Prune "always colliding" pairs: some non-adjacent links have nested or oversized COLLISION meshes
            # by design (e.g. wrist links) and collide in nearly every configuration, so a naive adjacency rule
            # yields 0% free configurations. Pairs that collide in nearly all samples are disabled (MoveIt practice).
            gdtmp = self.gm.createData()
            K = 80; cnt = np.zeros(len(cand), int); rng0 = np.random.default_rng(0)
            loq = self.model.lowerPositionLimit; hiq = self.model.upperPositionLimit
            for _ in range(K):
                q = loq + rng0.random(self.model.nq) * (hiq - loq)
                pin.computeCollisions(self.model, self.data, self.gm, gdtmp, q, False)
                for k in range(len(cand)):
                    if gdtmp.collisionResults[k].isCollision():
                        cnt[k] += 1
            always = {cand[k] for k in range(len(cand)) if cnt[k] >= 0.95 * K}
            if always:
                self.gm.removeAllCollisionPairs()
                for (i, j) in cand:
                    if (i, j) not in always:
                        self.gm.addCollisionPair(pin.CollisionPair(i, j))
                self.disabled_self_pairs = [(self.gm.geometryObjects[i].name, self.gm.geometryObjects[j].name) for (i, j) in always]
        # Robot points for the WorldModel path: subsampled collision-mesh vertices per robot geometry (surface
        # coverage) rather than one bounding sphere per link, which would be over-conservative. Sampling gaps are
        # a point-sampling limitation; the exact-mesh narrow phase is the verifier. ~48 points per geometry.
        self._rpts = []
        self.n_robot_points = int(n_robot_points)
        for i in range(nrobot):
            g = self.gm.geometryObjects[i].geometry
            try:
                V = self._surface_points(g, self.n_robot_points)   # vertices + triangle centroids, FPS-reduced
            except Exception:                          # primitive geometry (no mesh) -> AABB corners
                g.computeLocalAABB(); c = np.asarray(g.aabb_center, float); r = float(g.aabb_radius) / np.sqrt(3)
                V = c + r * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
            self._rpts.append((i, V))
        # Per-point parent joint (aligned with robot_points() order) so the CBF servo can build a Jacobian per surface point
        self._pt_parent_joint = (np.concatenate([np.full(len(V), self.gm.geometryObjects[i].parentJoint, int)
                                                 for i, V in self._rpts]) if self._rpts else np.array([], int))
        # Environment: either coal primitives (default) OR a WorldModel (SDF/voxel). With world_model set, coal
        # obstacles are not added (coal then only does self-collision); world_model=None keeps the coal behaviour.
        self.world_model = world_model
        specs = []
        if obstacle_box is not None:
            specs.append({"type": "box", "dims": list(obstacle_box), "pose": (box_pose if box_pose is not None else [0, 0, 0])})
        if obstacles:
            specs.extend(obstacles)
        if world_model is None:
            for k, spec in enumerate(specs):
                oid = self.gm.addGeometryObject(self._obstacle_geom(spec, k))
                for g in range(nrobot):
                    self.gm.addCollisionPair(pin.CollisionPair(g, oid))
        self.n_obstacles = 0 if world_model is not None else len(specs)
        self.gd = self.gm.createData()                         # after all pairs are finalised (self-pruned + obstacles)
        # EE frame: 'tool0' is not universal (some models lack it -> getFrameId sentinel -> IndexError).
        # Auto-detect the tip if the requested frame is missing: common names, else the frame on the deepest joint.
        # Dual-arm: prefer the TARGET arm's tip (otherwise farthest-tip can pick the now-locked other arm's tip).
        ee_pref = self._iso_tip if (self.isolated_arm and self._iso_tip) else ee_frame
        if self.model.existFrame(ee_pref):
            self.ee_frame, self.fid = ee_pref, self.model.getFrameId(ee_pref)
        else:
            self.ee_frame, self.fid = self._auto_tip()
        self.nq = self.model.nq; self.nv = self.model.nv       # efter reduktion: nq==nv (validerad Euklidisk stack OK)
        self.nj = self.nv                                       # alias = antal aktuerade DOF (effort/hastighet/RNEA)
        self.lo = self.model.lowerPositionLimit[:self.nq]; self.hi = self.model.upperPositionLimit[:self.nq]
        # Reject inverted model position limits before any clipping or sampling.
        if not np.all(self.lo <= self.hi):
            raise ValueError(f"model has lowerPositionLimit>upperPositionLimit for some joint: lo={self.lo}, hi={self.hi}")
        # PAYLOAD at the end effector: add payload inertia on the EE joint so RNEA/effort feasibility accounts for
        # the carried load (without it the effort margin of a loaded robot is overestimated). payload: float (kg) or
        # dict {mass, com=[x,y,z] in the EE frame, inertia=[Ixx,Iyy,Izz]}. Point mass by default.
        self.payload = payload; self.payload_kg = None
        if payload is not None:
            pm = float(payload["mass"]) if isinstance(payload, dict) else float(payload)
            self.payload_kg = pm                               # declared: the verdict applies to the LOADED robot
            com = np.asarray(payload.get("com", [0.0, 0.0, 0.0]), float) if isinstance(payload, dict) else np.zeros(3)
            Idiag = np.asarray(payload.get("inertia", [0.0, 0.0, 0.0]), float) if isinstance(payload, dict) else np.zeros(3)
            jid = self.model.frames[self.fid].parentJoint
            fpl = self.model.frames[self.fid].placement        # EE frame relative to its parent joint
            self.model.inertias[jid] = self.model.inertias[jid] + fpl.act(pin.Inertia(pm, com, np.diag(Idiag)))
            self.data = self.model.createData()                # rebuild data after the inertia change
        self.twin_name = twin_name
        if friction is None and twin_name:                     # auto-load identified friction from the twin contract
            friction = reality_door_friction(twin_name, self.nv)
        self.friction_source = "reality_door_validated" if friction is not None else "none(rigid)"
        self.Fc, self.Fv = (friction if friction is not None else (np.zeros(self.nv), np.zeros(self.nv)))
        self.accel_limit = None if accel_limit is None else np.broadcast_to(np.asarray(accel_limit, float), (self.nv,)).copy()
        self.jerk_limit = None if jerk_limit is None else np.broadcast_to(np.asarray(jerk_limit, float), (self.nv,)).copy()
        # Learned dynamics twin for effort: _effort uses twin.predict_torque when twin.dof == nv, otherwise the
        # analytic model. A string is resolved through DynamicsTwin.load.
        self.dyn_twin = None
        if dynamics_twin is not None:
            try:
                if isinstance(dynamics_twin, str):
                    from motion_engine.dynamics_twin import DynamicsTwin
                    dynamics_twin = DynamicsTwin.load(dynamics_twin)
                self.dyn_twin = dynamics_twin if getattr(dynamics_twin, "dof", None) == self.nv else None
            except Exception:
                self.dyn_twin = None
        self.rng = np.random.default_rng(seed)

    def _analytic_peaks(self, cs, S, T):
        """Analytic continuous-path kinodynamics (exact, not finite-difference). The path runs as a cubic spline
        q(s) with linear time parametrisation s(t)=S*t/T, so vel=(S/T)|q'(s)|, acc=(S/T)^2|q''(s)|, jerk=(S/T)^3|q'''(s)|.
        The cubic-spline third derivative is piecewise constant, so 4000 s-samples capture each segment exactly.
        -> (peak_acc (nq,), peak_jerk (nq,)) in physical units."""
        s = np.linspace(0, S, 4000); r = S / max(T, 1e-9)
        return r ** 2 * np.max(np.abs(cs(s, 2)), axis=0), r ** 3 * np.max(np.abs(cs(s, 3)), axis=0)

    def _retime(self, traj_q, T, cs=None, S=None):
        """Raise the total time until the accel/jerk limits hold (when set); configurations are unchanged, only the
        time scaling (acc ~ 1/T^2, jerk ~ 1/T^3). Velocity is already handled by time_param. -> bumped T.
        With spline cs and arclength S the kinodynamics are analytic (direct solution); the finite-difference
        variant over-reads jerk at spline knots by 2.5-30x. Without cs the legacy finite-difference path is used.
        Any constraint whose required-T candidate is non-finite raises: a bare max(accumulator, candidate)
        would drop that constraint silently (NaN comparisons are always False), and this function's contract
        is to guarantee hardware-safe timing, so an unmeasurable constraint must fail loudly."""
        if (self.accel_limit is None and self.jerk_limit is None) or traj_q is None or len(traj_q) < 3:
            return T
        if cs is not None and S is not None:                       # analytic (exact) — direct solution for T
            s = np.linspace(0, S, 4000)
            Treq = T
            if self.accel_limit is not None:                       # (S/T)^2*max|q''| <= a_lim -> T >= S*sqrt(max|q''|/a_lim)
                a2 = np.max(np.abs(cs(s, 2)), axis=0)
                cand = S * float(np.max(np.sqrt(a2 / np.maximum(self.accel_limit, 1e-9))))
                if not np.isfinite(cand):
                    raise ValueError("_retime: accel-constraint required-T is non-finite (degenerate spline segment "
                                      "or corrupted waypoint) -- cannot verify hardware-safe timing, refusing to "
                                      "silently return a possibly-unsafe T")
                Treq = max(Treq, cand)
            if self.jerk_limit is not None:                        # (S/T)^3*max|q'''| <= j_lim -> T >= S*cbrt(max|q'''|/j_lim)
                a3 = np.max(np.abs(cs(s, 3)), axis=0)
                cand = S * float(np.max(np.cbrt(a3 / np.maximum(self.jerk_limit, 1e-9))))
                if not np.isfinite(cand):
                    raise ValueError("_retime: jerk-constraint required-T is non-finite (degenerate spline segment "
                                      "or corrupted waypoint) -- cannot verify hardware-safe timing, refusing to "
                                      "silently return a possibly-unsafe T")
                Treq = max(Treq, cand)
            return float(Treq)
        for _ in range(8):                                         # legacy finite-difference (no spline given)
            dt = T / (len(traj_q) - 1)
            dq = np.gradient(traj_q, axis=0) / dt; ddq = np.gradient(dq, axis=0) / dt
            k = 1.0
            if self.accel_limit is not None:
                cand = float(np.sqrt(np.max(np.max(np.abs(ddq), 0) / np.maximum(self.accel_limit, 1e-9))))
                if not np.isfinite(cand):
                    raise ValueError("_retime: legacy accel scaling factor is non-finite (corrupted trajectory) -- "
                                      "cannot verify hardware-safe timing, refusing to silently return a possibly-"
                                      "unsafe T")
                k = max(k, cand)
            if self.jerk_limit is not None:
                jk = np.gradient(ddq, axis=0) / dt
                cand = float(np.cbrt(np.max(np.max(np.abs(jk), 0) / np.maximum(self.jerk_limit, 1e-9))))
                if not np.isfinite(cand):
                    raise ValueError("_retime: legacy jerk scaling factor is non-finite (corrupted trajectory) -- "
                                      "cannot verify hardware-safe timing, refusing to silently return a possibly-"
                                      "unsafe T")
                k = max(k, cand)
            if k <= 1.001:
                break
            T *= k * 1.02
        return float(T)

    def _auto_tip(self):
        """Find a tip frame when the requested EE frame is missing: (1) exact common name; (2) name-matched
        (tool0/tcp/flange/hand/gripper/ee substring) farthest-from-base; (3) otherwise farthest-from-base
        BODY/OP frame. Using FK distance rather than joint id avoids picking the torso on dual-arm models."""
        for nm in ("tool0", "flange", "tcp", "ee_link", "tool_link", "gripper", "tip", "link_tcp"):
            if self.model.existFrame(nm):
                return nm, self.model.getFrameId(nm)
        d = self.model.createData()
        pin.forwardKinematics(self.model, d, pin.neutral(self.model)); pin.updateFramePlacements(self.model, d)
        cand = [i for i in range(self.model.nframes)
                if int(self.model.frames[i].type) in (int(pin.FrameType.BODY), int(pin.FrameType.OP_FRAME))]
        named = [i for i in cand if any(k in self.model.frames[i].name.lower()
                                        for k in ("tool0", "tcp", "flange", "hand", "gripper", "_ee", "ee_"))]
        pool = named if named else cand
        best = _farthest_frame(pool, d)   # farthest from base = the real tip
        return self.model.frames[best].name, best

    def _arm_isolation_locks(self, model, ee_frame):
        """Dual-arm isolation: if the model has >=2 arm groups (left/right or arm1/arm2), pick the TARGET arm (from
        ee_frame, else the farthest-tip arm's side) -> return (target_side, target_tip_name, locks = all joints
        outside the target arm). Single-arm models return (None, None, [])."""
        import re
        movable = [model.names[j] for j in range(1, model.njoints)]

        def side(n):
            nl = (n or "").lower()
            if "left" in nl:
                return "left"
            if "right" in nl:
                return "right"
            m = re.search(r"arm_?([12])", nl) or re.search(r"_([lr])_", nl)
            return m.group(1) if m else None

        sides = {side(n) for n in movable}; sides.discard(None)
        if len(sides) < 2:
            return None, None, []                                  # not dual-arm
        d = model.createData(); pin.forwardKinematics(model, d, pin.neutral(model)); pin.updateFramePlacements(model, d)
        BODY, OPF = int(pin.FrameType.BODY), int(pin.FrameType.OP_FRAME)
        tips = [i for i in range(model.nframes) if int(model.frames[i].type) in (BODY, OPF)
                and any(k in model.frames[i].name.lower() for k in ("tool0", "tcp", "flange", "hand", "gripper", "tip", "_t"))]
        if not tips:
            tips = [i for i in range(model.nframes) if int(model.frames[i].type) in (BODY, OPF)]
        target = side(ee_frame)
        if target is None:                                         # no side in ee_frame -> use the farthest-tip arm's side
            target = side(model.frames[_farthest_frame(tips, d)].name)
        if target is None:
            return None, None, []
        tside = [i for i in tips if side(model.frames[i].name) == target]
        tip_name = (model.frames[_farthest_frame(tside, d)].name
                    if tside else None)
        locks = [n for n in movable if side(n) != target]          # lock the other arm and the shared torso joints
        return target, tip_name, locks

    def sample(self):
        """A random configuration (size nq) inside the joint limits (pure revolute model after joint locking)."""
        # Finite endpoint sentinels can overflow when subtracted. Use [-pi, pi]
        # for joints whose position-limit span is nonfinite.
        bad = ~np.isfinite(self.hi - self.lo)
        lo_s = np.where(bad, -np.pi, self.lo); hi_s = np.where(bad, np.pi, self.hi)
        return lo_s + self.rng.random(self.nq) * (hi_s - lo_s)

    def joint_names(self):
        """Joint names in Q-COLUMN order (matching trajectory_q columns), for ROS/CSV export.
        1-DOF joints sorted by idx_q. After dual-arm isolation these are the target arm's joints."""
        pairs = [(int(self.model.joints[j].idx_q), self.model.names[j])
                 for j in range(1, self.model.njoints) if self.model.joints[j].nq == 1]
        return [n for _, n in sorted(pairs)]

    def ee_trajectory(self, traj_q):
        """EE pose track (task space) along a joint path: each configuration -> EE pose [x,y,z, qx,qy,qz,qw] in the
        base frame. -> (N,7). For Cartesian export/visualisation/Cartesian controllers. FK via pinocchio."""
        out = []
        for q in np.asarray(traj_q, float):
            M = fk(self.model, self.data, self.fid, q)
            out.append(np.concatenate([M.translation, pin.Quaternion(M.rotation).coeffs()]))   # coeffs = xyzw
        return np.asarray(out)

    def robot_points(self, q):
        """Robot surface points at configuration q (subsampled mesh vertices per geometry, FK-transformed). -> (M,3).
        WorldModel path: these query the SDF world (world exact, robot point-sampled)."""
        pin.forwardKinematics(self.model, self.data, np.asarray(q, float))
        pin.updateGeometryPlacements(self.model, self.data, self.gm, self.gd)
        out = []
        for i, V in self._rpts:
            M = self.gd.oMg[i]
            out.append(V @ np.asarray(M.rotation).T + np.asarray(M.translation))
        return np.vstack(out)

    def free(self, q):
        if in_collision(self.model, self.data, self.gm, self.gd, q):   # self-collision (+ coal obstacles unless world_model)
            return False
        if self.world_model is not None:                                # environment via WorldModel: surface points vs SDF world
            if not bool((self.world_model.sd_batch(self.robot_points(q)) > 0.0).all()):
                return False
        return True

    def verify_path_exact(self, traj_q, dens=4):
        """Exact double-oracle final verification: check every (densely interpolated) configuration against the exact
        coal mesh (self + environment), closing the false-free residual (~3-5%, vertex-vs-face) of the SDF/GPU/point
        path. Workflow: plan fast (GPU/SDF, point-based) -> verify the path exactly here. Requires the stack to have
        been built with coal obstacles (obstacle_box/obstacles), not world_model.
        -> dict(collision_free, n_checked, first_collision). Touching the threshold counts as a collision."""
        traj = np.atleast_2d(np.asarray(traj_q, float))
        if len(traj) > 1 and dens > 1:
            seg = [np.linspace(a, b, dens, endpoint=False) for a, b in zip(traj[:-1], traj[1:])]
            Q = np.vstack(seg + [traj[-1:]])
        else:
            Q = traj
        for i, q in enumerate(Q):
            if in_collision(self.model, self.data, self.gm, self.gd, q):
                return dict(collision_free=False, n_checked=i + 1, first_collision=int(i))
        return dict(collision_free=True, n_checked=len(Q), first_collision=None)

    def _effort(self, traj_q, k_sigma=3.0):
        """Monotone-safe effort peak ratio (per joint, max(rigid, rigid+friction)) over the trajectory. 'N/A' when no
        real effort limits are declared. Returns (feasible, peak, plan_sigma), where plan_sigma holds per-joint sigma
        and the sigma-conservative peak (|tau|+k*sigma)/limit when a learned twin sigma exists, else None."""
        # k_sigma must be non-negative: a negative multiplier subtracts the uncertainty margin instead of adding it.
        if not (np.isfinite(k_sigma) and k_sigma >= 0):
            raise ValueError("_effort: k_sigma must be a non-negative finite sigma-multiplier -- a negative "
                             "k_sigma SUBTRACTS the uncertainty margin instead of adding it, defeating "
                             "plan_sigma's own anti-false-positive purpose")
        eff = np.asarray(self.model.effortLimit[:self.nv])
        # A `continuous` URDF joint with no <limit> tag makes pinocchio default effortLimit to +inf for that joint.
        # `(eff > 0).all()` alone lets inf through, so pj = |tau|/inf = 0 and that joint could never make
        # pj.max() >= 1.0 — it would be silently exempt from the feasibility check. inf means "no data declared",
        # not "physically unbounded torque"; require finite AND positive limits, as _eff_tier() does.
        if not (np.isfinite(eff).all() and (eff > 0).all()):
            return "N/A", None, None
        dq = np.gradient(traj_q, axis=0); ddq = np.gradient(dq, axis=0)
        tr = np.array([pin.rnea(self.model, self.data, traj_q[i], dq[i], ddq[i]) for i in range(len(traj_q))])
        tf = tr + (np.sign(dq) * self.Fc + dq * self.Fv)
        pj = np.maximum(np.max(np.abs(tr), 0) / eff, np.max(np.abs(tf), 0) / eff)   # per joint, monotone-safe
        plan_sigma = None
        if self.dyn_twin is not None:                                                # learned twin (real-data validated)
            epi = None
            if hasattr(self.dyn_twin, "predict_residual_decomposed"):                # ensemble twin: capture epistemic sigma
                resid, tw_sigma, epi, _ = self.dyn_twin.predict_residual_decomposed(traj_q, dq, ddq)
                tw = tr + resid                                                       # = predict_torque, with sigma decomposed
            else:
                tw, tw_sigma = self.dyn_twin.predict_torque(traj_q, dq, ddq, tr)      # keep sigma
            pj = np.maximum(pj, np.max(np.abs(tw), 0) / eff)                          # monotone-safe: never below the analytic value
            tws = np.abs(np.asarray(tw_sigma))                                        # sigma per (sample, joint)
            pj_cons = np.max((np.abs(tw) + k_sigma * tws), 0) / eff                   # sigma-conservative peak (anti-false-positive)
            plan_sigma = dict(per_joint_sigma_nm=[round(float(s), 4) for s in np.max(tws, 0)],
                              peak_ratio_point=round(float(pj.max()), 3),
                              peak_ratio_sigma_cons=round(float(pj_cons.max()), 3), k_sigma=k_sigma,
                              sigma_feasible=bool(pj_cons.max() < 1.0))               # executable even under the k*sigma margin
            if epi is not None:                                                       # ensemble epistemic sigma as a validity signal
                # grows where the twin extrapolates (members disagree) -> plan_sigma is less trustworthy there
                # Magnitude always; the boolean coverage flag only when the twin carries an epi_ref sidecar.
                epi_abs = np.abs(np.asarray(epi))                                      # (waypoints, joints)
                plan_sigma["epistemic_sigma_nm"] = [round(float(s), 4) for s in np.max(epi_abs, 0)]
                ref = getattr(self.dyn_twin, "epi_ref", None)                      # training-calibrated reference (99th-pct sidecar)
                if ref is not None:
                    per_wp = (epi_abs / np.maximum(np.asarray(ref, float), 1e-12)[None, :]).max(1)  # worst joint per waypoint
                    # Count nonfinite uncertainty ratios as out of distribution. A NaN
                    # comparison alone would omit invalid waypoints from the coverage fraction.
                    ood = (~np.isfinite(per_wp)) | (per_wp > 1.0)
                    frac = float(np.mean(ood))                                         # andel waypoints OOD
                    plan_sigma["twin_covered"] = bool(frac <= 0.05)                    # boolean validity certificate (>=95% in-distribution)
                    plan_sigma["twin_frac_extrap"] = round(frac, 3)
                    plan_sigma["twin_epi_ratio"] = round(float(per_wp.max()), 3)       # topp-ratio (>1 = waypoint extrapolerar)
        return bool(pj.max() < 1.0), float(pj.max()), plan_sigma

    def _sigma_safe_retime(self, cs, S, k=3.0, n=80):
        """Sigma-safe retime: fastest time profile that respects (|tau|+k*sigma) <= limit using the learned twin sigma,
        slowing down at the binding corners (per-point u-dot) rather than uniformly. Reuses
        dynamic_feasibility.torque_aware_retime. Returns the sigma-safe time, the nominal-torque time (k=0) and the
        sigma cost; None when there is no twin/spline/effort. Reported in plan_sigma; the path itself is unchanged."""
        if self.dyn_twin is None or cs is None or S is None:
            return None
        eff = np.asarray(self.model.effortLimit[:self.nv])
        # Missing or nonfinite effort limits cannot support this retiming result.
        # Return None before the torque-aware retimer rejects these limits.
        if not (np.isfinite(eff).all() and (eff > 0).all()):
            return None
        from motion_engine.dynamic_feasibility import torque_aware_retime    # lazy import (avoid a circular import)
        s = np.linspace(0.0, S, n)
        q_u = cs(s, 0); qp_u = cs(s, 1) * S; qpp_u = cs(s, 2) * (S ** 2)    # derivata wrt u=s/S ∈[0,1]
        vmax = np.asarray(self.model.velocityLimit[:self.nv])
        if not (np.isfinite(vmax).all() and (vmax > 0).all()):
            vmax = np.full(self.nv, 10.0)
        def pt(q, dq, ddq):                                                 # twin wrapper: predict_torque needs the analytic tau
            tr = np.array([pin.rnea(self.model, self.data, q[i], dq[i], ddq[i]) for i in range(len(q))])
            return self.dyn_twin.predict_torque(q, dq, ddq, tr)
        rs = torque_aware_retime(pt, q_u, qp_u, qpp_u, eff, vmax, k=k)      # σ-konservativ (k·σ-marginal)
        rn = torque_aware_retime(pt, q_u, qp_u, qpp_u, eff, vmax, k=0.0)    # null run: no sigma margin
        inf = bool((not rs.feasible) or rs.total_time > 1e6)                # infeasible under k*sigma within the velocity limit -> time undefined
        return dict(sigma_safe_time_s=(round(float(rs.total_time), 3) if not inf else None), sigma_safe_feasible=bool(rs.feasible),
                    nominal_torque_time_s=round(float(rn.total_time), 3), sigma_infeasible=inf,
                    sigma_time_cost_s=(round(float(rs.total_time - rn.total_time), 3) if not inf else None), retime_k_sigma=k)

    def plan_cartesian(self, target, q_start=None):
        """Cartesian EE goal (SE3) -> MotionPlan (smoothed, collision-checked, effort-checked, confidence-capped)."""
        if q_start is None:
            q_start = next((q for q in (self.sample() for _ in range(1500)) if self.free(q)), None)
            if q_start is None:
                return MotionPlan(ok=False, stage="no_free_start")
        cf = self.free if self.world_model is not None else None    # WorldModel path -> the planner sees the environment via self.free
        # 1) IK: Cartesian goal -> collision-free joint configuration
        q_goal, conv, ikfree, _ = ik_multistart(self.model, self.data, self.fid, target, self.lo, self.hi, self.rng, gm=self.gm, gd=self.gd, cfree=cf)
        oMf = fk(self.model, self.data, self.fid, q_goal)
        pos_err = float(np.linalg.norm(oMf.translation - target.translation))
        if not (conv and ikfree and pos_err < 1e-3):
            return MotionPlan(ok=False, stage="ik_unreachable", ik_pos_err_mm=round(pos_err * 1000, 3))
        # 2) coal-RRT + 3) smooth+safe (shortcut→blend→kollisions-revaliderad)
        path = rrt_connect(self.model, self.data, self.gm, self.gd, q_start, q_goal, self.lo, self.hi, self.rng, n_iter=6000, cfree=cf)
        if not path:
            return MotionPlan(ok=False, stage="planner_no_path", ik_pos_err_mm=round(pos_err * 1000, 3))
        wp, cs, S, _, safe = smooth_safe(path, self.model, self.data, self.gm, self.gd, self.rng, cfree=cf)
        # Use declared effort limits only when all are positive and finite;
        # otherwise retain the existing synthetic fallback.
        _effL0 = np.asarray(self.model.effortLimit[:self.nj])
        # Nonpositive or nonfinite velocity limits invalidate division-based
        # retiming. Use the existing positive fallback for those inputs.
        _velL0 = np.asarray(self.model.velocityLimit[:self.nj])
        _velL0 = _velL0 if (np.isfinite(_velL0).all() and (_velL0 > 0).all()) else np.full(self.nj, 10.0)
        T = time_param(cs, S, self.model, self.data, _effL0 if (np.isfinite(_effL0).all() and (_effL0 > 0).all()) else np.full(self.nj, 1e6),
                       _velL0, dt=0.01)
        traj = cs(np.linspace(0, S, max(50, int(T / 0.02))))
        T = self._retime(traj, T, cs=cs, S=S)                  # raise T for accel/jerk (analytic kinodynamics); configurations unchanged
        pk_acc, pk_jrk = self._analytic_peaks(cs, S, T)        # continuous-path peaks, for the report/compliance check
        coll_free = bool(safe) and all(self.free(q) for q in traj[::3])
        # 4) effort feasibility (monotone-safe friction)
        eff_feasible, peak, plan_sigma = self._effort(traj)
        # 5) plan confidence = MIN(twin fidelity, effort provenance, collision), capped by the weakest link
        tw_tier = wm_trust_cap("high", self.twin_name) if self.twin_name else "unvalidated"
        ef_tier, ef_prov = _eff_tier(self.model, self.nj)
        co_tier = "high" if coll_free else "unvalidated"
        # qualify the collision-free claim with the environment geometry: primitive-environment collision-free does
        # not transfer to real CAD meshes (mesh/SDF path via world_model)
        co_geom = "world_model_mesh" if self.world_model is not None else ("primitives_only" if self.n_obstacles > 0 else "free_space")
        conf = TIERS[min(TIERS.index(tw_tier), TIERS.index(ef_tier), TIERS.index(co_tier))]
        capped = min([("twin", tw_tier), ("effort", ef_tier), ("collision", co_tier)], key=lambda x: TIERS.index(x[1]))[0]
        ok = bool(coll_free and eff_feasible is not False)
        stage = "complete" if ok else ("collision_revalidation_failed" if not coll_free else "effort_infeasible")  # honest reason
        if plan_sigma is not None and cs is not None and S is not None:    # augment with the sigma-safe retime time
            _ssr = self._sigma_safe_retime(cs, S)
            if _ssr is not None:
                plan_sigma = {**plan_sigma, **_ssr}
        return MotionPlan(ok=ok, stage=stage, trajectory_q=traj, waypoints=len(wp),
                          total_time_s=round(float(T), 3), effort_feasible=eff_feasible, collision_free=coll_free,
                          plan_confidence=conf, plan_sigma=plan_sigma, capped_by=capped, ik_pos_err_mm=round(pos_err * 1000, 3),
                          provenance=dict(twin_tier=tw_tier, effort_tier=ef_tier, effort_provenance=ef_prov, collision_tier=co_tier, collision_geometry=co_geom, peak_ratio=peak,
                                          peak_accel=round(float(np.max(pk_acc)), 4), peak_jerk=round(float(np.max(pk_jrk)), 4),
                                          payload_kg=self.payload_kg))

    def _confidence(self, coll_free):
        """plan confidence = MIN(twin fidelity, effort provenance, collision), capped by the weakest link. -> (conf, capped_by)."""
        tw = wm_trust_cap("high", self.twin_name) if self.twin_name else "unvalidated"
        et, _ = _eff_tier(self.model, self.nv)
        ct = "high" if coll_free else "unvalidated"
        conf = TIERS[min(TIERS.index(tw), TIERS.index(et), TIERS.index(ct))]
        capped = min([("twin", tw), ("effort", et), ("collision", ct)], key=lambda x: TIERS.index(x[1]))[0]
        return conf, capped, tw, et, ct

    def plan_line(self, target, q_start=None, n=40, ori="slerp"):
        """Straight-line Cartesian motion: the EE follows a straight line in task space from the start pose to the
        target; orientation is SLERPed along the SO(3) geodesic (ori='slerp') or held at the target (ori='locked').
        Per-step warm-started full-pose IK, collision-checked (verifies free, not just convergence) plus time
        parametrisation. Distinct from plan_cartesian (free-space joint-space RRT). If a waypoint cannot be reached
        collision-free the result is ok=False with the location (cartesian_infeasible@s) rather than a shortcut."""
        if q_start is None:
            q_start = next((q for q in (self.sample() for _ in range(1500)) if self.free(q)), None)
            if q_start is None:
                return MotionPlan(ok=False, stage="no_free_start")
        oM0 = fk(self.model, self.data, self.fid, q_start)
        p0, R0 = oM0.translation.copy(), oM0.rotation.copy()
        p1, R1 = target.translation, target.rotation
        dR = pin.log3(R0.T @ R1)                                # SO(3) geodesic start -> target
        qs = [np.asarray(q_start, float)]; seed = np.asarray(q_start, float); fail_s = None
        ss = np.linspace(0, 1, n)
        for s in ss[1:]:
            Rs = R1 if ori == "locked" else R0 @ pin.exp3(s * dR)
            pose = pin.SE3(Rs, p0 + s * (p1 - p0))
            q, conv, free = ik_solve(self.model, self.data, self.fid, pose, seed, self.lo, self.hi, gm=self.gm, gd=self.gd)
            if not (conv and free):                            # require BOTH convergence AND collision-free
                fail_s = float(s); break
            qs.append(q); seed = q
        if fail_s is not None:
            return MotionPlan(ok=False, stage=f"cartesian_infeasible@s={fail_s:.2f}", waypoints=len(qs))
        traj = np.array(qs)
        # straightness and goal error (how well was the line followed?)
        devs = [float(np.linalg.norm(fk(self.model, self.data, self.fid, traj[i]).translation - (p0 + ss[i] * (p1 - p0)))) for i in range(len(traj))]
        oMf = fk(self.model, self.data, self.fid, traj[-1])
        pos_err = float(np.linalg.norm(oMf.translation - p1))
        coll_free = all(self.free(q) for q in traj)
        eff_feasible, peak, plan_sigma = self._effort(traj)
        # time parametrisation via blend (not smooth_safe: no shortcut that would cut the straight line)
        cs = S = None
        try:
            cs, S = blend([q for q in traj]); velL = np.asarray(self.model.velocityLimit[:self.nv])
            # Nonpositive or nonfinite velocity limits invalidate division-based
            # retiming. Use the existing positive fallback for those inputs.
            velL = velL if (np.isfinite(velL).all() and (velL > 0).all()) else np.full(self.nv, 10.0)
            # Use declared effort limits only when all are positive and finite;
            # otherwise retain the existing synthetic fallback.
            effL = np.asarray(self.model.effortLimit[:self.nv])
            effL = effL if (np.isfinite(effL).all() and (effL > 0).all()) else np.full(self.nv, 1e6)
            T = time_param(cs, S, self.model, self.data, effL, velL, dt=0.01)
        except Exception:
            T = 0.05 * len(traj); cs = S = None
        T = self._retime(traj, T, cs=cs, S=S)                 # accel/jerk limits (analytic when a spline is available)
        pk = dict()
        if cs is not None and S is not None:
            pa, pj = self._analytic_peaks(cs, S, T); pk = dict(peak_accel=round(float(np.max(pa)), 4), peak_jerk=round(float(np.max(pj)), 4))
        conf, capped, tw, et, ct = self._confidence(coll_free)
        ok = bool(coll_free and eff_feasible is not False and pos_err < 1e-3)
        stage = "complete" if ok else ("collision" if not coll_free else ("missed_target" if pos_err >= 1e-3 else "effort_infeasible"))
        return MotionPlan(ok=ok, stage=stage, trajectory_q=traj, waypoints=len(traj), total_time_s=round(float(T), 3),
                          effort_feasible=eff_feasible, collision_free=coll_free, plan_confidence=conf, plan_sigma=plan_sigma, capped_by=capped,
                          ik_pos_err_mm=round(pos_err * 1000, 3),
                          provenance=dict(twin_tier=tw, effort_tier=et, collision_tier=ct, peak_ratio=peak,
                                          cartesian_max_dev_mm=round(max(devs) * 1000, 3), orientation=ori, payload_kg=self.payload_kg, **pk))

    def servo_step(self, target, q, dq, dt=0.02, kp=6.0):
        """One reactive (closed-loop) control step towards a possibly MOVING Cartesian target, on pinocchio dynamics.
        Resolved-rate (damped J+) on the 6D pose error -> desired joint velocity -> acceleration -> inverse dynamics
        (rnea + friction) clipped against the effort limits -> integrate on the manifold with a collision brake.
        -> (q_next, dq_next, info)."""
        pin.forwardKinematics(self.model, self.data, q); pin.updateFramePlacements(self.model, self.data)
        err = pin.log6(self.data.oMf[self.fid].actInv(target)).vector        # 6D twist error in the EE (LOCAL) frame
        J = pin.computeFrameJacobian(self.model, self.data, q, self.fid)     # LOCAL 6xnv (same frame as err)
        v_des = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), kp * err)  # damped resolved-rate -> desired joint velocity
        # np.linalg.solve can silently return partial NaN on a degenerate J (e.g. from a corrupted q) instead of
        # raising. v_des also feeds the CBF safety filter's `slack = a@v_des-b` comparison below, where a NaN slack
        if not np.all(np.isfinite(v_des)):
            raise ValueError("servo_step: resolved-rate v_des is non-finite (NaN/inf) -- degenerate "
                             "Jacobian or corrupted q; refusing to command from a corrupted rate")
        if self.world_model is not None:                                     # SDF-gradient control-barrier safety filter (not
            # additive repulsion, which would create potential-field local minima): project dq_des minimally so the
            # nearest obstacle clearance cannot shrink too fast: d(sd)/dt = grad*(Jp*dq) >= -alpha*(sd-d_safe).
            d_safe, alpha = 0.04, 2.0
            P = self.robot_points(q)                                         # surface points (not geometry centres)
            pin.computeJointJacobians(self.model, self.data, q)              # robot_points already set FK; now joint Jacobians
            sd = self.world_model.sd_batch(P); g = self.world_model.grad_batch(P)
            worst = None; any_corrupted = False
            for k in np.where(sd < d_safe + 0.12)[0]:                        # only nearby surface points -> build the CBF constraints
                jid = int(self._pt_parent_joint[k])
                J6 = pin.getJointJacobian(self.model, self.data, jid, pin.LOCAL_WORLD_ALIGNED)
                r = P[k] - self.data.oMi[jid].translation
                rx = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
                a = g[k] @ (J6[:3] - rx @ J6[3:6])                           # 1×nv: a·dq = d(sd)/dt
                b = -alpha * (float(sd[k]) - d_safe)
                slack = float(a @ v_des - b)
                # A corrupted (NaN) slack must not take part in the worst-of-real-violations comparison at all:
                # coercing it to -inf would let it win permanently and mask a genuine later violation. Track it
                # separately and fail closed instead of projecting with unverifiable clearance data.
                if not np.isfinite(slack):
                    any_corrupted = True
                    continue
                if worst is None or slack < worst[2]:
                    worst = (a, b, slack)
            if any_corrupted:
                raise ValueError("servo_step: CBF safety filter found a corrupted (non-finite) gradient "
                                 "near an obstacle -- refusing to command with unverifiable clearance data")
            if worst is not None and worst[2] < 0.0:                         # violated -> minimal projection onto the half-space
                a, b, _ = worst
                v_des = v_des + (b - a @ v_des) / (float(a @ a) + 1e-9) * a
                # The projection itself can still produce a non-finite v_des if the winning gradient has a component
                # that stayed non-finite despite a finite dot product; re-check after projection.
                if not np.all(np.isfinite(v_des)):
                    raise ValueError("servo_step: CBF-projected v_des is non-finite (NaN/inf) -- "
                                     "corrupted world-model gradient at the winning worst-case point")
        a_cmd = (v_des - dq) / dt
        tau = pin.rnea(self.model, self.data, q, dq, a_cmd) + np.sign(dq) * self.Fc + dq * self.Fv
        eff = np.asarray(self.model.effortLimit[:self.nv]); escale = 1.0
        # Replace each invalid effort limit with the existing synthetic bound.
        # This preserves valid joint limits and includes every joint in scaling;
        # the fallback does not certify an undeclared physical actuator limit.
        eff_safe = np.where(np.isfinite(eff) & (eff > 0), eff, 1e6)
        escale = float(min(1.0, np.min(eff_safe / (np.abs(tau) + 1e-9))))     # scale acceleration down if effort is exceeded
        a_cmd = a_cmd * escale
        dq_next = dq + a_cmd * dt
        q_next = np.clip(pin.integrate(self.model, q, dq_next * dt), self.lo, self.hi)
        # Clipping does not sanitize nonfinite state. Reject it explicitly and
        # return zeros_like(dq), since multiplying a NaN velocity by zero remains NaN.
        if not (np.all(np.isfinite(q_next)) and np.all(np.isfinite(dq_next))):
            return q, np.zeros_like(dq), dict(pos_err=float(np.linalg.norm(err[:3])), ori_err=float(np.linalg.norm(err[3:])),
                                     collided=True, effort_scale=round(escale, 3))
        collided = not self.free(q_next)
        if collided:                                                        # reactive safety: stop at the obstacle
            q_next, dq_next = q, np.zeros_like(dq)
        return q_next, dq_next, dict(pos_err=float(np.linalg.norm(err[:3])), ori_err=float(np.linalg.norm(err[3:])),
                                     collided=collided, effort_scale=round(escale, 3))

    def servo(self, target, q_start=None, max_steps=600, dt=0.02, tol=1e-3):
        """Closed-loop reactive reach/track: iterates servo_step. target = SE3 (static) OR callable(t)->SE3 (moving
        target, which the open-loop planners cannot follow). -> (trajectory, reached, info). Reactive layer on top of
        (and independent from) the planner; the dynamics model is the analytic pinocchio one."""
        q = q_start if q_start is not None else next((s for s in (self.sample() for _ in range(1500)) if self.free(s)), None)
        if q is None:
            return None, False, {"stage": "no_free_start"}
        q = np.asarray(q, float); dq = np.zeros(self.nv); traj = [q.copy()]; info = {}
        moving = callable(target)
        for k in range(max_steps):
            tgt = target(k * dt) if moving else target
            q, dq, info = self.servo_step(tgt, q, dq, dt)
            traj.append(q.copy())
            if not moving and info["pos_err"] < tol and info["ori_err"] < 1e-2:
                return np.array(traj), True, info
        reached = info.get("pos_err", 9) < max(tol, 5e-3) and info.get("ori_err", 9) < 2e-2
        return np.array(traj), bool(reached), info
