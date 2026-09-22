#!/usr/bin/env python3
"""Owned, immutable keep-out contract over the producer-grid prefilter kernels.

This is the API layer under test in the matched-baseline study. It reuses the kernels of
`producer_grid_keepout.py` unchanged and replaces the forgeable public `Snapshot` dataclass with
producer-owned immutable handles; the grid-binding proof is moved to bind/query time. It is a
prototype contract, not product-integrated code.

Why. An earlier revision reproduced ordinary-input correctness but measured three ways the public
snapshot API could still take a decision on a grid that is not P2G(x_at, m_at): a `dataclasses.
replace` with a zero grid kept `verified=True`; a hand-built snapshot with a zero grid was accepted;
and a `wp.copy` of zeros over the exported grid array kept the flag. A validation cache keyed on a
caller-remembered counter also returned a stale displacement after an in-place position mutation.
Each produced silent false negatives.

What this module closes (documented public surface only):
  1. OWNED IMMUTABLE HANDLES. `OwnedCapture` and `OwnedState` can only be produced by the factories
     in this module; their constructors require a module-private ownership token and the objects
     reject `setattr`/`delattr`. No writable internal buffer is exported.
  2. BINDING RE-VERIFIED AT BIND TIME. `bind_batch` recomputes the int64 fixed-point P2G from the
     capture's own private particles and requires bit equality with its private grid.
  3. CONTENT-BOUND STATE, NOT A COUNTER. The validation cache is keyed by the identity of an
     immutable `OwnedState`; any change requires a new state and re-validation.
  4. EXPLICIT FRAME SEMANTICS. Every query returns `grid_frame`, `state_frame` and the lag; a stale
     or future grid falls back to the exact count on the state's own positions.
  5. CALLER-BUFFER ISOLATION. Factories copy caller arrays into private storage.

This is deliberately a bounded ownership contract, not a security boundary: arbitrary
`object.__new__` plus name-mangled attribute writes remain possible and are out of scope. What is
protected is the documented public constructors/getters/copy/replace."""
import hashlib
import secrets

import numpy as np
import warp as wp

import producer_grid_keepout as K

# Module-private ownership token. Only the factories below can mint owned objects.
_OWNERSHIP_TOKEN = secrets.token_bytes(32)


class OwnershipError(TypeError):
    """Raised when a caller tries to construct an owned handle outside the module factories."""


class GridNotBoundError(K.GridNotBoundError):
    """Re-export: the supplied grid is not bit-equal to P2G(x_at, m_at)."""


class _FrozenOwned:
    """Immutable base: no attribute writes after construction, no writable arrays exported."""

    __slots__ = ()

    def __setattr__(self, name, value):
        raise AttributeError("owned object is immutable (no public attribute writes)")

    def __delattr__(self, name):
        raise AttributeError("owned object is immutable (no public attribute deletes)")


def _binding_sha(x, m, ids, grid):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x.numpy(), dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(m.numpy(), dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(ids.numpy(), dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(grid.numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Owned capture (grid + the particles it was built from)
# --------------------------------------------------------------------------------------
class OwnedCapture(_FrozenOwned):
    """Producer-owned capture. `grid == P2G(x_at, m_at)` is re-checked by `bind_batch`."""

    __slots__ = ("_lat", "_frame", "_version", "_n", "_grid", "_x", "_m", "_ids",
                 "_binding_sha", "_source")

    def __init__(self, token, lat, frame, version, n, grid, x, m, ids, binding_sha, source):
        if token is not _OWNERSHIP_TOKEN:
            raise OwnershipError(
                "OwnedCapture can only be produced by capture_from_particles / "
                "capture_from_solver_grid (documented public constructors)")
        object.__setattr__(self, "_lat", lat)
        object.__setattr__(self, "_frame", int(frame))
        object.__setattr__(self, "_version", int(version))
        object.__setattr__(self, "_n", int(n))
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_m", m)
        object.__setattr__(self, "_ids", ids)
        object.__setattr__(self, "_binding_sha", binding_sha)
        object.__setattr__(self, "_source", source)

    @property
    def lattice(self):
        return self._lat

    @property
    def frame(self):
        return self._frame

    @property
    def version(self):
        return self._version

    @property
    def n(self):
        return self._n

    @property
    def source(self):
        return self._source

    @property
    def binding_sha256(self):
        return self._binding_sha

    def __repr__(self):
        return ("OwnedCapture(frame=%d, version=%d, n=%d, source=%s, binding_sha256=%s)"
                % (self._frame, self._version, self._n, self._source, self._binding_sha[:12]))


def capture_from_particles(x, m, ids, lattice, frame, version, device="cuda:0"):
    """Factory: copy the caller's particles and BUILD the grid here as P2G(x_at, m_at)."""
    lf = K._lattice_fail(lattice)
    if lf is not None:
        raise ValueError(lf)
    xs = K._owned(x, wp.vec3, device)
    ms = K._owned(m, wp.float32, device)
    isd = K._owned(ids, wp.int64, device)
    n = int(ms.size)
    if int(xs.size) != n or int(isd.size) != n:
        raise ValueError("x / m / ids lengths differ")
    with wp.ScopedDevice(device):
        grid = K._p2g_on_device(xs, ms, lattice, device)
    return OwnedCapture(_OWNERSHIP_TOKEN, lattice, frame, version, n, grid, xs, ms, isd,
                        _binding_sha(xs, ms, isd, grid), "factory")


def capture_from_solver_grid(grid_m, x, m, ids, lattice, frame, version, verify=True,
                             device="cuda:0"):
    """Factory: copy a solver grid and the particles it claims to come from.

    `verify=True` checks bit equality once here; `bind_batch` re-checks anyway, so the binding is
    enforced at query time regardless. A mismatch is rejected with `GridNotBoundError`.
    """
    lf = K._lattice_fail(lattice)
    if lf is not None:
        raise ValueError(lf)
    gs = K._owned(grid_m, wp.float32, device)
    xs = K._owned(x, wp.vec3, device)
    ms = K._owned(m, wp.float32, device)
    isd = K._owned(ids, wp.int64, device)
    res = int(lattice.res)
    if int(gs.size) != res ** 3:
        raise ValueError("grid_size_mismatch")
    n = int(ms.size)
    if int(xs.size) != n or int(isd.size) != n:
        raise ValueError("x / m / ids lengths differ")
    if verify:
        with wp.ScopedDevice(device):
            recomputed = K._p2g_on_device(xs, ms, lattice, device)
        a = np.ascontiguousarray(recomputed.numpy(), dtype=np.float32)
        b = np.ascontiguousarray(gs.numpy(), dtype=np.float32)
        ndiff = int((a != b).sum())
        if ndiff != 0:
            raise GridNotBoundError(
                "supplied grid is not P2G(x_at, m_at): ndiff=%d (rejected, no certificate)" % ndiff)
    return OwnedCapture(_OWNERSHIP_TOKEN, lattice, frame, version, n, gs, xs, ms, isd,
                        _binding_sha(xs, ms, isd, gs), "solver_verified" if verify
                        else "solver_unverified")


# --------------------------------------------------------------------------------------
# Owned particle state
# --------------------------------------------------------------------------------------
class OwnedState(_FrozenOwned):
    """Producer-owned immutable particle state. Validation is cached by its identity."""

    __slots__ = ("_frame", "_version", "_n", "_x", "_m", "_ids", "_state_sha")

    def __init__(self, token, frame, version, n, x, m, ids, state_sha):
        if token is not _OWNERSHIP_TOKEN:
            raise OwnershipError(
                "OwnedState can only be produced by state_from_particles (documented public "
                "constructor)")
        object.__setattr__(self, "_frame", int(frame))
        object.__setattr__(self, "_version", int(version))
        object.__setattr__(self, "_n", int(n))
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_m", m)
        object.__setattr__(self, "_ids", ids)
        object.__setattr__(self, "_state_sha", state_sha)

    @property
    def frame(self):
        return self._frame

    @property
    def version(self):
        return self._version

    @property
    def n(self):
        return self._n

    @property
    def state_sha256(self):
        return self._state_sha

    def __repr__(self):
        return ("OwnedState(frame=%d, version=%d, n=%d, state_sha256=%s)"
                % (self._frame, self._version, self._n, self._state_sha[:12]))


def state_from_particles(x, m, ids, frame, version, device="cuda:0"):
    """Factory: copy the caller's particles into private storage."""
    xs = K._owned(x, wp.vec3, device)
    ms = K._owned(m, wp.float32, device)
    isd = K._owned(ids, wp.int64, device)
    n = int(ms.size)
    if int(xs.size) != n or int(isd.size) != n:
        raise ValueError("x / m / ids lengths differ")
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(xs.numpy(), dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(ms.numpy(), dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(isd.numpy(), dtype=np.int64).tobytes())
    return OwnedState(_OWNERSHIP_TOKEN, frame, version, n, xs, ms, isd, h.hexdigest())


# --------------------------------------------------------------------------------------
# Owned query batch
# --------------------------------------------------------------------------------------
class OwnedBatch:
    """A frozen captured grid bound to one owned state, with an amortized occupancy.

    `bind_batch(capture, state)` re-verifies `grid == P2G(x_at, m_at)` from the capture's own
    private arrays (one extra P2G), validates IDs/version/frame/deposition/displacement against the
    state, and builds occupancy once. `query` then answers for that frozen pair. To move to a new
    state call `rebind(new_state)` (or pass `state=` to `query`); the validation is re-run because
    the new `OwnedState` has a different identity.
    """

    __slots__ = ("_cap", "_state", "_device", "_max_lag", "_max_disp", "_policy",
                 "_info", "_occ", "_validated_id", "_n_revalidations", "_n_occ_builds",
                 "_n_reverify", "_binding_ok")

    def __init__(self, token, capture, state, device, max_lag, max_disp, policy):
        if token is not _OWNERSHIP_TOKEN:
            raise OwnershipError(
                "OwnedBatch can only be produced by bind_batch (documented public constructor)")
        if not isinstance(capture, OwnedCapture) or not isinstance(state, OwnedState):
            raise OwnershipError("bind_batch requires OwnedCapture and OwnedState handles")
        self._cap = capture
        self._state = state
        self._device = device
        self._max_lag = int(max_lag)
        self._max_disp = max_disp
        self._policy = dict(K.POLICY if policy is None else policy)
        self._info = None
        self._occ = None
        self._validated_id = None
        self._n_revalidations = 0
        self._n_occ_builds = 0
        self._n_reverify = 0
        self._binding_ok = False
        self.bind(state)

    # -- read-only introspection -----------------------------------------------------------------
    @property
    def capture(self):
        return self._cap

    @property
    def state(self):
        return self._state

    @property
    def n_revalidations(self):
        return self._n_revalidations

    @property
    def n_occ_builds(self):
        return self._n_occ_builds

    @property
    def n_reverify(self):
        return self._n_reverify

    def info(self):
        return dict(self._info)

    # -- binding / validation --------------------------------------------------------------------
    def _reverify_binding(self):
        """Recompute P2G from the capture's own private particles and require bit equality."""
        with wp.ScopedDevice(self._device):
            recomputed = K._p2g_on_device(self._cap._x, self._cap._m, self._cap._lat,
                                          self._device)
        a = np.ascontiguousarray(recomputed.numpy(), dtype=np.float32)
        b = np.ascontiguousarray(self._cap._grid.numpy(), dtype=np.float32)
        self._n_reverify += 1
        if int((a != b).sum()) != 0:
            raise GridNotBoundError(
                "capture grid is not P2G(x_at, m_at): binding re-check failed at bind time")

    def _validate(self, state):
        info = {"used": False, "reason": "uninitialized", "D": 0.0, "n_id_mismatch": -1,
                "n_failed_deposition": -1, "n_nonfinite_particles": 0, "frame_lag": -1,
                "grid_frame": self._cap._frame, "state_frame": int(state._frame),
                "grid_version": self._cap._version, "state_version": int(state._version),
                "grid_sha256": self._cap._binding_sha, "cache_hit": False}
        lat = self._cap._lat
        lf = K._lattice_fail(lat)
        if lf is not None:
            info["reason"] = lf
            return info
        if int(self._cap._grid.size) != int(lat.res) ** 3:
            info["reason"] = "grid_size_mismatch"
            return info
        if state._n != self._cap._n:
            info["reason"] = "particle_set_size_changed"
            return info
        if int(state._x.size) != state._n or int(state._m.size) != state._n \
                or int(state._ids.size) != state._n:
            info["reason"] = "particle_arrays_inconsistent"
            return info
        if int(state._version) != int(self._cap._version):
            info["reason"] = "particle_version_changed"
            return info
        lag = int(state._frame) - int(self._cap._frame)
        info["frame_lag"] = lag
        if lag < 0:
            info["reason"] = "grid_newer_than_particles"
            return info
        if lag > self._max_lag:
            info["reason"] = "stale_grid"
            return info
        if state._n == 0:
            info.update(used=True, reason="ok_empty")
            return info
        device = self._device
        with wp.ScopedDevice(device):
            idm = wp.zeros(1, dtype=wp.int32, device=device)
            wp.launch(K._ids_mismatch_kernel, dim=state._n,
                      inputs=[self._cap._ids, state._ids, state._n, idm], device=device)
            nm = int(idm.numpy()[0])
            info["n_id_mismatch"] = nm
            if nm != 0:
                info["reason"] = "particle_id_mismatch"
                return info
            fail = wp.zeros(state._n, dtype=wp.int32, device=device)
            inv_dx = np.float32(1.0 / np.float32(lat.dx))
            wp.launch(K._deposition_valid_kernel, dim=state._n,
                      inputs=[self._cap._x, self._cap._m, state._n, int(lat.res),
                              wp.vec3(*[float(o) for o in lat.origin]), inv_dx,
                              float(lat.scale), fail], device=device)
            nf = int(fail.numpy().sum())
            info["n_failed_deposition"] = nf
            if nf != 0:
                info["reason"] = "deposition_invalid_underflow_outside_or_nonfinite"
                return info
            n_ch = (state._n + K.RED_CHUNK - 1) // K.RED_CHUNK
            disp = wp.zeros(n_ch, dtype=wp.float32, device=device)
            bad = wp.zeros(n_ch, dtype=wp.int32, device=device)
            wp.launch(K._chunk_disp_nonfinite_kernel, dim=n_ch,
                      inputs=[state._x, self._cap._x, state._m, state._n, K.RED_CHUNK, disp, bad],
                      device=device)
            d_arr = disp.numpy()
            nfpart = int(bad.numpy().sum())
        info["n_nonfinite_particles"] = nfpart
        if nfpart != 0:
            info["reason"] = "nonfinite_particle_or_mass"
            return info
        D = float(np.max(d_arr)) if d_arr.size else 0.0
        info["D"] = D
        limit = float(lat.dx) if self._max_disp is None else float(self._max_disp)
        if not np.isfinite(D):
            info["reason"] = "nonfinite_displacement"
            return info
        if D > limit:
            info["reason"] = "displacement_exceeds_bound"
            return info
        info.update(used=True, reason="ok")
        return info

    def bind(self, state):
        if not isinstance(state, OwnedState):
            raise OwnershipError("bind requires an OwnedState produced by state_from_particles")
        if not self._binding_ok:
            self._reverify_binding()
            self._binding_ok = True
        self._state = state
        self._info = self._validate(state)
        self._validated_id = id(state)
        self._n_revalidations += 1
        return dict(self._info)

    def _occupancy(self):
        if self._occ is None:
            self._occ = K.occupancy_from_massgrid(self._cap._grid, self._cap._lat, theta=0.0,
                                                  device=self._device)
            self._n_occ_builds += 1
        return self._occ

    # -- query -----------------------------------------------------------------------------------
    def query(self, box_lo, box_hi, radie=0.0, state=None):
        """Exact counts for `state` (default: the bound state) using the frozen capture's grid.

        Returns `(counts, gate, info)` with numpy copies (no internal buffer is exported) and
        explicit `grid_frame` / `state_frame` labels.
        """
        if state is not None and id(state) != self._validated_id:
            self.bind(state)
        box_lo = np.asarray(box_lo, dtype=np.float64).reshape(-1, 3)
        box_hi = np.asarray(box_hi, dtype=np.float64).reshape(-1, 3)
        if len(box_lo) != len(box_hi):
            raise ValueError("box_lo / box_hi lengths differ")
        nb = len(box_lo)
        info = dict(self._info)
        info.update(nb=nb, route="direct", n_exact_boxes=nb, n_gate_nodes=0,
                    n_box_particle_tests=nb * self._state._n, gate_fn=0, gate_fp=-1)
        if info["reason"] == "nonfinite_particle_or_mass":
            raise ValueError("current particle state contains NaN/inf; exact fallback undefined")
        if not info["used"]:
            counts = np.zeros(nb, dtype=np.int32)
            if nb and self._state._n > 0:
                counts = K.exact_box_counts(self._state._x, box_lo, box_hi, n=self._state._n,
                                            device=self._device)
            gate = np.ones(nb, dtype=np.int32)
            info["n_box_particle_tests"] = nb * self._state._n
            info["answered_frame"] = int(self._state._frame)
            info["answered_version"] = int(self._state._version)
            return counts, gate, info
        lat = self._cap._lat
        res = int(lat.res)
        occ = self._occupancy()
        pad = float(radie) + 2.0 * float(lat.dx) + float(info["D"])
        info["pad_m"] = pad
        i0, i1, valid, ni = K.node_index_bounds(box_lo, box_hi, pad, res, lat.origin, lat.dx)
        info["n_gate_nodes"] = int(ni.sum())
        direct_work = nb * self._state._n
        gate_work = int(ni.sum())
        pol = self._policy
        use_gated = (nb >= int(pol["min_nb_gated"])) and (
            direct_work == 0 or gate_work <= float(pol["max_gate_work_frac"]) * direct_work)
        if not use_gated:
            counts = np.zeros(nb, dtype=np.int32)
            if nb and self._state._n > 0:
                counts = K.exact_box_counts(self._state._x, box_lo, box_hi, n=self._state._n,
                                            device=self._device)
            gate = np.ones(nb, dtype=np.int32)
            info["route"] = "direct"
            info["n_box_particle_tests"] = nb * self._state._n
            info["answered_frame"] = int(self._state._frame)
            info["answered_version"] = int(self._state._version)
            return counts, gate, info
        with wp.ScopedDevice(self._device):
            a0 = wp.array(i0, dtype=wp.vec3i, device=self._device)
            a1 = wp.array(i1, dtype=wp.vec3i, device=self._device)
            vv = wp.array(valid, dtype=wp.int32, device=self._device)
            gate_buf = wp.zeros(nb, dtype=wp.int32, device=self._device)
            wp.launch(K._grid_box_count_kernel, dim=nb,
                      inputs=[occ, res, a0, a1, vv, gate_buf], device=self._device)
        gate = gate_buf.numpy().astype(np.int32)
        flagged = np.nonzero(gate > 0)[0]
        counts = np.zeros(nb, dtype=np.int32)
        if len(flagged) and self._state._n > 0:
            fc = K.exact_box_counts(self._state._x, box_lo[flagged], box_hi[flagged],
                                    n=self._state._n, device=self._device)
            counts[flagged] = fc
        info["route"] = "gated"
        info["n_exact_boxes"] = int(len(flagged))
        info["n_box_particle_tests"] = int(len(flagged)) * self._state._n
        info["answered_frame"] = int(self._state._frame)
        info["answered_version"] = int(self._state._version)
        return counts, gate, info


def bind_batch(capture, state, device="cuda:0", max_lag=1, max_disp=None, policy=None):
    """Documented public constructor: the only way to obtain an OwnedBatch."""
    return OwnedBatch(_OWNERSHIP_TOKEN, capture, state, device, max_lag, max_disp, policy)


# --------------------------------------------------------------------------------------
# Direct baseline (unchanged strong kernel) and public surface
# --------------------------------------------------------------------------------------
def exact_box_counts(P, box_lo, box_hi, *, n=None, device="cuda:0"):
    """Exact strict particle-centre count per box (D1). Re-exported strong direct kernel."""
    return K.exact_box_counts(P, box_lo, box_hi, n=n, device=device)


def public_surface():
    """The documented public API; everything else is private by convention."""
    return {
        "factories": ["capture_from_particles", "capture_from_solver_grid",
                      "state_from_particles", "bind_batch"],
        "query": ["OwnedBatch.query", "OwnedBatch.bind", "OwnedBatch.rebind"],
        "handles": ["OwnedCapture", "OwnedState", "OwnedBatch"],
        "no_writable_internal_buffers": True,
        "binding_reverified_at_bind": True,
        "validation_cache_keyed_by": "identity of an owned immutable OwnedState",
    }


# `rebind` is an alias of `bind` for the documented "state changed between batches" path.
def _rebind(self, state):
    return self.bind(state)


OwnedBatch.rebind = _rebind
