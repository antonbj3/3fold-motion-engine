#!/usr/bin/env python3
"""World model: hot-swappable collision/scene world behind one data-oriented Protocol.

The planner and the servo query one `WorldModel` surface (sd/grad/free, single and batched) instead of
being hard-wired to coal. The contract is data in, data out (numpy arrays, poses) and never leaks coal or
pinocchio objects, so another backend (Rust/C++/Warp GPU/JAX) drops in behind the same signature.

Two CPU backends are included:
  * PrimitiveExactWorld — analytic SDFs (box/sphere/cylinder/capsule), exact, near-closed-form gradient.
  * EdtVoxelWorld — voxelised ESDF via scipy distance_transform_edt (the CPU twin of the GPU grid path).
EdtVoxel.sd must agree with PrimitiveExact.sd within the voxel tolerance (cross-check).
"""
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class WorldModel(Protocol):
    """KONTRAKT-YTA (konsumenten ser ALDRIG impl). Allt DATA in/ut → backend-agnostisk (Python/Rust/C++/Warp/JAX)."""
    name: str
    def sd_batch(self, P) -> np.ndarray: ...        # (N,3)->(N,) signed distance, <0 inuti hinder
    def grad_batch(self, P) -> np.ndarray: ...      # (N,3)->(N,3) outward gradient (d sd / d p)
    def free(self, P, margin: float = 0.0) -> bool: ...  # all points have sd > margin


# ───────────────────────── analytiska primitiv-SDF:er (exakt) ─────────────────────────
def _sd_box(pl, h):
    q = np.abs(pl) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)

def _sd_sphere(pl, r):
    return np.linalg.norm(pl, axis=-1) - r

def _sd_cylinder(pl, r, hh):                          # z axis, radius r, half-height hh
    d_xy = np.linalg.norm(pl[..., :2], axis=-1) - r
    d_z = np.abs(pl[..., 2]) - hh
    out = np.linalg.norm(np.stack([np.maximum(d_xy, 0), np.maximum(d_z, 0)], -1), axis=-1)
    return out + np.minimum(np.maximum(d_xy, d_z), 0.0)

def _sd_capsule(pl, r, hh):                           # segment along z in [-hh,hh], radius r
    pz = np.clip(pl[..., 2], -hh, hh)
    seg = np.stack([np.zeros_like(pz), np.zeros_like(pz), pz], -1)
    return np.linalg.norm(pl - seg, axis=-1) - r

_SD = {"box": lambda pl, d: _sd_box(pl, np.asarray(d, float) / 2.0),
       "sphere": lambda pl, d: _sd_sphere(pl, float(d[0] if hasattr(d, "__len__") else d)),
       "cylinder": lambda pl, d: _sd_cylinder(pl, float(d[0]), float(d[1]) / 2.0),
       "capsule": lambda pl, d: _sd_capsule(pl, float(d[0]), float(d[1]) / 2.0)}


class PrimitiveExactWorld:
    """Reference backend: analytic SDFs for box/sphere/cylinder/capsule. Exact sd; gradient by central difference.
    obstacles=[{type,dims,pose,rot?}] (the same spec as MotionStack._obstacle_geom)."""
    name = "primitive_exact"

    def __init__(self, obstacles):
        self.obs = []
        for o in obstacles:
            R = np.asarray(o["rot"], float) if "rot" in o else np.eye(3)
            c = np.asarray(o.get("pose", [0, 0, 0]), float)
            self.obs.append((o.get("type", "box"), o["dims"], c, R))

    def sd_batch(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        if not self.obs:
            return np.full(len(P), np.inf)
        sds = []
        for t, dims, c, R in self.obs:
            pl = (P - c) @ R                          # world -> obstacle-local (R orthonormal: p_local = R^T (p-c))
            sds.append(_SD[t](pl, dims))
        return np.min(np.stack(sds, -1), axis=-1)

    def grad_batch(self, P, eps=1e-5):
        P = np.atleast_2d(np.asarray(P, float)); g = np.zeros_like(P)
        for k in range(3):
            d = np.zeros(3); d[k] = eps
            g[:, k] = (self.sd_batch(P + d) - self.sd_batch(P - d)) / (2 * eps)
        return g

    def free(self, P, margin=0.0):
        return bool((self.sd_batch(P) > margin).all())


class EdtVoxelWorld:
    """Grid backend: voxelised ESDF via scipy distance_transform_edt (CPU twin of the GPU grid path).
    Built from an occupancy function (e.g. a PrimitiveExactWorld). Trilinear sd and grad. res = voxel size (m)."""
    name = "edt_voxel"

    def __init__(self, bounds_min, bounds_max, res, occ_fn):
        from scipy.ndimage import distance_transform_edt
        self.lo = np.asarray(bounds_min, float); self.hi = np.asarray(bounds_max, float); self.res = float(res)
        n = np.maximum(np.ceil((self.hi - self.lo) / self.res).astype(int), 1)
        self.n = n
        xs = [self.lo[i] + (np.arange(n[i]) + 0.5) * self.res for i in range(3)]
        gx, gy, gz = np.meshgrid(xs[0], xs[1], xs[2], indexing="ij")
        centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1)
        occ = np.asarray(occ_fn(centers), bool).reshape(n)
        out = distance_transform_edt(~occ) * self.res
        inn = distance_transform_edt(occ) * self.res
        self.sdf = (out - inn).astype(float)          # >0 outside, <0 inside
        self.gsdf = np.stack(np.gradient(self.sdf, self.res), -1)  # (nx,ny,nz,3)

    def _idx(self, P):
        f = (np.asarray(P, float) - self.lo) / self.res - 0.5
        return np.clip(f, 0, self.n - 1.0)

    def sd_batch(self, P):
        from scipy.ndimage import map_coordinates
        c = self._idx(P).T
        return map_coordinates(self.sdf, c, order=1, mode="nearest")

    def grad_batch(self, P):
        from scipy.ndimage import map_coordinates
        c = self._idx(P).T
        return np.stack([map_coordinates(self.gsdf[..., k], c, order=1, mode="nearest") for k in range(3)], -1)

    def free(self, P, margin=0.0):
        return bool((self.sd_batch(P) > margin).all())
