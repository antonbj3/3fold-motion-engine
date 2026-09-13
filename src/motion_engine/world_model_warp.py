#!/usr/bin/env python3
"""GPU backend for the WorldModel contract: a dense signed-distance grid, sampled at voxel corners from any exact
world, is loaded into a NanoVDB volume; sd_batch and grad_batch run as one Warp kernel with trilinear sampling.
Inputs and outputs are numpy arrays; the device (Warp CPU or CUDA) is a constructor argument.
"""
import numpy as np

import warp as wp

wp.init()


@wp.kernel
def _sample_sd(vol: wp.uint64, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    uvw = wp.volume_world_to_index(vol, pts[i])              # world -> index (NanoVDB intrinsic affine transform)
    out[i] = wp.volume_sample_f(vol, uvw, wp.Volume.LINEAR)  # trilinear sd


@wp.kernel
def _sample_sd_grad(vol: wp.uint64, pts: wp.array(dtype=wp.vec3), inv_vs: float,
                    out: wp.array(dtype=wp.float32), grad: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    uvw = wp.volume_world_to_index(vol, pts[i])
    g = wp.vec3(0.0, 0.0, 0.0)
    out[i] = wp.volume_sample_grad_f(vol, uvw, wp.Volume.LINEAR, g)
    grad[i] = g * inv_vs                                     # index gradient -> world gradient (isotropic voxel: /voxel_size)


class WarpVolumeWorld:
    """GPU-backend (NanoVDB) bakom WorldModel-Protocol. Bygg via from_world_model (orakel-grid) eller direkt (sdf_grid).
    device='cuda:0' (GPU-throughput) el. 'cpu' (Warp-CPU-tvilling, samma kod). numpy in/ut — konsumenten ser aldrig warp."""
    name = "warp_volume"

    def __init__(self, sdf_grid, min_world, voxel_size, device=None, bg_value=None):
        if device is None:
            device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
        self.device = device
        self.voxel_size = float(voxel_size)
        self.min_world = np.asarray(min_world, float)
        grid = np.ascontiguousarray(np.asarray(sdf_grid, np.float32))
        self.shape = grid.shape
        if bg_value is None:
            bg_value = float(grid.max())                     # outside active voxels = far from any obstacle (large positive sd)
        with wp.ScopedDevice(device):
            self.vol = wp.Volume.load_from_numpy(grid, min_world=tuple(self.min_world),
                                                 voxel_size=self.voxel_size, bg_value=bg_value)
        self._vid = self.vol.id

    @classmethod
    def from_world_model(cls, oracle, bounds_min, bounds_max, res, device=None):
        """Build the GPU volume from an oracle (any WorldModel, e.g. PrimitiveExactWorld). Grid values are the true sd at
        voxel corners (i,j,k) <-> min_world + (i,j,k)*res (NanoVDB load_from_numpy corner convention, so world_to_index matches)."""
        lo = np.asarray(bounds_min, float); hi = np.asarray(bounds_max, float); res = float(res)
        # H281 (portad C/D->I av S4 2026-08-22, BRYGGA 4; D 7a3def57d): en NaN eller inverterad (lo>hi)
        # a degenerate bound on any axis, e.g. a caller deriving bounds from V.min(0)/V.max(0) on corrupt input,
        # vertexdata -- avvisas INTE; np.maximum(...,2) klampar tyst axelns voxelantal till degenererat 2
        # collapses the GPU volume along that dimension. `lo<=hi` is False for NaN, which catches both.
        if not np.all(lo <= hi):
            raise ValueError(f"WarpVolumeWorld.from_world_model: bounds_min>bounds_max (or non-finite) "
                             f"on some axis: lo={lo}, hi={hi}")
        n = np.maximum(np.ceil((hi - lo) / res).astype(int) + 1, 2)
        xs = [lo[i] + np.arange(n[i]) * res for i in range(3)]
        gx, gy, gz = np.meshgrid(xs[0], xs[1], xs[2], indexing="ij")
        centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1)
        sdf = np.asarray(oracle.sd_batch(centers), np.float32).reshape(tuple(n))
        return cls(sdf, lo, res, device=device)

    def sd_batch(self, P):
        P = np.atleast_2d(np.asarray(P, np.float32))
        with wp.ScopedDevice(self.device):
            pts = wp.array(P, dtype=wp.vec3)
            out = wp.empty(len(P), dtype=wp.float32)
            wp.launch(_sample_sd, dim=len(P), inputs=[self._vid, pts], outputs=[out])
            return out.numpy().astype(float)

    def grad_batch(self, P):
        P = np.atleast_2d(np.asarray(P, np.float32))
        with wp.ScopedDevice(self.device):
            pts = wp.array(P, dtype=wp.vec3)
            out = wp.empty(len(P), dtype=wp.float32)
            grad = wp.empty(len(P), dtype=wp.vec3)
            wp.launch(_sample_sd_grad, dim=len(P), inputs=[self._vid, pts, 1.0 / self.voxel_size],
                      outputs=[out, grad])
            return grad.numpy().astype(float)

    def free(self, P, margin=0.0):
        return bool((self.sd_batch(P) > margin).all())
