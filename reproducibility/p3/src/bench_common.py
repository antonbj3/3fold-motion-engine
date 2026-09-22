#!/usr/bin/env python3
"""Shared constants, independent strict-D1 oracle and frozen-state loader.

This module is used by the correctness and benchmark replay scripts. It imports the independent
reference (`range_reference`) but neither the producer module nor the owned contract. The truth
reference is the NumPy D1 predicate written here (`d1_all`), cross-checked against the independent
`range_reference.d1_counts_all`; the path under test is never used as the truth.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import range_reference as C  # independent reference: constants and box generators

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = PKG / "data"
STATES = DATA / "states"
MEAS = DATA / "measurements"

RES = int(C.RES)                       # 64
ORIGIN = np.asarray(C.ORIGIN, dtype=np.float64)
DX = float(C.DX)                       # 0.0125
RHO = float(C.RHO)
DOMAIN = float(C.DOMAIN)

STATE_FILES = {
    "120": "state_frame_120.npz",
    "140": "state_frame_140.npz",
    "150": "state_frame_150.npz",
    "160": "state_frame_160.npz",
    "160geom": "state_frame_160_shifted.npz",
}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def state_hashes():
    p = STATES / "state_hashes.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def load_frame(frame_key):
    """Load a frozen state. Returns dict with x_at, x_now, mass, ids, grid, n and its hash."""
    npz = STATE_FILES[str(frame_key)]
    p = STATES / npz
    h = sha_file(p)
    exp = state_hashes().get(npz)
    if exp is not None:
        assert h == exp, "state moved: %s %s != %s" % (npz, h, exp)
    d = np.load(p, allow_pickle=False)
    return {
        "key": str(frame_key), "path": npz, "sha256": h,
        "x_at": np.ascontiguousarray(d["x_at"], dtype=np.float32),
        "x_now": np.ascontiguousarray(d["x_now"], dtype=np.float32),
        "mass": np.ascontiguousarray(d["mass"], dtype=np.float32),
        "ids": np.ascontiguousarray(d["ids"], dtype=np.int64),
        "grid": np.ascontiguousarray(d["grid_resident"], dtype=np.float32),
        "n": int(d["n"]),
    }


def box_config():
    """The frozen box-geometry families and lattice (public subset of the pre-registration)."""
    return json.loads((MEAS / "box_geometry_families.json").read_text())


def d1_all(P, box_lo, box_hi):
    """Independent strict-D1 count per box: particle CENTRE in the closed box [lo, hi]."""
    p = np.asarray(P, dtype=np.float32).astype(np.float64).reshape(-1, 3)
    blo = np.asarray(box_lo, dtype=np.float64).reshape(-1, 3)
    bhi = np.asarray(box_hi, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(len(blo), dtype=np.int64)
    if p.size == 0:
        return out
    for b in range(len(blo)):
        out[b] = int(np.all((p >= blo[b]) & (p <= bhi[b]), axis=1).sum())
    return out


def boxes_for_dist(cfg, nb, seed):
    """Same generator dispatch as the pre-registration, from the frozen family config."""
    region = (np.asarray(cfg["region"][0], dtype=np.float64),
              np.asarray(cfg["region"][1], dtype=np.float64))
    shape = cfg.get("shape", "box")
    if shape == "slab":
        return C.boxes_slab(nb, seed, region, thin=cfg.get("thin", 0.02),
                            min_frac=cfg["min_frac"], max_frac=cfg["max_frac"])
    if shape == "corner":
        return C.boxes_corner(nb, seed, region, cfg["min_frac"], cfg["max_frac"])
    if shape == "empty":
        return C.boxes_empty(nb, seed)
    return C.boxes_for(nb, seed, region, cfg["min_frac"], cfg["max_frac"])
