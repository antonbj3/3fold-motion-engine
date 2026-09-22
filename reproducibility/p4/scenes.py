#!/usr/bin/env python3
"""Static contact scenes and their 6D wrench maps.

A scene is a list of rigid bodies plus a list of point contacts.  The wrench
map is assembled from first principles: a contact force f applied at world
point p to body a (and -f to body b) contributes to the bodies' 6D generalized
forces as [f; (p-com_a) x f] and -[f; (p-com_b) x f].  With per-contact rays
from friction_polygon, the columns of W are the generalized force of one unit
ray; the equilibrium is W @ lam = f0 + L * F @ d, where f0 is the supporting
weight wrench and -F @ d * L is the prescribed inertial load.

The script reads only its own raw/static_scenes.json; it does not import any
external scene code.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from friction_polygon import POLY_SCALE, contact_rays, K

HERE = Path(__file__).resolve().parent
G = 9.81


@dataclass
class Scene:
    name: str
    bodies: list          # [{"mass": float, "com": [x,y,z]}]
    contacts: list        # [{"body": int, "p": [x,y,z], "n": [x,y,z], "mu": float}]
    n_contacts: int

    @property
    def n_bodies(self):
        return len(self.bodies)

    @property
    def n_dof(self):
        return 6 * self.n_bodies


def _tangential_basis(n):
    n = np.asarray(n, float)
    hint = np.array([1.0, 0.0, 0.0]) if abs(n[2]) > 0.9 else np.array([0.0, 1.0, 0.0])
    t2 = np.cross(n, hint)
    t2 /= np.linalg.norm(t2)
    t1 = np.cross(t2, n)
    t1 /= np.linalg.norm(t1)
    return t1, t2


def contact_velocity_rows(scene):
    """(n_contacts*3, n_dof) kinematic Jacobian: generalized velocity -> v_rel."""
    nb = scene.n_bodies
    com = [np.asarray(b["com"], float) for b in scene.bodies]
    rows = []
    for c in scene.contacts:
        a = int(c["body"])
        b = int(c.get("support_body", -1))
        p = np.asarray(c["p"], float)
        n = np.asarray(c["n"], float)
        ra = p - com[a]
        rb = None if b < 0 else p - com[b]
        t1, t2 = _tangential_basis(n)
        for u in (n, t1, t2):
            row = np.zeros(6 * nb)
            row[6 * a:6 * a + 3] = u
            row[6 * a + 3:6 * a + 6] = np.cross(ra, u)
            if b >= 0:
                row[6 * b:6 * b + 3] = -u
                row[6 * b + 3:6 * b + 6] = -np.cross(rb, u)
            rows.append(row)
    return np.array(rows)


def assemble(scene):
    """Return (W, f0, F) with W @ lam = f0 + L * F @ d."""
    J = contact_velocity_rows(scene)
    W = np.zeros((scene.n_dof, scene.n_contacts * K))
    for i, c in enumerate(scene.contacts):
        rays = np.array(contact_rays(float(c["mu"])), float).T   # (3, K)
        W[:, i * K:(i + 1) * K] = J.T[:, 3 * i:3 * i + 3] @ rays
    f0 = np.zeros(scene.n_dof)
    F = np.zeros((scene.n_dof, 2))
    for k, b in enumerate(scene.bodies):
        f0[6 * k + 2] = b["mass"] * G
        F[6 * k, 0] = -b["mass"]
        F[6 * k + 1, 1] = -b["mass"]
    return W, f0, F


def load_raw(path=None):
    path = Path(path) if path else HERE / "raw" / "static_scenes.json"
    return json.loads(Path(path).read_text())


def iter_scenes(path=None):
    raw = load_raw(path)
    for s in raw["scenes"]:
        yield Scene(name=s["name"], bodies=s["bodies"], contacts=s["contacts"],
                    n_contacts=s["n_contacts"])


def polygon_info(path=None):
    return load_raw(path)["polygon"]
