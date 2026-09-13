"""Batched analytic gravity vector g(q) in pure torch (GPU) for serial arms.

Two parts, two environments:
  1) extract_gravparams(): runs where pinocchio is available and extracts, per joint, the constants M_i
     (SE3 joint placement in the parent joint frame), joint axis a_i, link mass m_i, COM c_i (in the joint
     frame), the gravity vector and the ancestor mask. Saved to data/<robot>_gravparams.npz.
  2) TorchGravity: runs where torch is available; loads the npz, does batched FK through the chain and
     applies the moment-arm formula
        g_j(q) = -sum_{i: j is an ancestor of i} m_i * gvec . (z_j x (p_com_i - p_j))
     where z_j is the world axis of joint j, p_j the joint origin and p_com_i the world COM of link i.
     This is exactly dE_pot/dq_j and is verified against pin.computeGeneralizedGravity before use.

CLI (extraction):
  python -m motion_engine.torch_gravity <robot.urdf> data/<robot>_gravparams.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

_AXIS_BY_NAME = {"RX": (1.0, 0.0, 0.0), "RY": (0.0, 1.0, 0.0), "RZ": (0.0, 0.0, 1.0)}


def extract_gravparams(urdf_path: str | Path, npz_out: str | Path) -> dict:
    """Extrahera gravitationskonstanter ur URDF via Pinocchio. Kraver pinocchio."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    nj = model.njoints - 1  # exkl. universe
    M_rot = np.zeros((nj, 3, 3))
    M_trans = np.zeros((nj, 3))
    axis = np.zeros((nj, 3))
    mass = np.zeros(nj)
    com = np.zeros((nj, 3))
    inertia = np.zeros((nj, 3, 3))  # rotationstroghet KRING COM, i ledramen
    parent = np.zeros(nj, dtype=np.int64)  # 0-baserat; -1 = varlden
    names = []
    for i in range(1, model.njoints):
        jm = model.joints[i]
        sn = jm.shortname()  # ex "JointModelRZ", "JointModelRevoluteUnaligned"
        if "Revolute" not in sn and not any(k in sn for k in ("RX", "RY", "RZ")):
            raise ValueError(f"led {model.names[i]}: {sn} stods ej (endast revolute)")
        if "Unaligned" in sn:
            # generiska JointModel-wrappern saknar .axis i python-bindningen →
            # read the axis (joint frame = URDF joint frame) straight from the URDF XML
            import xml.etree.ElementTree as ET
            axmap = {j.get("name"): j.find("axis").get("xyz")
                     for j in ET.parse(str(urdf_path)).getroot().iter("joint")
                     if j.get("type") in ("revolute", "continuous")
                     and j.find("axis") is not None}
            ax = np.array([float(v) for v in axmap[model.names[i]].split()])
            ax /= np.linalg.norm(ax)
        else:
            key = sn.replace("JointModel", "")
            if key not in _AXIS_BY_NAME:
                raise ValueError(f"okand revolute-typ {sn}")
            ax = np.array(_AXIS_BY_NAME[key])
        k = i - 1
        M = model.jointPlacements[i]
        M_rot[k] = M.rotation
        M_trans[k] = M.translation
        axis[k] = ax
        ine = model.inertias[i]  # inkl. ihopslagna fixed-lankar
        mass[k] = ine.mass
        com[k] = ine.lever
        inertia[k] = ine.inertia  # pin lagrar I kring COM i ledramen
        parent[k] = model.parents[i] - 1
        names.append(model.names[i])
    # anfader-mask: anc[j, i] = 1 om led j ar anfader-eller-sjalv till led i
    anc = np.zeros((nj, nj))
    for i in range(nj):
        j = i
        while j >= 0:
            anc[j, i] = 1.0
            j = parent[j]
    gvec = np.asarray(model.gravity.linear, dtype=float).reshape(3)
    out = dict(M_rot=M_rot, M_trans=M_trans, axis=axis, mass=mass, com=com,
               inertia=inertia, parent=parent, anc=anc, gvec=gvec,
               names=np.array(names), urdf=str(urdf_path))
    np.savez(Path(npz_out), **out)
    return out


class TorchGravity:
    """g(q) batchad i ren torch. q: (B, nj) -> g: (B, nj) [Nm].

    Konvention som Pinocchio: tau = M(q) qdd + C(q,qd) qd + g(q),
    dvs statisk hallning kraver tau = g(q).
    """

    def __init__(self, npz_path: str | Path, device="cuda", dtype=None):
        import torch
        self.torch = torch
        dtype = dtype or torch.float32
        d = np.load(npz_path, allow_pickle=False)
        t = lambda x: torch.tensor(np.asarray(x), dtype=dtype, device=device)
        self.R0 = t(d["M_rot"])        # (n,3,3)
        self.p0 = t(d["M_trans"])      # (n,3)
        self.axis = t(d["axis"])       # (n,3)
        self.mass = t(d["mass"])       # (n,)
        self.com = t(d["com"])         # (n,3)
        self.anc = t(d["anc"])         # (n,n)
        self.gvec = t(d["gvec"])       # (3,)
        self.parent = d["parent"]      # (n,) int, -1 = varld
        self.n = int(self.mass.shape[0])
        # precomputerad skevsymmetrisk K per led (for Rodrigues)
        K = torch.zeros(self.n, 3, 3, dtype=dtype, device=device)
        a = self.axis
        K[:, 0, 1], K[:, 0, 2] = -a[:, 2], a[:, 1]
        K[:, 1, 0], K[:, 1, 2] = a[:, 2], -a[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -a[:, 1], a[:, 0]
        self.K = K
        self.K2 = K @ K
        self.eye3 = torch.eye(3, dtype=dtype, device=device)

    def __call__(self, q):
        torch = self.torch
        B, n = q.shape
        assert n == self.n, f"q har {n} leder, modellen {self.n}"
        Rws = [None] * n
        zs, ps = [None] * n, [None] * n
        s, c = torch.sin(q), torch.cos(q)
        for i in range(n):
            par = self.parent[i]
            if par >= 0:
                Rp, pp = Rws[par], ps[par]
            else:
                Rp = self.eye3.expand(B, 3, 3)
                pp = torch.zeros(B, 3, dtype=q.dtype, device=q.device)
            # T_parent o M_i : ledorigo + ram FORE ledrotationen
            Rpre = Rp @ self.R0[i]                              # (B,3,3)
            p_i = pp + (Rp @ self.p0[i].unsqueeze(-1)).squeeze(-1)
            # Rodrigues kring axis_i
            Rot = (self.eye3 + s[:, i, None, None] * self.K[i]
                   + (1.0 - c[:, i, None, None]) * self.K2[i])
            Rw_i = Rpre @ Rot
            Rws[i] = Rw_i
            ps[i] = p_i
            zs[i] = (Rpre @ self.axis[i].unsqueeze(-1)).squeeze(-1)  # = Rw_i @ axis
        Rw_all = torch.stack(Rws, 1)              # (B,n,3,3)
        p_all = torch.stack(ps, 1)                # (B,n,3)
        z_all = torch.stack(zs, 1)                # (B,n,3)
        pcom = p_all + (Rw_all @ self.com.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        # momentarm: cross[b,j,i] = z_j x (pcom_i - p_j)
        diff = pcom[:, None, :, :] - p_all[:, :, None, :]        # (B,j,i,3)
        cr = torch.cross(z_all[:, :, None, :].expand_as(diff), diff, dim=-1)
        proj = -(cr * self.gvec).sum(-1)                          # (B,j,i)
        return (proj * self.mass * self.anc).sum(-1)              # (B,j)


def _main():
    import sys
    urdf = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "assets/robots/ur_description/ur10e.urdf")
    npz = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "data/ur10e_gravparams.npz")
    out = extract_gravparams(urdf, npz)
    print(f"{len(out['mass'])} leder -> {npz}")
    for nm, m, cv, ax in zip(out["names"], out["mass"], out["com"], out["axis"]):
        print(f"  {nm}: m={m:.3f} kg  com={np.round(cv,4)}  axis={ax}")
    print(f"  gvec={out['gvec']}")


if __name__ == "__main__":
    _main()
