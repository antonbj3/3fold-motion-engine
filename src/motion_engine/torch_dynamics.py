"""Batched analytic mass matrix M(q) in pure torch (GPU) for serial revolute arms.

Builds on the verified FK in torch_gravity.py and the same npz from extract_gravparams(), which also
carries each link's rotational inertia about its COM in the joint frame.

Jacobian form of the composite-rigid-body algorithm (identical result to pin.crba, but batch-friendly):

    M(q) = sum_i [ m_i Jv_i^T Jv_i + Jw_i^T (R_i I_i R_i^T) Jw_i ]

  where for revolute joint j (an ancestor of link i):
    Jw_i[:, j] = z_j
    Jv_i[:, j] = z_j x (p_com_i - p_j)
  and I_i is link i's rotational inertia about its COM in the joint frame, R_i the world rotation.

Verified against pin.crba on 1000 random configurations (float64, max error < 1e-6) before use.
M_and_g(q) shares one FK pass between M(q) and g(q).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .torch_gravity import TorchGravity

ROOT = Path(__file__).resolve().parents[2]


class TorchDynamics(TorchGravity):
    """M(q): (B,nj) -> (B,nj,nj); M_and_g(q) -> ((B,nj,nj), (B,nj)).

    Konvention som Pinocchio: tau = M(q) qdd + C(q,qd) qd + g(q).
    """

    def __init__(self, npz_path: str | Path, device="cuda", dtype=None):
        super().__init__(npz_path, device=device, dtype=dtype)
        torch = self.torch
        d = np.load(npz_path, allow_pickle=False)
        if "inertia" not in d:
            raise KeyError(
                f"{npz_path} saknar 'inertia' — kor om extraktionen "
                "(python -m motion_engine.torch_gravity <urdf> <npz>) i pinocchio-miljon")
        self.inertia = torch.tensor(np.asarray(d["inertia"]),
                                    dtype=self.eye3.dtype, device=device)  # (n,3,3)

    def _fk(self, q):
        """Batchad FK (samma slinga som TorchGravity.__call__).

        Returnerar Rw (B,n,3,3), p (B,n,3), z (B,n,3), pcom (B,n,3)."""
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
            Rpre = Rp @ self.R0[i]
            p_i = pp + (Rp @ self.p0[i].unsqueeze(-1)).squeeze(-1)
            Rot = (self.eye3 + s[:, i, None, None] * self.K[i]
                   + (1.0 - c[:, i, None, None]) * self.K2[i])
            Rws[i] = Rpre @ Rot
            ps[i] = p_i
            zs[i] = (Rpre @ self.axis[i].unsqueeze(-1)).squeeze(-1)
        Rw = torch.stack(Rws, 1)
        p = torch.stack(ps, 1)
        z = torch.stack(zs, 1)
        pcom = p + (Rw @ self.com.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        return Rw, p, z, pcom

    def _mass_from_fk(self, Rw, p, z, pcom):
        torch = self.torch
        # diff[b,j,i,:] = pcom_i - p_j ; cr[b,j,i,:] = z_j x (pcom_i - p_j)
        diff = pcom[:, None, :, :] - p[:, :, None, :]
        cr = torch.cross(z[:, :, None, :].expand_as(diff), diff, dim=-1)
        ancm = self.anc[None, :, :, None]                       # (1,j,i,1)
        Jv = (cr * ancm).permute(0, 2, 3, 1)                    # (B,i,3,j)
        Jw = (z[:, :, None, :].expand_as(diff) * ancm).permute(0, 2, 3, 1)
        Iw = Rw @ self.inertia.unsqueeze(0) @ Rw.transpose(-1, -2)  # (B,i,3,3)
        Mv = torch.einsum("bixj,i,bixk->bjk", Jv, self.mass, Jv)
        Mw = torch.einsum("bixj,bixy,biyk->bjk", Jw, Iw, Jw)
        return Mv + Mw

    def _grav_from_fk(self, p, z, pcom):
        torch = self.torch
        diff = pcom[:, None, :, :] - p[:, :, None, :]
        cr = torch.cross(z[:, :, None, :].expand_as(diff), diff, dim=-1)
        proj = -(cr * self.gvec).sum(-1)
        return (proj * self.mass * self.anc).sum(-1)

    def _bias_from_fk(self, Rw, p, z, pcom, qd):
        """RNEA med qdd=0 -> bias b(q,qd) = C(q,qd)qd + g(q)  (B,n) [Nm].

        Varldsram-rekursion; gravitation via bastricket a_0 = -gvec.
        Verified against pin.rnea(q, qd, 0)."""
        torch = self.torch
        B, n = qd.shape
        Iw = Rw @ self.inertia.unsqueeze(0) @ Rw.transpose(-1, -2)  # (B,n,3,3)
        zero3 = torch.zeros(B, 3, dtype=qd.dtype, device=qd.device)
        acc0 = (-self.gvec).expand(B, 3)
        w, al, a = [None] * n, [None] * n, [None] * n
        for i in range(n):
            par = self.parent[i]
            if par >= 0:
                wp_, ap_, aa_, pp_ = w[par], al[par], a[par], p[:, par]
            else:
                wp_, ap_, aa_, pp_ = zero3, zero3, acc0, zero3
            r = p[:, i] - pp_
            a[i] = (aa_ + torch.cross(ap_, r, dim=-1)
                    + torch.cross(wp_, torch.cross(wp_, r, dim=-1), dim=-1))
            w[i] = wp_ + z[:, i] * qd[:, i:i + 1]
            al[i] = ap_ + torch.cross(wp_, z[:, i], dim=-1) * qd[:, i:i + 1]
        W_ = torch.stack(w, 1); AL = torch.stack(al, 1); A = torch.stack(a, 1)
        d = pcom - p
        acom = (A + torch.cross(AL, d, dim=-1)
                + torch.cross(W_, torch.cross(W_, d, dim=-1), dim=-1))
        F = self.mass.view(1, -1, 1) * acom                       # (B,n,3)
        N = ((Iw @ AL.unsqueeze(-1)).squeeze(-1)
             + torch.cross(W_, (Iw @ W_.unsqueeze(-1)).squeeze(-1), dim=-1))
        diff = pcom[:, None, :, :] - p[:, :, None, :]
        cr = torch.cross(z[:, :, None, :].expand_as(diff), diff, dim=-1)
        term = (cr * F[:, None, :, :]).sum(-1) + (z[:, :, None, :] * N[:, None, :, :]).sum(-1)
        return (term * self.anc).sum(-1)

    def mass_matrix(self, q):
        Rw, p, z, pcom = self._fk(q)
        return self._mass_from_fk(Rw, p, z, pcom)

    def bias(self, q, qd):
        Rw, p, z, pcom = self._fk(q)
        return self._bias_from_fk(Rw, p, z, pcom, qd)

    def M_and_g(self, q):
        """One FK pass -> (M (B,n,n), g (B,n))."""
        Rw, p, z, pcom = self._fk(q)
        return self._mass_from_fk(Rw, p, z, pcom), self._grav_from_fk(p, z, pcom)

    def M_and_bias(self, q, qd):
        """En FK-pass -> (M (B,n,n), b (B,n)) dar b = C qd + g. For v5 --coriolis."""
        Rw, p, z, pcom = self._fk(q)
        return self._mass_from_fk(Rw, p, z, pcom), self._bias_from_fk(Rw, p, z, pcom, qd)
