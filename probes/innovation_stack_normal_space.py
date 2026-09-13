"""Measure the frozen normal-contact row space; never apply a correction."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from engine_metrics import DIMS, m5_penetration
from innovation_stack_probe import idle
from motion_engine.contact_engine_gpu_colored import GraphColoredContactEngine

SAMPLES = (1, 20, 100, 300, 590)
RANK_REL = 1e-10


class SnapshotEngine(GraphColoredContactEngine):
    def __init__(self):
        super().__init__(dims=DIMS, mu=.5, vit=40, pit=10,
                         manifold_reduce=True, graph_capture=True, pair_chunks=True)
        self.calls = 0
        self.snapshots = []

    def _color_pairs(self, pairs):
        colors = super()._color_pairs(pairs)
        self.calls += 1
        if self.calls in SAMPLES:
            c = self.n_contacts_solved
            snapshot = {"call": self.calls, "pairs": pairs, "contacts": c}
            for name in ("sbi", "sbj", "spA", "spB", "sn", "pvalP"):
                snapshot[name] = getattr(self, name).numpy()[:c].copy()
            for name in ("pstart", "pcount"):
                snapshot[name] = getattr(self, name).numpy()[:pairs].copy()
            for name in ("xc", "invM", "invIw", "v", "w"):
                snapshot[name] = getattr(self, name).numpy().copy()
            self.snapshots.append(snapshot)
        return colors


def basis(a):
    _, singular, vh = np.linalg.svd(a, full_matrices=False)
    rank = int(np.count_nonzero(singular > RANK_REL * singular[0])) if len(singular) and singular[0] else 0
    return vh[:rank], singular, rank


def measure(s):
    n, c, p = len(s["xc"]), s["contacts"], s["pairs"]
    jac = np.zeros((c, n, 6))
    for i in range(c):
        a, b, normal = s["sbi"][i], s["sbj"][i], s["sn"][i].astype(float)
        jac[i, a, :3] = normal
        jac[i, a, 3:] = np.cross(s["spA"][i].astype(float)-s["xc"][a], normal)
        if b >= 0:
            jac[i, b, :3] = -normal
            jac[i, b, 3:] = -np.cross(s["spB"][i].astype(float)-s["xc"][b], normal)
    # The symmetric mass square root makes Euclidean projections comparable.
    eig, vec = np.linalg.eigh(s["invIw"].astype(float))
    if np.any(eig <= 0) or np.any(s["invM"] <= 0):
        raise RuntimeError("Positive dynamic inverse mass/inertia required")
    root_i = (vec * np.sqrt(eig)[:, None, :]) @ vec.transpose(0, 2, 1)
    weighted = jac.copy()
    weighted[:, :, :3] *= np.sqrt(s["invM"].astype(float))[None, :, None]
    weighted[:, :, 3:] = np.einsum("cni,nij->cnj", jac[:, :, 3:], root_i)
    full = weighted.reshape(c, -1)
    mean = []
    for start, count in zip(s["pstart"], s["pcount"]):
        ids = s["pvalP"][start:start+count]
        mean.append(full[ids].mean(axis=0))
    coarse = np.asarray(mean)
    bfull, _, rank_full = basis(full)
    bmean, _, rank_mean = basis(coarse)
    relative_loss = np.linalg.norm(full-(full @ bmean.T) @ bmean) / np.linalg.norm(full)
    velocity = np.concatenate((s["v"], s["w"]), axis=1).astype(float)
    scaled_velocity = velocity.copy()
    scaled_velocity[:, :3] /= np.sqrt(s["invM"].astype(float))[:, None]
    scaled_velocity[:, 3:] = np.linalg.solve(root_i, velocity[:, 3:, None])[:, :, 0]
    u = scaled_velocity.ravel()
    full_component = (u @ bfull.T) @ bfull
    missed = full_component - (full_component @ bmean.T) @ bmean
    digest = hashlib.sha256()
    for key in sorted(s):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(s[key]).tobytes())
    return {"contact_call": s["call"], "contacts": c, "pairs": p,
            "normal_rank": rank_full, "pair_mean_rank": rank_mean,
            "jacobian_relative_projection_loss": float(relative_loss),
            "normal_velocity_component_norm": float(np.linalg.norm(full_component)),
            "missed_velocity_component_norm": float(np.linalg.norm(missed)),
            "snapshot_sha256": digest.hexdigest()}


def main():
    idle()
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA required")
    legs = []
    for _ in range(2):
        e = SnapshotEngine()
        m5 = m5_penetration(lambda: e, "gpu", 16, 600)
        legs.append({"m5": m5, "rows": [measure(s) for s in e.snapshots]})
    baseline = json.loads((ROOT / "reports/innovation_stack_baseline.json").read_text())["legs"][0]["m5"]["40"]
    gates = {"bit_identical": legs[0] == legs[1],
             "unchanged_m5": all(x["m5"] == baseline for x in legs),
             "all_samples": all(tuple(r["contact_call"] for r in x["rows"]) == SAMPLES for x in legs),
             "finite": all(np.isfinite([r[k] for k in ("jacobian_relative_projection_loss", "normal_velocity_component_norm", "missed_velocity_component_norm")]).all() for x in legs for r in x["rows"])}
    result = {"rank_relative_threshold": RANK_REL, "legs": legs, "gates": gates}
    text = json.dumps(result, indent=2) + "\n"
    (ROOT / "reports/innovation_stack_normal_space.json").write_text(text)
    print(text)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
