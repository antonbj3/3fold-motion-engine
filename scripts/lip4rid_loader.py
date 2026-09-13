#!/usr/bin/env python3
"""LIP4RID loader: opens the public Zenodo-12516500 (LIP4RID) real-robot pickles in modern pandas.

The recordings (a real Panda 7-DOF and a MELFA 6-DOF with measured joint torque) are pickled with
pandas 1.x and do not load in pandas 2.2/3.0. This shim unpickles them without installing old pandas,
via two patches applied during unpickling: (1) new_block coerces slice -> BlockPlacement; (2) the removed
Int64Index/Float64Index map to the current pd.Index.

The dataset itself is not shipped with this repository; unpack it under data/real_external/.
"""
import glob
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

MELFA_DIR = Path(__file__).resolve().parents[1] / "data" / "real_external" / "lip4rid_melfa"
PANDA_DIR = Path(__file__).resolve().parents[1] / "data" / "real_external" / "lip4rid_panda"


def _install_shim():
    """Idempotent: patch pandas so 1.x pickles load in 2.2/3.0."""
    from pandas._libs.internals import BlockPlacement
    import pandas.core.internals.blocks as B
    if not getattr(B, "_lip4rid_patched", False):
        _orig = B.new_block

        def patched(values, placement, ndim, refs=None):
            if isinstance(placement, slice):
                placement = BlockPlacement(placement)
            try:
                return _orig(values, placement, ndim=ndim, refs=refs)
            except TypeError:
                return _orig(values, placement, ndim=ndim)

        B.new_block = patched
        B._lip4rid_patched = True
    if "pandas.core.indexes.numeric" not in sys.modules:
        m = types.ModuleType("pandas.core.indexes.numeric")
        m.Int64Index = m.Float64Index = m.UInt64Index = pd.Index
        sys.modules["pandas.core.indexes.numeric"] = m


def load_pkl(path):
    """Ladda EN LIP4RID-pkl → pandas DataFrame (shim installeras automatiskt)."""
    _install_shim()
    with open(path, "rb") as f:
        return pickle.load(f)


def load_melfa(measured=True):
    """Load all MELFA experiments -> a list of per-file dicts {column: ndarray}. measured=True returns the raw q/dq/ddq/tau."""
    files = sorted(glob.glob(str(MELFA_DIR / "*.pkl")))
    if not files:
        raise SystemExit(f"no MELFA pickles in {MELFA_DIR} (unpack LIP4RID first)")
    out = []
    for fp in files:
        d = load_pkl(fp)
        rec = {}
        for j in range(1, 7):
            for base in ("q", "dq", "ddq", "tau"):
                col = f"{base}_{j}"
                if col in d.columns:
                    rec[col] = np.asarray(d[col].values, float)
        rec["_file"] = Path(fp).name
        out.append(rec)
    return out


def load_panda():
    """Ladda ALLA Panda 7-DOF filtered experiment + dynamics_components → lista per-seed-dict.
    Varje: q/dq/ddq/tau_j (j=1..7) + M (N,7,7), c (N,7), g (N,7) [stel-kropps-komponenter, friktionsfria]."""
    data_files = sorted(f for f in glob.glob(str(PANDA_DIR / "*_as_filtered_fcut4.0.pkl")) if "dynamics_components" not in f)
    if not data_files:
        raise SystemExit(f"no Panda pickles in {PANDA_DIR} (unpack LIP4RID first)")
    out = []
    for fp in data_files:
        comp_fp = fp.replace(".pkl", "_dynamics_components.pkl")
        if not Path(comp_fp).exists():
            continue
        d = load_pkl(fp); comp = load_pkl(comp_fp)
        rec = {"_file": Path(fp).name}
        for j in range(1, 8):
            for base in ("q", "dq", "ddq", "tau"):
                col = f"{base}_{j}"
                if col in d.columns:
                    rec[col] = np.asarray(d[col].values, float)
        rec["M"] = np.asarray(comp["M"], float)   # (N,7,7)
        rec["c"] = np.asarray(comp["c"], float)   # (N,7)
        rec["g"] = np.asarray(comp["g"], float)   # (N,7)
        out.append(rec)
    return out


if __name__ == "__main__":
    recs = load_melfa()
    print(f"LIP4RID loader ok: {len(recs)} MELFA files opened (pandas {pd.__version__})")
    r = recs[0]
    print(f"  fil[0]={r['_file']} kolumner={[k for k in r if not k.startswith('_')][:6]}... n={len(r['tau_1'])}")
    tot = sum(len(r["tau_1"]) for r in recs)
    print(f"  {tot} samples in total across {len(recs)} files")
