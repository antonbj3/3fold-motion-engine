#!/usr/bin/env python3
"""run shared helpers: pre-launch provenance, run_id paths, hashes.

The provenance is taken in two phases so the frozen source copies and the serialized
inputs are pinned BEFORE warp is imported or any kernel is compiled/launched:

  phase 1 (`capture_pre`, no warp import)  source-file sha256, input-file sha256,
                                            declared graph budget, run_id, timestamp.
  phase 2 (`capture_post`, after import)   the ACTUAL loaded `module.__file__`, its
                                            on-disk sha256, the warp module options
                                            (`get_module_options`), warp/python/numpy
                                            versions and the CUDA device.

A post-hoc hash of a delivered file never certifies an earlier execution: phase 1 is
written to `out/<run_id>/provenance_pre.json` before any GPU call and is immutable for
that run_id.  Every run_id has its own output directory; repetitions never overwrite.
"""
import hashlib
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)

FROZEN_SOURCES = [
    "batch_adjoint_gpu.py", "batch_adjoint_gpu_f32.py", "talos_ident_gn.py", "talos_ident_scene.py", "ncp_ref.py",
    "estimator_gpu.py",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def arr_sha256(*arrs):
    import numpy as np
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def theta_sha256(theta):
    import numpy as np
    return arr_sha256(np.asarray(theta, np.float64))


def sp_digest(sp):
    return arr_sha256(sp["J"], sp["b_off"], sp["vfy"], sp["P"], sp["y"], sp["env"],
                      sp["vel"], sp["omg"], sp["R"], sp["hand"], sp["pos"])


def out_dir(run_id):
    d = os.path.join(HERE, "out", str(run_id))
    os.makedirs(d, exist_ok=True)
    return d


def source_hashes():
    return {m: sha256_file(os.path.join(SRC, m)) for m in FROZEN_SOURCES}


def capture_pre(run_id, inputs, budget, extra=None):
    """Written BEFORE warp import / compile / launch.  No warp import here."""
    rec = dict(run_id=str(run_id), phase="pre_launch", t_wall_unix=time.time(),
               python=sys.version.split()[0], platform=platform.platform(),
               frozen_sources=source_hashes(),
               inputs=dict(inputs), budget=dict(budget),
               note="hashes taken before import/compile/launch; post-hoc hashes certify "
                    "nothing about this run")
    if extra:
        rec.update(extra)
    d = out_dir(run_id)
    with open(os.path.join(d, "provenance_pre.json"), "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    return rec


def capture_post(run_id, modules):
    """After import, before/around the measured launch.  Records loaded files + flags."""
    import numpy as np
    import warp as wp
    mods = {}
    for name, mod in modules.items():
        path = getattr(mod, "__file__", None)
        opts = None
        try:
            opts = wp.get_module_options(module=mod)
        except Exception as e:  # pragma: no cover
            opts = {"error": repr(e)}
        mods[name] = dict(file=path,
                          file_sha256=(sha256_file(path) if path and os.path.exists(path)
                                       else None),
                          source_frozen_sha256=sha256_file(os.path.join(SRC, name + ".py"))
                          if os.path.exists(os.path.join(SRC, name + ".py")) else None,
                          module_options=opts)
    dev = wp.get_device()
    rec = dict(run_id=str(run_id), phase="post_import", t_wall_unix=time.time(),
               numpy=np.__version__, warp=wp.__version__,
               device=str(dev), device_name=getattr(dev, "name", None),
               modules=mods)
    d = out_dir(run_id)
    with open(os.path.join(d, "provenance_post.json"), "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    return rec


def gpu_memory_mib():
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            text=True, timeout=10).strip().splitlines()
        return [int(x.split()[0]) for x in out]
    except Exception:
        return None


STEP_KEYS = {"tau", "q", "v", "lam", "w", "Jh", "D", "A", "Jhv", "nle", "hand",
             "box", "res", "gap0", "gaps"}
ENV_KEYS = {"m_true", "I_zz_true", "mu_true", "L", "sel"}


def subset_env(d, nB, off):
    """Slice the leading env axis of a scene dict (axis 1 for step arrays, axis 0 for
    per-env arrays).  Logic copied verbatim from talos_adjoint_run.py."""
    import numpy as np
    out = {}
    for k, v in d.items():
        v = np.asarray(v)
        if k in STEP_KEYS and v.ndim >= 2:
            out[k] = np.ascontiguousarray(v[:, off:off + nB])
        elif k in ENV_KEYS and v.ndim >= 1:
            out[k] = np.ascontiguousarray(v[off:off + nB])
        else:
            out[k] = v
    return out
