"""Corrupt-data negative controls for the loader / manifest.

Each control makes one deliberate corruption in a throwaway copy of the package
and checks that `verify.py --all` DETECTS it (non-zero exit).  A control that
passes is a failed negative control.

    python negative_controls.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run_verify(root):
    p = subprocess.run([PY, "verify.py", "--all"], cwd=root,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def fresh_copy(root):
    dst = tempfile.mkdtemp(prefix="ncp_ctrl_")
    shutil.copytree(root, os.path.join(dst, "pkg"))
    return os.path.join(dst, "pkg")


def ctl_json_bias(root):
    pkg = fresh_copy(root)
    p = os.path.join(pkg, "scenes", "00.json")
    d = json.load(open(p))
    d["b"][0] = float(d["b"][0]) + 1.0
    json.dump(d, open(p, "w"))
    return run_verify(pkg)


def ctl_hdf5_value(root):
    pkg = fresh_copy(root)
    import h5py
    import numpy as np
    p = os.path.join(pkg, "scenes", "00.hdf5")
    with h5py.File(p, "r+") as f:
        x = f["fclib_local/W/x"]
        x[0] = np.float64(x[0]) + 1e-3
    return run_verify(pkg)


def ctl_manifest_hash(root):
    pkg = fresh_copy(root)
    p = os.path.join(pkg, "scenes", "manifest.json")
    d = json.load(open(p))
    d["scenes"]["00"]["json_sha256"] = "0" * 64
    json.dump(d, open(p, "w"))
    return run_verify(pkg)


def ctl_sha256sums(root):
    pkg = fresh_copy(root)
    p = os.path.join(pkg, "SHA256SUMS")
    lines = open(p).read().splitlines()
    for k, line in enumerate(lines):
        if "scenes/01.json" in line:
            lines[k] = "0" * 64 + "  scenes/01.json"
            break
    open(p, "w").write("\n".join(lines) + "\n")
    return run_verify(pkg)


def ctl_json_structure(root):
    pkg = fresh_copy(root)
    p = os.path.join(pkg, "scenes", "00.json")
    d = json.load(open(p))
    del d["bodies"]           # rebuild must fail loudly
    json.dump(d, open(p, "w"))
    return run_verify(pkg)


CONTROLS = [
    ("json_numeric_bias", ctl_json_bias),
    ("hdf5_numeric_value", ctl_hdf5_value),
    ("manifest_hash", ctl_manifest_hash),
    ("sha256sums_entry", ctl_sha256sums),
    ("json_structure_missing_bodies", ctl_json_structure),
]


def main():
    rc0, _ = run_verify(HERE)
    print(f"baseline verify.py --all exit={rc0} (expected 0)")
    all_ok = rc0 == 0
    for name, fn in CONTROLS:
        try:
            rc, out = fn(HERE)
        except Exception as exc:                 # a raised error is also detection
            rc, out = 1, f"exception: {exc!r}"
        detected = rc != 0
        all_ok = all_ok and detected
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        print(f"control {name:32s} exit={rc} detected={detected} :: {tail}")
    print("NEGATIVE_CONTROLS_OK" if all_ok else "NEGATIVE_CONTROLS_FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
