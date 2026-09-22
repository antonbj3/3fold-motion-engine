#!/usr/bin/env python3
"""Verify the release package: manifest hashes, state loading, and exact table match.

CPU only. Content hygiene (absence of internal identifiers in names, text and binary payloads) is
checked by the separate private release auditor and is not part of this public check.

Usage: python scripts/verify_package.py
"""
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_manifest(failures):
    man = PKG / "MANIFEST.sha256"
    if not man.exists():
        failures.append("MANIFEST.sha256 missing")
        return 0
    n = 0
    for line in man.read_text().splitlines():
        if not line.strip():
            continue
        h, rel = line.split("  ", 1)
        p = PKG / rel
        if not p.exists():
            failures.append("manifest member missing: %s" % rel)
        elif sha(p) != h:
            failures.append("manifest hash mismatch: %s" % rel)
        else:
            n += 1
    return n


def verify_states(failures):
    import numpy as np
    import bench_common as U
    n = 0
    for key in ["120", "140", "150", "160", "160geom"]:
        fr = U.load_frame(key)
        d = np.load(U.STATES / U.STATE_FILES[key], allow_pickle=False)
        for k in ("x_at", "x_now", "mass", "ids", "grid_resident", "n"):
            if k not in d.files:
                failures.append("state %s missing array %s" % (key, k))
        if fr["n"] != int(d["n"]):
            failures.append("state %s n mismatch" % key)
        n += 1
    return n


def verify_tables(failures):
    """Regenerate every table from frozen raw and require an exact match with expected/."""
    exp = PKG / "expected"
    with tempfile.TemporaryDirectory() as td:
        subprocess.check_call([sys.executable, str(HERE / "regenerate_tables.py"), "--out", td])
        names = sorted(p.name for p in exp.glob("*.csv"))
        for name in names:
            a = (exp / name).read_bytes()
            b = (Path(td) / name).read_bytes()
            if a != b:
                failures.append("regenerated table differs: %s" % name)
    return len(names)


def verify_prose_controls(failures):
    """Recompute the supplementary prose-control numbers from the bundled raw inputs."""
    script = HERE / "verify_prose_controls.py"
    if not script.exists():
        return 0
    with tempfile.TemporaryDirectory() as td:
        rc = subprocess.run([sys.executable, str(script), "--out", str(Path(td) / "pc.json")],
                            capture_output=True, text=True)
    if rc.returncode != 0:
        failures.append("prose-control check failed: %s" % (rc.stdout + rc.stderr).strip())
        return 0
    return 1


def main():
    failures = []
    nm = verify_manifest(failures)
    ns = verify_states(failures)
    nt = verify_tables(failures)
    npc = verify_prose_controls(failures)
    print("manifest members verified:", nm)
    print("states loaded:", ns)
    print("tables regenerated and matched:", nt)
    print("prose-control checks:", npc)
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PACKAGE_OK = True")


if __name__ == "__main__":
    main()
