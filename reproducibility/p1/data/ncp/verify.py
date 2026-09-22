"""Meaningful verification for the `contact-scene-v1` package.

For every scene it

  1. reconstructs G = J M^-1 J^T + diag(eta) from the JSON,
  2. compares G, b and mu element-by-element with the HDF5 reference,
  3. checks the delivered file SHA-256 against the manifest and SHA256SUMS,
  4. checks every file listed in SHA256SUMS (including loader.py, verify.py and
     the manifest itself) against its bytes and flags unlisted delivered files.

Exit status is 0 only if all checks pass.

    python verify.py --all           # everything (default)
    python verify.py --scene 7       # one scene
    python verify.py --all --no-sha  # skip file hashes
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

import loader

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.join(HERE, "scenes")
G_TOL = 1e-15   # Frobenius-relative reconstruction tolerance
G_ABS_TOL = 1e-9  # absolute element bound (large-operator sanity check)
B_TOL = 1e-15   # absolute bias tolerance
MU_TOL = 1e-15  # absolute friction tolerance


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums():
    sums = {}
    with open(os.path.join(HERE, "SHA256SUMS")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split(None, 1)
            sums[name.strip()] = digest
    return sums


def check_manifest():
    manifest = json.load(open(os.path.join(SCENES, "manifest.json")))
    sums = read_sums()
    problems = []
    for key, e in sorted(manifest["scenes"].items()):
        for rel, digest in ((e["json"], e["json_sha256"]), (e["hdf5"], e["hdf5_sha256"])):
            got = sha256_file(os.path.join(HERE, rel))
            if got != digest:
                problems.append(f"manifest hash mismatch {rel}: {got} != {digest}")
            listed = sums.get(rel)
            if listed is None:
                problems.append(f"{rel} missing from SHA256SUMS")
            elif listed != got:
                problems.append(f"SHA256SUMS mismatch {rel}: {got} != {listed}")
    return manifest, problems


def check_self_hashes():
    """Check every file listed in SHA256SUMS against its bytes.

    This closes the gap where a descriptive manifest edit (or any other
    non-scene file change) would pass the scene reconstruction check but alter
    a delivered byte.  `SHA256SUMS` itself is not self-listed by construction.
    """
    sums = read_sums()
    problems = []
    for rel, digest in sorted(sums.items()):
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            problems.append(f"{rel} listed in SHA256SUMS but missing")
            continue
        got = sha256_file(path)
        if got != digest:
            problems.append(f"SHA256SUMS mismatch {rel}: {got} != {digest}")
    # every delivered file except SHA256SUMS must be listed
    for root, dirs, names in os.walk(HERE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), HERE)
            if rel == "SHA256SUMS":
                continue
            if rel not in sums:
                problems.append(f"{rel} delivered but not listed in SHA256SUMS")
    return problems


def main(argv):
    only = None
    do_sha = True
    quiet = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--scene":
            i += 1
            only = [int(argv[i])]
        elif a in ("--all",):
            only = None
        elif a == "--no-sha":
            do_sha = False
        elif a == "--quiet":
            quiet = True
        else:
            only = [int(a)]
        i += 1

    manifest, problems = ({}, [])
    if do_sha:
        manifest, problems = check_manifest()
        problems = problems + check_self_hashes()

    idxs = only if only is not None else list(range(loader.N_SCENES))
    failures = list(problems)
    for i in idxs:
        jp = os.path.join(SCENES, f"{i:02d}.json")
        hp = os.path.join(SCENES, f"{i:02d}.hdf5")
        r = loader.verify(jp, hp)
        bad = []
        if not (r["G_rel_fro"] <= G_TOL and r["G_abs"] <= G_ABS_TOL):
            bad.append("G")
        if not (r["b_abs"] <= B_TOL):
            bad.append("b")
        if not (r["mu_abs"] <= MU_TOL):
            bad.append("mu")
        status = "PASS" if not bad else "FAIL(" + ",".join(bad) + ")"
        if not quiet or bad:
            print(f"{status} {i:2d} {r['scene']:14s} n_c={r['n_c']:5d} "
                  f"model={r['model']:16s} G_rel_fro={r['G_rel_fro']:.3e} "
                  f"b_abs={r['b_abs']:.3e} mu_abs={r['mu_abs']:.3e}")
        if bad:
            failures.append(f"scene {i} reconstruction: {bad}")

    if do_sha and not problems:
        print(f"manifest + SHA256SUMS OK for {len(manifest.get('scenes', {}))} scenes")
    if failures:
        print("VERIFY_FAILED")
        for p in failures:
            print("  -", p)
        return 1
    print("VERIFY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
