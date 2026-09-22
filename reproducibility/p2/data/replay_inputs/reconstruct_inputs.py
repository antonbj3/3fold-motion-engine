#!/usr/bin/env python3
"""Reconstruct the large replay NPZ inputs from their shipped .gz members.

Uses only the Python standard library.  Run from this directory:
    python reconstruct_inputs.py
Each reconstructed file is hashed and compared with the full sha256 recorded in
INPUT_MANIFEST.json.  Extra disk: a reconstructed file is ~its original size and
coexists with its .gz until you delete the .gz.
"""
import gzip
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
man = json.load(open(os.path.join(HERE, "INPUT_MANIFEST.json"), encoding="utf-8"))
bad = 0
for name, meta in sorted(man["files"].items()):
    gz = name + ".gz"
    with gzip.open(os.path.join(HERE, gz), "rb") as f:
        data = f.read()
    h = hashlib.sha256(data).hexdigest()
    ok = h == meta["sha256"] and len(data) == meta["bytes"]
    with open(os.path.join(HERE, name), "wb") as f:
        f.write(data)
    print("%-40s %s %s" % (name, "OK" if ok else "MISMATCH", h))
    bad += 0 if ok else 1
raise SystemExit(1 if bad else 0)
