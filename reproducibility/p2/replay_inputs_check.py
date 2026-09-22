#!/usr/bin/env python3
"""Verify the replay inputs without executing any GPU work.

Large NPZ inputs are shipped as deterministic ``.gz`` members; this check
decompresses each in memory and compares the FULL sha256 against
``data/replay_inputs/INPUT_MANIFEST.json``.  Small inputs are hashed directly.
"""
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RID = os.path.join(HERE, "data", "replay_inputs")
man = json.load(open(os.path.join(HERE, "MANIFEST.json"), encoding="utf-8"))
inp = json.load(open(os.path.join(RID, "INPUT_MANIFEST.json"), encoding="utf-8"))

missing, bad = [], []
shipped_gz = {"data/replay_inputs/" + n for n in inp["files"]}
for relp, meta in man["bundled"].items():
    if not relp.startswith("data/replay_inputs/") or relp.endswith("INPUT_MANIFEST.json"):
        continue
    if relp in shipped_gz:
        continue  # verified through INPUT_MANIFEST after decompression
    p = os.path.join(HERE, relp)
    if not os.path.isfile(p):
        missing.append(relp)
        continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != meta["sha256"]:
        bad.append(relp)

n = 0
for name, meta in sorted(inp["files"].items()):
    n += 1
    gz = os.path.join(RID, name + ".gz")
    plain = os.path.join(RID, name)
    if os.path.isfile(plain):
        h = hashlib.sha256(open(plain, "rb").read()).hexdigest()
    elif os.path.isfile(gz):
        h = hashlib.sha256(gzip.open(gz, "rb").read()).hexdigest()
    else:
        missing.append(name)
        continue
    if h != meta["sha256"]:
        bad.append(name)

code = [r for r in man["bundled"] if r.startswith("solver/")]
print("replay inputs: %d compressed + bundled, missing: %d, hash mismatch: %d, "
      "producer files: %d" % (n, len(missing), len(bad), len(code)))
print("physics_replay_complete=false (not executed here)")
if missing or bad:
    print("MISSING", missing, "BAD", bad)
    sys.exit(1)
sys.exit(0)
