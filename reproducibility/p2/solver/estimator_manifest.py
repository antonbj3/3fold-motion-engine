#!/usr/bin/env python3
"""run manifest: full sha256 of frozen sources, inputs, scripts and artifacts, and a
canonical rebuild of run_records.jsonl from the per-run record.json files.

No GPU, no warp import.
"""
import glob
import json
import os

import estimator_common as C

SCRIPTS = ["estimator_common.py", "estimator_state.py", "estimator_run.py", "estimator_chunk.py",
           "estimator_residual.py", "estimator_fd.py", "estimator_summary.py", "estimator_manifest.py"]
INPUTS = ["identification_sim_B512.npz", "identification_sim_win7_B512.npz"]


def main():
    man = dict(
        frozen_sources=C.source_hashes(),
        scripts={s: C.sha256_file(os.path.join(C.HERE, s)) for s in SCRIPTS},
        inputs={s: C.sha256_file(os.path.join(C.HERE, s)) for s in INPUTS},
        artifacts={},
        records=[],
    )
    for f in sorted(glob.glob(os.path.join(C.HERE, "*.csv")) +
                    glob.glob(os.path.join(C.HERE, "*.json")) +
                    glob.glob(os.path.join(C.HERE, "chunk_accepted_*.csv")) +
                    glob.glob(os.path.join(C.HERE, "fd_check_*.csv"))):
        base = os.path.basename(f)
        if base in ("summary.json",):
            continue
        man["artifacts"][base] = C.sha256_file(f)
    recs = []
    for f in sorted(glob.glob(os.path.join(C.HERE, "out", "*", "record.json"))):
        d = json.load(open(f))
        if d["run_id"].startswith(("series_", "smoke_", "bench_")):
            recs.append(d)
    json.dump(man, open(os.path.join(C.HERE, "MANIFEST.json"), "w"), indent=1,
              sort_keys=True)
    with open(os.path.join(C.HERE, "run_records.jsonl"), "w") as f:
        for d in recs:
            f.write(json.dumps(d) + "\n")
    print("MANIFEST.json written;", len(man["artifacts"]), "artifacts,",
          len(recs), "run records")


if __name__ == "__main__":
    main()
