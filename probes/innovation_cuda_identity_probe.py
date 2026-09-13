"""Record process-namespace evidence around completed CUDA context creation."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
import os
from pathlib import Path
import subprocess


def snapshot():
    status = Path("/proc/self/status").read_text().splitlines()
    ids = [line for line in status if line.startswith(("Pid:", "NSpid:"))]
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return {"python_pid": os.getpid(), "namespace_ids": ids,
            "compute_processes": result.stdout.strip().splitlines()}


def main():
    before = snapshot()
    import warp as wp
    wp.init()
    a = wp.zeros(1, dtype=float, device="cuda:0")
    wp.synchronize()
    after = snapshot()
    values = [a.numpy().tolist(), a.numpy().tolist()]
    result = {"before": before, "after": after, "readbacks": values,
              "readback_identity": values[0] == values[1], "timing_samples": 0}
    root = Path(__file__).resolve().parents[1]
    (root/"reports"/"innovation_cuda_identity_probe.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
    return 0 if result["readback_identity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
