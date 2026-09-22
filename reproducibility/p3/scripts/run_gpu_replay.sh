#!/usr/bin/env bash
# GPU replay of the physical measurements (not required to check the tables).
# Every step takes the device lock and a hard timeout, so a shared device is not left occupied.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
LOCK=/tmp/gpu.lock
mkdir -p replay_out
flock -w180 "$LOCK" timeout 90 python scripts/replay_correctness.py --out replay_out/correctness.json
flock -w180 "$LOCK" timeout 90 python scripts/replay_benchmark.py --tag a --out replay_out/benchmark_a.json
echo "replay outputs written to replay_out/"
