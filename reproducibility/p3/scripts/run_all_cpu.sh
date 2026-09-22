#!/usr/bin/env bash
# CPU-only reproduction: regenerate every displayed table from frozen raw data and verify the package.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
python scripts/regenerate_tables.py
python scripts/verify_package.py
