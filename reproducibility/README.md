# Reproduction bundles

Self-contained bundles for the four papers. Each `pN/` directory carries its
own manifest and checksum file and imports no project build tree. Copy a bundle
anywhere and run its commands.

| bundle | manifest | max single file |
| --- | --- | --- |
| `p1/` | `SHA256SUMS` (428 entries) | 16 MB |
| `p2/` | `SHA256SUMS` (313) and `MANIFEST.json` | 90 MB (`.npz.gz`) |
| `p3/` | `MANIFEST.sha256` (59) | 5.6 MB |
| `p4/` | `MANIFEST.sha256` (73) | 170,470 B (`f2_reached.png`) |

No single file exceeds 100 MB. The three P2 replay inputs that exceed 100 MB
uncompressed ship as deterministic standard-library gzip members below 90 MB;
`p2/data/replay_inputs/reconstruct_inputs.py` restores them and verifies full
SHA-256 identity.

## Commands

P1 — rebuild and verify the 21 contact scenes:

```sh
cd p1/data/ncp
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python loader.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python verify.py --all
```

P2 — regenerate every table from the bundled raw records:

```sh
cd p2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python run_tables.py
```

P3 — regenerate the tables and the numerical prose controls from frozen raw:

```sh
cd p3
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/regenerate_tables.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/verify_package.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/verify_prose_controls.py
```

P4 — regenerate figure payloads, tables and the frozen-payload checks:

```sh
cd p4
export SOURCE_DATE_EPOCH=1700000000 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python3 gen_data.py
python3 gen_tables.py
python3 verify_regions.py
python3 render_figures.py
python3 check_release.py
```

`p4/verify_static_lp.py` is the independent LP-versus-enumeration check; it
needs SciPy and takes about a minute on a shared CPU host. It is not required to
check the tables.

## Dependencies

NumPy throughout; h5py for the P1 scene containers; SciPy and Matplotlib for the
P4 bundle. See each bundle's own README for exact versions and the shared-device
timing caveats. No GPU and no physics replay is executed by the commands above;
the bounded GPU replay entry points ship with the P2 and P3 bundles and are
documented in their READMEs.
