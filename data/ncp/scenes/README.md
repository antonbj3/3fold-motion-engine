# Canonical contact-scene package (`contact-scene-v1`)

The 21 contact problems behind the P1 manuscript's penalty and contact-law
experiments, each as a JSON reconstruction source (`NN.json`) and a compact
symmetric-packed HDF5 reference (`NN.hdf5`), with per-scene metadata and hashes
in `manifest.json` and the scene table in `table.csv` / `table.md`.

Author: Anton Björkegren.

Rebuild each operator from its JSON alone and compare it with its HDF5
reference:

```sh
cd data/ncp
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python loader.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python loader.py 0 1 12   # selected scenes
```

`loader.py` uses only NumPy and h5py, imports nothing from the project build
tree and needs no generator code. `python loader.py --help`-style listing is
`python loader.py --list`.

The HDF5 container stores only the lower triangle including the diagonal in CSC
order with `nz = -1` and `symmetric_packed = 1`; recover the full operator with
`W = L + L^T - diag(L)`, which is what the loader does. An unmodified standard
FCLIB reader must not treat the stored triangle as the full operator.

The same package is reproduced verbatim in the P1 bundle at
`reproducibility/p1/data/ncp/`, which additionally ships `verify.py`,
`negative_controls.py`, `iso_run.py`, `numeric_identity.json` and the package
`SHA256SUMS`. The manuscript names this directory and `data/ncp/loader.py`; the
scene JSON/HDF5 bytes are identical between the two locations.
