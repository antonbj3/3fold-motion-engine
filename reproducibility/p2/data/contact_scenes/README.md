# NCP contact-scene dataset (`contact-scene-v1`)

This directory is the scene package promised by the manuscript: 21 contact
problems, each as a JSON reconstruction source and a compact HDF5 reference for
the contact operator.

Author: Anton Björkegren.

## Layout

```
README.md            this file
loader.py            standalone reconstruction + verification CLI
verify.py            full verification (reconstruction, manifest, SHA-256)
manifest.json        per-scene metadata and delivered-file hashes
SHA256SUMS           SHA-256 of every delivered file
table.csv, table.md   scene table (sizes, spectra, conditioning)
scenes/NN.json        `contact-scene-v1` description of scene NN (00..20)
scenes/NN.hdf5        same operator in a compact FCLIB local-problem container
```

## Running it

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python loader.py          # all 21
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python verify.py --all
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python negative_controls.py
```

Dependencies: NumPy and h5py only. The loader imports nothing from any project
build tree and uses no absolute path. `verify.py` also checks manifest and
`SHA256SUMS`.

## What a scene contains

Each JSON has `schema = "contact-scene-v1"`, `index`, `scene`, `dt`, `n_b`,
`n_c`, `dof`, `mu`, `b`, `eta`, a public `provenance` block, and one of

* `model = "body_contacts"`: `bodies` (mass/inertia, inverse blocks) and
  `contacts` (normal and tangent frame `t1,t2`, body pair `a,b`, lever arms,
  friction coefficient). The loader rebuilds

      J = per-contact block [D ; r x D]   (rows n,t1,t2; cols v,omega)
      M^-1 = block diagonal of per-body inverse mass and inertia

* `model = "operator"`: `J` and `M^-1` are supplied directly (dense random
  operator, or a floating-base multibody whose columns are generalized
  velocities).

In both cases the operator is

    W = G = J M^-1 J^T + diag(eta).

Here `eta` is the per-contact compliance (zero for the rigid scenes; nonzero for
the packing and granular-step scenes). Scene names use descriptive public aliases, indexed consistently
across the manifest, JSON, HDF5 metadata and study tables.

## The HDF5 container

`scenes/NN.hdf5` uses the FCLIB `fclib_local` layout, but it is **not** an
untouched standard full-matrix FCLIB export. The matrix `W` is stored in a
documented compact symmetric-packed form: only the lower triangle including the
diagonal is written, in CSC order, with the dataset attributes
`nz = -1` and `symmetric_packed = 1`. To recover the full operator use

    W_full = L + L^T - diag(L),

which is exactly what `loader.read_hdf5_problem` does. A standard full-matrix
FCLIB file would store both triangles and would be substantially larger. The
`W` dataset also carries a human-readable `description` attribute stating the
packing. `/fclib_local/vectors/{q,mu}` hold `q = b` and the friction vector.

The compact container is a lossless representation of the same numeric payload.
For this public release every container was **rebuilt from scratch**
from the original numeric datasets: the two free-text string datasets
`fclib_local/info/title` and `fclib_local/info/description` carry reviewed
public text, and no internal build-tree string is present anywhere in the
file bytes (a fresh `h5py.File(path, 'w')` container, not an in-place string
overwrite). Every numeric dataset (`W/{x,i,p,n,m,nz,nzmax}`,
`vectors/{q,mu}`), every numeric attribute (including `symmetric_packed`) and the
remaining string dataset (`info/math_info`) are bit-identical to the
original. The source-to-release numeric comparison is recorded in
`numeric_identity.json`: zero numeric dataset differences across all
21 scenes. Run `python verify.py --all` for the public JSON/HDF5 consistency check.

## Reference

The dataset accompanies the manuscript's NCP contact-operator study and is the
`data/ncp/` component of the reproduction bundle. The contact-graph reproduction
supplement (patch-wrench, load-ordering, ramp experiments, and the dimensionless
stopping comparison) is provided in a separate companion supplement. The FCLIB
container layout follows the FCLIB benchmark format; FCLIB itself is not
redistributed here.
