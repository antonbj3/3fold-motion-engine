# Papers — four-paper assembly

Four PDFs, one checksum file (`SHA256SUMS`), and self-contained reproduction
bundles under [`../../reproducibility/`](../../reproducibility/). This index
records the exact artifacts, titles and measured limits.

| file | pages | SHA-256 |
| --- | --- | --- |
| `P1.pdf` | 51 | `f2939f837cdf71dd020d4856d2191b23526f72174cbad6fe22ddd3a61c5e7039` |
| `P2.pdf` | 37 | `d23ebac030bf68974a0d7babf0498679f6204d0e69cbd23b0a0294808f35f72d` |
| `P3.pdf` | 9 | `5bdbb99f7713843bb42653afe20b0f5474a3cb50ff0501b4525cb63859c810d0` |
| `P4.pdf` | 20 | `a8eb62e001d4eaaf9992071c5765cd1b71602e8fee6436e6ed929ebbf24071b5` |

Verify the PDFs:

```sh
cd docs/papers && sha256sum -c SHA256SUMS
```

## Exact titles and revisions

* **P1, revision 3** — *Incidence structure and solver limits of contact
  Delassus operators* (Anton Björkegren, 22 September 2026).
* **P2, revision 3** — *Deterministic sensitivities of Coulomb contact on GPUs*
  (Anton Björkegren, 22 September 2026).
* **P3, revision 3** — *When does reusing a solver grid pay for an exact
  point-in-box particle count? A controlled comparative study with a measured
  negative answer* (Anton Björkegren, 22 September 2026).
* **P4, revision 4** — *Static support boundaries, reached branches and local
  events: a measured scope map with a real-arithmetic full-LP witness*
  (Anton Björkegren, 22 September 2026).

Each PDF is rendered from its own frozen manuscript source; the corrected P4
source is shipped as [`PAPER_SOURCE.md`](../../reproducibility/p4/PAPER_SOURCE.md).
The PDF hashes above identify the reviewed release artifacts; a changed render
requires review.

## Reproduction

Each bundle is standalone: it copies its own code, data and manifest and imports
no project build tree.

| paper | bundle | primary command | dependencies |
| --- | --- | --- | --- |
| P1 | `reproducibility/p1/` | `python data/ncp/loader.py` and `python data/ncp/verify.py --all` | NumPy, h5py |
| P2 | `reproducibility/p2/` | `python run_tables.py` | NumPy |
| P3 | `reproducibility/p3/` | `python scripts/regenerate_tables.py` then `python scripts/verify_package.py` | NumPy |
| P4 | `reproducibility/p4/` | `python gen_data.py && python gen_tables.py && python verify_regions.py` | NumPy, SciPy, Matplotlib |

The canonical scene package the P1 manuscript names is also available in the
repository at [`../../data/ncp/scenes/`](../../data/ncp/scenes/) with the loader
at [`../../data/ncp/loader.py`](../../data/ncp/loader.py).

## Numerical source-availability limits

These limits are declared in each bundle and are not softened here.

* **P1.** The manuscript text is not part of the bundle; only the PDF and the
  reproduction data/code are. The 21 exported contact operators and the tabulated
  studies are regenerated from stored row-level measurements or delivered as
  labelled aggregates; the original grain-scale physics producers are not in the
  bundle. The dimensionless stop reproduces 45/45 stored outer counts with a
  sparse-LU solver and 44/45 with dense Cholesky (one rounding-sensitive cell).
  No benchmark or physics producer is rerun.
* **P2.** Thirty printed numeric measurement cells have no bundled original
  per-run source, eighteen cells are em-dash placeholders, and one Table P digest
  identifies the original pre-cleanup serialization. These are declared in
  `reproducibility/p2/SOURCE_AVAILABILITY.json`. The bundle regenerates the
  tables from stored measurements and packaged producer code; it is not a fresh
  GPU benchmark. A bounded physics replay is possible on a GPU host but is **not
  executed** here.
* **P3.** The tables and numerical prose controls are recomputed from frozen raw
  measurements. The historical `84.66`, `4.85` and `66.52 ms` values are archival
  reports whose original timing runs are not retained and cannot be regenerated;
  the retained later incremental-flow record has median `70.2591 ms` and its
  `3.5 %` within-10 ms fraction is recomputable. All timings are on a shared
  device and carry that caveat.
* **P4.** The manuscript text is `reproducibility/p4/PAPER_SOURCE.md`. Retained numbers are
  `raw_backed`, `derived_from_raw` or `archived_report` per
  `reproducibility/p4/CLAIM_SOURCE_MAP.csv`. Ten claim groups remain archived
  reports without shipped raw reruns. The full-LP witness is a **conditional**
  real-arithmetic lemma; the outward-rounded interval implementation was audited
  and its strict outward obligation was **not** established
  (`outward_rounding_verified = false`). Two figure sets (humanoid double-support
  boundary, placement field) carry qualified independent audits.

None of the four bundles claims a full physics replay. A replay is a repetition
check on a recorded configuration, not a new accuracy certificate.

## Artifact scope

The four PDF hashes and the bundle manifests identify this assembly. The
source-availability limits and audit reservations above remain part of its
interpretation; this index does not extend the measured claims.
