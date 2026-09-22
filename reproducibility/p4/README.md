# P4 release — boundary/scope study of static support feasibility

Self-contained reproduction package for the empirical/theoretical study of the
boundaries between static support feasibility, the reached quasistatic branch,
the first local contact event, and macroscopic support loss, together with the
constructive real-arithmetic primal witness for the fixed coplanar full-LP
family.

The directory contains only public inputs, public data and standalone Python.
It needs no private workspace, no absolute imports, no symlinks and no network.
Copy this directory anywhere and run the commands below.

## Contents

| path | what |
|---|---|
| `data/*.csv` | public numeric payloads used by the seven figures |
| `raw/*.json` | frozen input payloads from the audited runs |
| `render_figures.py` | renders F1–F7 (vector PDF/SVG + PNG proofs) from `data/` only |
| `gen_data.py` | regenerates `data/*.csv` from `raw/*.json` |
| `gen_tables.py` | regenerates `tables/*.csv` from `raw/*.json` |
| `tables/*.csv` | machine-readable version of the manuscript's numeric tables |
| `PAPER_SOURCE.md` | exact corrected manuscript source for `docs/papers/P4.pdf` |
| `scenes.py` | static contact scenes and the 6D wrench map assembled from geometry |
| `friction_polygon.py` | minimax 16-ray friction polygon (straddles the circular cone) |
| `lp_reference.py` | independent LP and subset-enumeration reference |
| `verify_static_lp.py` | rebuilds the wrench map from geometry and checks LP vs enumeration |
| `verify_regions.py` | internal consistency of the certified-region/event payloads |
| `WITNESS_LEMMA.md` | self-contained constructive full-LP witness derivation |
| `CAPTIONS.md` | the manuscript figure captions |
| `PROVENANCE.md` | claim -> public raw field / generator map and declared gaps |
| `CLAIM_SOURCE_MAP.csv` | machine-readable version of the provenance map |
| `RUNTIME.json` | verified interpreter/library versions |
| `make_manifest.py` | rebuilds SHA-256 manifest after reproduction |

## Run

```sh
export SOURCE_DATE_EPOCH=1700000000 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
python3 verify_static_lp.py      # independent LP vs enumeration (CPU)
python3 verify_regions.py        # frozen payload consistency
python3 gen_data.py              # regenerate data/*.csv
python3 gen_tables.py            # regenerate tables/*.csv
python3 render_figures.py        # regenerate figures/*.pdf/svg/png
python3 make_manifest.py          # hash the regenerated public package
python3 check_release.py          # check manifest and private-name scan
```

`verify_static_lp.py` needs SciPy; the others need NumPy and Matplotlib only.
The verified environment is Python 3.10, NumPy 2.2.6, SciPy 1.15.3,
Matplotlib 3.10.9 on Linux (see `RUNTIME.json`).
The commands above regenerate numeric tables and figures; they do not typeset
the paper PDF. `PAPER_SOURCE.md` records its exact manuscript text.

## Scope and status

The figures are static visualisations of measured numerical payloads. They do
not regenerate physics. The static-method exactness check is a numerical parity
result on the same discretized friction law, not a statement about the
continuous circular cone. The full-LP witness is a real-arithmetic *lemma*
(`WITNESS_LEMMA.md`); the attempted outward-rounded interval implementation was
independently audited and its strict outward obligation was **not** established
(`outward_rounding_verified = false`, `floating_certificate_supported = false`).
Two figure sets (the humanoid double-support boundary and the placement field)
are results whose independent audit is qualified; they are labelled
accordingly in `CAPTIONS.md` and must not be promoted to unqualified positive
claims. `PROVENANCE.md` and `CLAIM_SOURCE_MAP.csv` state which retained numbers
are backed by shipped raw inputs and which remain archived reported values.
The N12 platform is an elliptical, 3-fold-perturbed 12-contact ring. The
F4/F5 in-plot labels read “Independently checked; limitations apply”; their
captions state the audit reservations.
