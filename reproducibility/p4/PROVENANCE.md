# Provenance of the retained numbers

This note maps every retained numeric claim in the manuscript to a shipped public
input and, where possible, to a shipped executable computation. It exists so that
the availability statement is checkable rather than asserted.

## What is shipped

- `raw/*.json` — frozen numeric inputs. Each field is either (a) copied from a
  genuine producer artifact, (b) computed from bundled geometry/parameters by an
  executable formula in a shipped script, or (c) declared as an
  `archived_report` value with no raw rerun. **No value is transcribed from the
  printed manuscript.**
- `data/*.csv` — figure payloads, regenerated from `raw/` by `gen_data.py`.
- `tables/*.csv` — manuscript tables T1–T12, regenerated from `raw/` by
  `gen_tables.py`.
- `figures/*` — F1–F7 (PDF/SVG/PNG), regenerated from `data/` by
  `render_figures.py`; the shipped checksums are recorded in `MANIFEST.sha256`.
- `verify_static_lp.py`, `verify_regions.py`, `check_release.py` — independent
  checks over the frozen geometry and the shipped payloads.
- `CLAIM_SOURCE_MAP.csv` — the machine-readable map summarised below.

## Source kinds

`CLAIM_SOURCE_MAP.csv` classifies each claim:

| kind | meaning |
|---|---|
| `raw_backed` | the retained value is a field of a shipped `raw/*.json` (and is emitted by a generator) |
| `derived_from_raw` | the retained value is computed from shipped `raw/*.json` or bundled geometry/parameters by a shipped generator or formula |
| `archived_report` | only an archived reported value exists; **no raw rerun is shipped** |
| `no_numeric` | attribution / no numeric value |

## Field status in `raw/certified_regions.json`

| field | kind | source / formula |
|---|---|---|
| `reduced`, `full`, `outward` areas | `raw_backed` | producer region payloads |
| `witness_max_residual` | `raw_backed` | producer readiness key number |
| `per_direction_yaw_gap` | `raw_backed` | producer balance payload (worst geometry) |
| `reduced_inradius_overestimate` | `raw_backed` | producer balance payload |
| `holdouts` | `raw_backed` | producer holdout `safe_fraction`, full precision |
| `smallest_safe_margin` | `derived_from_raw` | minimum of the producer per-geometry `min_safe_margin_over_tau` over the three shipped geometries |
| `interval_vs_float_gap` | `derived_from_raw` | maximum of the producer per-case `interval_minus_float` over the locked cases |
| `cover_remainder` | `derived_from_raw` | `(g/h)·R_max·ε`, recomputed by `verify_regions.py` |
| `direction_cover.*` | `derived_from_raw` | `R_max = max_i‖p_i(u_c)−c‖ + hs·√2` over the shipped placement grid, from the bundled exact contact geometry; `verify_regions.py` recomputes and checks every value |
| `global_fallback` | `derived_from_raw` | `(g/h)·hs·√2` (cell half-diagonal); `verify_regions.py` recomputes and checks it |

## Field status in `raw/isotropic_controls.json`

Three controls are shipped and kept separate:

| key | identity | kind |
|---|---|---|
| `box_s1_sampled_24dir` | four-contact box scene (S1), 24 sampled static LP radii | `raw_backed` |
| `n12_platform_12dir` | genuine 12-contact elliptical 3-fold-perturbed ring control, reached T=50 s boundary vs LP; ratio recomputed from its own LP radii | `raw_backed` / `derived_from_raw` |
| `n20_exact_scalar` | exact 3D scalar LP control, not a 12-direction sample | `raw_backed` |

## Declared gaps (archived reported values, not raw reruns)

The following retained numbers are reported in archived audit reports and are
**not** backed by a shipped raw rerun. They are retained in the manuscript as
reported values, and the availability statement is narrowed accordingly:

- C06b — compliant-stiff convergence slope `-0.60` and floor `5.48e-3`.
- C07 — compliant-stiff branch `1.36–8.82 %` below the static LP radius.
- C09 qualifiers — placement-field reproduction `9.048e-15` and the
  `3.27 %`/`3.29 %` sampling/reference bounds.
- C10b — active-set certificate outside bracket `144/144` and mean relative
  error range `0.198–2.979`.
- C11 — support-rectangle mean relative errors `54.7/86.5/70.5 %` and the full
  geometry `0.66/0.66/1.13 %`.
- C15 — the outward-interval audit conclusion (status flag). Its numeric inputs
  (`interval_vs_float_gap`, `smallest_safe_margin`) are shipped derived values.
- C16 qualifiers — continuation load-error range `5.4e-6…2.99 %` (median
  `0.51 %`) and the `+623 %` miss.
- C17 — material-tangent error `1.26e-2/1.68e-2`; consistent tangent vs secant
  `3–4e-8`.
- C18 — path-derivative discrepancy `1.505e-2`.
- C19 qualifiers — the individual graze ratios `0.9926/0.9996/0.9916/0.9766`.
- C20 qualifier — the Z300 branch max ratio `0.992566`.
- C21 qualifiers — the `18/18` count, `+5.2…+1462 %` range and the `-6.28 %`
  threshold shift.
- C22 — macro overestimate `1.91–20.87×` and the rate-stability ranges.
- C03 qualifiers — the N12/N20 T=50 s scalar reached-boundary percentages are
  archived reported values; the box/stack/pallet per-scene and T=20/T=100 s
  values are shipped raw (`raw/reached_ramp.json`).
- Appendix A — the remaining supporting ledger values (seed misses, scaling,
  at-cap discrepancy, micro-creep, `0/120°` cold-vs-seed, safe fractions,
  tangent-vs-secant), except those already fields of
  `raw/static_exactness.json`, `raw/anisotropic.json` and
  `raw/certified_regions.json`.

Where a claim is `raw_backed` or `derived_from_raw`, the raw field is the
primary frozen input or the bundled geometry and the generator/formula is named
in the map. The witness-region classifications (C14) are float-computed
observations over the discrete `w0` grid; the lemma itself is a conditional
real-arithmetic statement (`WITNESS_LEMMA.md`) and is not a machine-checked
interval certificate.

## Correction ledger (versus the previous revision)

| field | previous value | shipped value | reason |
|---|---|---|---|
| `isotropic_controls.n12_sampled` | historically mislabelled as an N12 sample despite `n=24` | renamed `box_s1_sampled_24dir` (four-contact box S1, 24 dirs) | the source is the box scene, not a 12-leg platform |
| `isotropic_controls` N12 | absent (box mislabelled as N12) | `n12_platform_12dir` added as a separate control | genuine N12 control kept separate from the box and N20 |
| `global_fallback` | `0.13610726901519536` (unsourced prose) | `0.1361112120652152` = `(g/h)·hs·√2` | derivation-backed value from the cell half-diagonal |
| `cover_remainder` | `0.004999399292698012` (literal) | same value, now derived in `verify_regions.py` | made an executable formula of bundled inputs |
| `holdouts` | `0.990312 / 0.989415` (6-digit roundings) | `0.9903121636167922 / 0.9894154818325435` | producer field at full precision |
| `smallest_safe_margin` | `2.304e-5` (report rounding) | `2.304220376291022e-05` | derived from the producer region payload |
| `interval_vs_float_gap` | `2e-12` (report rounding) | `3.0393465522138285e-12` | maximum of the producer per-case gap field |
