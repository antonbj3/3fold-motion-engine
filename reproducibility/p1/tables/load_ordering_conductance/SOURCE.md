# Section 5.4 - conductance ranking (iterative reference)

Row-level source for the three-row Section 5.4 conductance table. The values are
the archived load-ordering benchmark for the finest foot-loaded packing
(N=63365 grains, 124311 contacts), the delivered iterative reference. They are
distinct from the corrected direct-solve table (S5.5).

## Delivered files

- `benchmark_results.json` - the archived 8-scene load-ordering benchmark
  (verbatim).
- `table_conductance.py` - public builder: prints the three conductance Spearman
  values for the finest packing.

## Source identity

| file | sha256 |
| --- | --- |
| archived benchmark | `f3e7a2add09595f9a53b316b2313957541e89079ece278c7a3af3d883eb0b9c4` |

The producer requires the grain-scale scenes; the physics is not bundled and is
not rerun here. The delivered benchmark file is the exported measurement, and
the `Resistans (H13 1/m)`, `Resistans (unweighted)` and `Resistans (Hertz R)`
entries are the printed rows.

## Limits

- This is the iterative reference; the manuscript states that the corrected
  direct-solve recomputation moves the rank statistic by up to 5.2e-2 and that
  S5.4 and S5.5 must not be combined.
- The benchmark covers 8 scenes; only the finest packing is printed in S5.4.
