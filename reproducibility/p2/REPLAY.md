# Bounded physics replay entry point

The table scripts regenerate the manuscript values from stored measurements. The
producer code under `solver/` and the large immutable inputs under
`data/replay_inputs/` are supplied so that the underlying physics can be replayed
on a GPU host.  The replay is **not executed** in this package, and no fresh
benchmark is invented here; the physics-replay gate therefore stays false.

## Layers

| layer | producer entry point | inputs |
| --- | --- | --- |
| matched single-graph accuracy/repeatability | `solver/matched_graph_protocol.py` | `data/replay_inputs/branch_reference_cpu.npz`, `branch_reference_sphere_packing_200.npz` |
| matched accumulation | `solver/batch_adjoint_table.py` | `data/replay_inputs/branch_reference_cpu.npz` |
| identification Gauss-Newton | `solver/talos_ident_gn.py`, `solver/estimator_run.py` | `data/replay_inputs/identification_sim_B512.npz`, `identification_sim_win7_B512.npz` |
| identification batch | `solver/estimator_gpu.py`, `solver/estimator_state.py` | `data/replay_inputs/identification_batch_B4096.npz` |
| single-contact slip sweep | `solver/talos_adjoint_gpu.py` | `data/replay_inputs/slip_sweep_w50.npz` |

The driver scripts use the recorded Warp numerical settings and public selectors; the numeric path is unchanged.  A replay
requires an NVIDIA GPU, the Warp runtime and the same numerical configuration
recorded in `data/reference/matched_protocol.json` and
`data/reference/identification_protocol.json`.  Byte identity of a replay is a
repetition check on the recorded configuration, not a new accuracy certificate.

Run `python replay_inputs_check.py` to verify that every replay input is present
and matches the package manifest before attempting a replay.

The delivered matched-graph and estimator entry points import on a CPU host. The scene-key names in the public replay NPZ were cleaned; its numeric NPY members are byte-identical to the archived input, while saved per-run input hashes continue to describe the original run. No GPU physics replay was run.
