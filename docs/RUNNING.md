# Running

Status values: VERIFIED-FRESH = the module's own gates passed in the recorded run; OWN-GATE-FAIL = one or more own gates failed, result retained and unpromoted; CUDA-ONLY = GPU execution not yet verified in the recorded scope; SYNTHETIC-ONLY = passing evidence limited to synthetic inputs. Earlier attempts are notes, not additional current module rows.

1. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` (numpy, scipy, pinocchio via `pin`, coal, pytest; `warp-lang` only for the GPU modules).
2. `PYTHONPATH=src .venv/bin/python -m motion_engine.selftest_motion_stack` — UR10e: IK -> RRT -> smoothing -> time parametrisation -> RNEA effort -> plan confidence, plus an honest non-plan on an unreachable goal.
3. `PYTHONPATH=src .venv/bin/python -m motion_engine.contact_engine` and `-m motion_engine.federation` — contact resolution (ramp stick/slide at atan(mu), K=2 stack, contact force = Mg, friction grip lift) and the solver router.
4. `PYTHONPATH=src .venv/bin/python -m pytest -q tests` — wraps the selftests above plus the planner, smoothing, world-model, dynamics-twin and grasp-certificate checks.
5. GPU modules (`fk_warp`, `rnea_warp`, `trajopt_warp`, `contact_engine_gpu`, `contact_engine_sdf`) need `warp-lang` with a CUDA device; the Newton cells (`probe_friction_grip`, `newton_friction_pick`, `motion_engine.manip.*`) additionally need `newton`, `mujoco-warp` and `torch`, and the LIP4RID friction scripts need the public Zenodo-12516500 recordings under `data/real_external/`.

## Robot cell: kinematics, swept collision, dynamics, pick and place

The cell scripts are fleet agnostic: the kinematic chain comes from a URDF and the geometry from a parts
manifest generated from that URDF. The default robot is the vendored UR10e, the default cell is synthetic.

```
.venv/bin/python scripts/robotcell_leder_v1.py --check                  # fk() vs pinocchio on the same URDF
.venv/bin/python scripts/urdf_parts_manifest.py \
      --static data/cells/synthetic_cell_static_v1.json                 # -> data/cells/ur10e_parts_manifest_v1.json
.venv/bin/python scripts/robotcell_bansvep_v1.py --ref-only             # calibration: 0 candidates expected
.venv/bin/python scripts/robotcell_bansvep_v1.py                        # full-envelope sweep (exit 2 by design)
.venv/bin/python scripts/robotcell_pickplace_v1.py                      # -> data/banor/robotcell_pickplace_v1.json
.venv/bin/python scripts/robotcell_bansvep_v1.py \
      --bana data/banor/robotcell_pickplace_v1.json                     # the planned path must be GREEN_BANA
.venv/bin/python scripts/robotlaster_v1.py \
      --bana data/banor/robotcell_pickplace_v1.json                     # inverse dynamics over that path
```

Point them at another robot with `--urdf` (for the chain) and `--parts-manifest` (for the geometry); any of the
165 descriptions under `assets/robots/fleet_urdf/` works for kinematics and inertials, but only the UR10e ships
with collision meshes, so `urdf_parts_manifest.py` needs the meshes of whichever robot you target.

## Swept CCD, contact, friction

```
.venv/bin/python scripts/tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py
.venv/bin/python scripts/collision_free_trajectory_cert_swept_ccd_plus_margin_binding_segment_witness.py
.venv/bin/python scripts/prbe_impulse_validate.py            # add --full for the full substep counts
.venv/bin/python scripts/fleet_urdf_physics_quality.py       # inertia physics gate over 165 URDF
.venv/bin/python scripts/make_synthetic_friction_cache.py    # -> data/friction/, then the three friction cells
```

## Contact softness and the midpoint-VBD backend

```
PYTHONPATH=src .venv/bin/python -m motion_engine.soft_step             # softness invariants
PYTHONPATH=src .venv/bin/python -m motion_engine.midpoint_vbd_engine   # free fall, flat rest, incline friction
.venv/bin/python scripts/midpoint_vbd_core.py                  # integration order, stability, settle force
.venv/bin/python scripts/vbd_energy_drift_symplectic.py        # secular energy drift, integrator and engine
.venv/bin/python scripts/vbd_nt2_within_step_sliding_impact_gate.py   # within-step normal/tangential impact
.venv/bin/python scripts/vbd_nt_friction_gate_result.py        # the friction battery, folds the NT2 gate in
.venv/bin/python scripts/symplectic_vbd_scheme_crux.py         # which schemes admit per-block descent (~85 s)
```

Each cell writes its numbers to `reports/<cell>.json`. `vbd_nt2_within_step_sliding_impact_gate.py` can
additionally be pointed at an external reference integrator with `NT2_REFERENCE_PATH=<dir>`, where `<dir>`
holds a module `nt_validator` exporting `BrokenNT2` and `AnalyticCoulomb` with an
`.impact(vt, mu, Jn) -> (vt_post, ke_ratio)` method; without it the discrimination row is reported as
unavailable and the gate verdict is unchanged.

`soft_step` is wired into `contact_engine.py` as an opt-in path: `SplitImpulseEngine(..., soft=True)`
replaces the fixed-Baumgarte position pass with a TGS soft substep, and `soft=False` (the default)
leaves the shipped behaviour bit-identical. The soft substep computes `contact_hertz(inv_h, soft_hertz)`
once and builds two `make_soft` softnesses, a stiffer one (twice the hertz, half the damping ratio) for
static and kinematic contacts. `bias_velocity` is called in the normal solve with the separation passed
as `-penetration`; its `mass_scale` multiplies the normal velocity and its `impulse_scale` multiplies the
accumulated impulse. The substep sequence is warm start, one biased solve, integrate positions, one relax
solve. Accuracy comes from the substep count (`soft_substeps`, default 4, enforced as a lower bound on
the caller's substeps) and from the cross-step warm start, not from inner iterations: inner sweeps
re-apply `impulse_scale` and break the mass/impulse-scale partition the primitive guarantees, which is
why the soft path is vel_iters independent (checked in the harness below).

```
.venv/bin/python scripts/engine_metrics.py            # -> reports/engine_metrics.json (about 500 s)
.venv/bin/python scripts/engine_metrics.py --quick    # short scenes, what the test wrapper runs
```

## Engine metrics (2026-09-12)

Measured with `scripts/engine_metrics.py` on this machine, CPU only; the GPU backend was skipped (no CUDA
device). Scenes and thresholds are the pre-registered set: M1 = per-body net contact force equal to Mg
across a K=4 stack, worst-body relative error, threshold < 0.10; M4 = minimum `vel_iters` for a resting
force residual < 0.10 on both bodies of a 2-stack, swept over mass ratio; M5 = maximum penetration in a
K=8 stack normalised by the corner radius R with a bounded-versus-secular drift class, threshold pen/R
< 0.5 and bounded; M6 = run-to-run spread of the final centres of mass, threshold < 1e-2 m. Every force
reading uses `substeps=1`, because `contact_forces()` divides the impulse accumulated over all substeps
by `dt/substeps`.

Sampling changed on 2026-09-12 (names and thresholds unchanged, see `scripts/m4_soft_diagnostic.py`):
every force residual is now the **mean of |residual| over the last 20 % of the steps** rather than the
final sample, and a row is reported **INVALID** — it cannot pass — when the stack it was read from has
collapsed, i.e. the final body spacing is below 0.8 x the 0.205 m rest pitch or a body-body penetration
exceeds the corner radius R = 0.05 m. Re-measured CPU rows (`engine_metrics.py --cpu`, 528 s):

| engine | M1 stack_load_K4 (<0.10) | M4 ratio 1: iters / mean residual / spacing | M4 ratio 1000: iters / mean residual / spacing / validity | M5 stack8 pen/R (<0.5) | M6 eps_nondet (<1e-2 m) |
|---|---|---|---|---|---|
| CPU `soft=False` (v0) | 0.102 FAIL | 4 / 0.000 / 0.200 m | none / 0.000 / 0.050 m / **INVALID** (spacing) | 0.721 bounded FAIL | 0.0 PASS |
| CPU `soft=True` | 0.011 PASS | 2 / 0.006 / 0.200 m | none / 1.506 / 0.112 m / **INVALID** (spacing, pen 0.052 m > R) | 0.127 bounded PASS | 0.0 PASS |
| CPU `soft=True, soft_hertz=60` (labelled row, M4 only) | - | 2 / 0.003 / 0.200 m | none / 0.161 / 0.144 m / **INVALID** (spacing) | - | - |
| GPU | skipped, no CUDA device | skipped | skipped | skipped | skipped |

The soft path keeps both of its wins under the new sampling: the K=4 load-propagation error is 0.011
against 0.102 for the default path (M1), and the K=8 stack penetration 0.127 against 0.721 pen/R (M5),
both with bounded drift and M6 unchanged at bit determinism. Its M4 residual is `vel_iters` independent
by construction and measured so (identical to the last bit at 2 and at 40 iterations).

The earlier v0 M4 "pass" at mass ratio 1000 was read off a collapsed stack — final spacing 0.050 m against
the 0.205 m rest pitch — and is now reported INVALID rather than reached; no engine reaches the M4
tolerance at that ratio on a geometrically valid scene. The default therefore stays `soft=False` on the
old grounds minus the false regression: the mass-ratio-1000 comparison no longer favours either path,
because both rows are INVALID there. The tail-mean also shows the old soft number was a sampling artefact:
the final-sample residual is 1.408 while the mean over the last 40 steps is 1.506, on a trace that visits
0.009 and 4.269 in the same run. The `soft_hertz=60` labelled row is the closest any single knob gets
(mean residual 0.161, spacing 0.144 m, the largest spacing of the three) and is a candidate for a later
measured flip, not a change made here. The flip stays blocked until a mass-ratio scene exists that both
paths keep geometrically valid.

The v0 numbers above are from `scripts/engine_metrics.py` in this repository. The source project's banked v0 values for the
same metric names were M1 1.011 and M5 2.178 (CPU), i.e. worse than measured here; the scenes in this harness were
rebuilt from the metric definitions and are not the original scenes, so the two sets are not comparable
number for number. Only within-harness comparisons (soft on vs off) are claimed.

## Status

| module | status |
|---|---|
| `scripts/robotcell_leder_v1.py` | VERIFIED-FRESH |
| `scripts/urdf_parts_manifest.py` | VERIFIED-FRESH |
| `scripts/robotcell_bansvep_v1.py` | VERIFIED-FRESH |
| `scripts/robotcell_pickplace_v1.py` | VERIFIED-FRESH |
| `scripts/robotlaster_v1.py` | VERIFIED-FRESH |
| `scripts/tropical_sdf_backend_min_of_capsules_beats_voxel_3x_bytes_matched_penetration_so_arm100.py` | VERIFIED-FRESH |
| `scripts/tropical_sdf_so_arm100_geometry_backend_v2_config_driven_posing_plus_contact_normal_gradient.py` | VERIFIED-FRESH |
| `scripts/tropical_sdf_backend_computes_full_v6_contact_geometry_ladder_2jet_curvature_anisotropy_on_so_arm100.py` | VERIFIED-FRESH |
| `scripts/v6_sigma_min_identifiability_climb_force_to_mean_to_spectrum_on_so_arm100_curvatures.py` | VERIFIED-FRESH |
| `scripts/so_arm100_contact_sim_cost_from_tropical_sdf_backend_hertz_stiffness_sets_stable_dt_stability_verified.py` | VERIFIED-FRESH |
| `scripts/tropical_sdf_swept_volume_continuous_collision_detection_min_over_trajectory.py` | VERIFIED-FRESH |
| `scripts/collision_free_trajectory_cert_swept_ccd_plus_margin_binding_segment_witness.py` | VERIFIED-FRESH |
| `scripts/prbe_impulse.py` | VERIFIED-FRESH |
| `scripts/prbe_impulse_validate.py` | VERIFIED-FRESH |
| `scripts/motion_occlusion_safe_plan.py` | VERIFIED-FRESH |
| `scripts/perception_observed_mask_occlusion.py` | VERIFIED-FRESH |
| `scripts/fleet_urdf_physics_quality.py` | VERIFIED-FRESH |
| `scripts/collision_false_free_audit.py` | VERIFIED-FRESH |
| `scripts/retime_jerk_resolution_audit.py` | VERIFIED-FRESH |
| `scripts/motion_footprint_collision.py` | VERIFIED-FRESH |
| `scripts/hum_seq_planner_v0.py` | VERIFIED-FRESH |
| `scripts/make_synthetic_friction_cache.py` | VERIFIED-FRESH |
| `scripts/movement_ik.py` | VERIFIED-FRESH |
| `scripts/fleet_agnostic_planner.py` | VERIFIED-FRESH |
| `scripts/smooth_safe_trajectory.py` | VERIFIED-FRESH |
| `scripts/fleet_waypoint_blending.py` | VERIFIED-FRESH |
| `scripts/twin_fidelity_cap.py` | VERIFIED-FRESH |
| `scripts/collision_aware_shortcut.py` | VERIFIED-FRESH |
| `scripts/motion_stack_contract.py` | VERIFIED-FRESH |
| `scripts/planner_contract.py` | VERIFIED-FRESH |
| `scripts/joint_trajectory_planner.py` | VERIFIED-FRESH |
| `scripts/moment_gate.py` | VERIFIED-FRESH |
| `scripts/constrained_cartesian_motion.py` | VERIFIED-FRESH |
| `scripts/extract_rnea_spec_franka.py` | VERIFIED-FRESH |
| `scripts/extract_kin_spec.py` | VERIFIED-FRESH |
| `scripts/friction_model_reality_gap_presliding_on_real_ur_robot_data.py` | SYNTHETIC-ONLY |
| `scripts/friction_gap_is_dynamic_presliding_superres_on_right_axis_not_static.py` | SYNTHETIC-ONLY |
| `scripts/deployable_friction_cert_error_band_validated_out_of_sample_regime_gated.py` | SYNTHETIC-ONLY |
| `scripts/friction_tournament_harness.py` | SYNTHETIC-ONLY |
| `scripts/friction_aware_effort_backend.py` | SYNTHETIC-ONLY |
| `scripts/lip4rid_loader.py` | SYNTHETIC-ONLY |
| `scripts/vamp_collision_planner.py` | VERIFIED-FRESH |
| `scripts/rnea_warp_parity.py` | VERIFIED-FRESH |
| `scripts/newton_friction_pick.py` | VERIFIED-FRESH |
| `src/motion_engine/motion_stack.py` | VERIFIED-FRESH |
| `src/motion_engine/world_model.py` | VERIFIED-FRESH |
| `src/motion_engine/dynamics_twin.py` | VERIFIED-FRESH |
| `src/motion_engine/dynamic_feasibility.py` | VERIFIED-FRESH |
| `src/motion_engine/contact_engine.py` | VERIFIED-FRESH |
| `src/motion_engine/federation.py` | VERIFIED-FRESH |
| `src/motion_engine/grasp_wrench_cert.py` | VERIFIED-FRESH |
| `src/motion_engine/identifiability_governor.py` | VERIFIED-FRESH |
| `src/motion_engine/selftest_motion_stack.py` | VERIFIED-FRESH |
| `src/motion_engine/selftest_dynamics_twin.py` | VERIFIED-FRESH |
| `src/motion_engine/torch_dynamics.py` | VERIFIED-FRESH |
| `src/motion_engine/torch_gravity.py` | OWN-GATE-FAIL |Final isolated L4 pytest: module exit1 writing data/ur10e_gravparams.npz because the source-only payload lacks data/. Historical extraction result below is not a fresh final-suite pass. |
| `src/motion_engine/fk_warp.py` | VERIFIED-FRESH |
| `src/motion_engine/rnea_warp.py` | VERIFIED-FRESH |
| `src/motion_engine/trajopt_warp.py` | VERIFIED-FRESH |
| `src/motion_engine/world_model_warp.py` | VERIFIED-FRESH |
| `src/motion_engine/contact_engine_gpu.py` | VERIFIED-FRESH |
| `src/motion_engine/contact_engine_gpu_colored.py` | VERIFIED-FRESH |
| `scripts/colored_vs_jacobi.py` | VERIFIED-FRESH |
| `tests/test_colored_solver.py` | VERIFIED-FRESH |
| `src/motion_engine/contact_engine_sdf.py` | VERIFIED-FRESH |
| `examples/compose/field_sdf_to_contacts_to_graph.py` | VERIFIED-FRESH |
| `tests/test_compose_example.py` | VERIFIED-FRESH |
| `src/motion_engine/manip/report.py` | VERIFIED-FRESH |
| `src/motion_engine/manip/profiles.py` | VERIFIED-FRESH |
| `src/motion_engine/manip/rplant.py` | VERIFIED-FRESH |
| `src/motion_engine/manip/scene.py` | VERIFIED-FRESH |
| `src/motion_engine/manip/grip.py` | VERIFIED-FRESH |
| `src/motion_engine/manip/servo.py` | VERIFIED-FRESH |
| `src/motion_engine/manip/sensors.py` | VERIFIED-FRESH |
| `src/motion_engine/soft_step.py` | VERIFIED-FRESH |
| `src/motion_engine/midpoint_vbd_engine.py` | VERIFIED-FRESH |
| `scripts/midpoint_vbd_core.py` | VERIFIED-FRESH |
| `scripts/vbd_energy_drift_symplectic.py` | VERIFIED-FRESH |
| `scripts/vbd_nt_friction_gate_result.py` | VERIFIED-FRESH |
| `scripts/vbd_nt2_within_step_sliding_impact_gate.py` | VERIFIED-FRESH |
| `scripts/engine_metrics.py` | VERIFIED-FRESH |
| `scripts/symplectic_vbd_scheme_crux.py` | VERIFIED-FRESH |

| `scripts/kinematic_validity_gate.py` | VERIFIED-FRESH |
| `scripts/fleet_cartesian_to_plan.py` | VERIFIED-FRESH |
| `scripts/fleet_motion_stack.py` | VERIFIED-FRESH |
| `scripts/fleet_multirobot_movement.py` | VERIFIED-FRESH |
| `scripts/fleet_cartesian_industrial.py` | VERIFIED-FRESH |
| `scripts/fleet_collision_false_free.py` | VERIFIED-FRESH |
| `scripts/verify_motion_sigma_e2e.py` | VERIFIED-FRESH |
| `scripts/fleet_grad_trajopt.py` | CUDA-ONLY |
| `scripts/fleet_mppi_benchmark.py` | VERIFIED-FRESH |
| `scripts/mppi_completeness_audit.py` | VERIFIED-FRESH |
| `scripts/fleet_plannability_metric.py` | VERIFIED-FRESH |
| `scripts/hard_benchmark_plannability.py` | VERIFIED-FRESH |
| `scripts/rrt_completeness_verify.py` | VERIFIED-FRESH |
| `scripts/motion_stack_integration.py` | VERIFIED-FRESH |
| `scripts/coacd_scene_to_planner.py` | VERIFIED-FRESH |

## int64 fixed-point accumulation (2026-09-12)

`contact_engine_gpu.py` accumulates per-body velocity/position deltas in int64 fixed point (scale 1e10, float64
rounding) instead of float32 atomics, so the per-step sum does not depend on the order in which contacts are
processed. Verified on the CPU backend (stack K=8 stable, Fz = Mg 0.0 %, two runs bit-identical). The CUDA run,
measured 2026-09-12 on an RTX 5070 (driver 580.178): max |Δxc| = 0.00e+00 m over two runs, bit-identical; stack K=8 stable,
Fz = Mg at 0.0 %, set_kinematic ok. Previous float32 measurement on CUDA: max |Δxc| = 1.09e-3 m over two runs.


## Planner and dynamics cells (2026-09-12)

All CPU cells below run in `.venv-motion` on CPU only; each is under 120 s.

```
python -u scripts/kinematic_validity_gate.py            # 165 fleet URDFs, joint axis/limit validity + effort provenance
python -u scripts/fleet_cartesian_to_plan.py            # Cartesian goal -> coal IK -> coal RRT -> effort-feasible plan
python -u scripts/fleet_motion_stack.py                 # plan -> time parametrisation -> torque gate on UR10e
python -u scripts/fleet_multirobot_movement.py          # the stack on every vendor description that ships meshes
python -u scripts/fleet_cartesian_industrial.py         # the same pipeline + the fleet-wide effort-provenance guard
python -u scripts/fleet_collision_false_free.py         # false-free rate of the points-vs-SDF route, per robot
python -u probes/honest_plannability_probe.py          # IK quality separated from reachability
python -u scripts/verify_motion_sigma_e2e.py            # twin sigma -> MotionPlan.plan_sigma, with a null control
```

The CUDA cells need `warp` with a CUDA device and a kinematics spec `data/kin_spec_<robot>.npz` in the form
`motion_engine.fk_warp.WarpBatchFK` reads. `scripts/extract_kin_spec.py` writes that spec from any URDF that
ships its collision meshes (the vendored UR10e by default); a description without meshes has no surface to
sample and therefore no spec, which is why the other fleet robots are reported as skipped:

```
python -u scripts/extract_kin_spec.py --robot ur10e   # -> data/kin_spec_ur10e.npz
```


```
python -u scripts/fleet_grad_trajopt.py
python -u scripts/fleet_mppi_benchmark.py
python -u scripts/mppi_completeness_audit.py            # also writes reports/mppi_completeness_scenes.npz
python -u scripts/fleet_plannability_metric.py
python -u scripts/hard_benchmark_plannability.py
python -u scripts/rrt_completeness_verify.py            # CPU code, consumes the npz above (producer: mppi_completeness_audit.py)
```

`scripts/motion_stack_integration.py`, `scripts/vamp_collision_planner.py` and `scripts/coacd_scene_to_planner.py`
need `vamp-planner` (and `coacd` + `trimesh` for the last); all three ship with their tests skipping when the
dependency is absent.

Measured here (2026-09-12, CPU, `nice -n 15`):

| cell | decisive number |
|---|---|
| `kinematic_validity_gate` | 165/165 URDFs loaded, 0 hard-broken; effort limits real 89 / placeholder 4 / missing 72; gate binds on a constructed broken URDF |
| `fleet_cartesian_to_plan` | 5/5 gates; IK residual 0.05 mm, 57 waypoints dense-validated, max peak ratio 0.800 of the effort limit |
| `fleet_motion_stack` | 45-node path collision-free, 44 segments, max peak ratio 0.800, aggressive scaling 10.91 (dynamics gate load-bearing) |
| `fleet_multirobot_movement` | 2/2 vendor descriptions plan collision-free (UR10e 6 DOF, Panda 9 DOF), collision binds on both |
| `fleet_cartesian_industrial` | pipeline peak ratios 0.800 on both real-effort robots; fleet scan 89 real / 4 placeholder / 72 missing, all non-real reported N/A |
| `fleet_collision_false_free` | worst 2.7 % false-free (UR10e), Panda 0.0 %, mean 1.4 % |
| `honest_plannability_probe` | R1 soft 100 %, R2 reachable full-6D 100 %, R3 arbitrary 78 % |
| `verify_motion_sigma_e2e` | point peak 0.458 vs sigma-conservative 1.469 at k=3; at sigma=0 both 0.458; sigma-safe retiming infeasible while the nominal 8.785 s is feasible |

## GPU cells (2026-09-12)

Measured on this machine (RTX 5070, driver 580.178, CUDA 12.9) in `.venv-motion` with `nice -n 15`, one cell at
a time. The Newton cells need `newton==1.2.1` with matching `warp-lang==1.13.0`, `mujoco==3.8.0` and
`mujoco-warp==3.8.0.3`; a newer warp/mujoco pair fails Newton's MuJoCo kernel codegen.

| cell | decisive number |
|---|---|
| `motion_engine.rnea_warp` | pinocchio parity max abs error 1.29e-05 Nm; 15.6M env-RNEA/s at N=4096, 55.5M at N=65536 |
| `scripts/rnea_warp_parity.py` | PARITY: max abs 1.29e-05 Nm, mean 1.10e-06, max rel 4.31e-05 over 64 samples, 7 joints |
| `motion_engine.contact_engine_sdf` | sphere ramp 0.911 m vs theory 0.907 (0 %), cylinder 0.850 vs 0.846, box sticks, lateral drift 3.1 cm braked |
| `motion_engine.torch_gravity` | Historical:6 joints extracted to `data/ur10e_gravparams.npz`, gravity [0,0,-9.81]; final isolated test failed on absent output directory. |
| `motion_engine.{fk_warp,trajopt_warp,world_model_warp,torch_dynamics}` | libraries; import clean on CUDA and exercised by the planner/Newton cells below |
| `scripts/extract_kin_spec.py` | UR10e spec: 6 joints, 2800 surface points, chain parity vs `robotcell_leder_v1.fk` max 4.44e-16 m over 64 poses |
| `probes/probe_friction_grip.py` | P1 ok, P2 grip 6/6 at F>=8 N (NaN-free), P3 dose response: slip 146.5/35.5 mm below 8 N vs 1.7-7.0 mm at/above; 58 substeps/s at W=8 |
| `scripts/newton_friction_pick.py` | F1 100 % hold (>=90 required), F2 median slip 3.3 mm (<=10), F3 pre-grip drift 0.0 mm (<5), W=256, 234 substeps/s |
| `motion_engine.manip.{scene,grip,servo,sensors}` | pick cycle on the vendored UR10e, W=2: task-space error 0.335 m -> 0.028 m in 60 servo steps, contact force finite, weld latched on world 0 only |
| `scripts/fleet_mppi_benchmark.py` | ALL SOLVED: 8/8 genuine scenes (single 4/4, clutter 4/4) on UR10e, clearance 56.7 mm, 1.18M cfg-eval/s |
| `scripts/fleet_plannability_metric.py` | SOLVED 12/12 (moderate 6/6, hard 6/6), fast path 10 at median 675 ms + 2 RRT fallbacks, 0 failures |
| `scripts/hard_benchmark_plannability.py` | FRONTIER FOUND: 36/36 solved over cage / cage_forced / corridor / pillars, 32 fast + 4 fallback, 0 failed |
| `scripts/mppi_completeness_audit.py` | 6/8 low-budget misses, 3 recovered at K1024x90, 3 unsolved -> genuinely hard scenes exist |
| `scripts/rrt_completeness_verify.py` | 4/8 scenes valid (start/goal free in the same model); MPPI-high misses 0, false-infeasible 0, genuinely hard 0 |
| `scripts/fleet_grad_trajopt.py` | own verdict PARTIAL: 3/4 scenes solved on UR10e, worst latency 131 ms; the cell's gate is 4/4, so the row stays CUDA-ONLY |
| `scripts/vamp_collision_planner.py` | VALIDATED: RRTC plan 4.44 ms, 6 waypoints, 320 dense configurations with 0 collisions against the exact geometry |
| `scripts/coacd_scene_to_planner.py` | VALIDATED: 8 convex pieces (15.4 s), 100 % surface coverage at 1.11x volume, plan 98.7 ms, 512 configurations 0 collisions |
| `scripts/motion_stack_integration.py` | VALIDATED: plan 4.2 ms collision-free, 5 segments in 1.44 s, peak ratio 0.800; at 4x speed 16.06 (gate load-bearing) |

Only the fleet robots with a kinematics spec are exercised: the repository vendors the UR10e collision meshes,
so `data/kin_spec_ur10e.npz` is the spec that exists and the other fleet entries report "no kinematics spec".


## Graph-coloured solver (2026-09-12)

`src/motion_engine/contact_engine_gpu_colored.py` (`GraphColoredContactEngine`) is a second GPU member of the
same `ContactEngine` Protocol: identical bodies, broad phase and contact model as
`contact_engine_gpu.RelaxedJacobiContactEngine`, but the solve is Gauss-Seidel by graph colouring. After
contact generation the contacts are sorted by feature id (owning point index on body A, and on body B or -1
for the ground), then coloured **on the GPU** by a Jones-Plassmann loop — each uncoloured contact claims its
dynamic bodies with an atomic minimum over a key that is a hash of its rank packed with the rank, and takes
the colour when it won every body it needs — and the colours are solved one after another. Inside a colour
the bodies are disjoint, so the per-contact impulses are applied with plain reads and writes: no atomics,
no fixed-point accumulator, no relaxation factor. Impulses are warm started by feature id across steps.
Sorting by feature id before colouring is what makes the atomic-counter order of the generation kernel
unable to reach the solve; the colour assignment and the sweep order are then a pure function of the contact
set. Defaults: 20 velocity and 20 position iterations (the lowest velocity count that keeps M5 bounded over
600 steps), against 40 and 20 for the Jacobi engine.

```
PYTHONPATH=src .venv/bin/python -m motion_engine.contact_engine_gpu_colored   # contract selftest (CUDA)
.venv/bin/python scripts/engine_metrics.py --gpu-engine both --gpu-only       # both GPU backends
.venv/bin/python scripts/colored_vs_jacobi.py            # -> reports/colored_vs_jacobi.json (315 s)
```

Measured on this machine (RTX 5070, warp 1.13.0, CUDA 12.9), fresh, two runs per timing number.

Contract selftest, coloured engine: K=8 stack stable (KE 0.002 J, spacing 0.200-0.200 m, 72 contacts in 28
colours), contact force on a box at rest Fz = Mg at 0.0 % error, `set_kinematic` holds and moves live, and
two runs of a K=4 stack over 200 steps are bit-identical (max |Δxc| = 0.00e+00 m).

(b) `engine_metrics.py --gpu-engine both --gpu-only`, each engine at its own default iteration counts:

| engine | velocity/position iters | M1 stack_load_K4 (<0.10) | M5 stack8 pen/R (<0.5) | M6 eps_nondet (<1e-2 m) | harness wall |
|---|---|---|---|---|---|
| GPU relaxed Jacobi | 40 / 20 | 1.1e-07 PASS | 0.146 bounded PASS | 0.0 PASS (bit-deterministic) | 8.1 s |
| GPU graph-coloured | 20 / 20 | 0.0019 PASS | 0.174 bounded PASS | 0.0 PASS (bit-deterministic) | 81.9 s |

M4 at mass ratio 1000 is not reached by either engine: the Jacobi row ends NOT REACHED with a final spacing
of 2.5e6 m (the 2-stack blows up), the coloured row ends INVALID with a final spacing of 0.050 m against the
0.205 m rest pitch (the 2-stack collapses but stays finite).

(a) Velocity iterations to the M5 tolerance (pen/R < 0.5 **and** bounded drift), position iterations fixed at
10 for both engines, 600 steps, sweep 2, 4, 6, 8, 10, 20, 40:

| scene | relaxed Jacobi | graph-coloured | best pen/R in the sweep (Jacobi / coloured) |
|---|---|---|---|
| K=8 stack | 6 | 20 | 0.14 at 40 / 0.17 at 20 |
| K=16 stack | 40 | not reached in the sweep | 0.41 at 40 / 0.85 at 40 |

(c) Colours per scene (coloured engine, maximum over the run):

| scene | bodies | contacts | colours |
|---|---|---|---|
| K=4 stack (M1) | 4 | 36 | 24 |
| K=8 stack (M5) | 8 | 72 | 28 |
| K=16 stack | 16 | 144 | 34 |
| lattice, 2 layers | 1000 | 9000 | 18 |

(d) Wall time per step, a lattice of boxes dropped onto the ground (two layers), each engine at its default
iteration counts, two runs each:

| N bodies | contacts | Jacobi ms/step (run 1, run 2) | Jacobi steps/s | coloured ms/step (run 1, run 2) | coloured steps/s |
|---|---|---|---|---|---|
| 1 000 | 8 999 | 2.906, 2.873 | 344 | 19.084, 19.121 | 52.4 |
| 10 000 | 89 997 | 5.605, 5.599 | 178 | 29.807, 29.718 | 33.6 |

Reading, plainly: colouring alone is **not** better than relaxed Jacobi on this repository's scenes.
It buys determinism by construction (no atomics in the solve at all, M6 = 0 by disjointness rather than by
fixed-point accumulation) and it reaches comparable M5 and M1 at half the velocity iterations at K=8, but it
needs *more* velocity iterations than Jacobi to pass M5 at a fixed position-iteration budget (20 against 6 at
K=8) and it does not reach the M5 tolerance at all on the K=16 stack inside the sweep, where Jacobi reaches
it at 40. The cause is visible in (c): a deep stack puts ~9 contacts on every body pair, so the contacts of
one interface cannot share a colour and the colour count grows with stack depth (24-34 colours for 4-16
bodies), which both serialises the sweep — 28 kernel launches per iteration at K=8 — and makes each colour a
small launch. The wall-clock cost follows: 3.4x (N=1e3) to 5.3x (N=1e4) slower per step than Jacobi, 33.6
steps/s at N=1e4 against the 60 fps target. The scaling direction is the one the design predicts (the ratio
of coloured to Jacobi cost falls as N grows, because the colour count is set by the local contact topology
and not by N: 18 colours at N=1e3 and at N=1e4), but at these depths the colour count, not the arithmetic, is
the bottleneck. That is the measured input for step 3 (chunkwise WY inside a colour): the thing to remove is
the number of colours forced by contacts that share a body.

## Manifold reduction on the coloured solver (2026-09-12)

`GraphColoredContactEngine(manifold_reduce=True)` adds one stage between contact generation and the
colouring, and changes nothing else. The contacts are grouped by body pair and each pair is reduced to at
most 4 representative points. Which points: the E-optimal rule of this repository's manifold-reduction
cells (`engine_contact_manifold_rank_twist_selfstress_2point_reduction`,
`engine_manifold_reduction_needs_3points_not_2_wrench_span`,
`engine_which_3_points_Eoptimal_sigmamin_manifold_not_max_area` in the kernel repository). A face contact
wrench is 3-dimensional (F_z, tau_x, tau_y), so the wrench map `G = [1, r_y, -r_x]` per point, taken about
the projected centre of mass of body A, needs rank 3: two points leave one torque unconstrained. Among the
subsets the E-optimal choice maximises `sigma_min(G)` -- the worst-direction wrench margin, equal to
`sqrt(min(npts, lambda_min(second moment of the points about the reference)))` -- which selects a round
straddle of the reference rather than the largest-area triple. The deepest point of the pair is always kept
and seeds a greedy that adds the next three points by `sigma_min(G)`. The pair's total normal impulse is
conserved: every dropped contact's warm-started normal impulse is added to the kept point nearest to it in
the contact plane. The colouring priority is ranked by slot within the pair, then by the parity of the
pair's rank, then by contact rank, so a chain of pairs is taken in two rounds per slot. Defaults with the
flag on: 40 velocity and 20 position iterations (the lowest of the sweep at which the reduced K=8 stack
keeps M5); the flag is off by default, so the plain coloured numbers above are unchanged.

```
.venv/bin/python scripts/engine_metrics.py --gpu-engine all --gpu-only   # three GPU rows
.venv/bin/python scripts/colored_vs_jacobi.py            # -> reports/colored_vs_jacobi.json (641 s)
PYTHONPATH=src .venv/bin/python -m pytest tests/test_colored_solver.py   # 7 passed
```

Measured on this machine (RTX 5070, warp 1.13.0, CUDA 12.9), fresh, two runs per timing number, the same
script and the same scenes as the step-1 table above (its sweep is extended by 60 and 80 velocity
iterations, which also gives the plain coloured column a K=16 result it did not have before).

(c) Colours and contacts per scene, maximum over the run, and the contacts actually solved:

| scene | bodies | contacts generated | coloured: solved / colours | + manifold: solved / colours |
|---|---|---|---|---|
| K=4 stack (M1) | 4 | 36 | 35 / 24 | 16 / 8 |
| K=8 stack (M5) | 8 | 72 | 68 / 28 | 32 / 8 |
| K=16 stack | 16 | 144 | 131 / 34 | 64 / 8 |
| lattice, 2 layers | 1000 | 9 000 | 9 000 / 18 | 4 000 / 8 |

(a) Velocity iterations to the M5 tolerance (pen/R < 0.5 **and** bounded drift), position iterations 10,
600 steps, sweep 2, 4, 6, 8, 10, 20, 40, 60, 80:

| scene | relaxed Jacobi | graph-coloured | + manifold | best pen/R (Jacobi / coloured / manifold) |
|---|---|---|---|---|
| K=8 stack | 6 | 20 | 40 | 0.14 at 40 / 0.17 at 20 / 0.17 at 40 |
| K=16 stack | 40 | 60 | 80 | 0.30 at 60 / 0.17 at 80 / 0.32 at 80 |

(b) `engine_metrics.py --gpu-engine all --gpu-only`, each column at its own default iteration counts:

| engine | velocity/position iters | M1 stack_load_K4 (<0.10) | M5 stack8 pen/R (<0.5) | M6 eps_nondet (<1e-2 m) |
|---|---|---|---|---|
| GPU relaxed Jacobi | 40 / 20 | 1.1e-07 PASS | 0.146 bounded PASS | 0.0 PASS (bit-deterministic) |
| GPU graph-coloured | 20 / 20 | 0.0019 PASS | 0.174 bounded PASS | 0.0 PASS (bit-deterministic) |
| GPU coloured + manifold | 40 / 20 | 0.0013 PASS | 0.169 bounded PASS | 0.0 PASS (bit-deterministic) |

Contact force on a box at rest with the reduction on: Fz = 123.606 N against Mg = 123.606 N, 0.0002 % error
(the reduction keeps 4 of the 8 ground contacts of that box and moves the dropped normal impulses onto the
kept points, so the sum is unchanged).

(d) Wall time per step, a lattice of boxes dropped onto the ground (two layers), each column at its default
iteration counts, two runs each:

| N bodies | Jacobi ms/step (run 1, run 2) | coloured ms/step | + manifold ms/step | manifold steps/s | manifold / Jacobi |
|---|---|---|---|---|---|
| 1 000 | 2.779, 2.779 | 18.641, 18.531 | 16.764, 16.787 | 59.6 | 6.0x |
| 10 000 | 5.553, 5.555 | 29.443, 29.436 | 52.410, 52.384 | 19.1 | 9.4x |

Reading, plainly: the reduction does what it was built to do to the colour count -- 28 and 34 colours on the
K=8 and K=16 stacks become 8, and 18 colours on the 1e3 lattice become 8, because a body pair now carries at
most 4 contacts and the pair graph, not the contact list, sets the number of rounds -- and it keeps the
physics: M1, M5 and M6 pass, the resting contact force is conserved to 0.0002 %, and the K=16 stack reaches
the M5 tolerance, which the coloured solver did at 60 velocity iterations and this column does at 80. It
costs iterations: half the contacts carry the same load, so the K=8 stack needs 40 velocity iterations where
the plain coloured solve needs 20 and Jacobi needs 6.

It is still slower than Jacobi in wall time, and at N=1e4 it is slower than the plain coloured solve: 6.0x
Jacobi at N=1e3 (where it is 10 % faster than plain colouring) and 9.4x at N=1e4 (78 % slower than plain
colouring). The cause is measured, not guessed: at N=1e4 the same scene takes 50.56 ms/step at 20 velocity
iterations and 53.10 ms/step at 40, so the 20 extra iterations over 8 colours cost 2.5 ms and the remaining
~48 ms is fixed per-step cost. The reduction stage runs on the host -- the 85 000 generated contacts are read
back, grouped by pair and selected in numpy every step -- and that readback and grouping, not the solve, is
what dominates at that size. The solve itself is now very cheap: 8 colour launches per iteration against 18
before. The next thing to remove is therefore the host round trip (grouping, the greedy sigma_min selection
and the nearest-point assignment are all per-pair work that fits a kernel), not the number of colours.

## Cross-repo compose example (2026-09-12)

`examples/compose/field_sdf_to_contacts_to_graph.py` composes three repositories in one run: the field
engine rasterises a part into a signed field, this engine drops a box on that field and measures the
resting contact, and the graph engine takes the measurement as a node. The sibling repositories are found
under `THREEFOLD_ROOT` (default: the parent of this repository); nothing outside this file is changed in
any of them, and the example skips (pytest) when the siblings are absent.

```
.venv/bin/python examples/compose/field_sdf_to_contacts_to_graph.py      # 3 stages, exit 1 on any FAIL
PYTHONPATH=src .venv/bin/python -m pytest tests/test_compose_example.py  # 1 passed
```

Measured here (RTX 5070, warp 1.13.0, CUDA 12.9; the field stage runs on the CPU backend):

| stage | gate | measured | verdict |
|---|---|---|---|
| field | voxel volume vs analytic volume, < 5 % (the field repository's own rasterisation band) | 31752.0 vs 31636.3 mm3, rel 3.66e-03, watertightness OK | PASS |
| motion | Fz = Mg within 1 % | Fz = Mg = 0.003516 N, rel 0.0e+00 | PASS |
| motion | penetration / R < 0.5 | 0.100 mm / 4.0 mm = 0.025 | PASS |
| motion | two runs bit-identical | xc, Rm, vc, om identical byte for byte | PASS |
| graph | the node loads and the query returns it | `GraphStore.load` returns the node, `neighbors(depth=1)` returns both producer nodes | PASS |

Three adapters were needed, all of them in the example and none in either engine; they are the measure of
how flush the seams are:

1. Units. The field engine works in mm, this engine in SI metres. Converted once at the boundary.
2. Contact surface. `contact_engine_sdf` resolves contact against an analytic half-space (plane or ramp);
   it has no input for a sampled grid, so the exported `.npz` cannot be handed to it. The adapter reads a
   support plane out of the grid instead: it samples the field trilinearly over the drop footprint, takes
   the zero crossing per column, and requires the crossing to be flat (0.000 mm over the footprint) and
   the material to continue two pitches below it before it gives the plane to the engine, which then runs
   in the plane frame. This is the one real gap: a grid-SDF contact path would remove it.
3. Result dict to plain arrays. The field entry point returns a dict (window origin `gmin`, shape, dense
   field, block counts); `sdf`, `origin` and `pitch` are assembled from it here.

One number worth keeping: the support plane read out of the grid lands at z = 2.500 mm against an analytic
top face at 3.000 mm, exactly -0.50 pitch. That is the zero crossing of the distance transform, not a
choice made in the example, and it is the same half-pitch the field module documents; a consumer that
wants the part surface rather than the grid surface has to know it.


## Manifold stage on the device (2026-09-12)

The grouping and the E-optimal selection of the section above ran in numpy on the host and cost 48 of the
53 ms of a step at N = 1e4. They are now GPU kernels and are the default of `manifold_reduce=True`;
`manifold_on_host=True` keeps the numpy stage for A/B. The stages, in order: a key kernel and a stable
radix sort (`wp.utils.radix_sort_pairs`) by feature id, the gather, a second key kernel and stable radix
sort by body pair, a run-length kernel for the segment boundaries, one thread per body pair for the
E-optimal pick of at most 4 points and the nearest-point redistribution of the dropped normal impulses, a
scan and a stream compaction of the kept contacts, and the colouring priority (slot, pair parity, rank)
built from a second run-length pass over the kept list. The warm start is a binary search in the kept keys
of the previous step, so no contact data crosses to the host: the only readback of the stage is two
integers (kept count, oversize-segment counter).

The selection kernel is compiled as its own module with floating-point contraction off (`fuse_fp=False`).
A flat contact manifold is full of exact score ties -- on a 3x3 ground patch seven of the eight candidates
score identically in the first greedy round -- so a fused multiply-add changes the last bit of one
candidate, the tie breaks the other way and a different (equally E-optimal) point set is kept. Without
contraction the device scores are the same doubles as the numpy scores. The solve kernels are untouched.

Identity to the stage it replaces, asserted, not assumed: same kept contacts in the same order and
`n_contacts_solved` equal at every step, and `max |delta xc| = 0` between the two paths, on the K=8 stack
(400 steps) and on the 1e3 lattice (12 steps). The (a), (b) and (c) numbers of the host column below are
therefore reproduced digit for digit by the device column.

```
PYTHONPATH=src .venv/bin/python -m motion_engine.contact_engine_gpu_colored   # contract selftest (CUDA)
.venv/bin/python scripts/colored_vs_jacobi.py            # -> reports/colored_vs_jacobi.json (716 s)
PYTHONPATH=src .venv/bin/python -m pytest tests/test_colored_solver.py   # 9 passed
```

Measured on this machine (RTX 5070, warp 1.13.0, CUDA 12.9), fresh, two runs per timing number, the same
script and scenes as the two sections above.

(d) Wall time per step, lattice dropped onto the ground, each column at its default iteration counts:

| N bodies | Jacobi ms/step | coloured ms/step | + manifold on host | + manifold on device | device steps/s | device / Jacobi |
|---|---|---|---|---|---|---|
| 1 000 | 3.091, 2.913 | 19.351, 20.273 | 17.666, 17.390 | 11.910, 11.693 | 84.0, 85.5 | 3.9x |
| 10 000 | 5.754, 5.757 | 31.523, 30.396 | 57.617, 57.188 | 14.772, 14.729 | 67.7, 67.9 | 2.6x |

(e) Per-stage wall time of one step, manifold on device, device-synchronised at every stage boundary (the
synchronisation is why the profiled total is 2 % above the ms/step of (d)):

| stage | N = 1e3 | N = 1e4 |
|---|---|---|
| generation (gravity, world points, hash grid, contacts) | 0.229 | 1.515 |
| sort (two radix sorts, gather, segment flags) | 0.204 | 0.399 |
| select (warm start, E-optimal pick, scan, compaction, priority) | 0.279 | 0.971 |
| colour (Jones-Plassmann rounds + the colour spans) | 1.035 | 1.608 |
| solve (40 velocity + 20 position sweeps over 8 colours) | 9.722 | 9.737 |
| integrate (+ the contact-force readback) | 0.172 | 0.627 |
| warm-start store | 0.041 | 0.038 |
| total (profiled) | 11.777 | 15.021 |

(a) Velocity iterations to the M5 tolerance and (c) colours per scene: identical to the host column, by the
identity above -- K=8 reaches M5 at 40 velocity iterations and K=16 at 80; 8 colours on the K=4, K=8 and
K=16 stacks and on the 1e3 lattice, 32 of 72 contacts solved on the K=8 stack and 4 000 of 8 160 on the
lattice. (b) `engine_metrics.py --gpu-engine all --gpu-only` row `gpu_manifold`, which now builds the
device path: M1 5.21e-04 PASS, M5 0.169 bounded PASS, M6 0.0 bit-deterministic -- the same three numbers as
the host stage produced, at 36.8 s instead of 42.3 s of harness wall time. Contact force on a box at rest,
device path: Fz = Mg = 123.61 N, 0.000 % error.

Reading, plainly: moving the stage to the device removes the host round trip it was built to remove.
At N = 1e4 the step goes from 57.4 to 14.75 ms (3.9x), which is also 2.1x faster than the plain coloured
solver (30.96 ms) that solves 2.1 times as many contacts, and the wall-time gap to relaxed Jacobi falls
from 9.4x to 2.6x. At N = 1e3 the gain is smaller (17.53 to 11.80 ms, 1.5x) and the ratio to Jacobi is
3.9x, because at that size the step is dominated by launch latency, not by the host stage.

It is still slower than Jacobi. The remaining time is measured, not guessed: at N = 1e4, 9.74 of the
15.02 ms is the solve, and that is 60 velocity+position sweeps over 8 colours = 480 kernel launches of
40 000 contacts each, i.e. launch-bound work of about 20 us per launch on 40 000 threads. The manifold
stage it replaced is now 1.37 ms of the step (sort 0.40 + select 0.97), the colouring 1.61 ms (measured split at
N = 1e4: 0.85 ms of Jones-Plassmann rounds on the device and 0.87 ms of colour-array readback, host argsort
and order upload), generation 1.52 ms and
integration 0.63 ms. The next thing to remove is therefore inside the solve: fewer, larger launches
(one launch over all colours with a device-side span table, or a chunkwise sweep that keeps a colour
resident), and after that the colour-span readback -- not the grouping, which no longer shows in the step.

## Graph-captured colour loop (2026-09-12)

The step above ended with the solve as the whole cost: 8 colours x (40 velocity + 20 position) sweeps =
480 kernel launches of about 20 us of host work each, 9.7 of the 14.7 ms of a step at N = 1e4, and the
launches, not the arithmetic, were the step. With `graph_capture=True` the whole per-step solve loop is
captured once with `wp.ScopedCapture` and replayed with `wp.capture_launch`.

The default capture (`graph_mode="device"`) bakes in nothing that changes between steps: each colour's
offset and count are read from device arrays by the sweep kernels (`k_gs_vel_d` / `k_gs_pos_d`), and those
arrays are built on the device by a counting sort of the contacts on the colour id, a histogram of the
colour ids and an exclusive scan, so the colour array is no longer read back and no permutation is uploaded.
The launch dimension is the contact count rounded up to a power of two, an upper bound for any colour, so
the graph is re-captured only when the number of colours or that bucket changes -- twice over the 40 timed
steps of the lattice runs. `graph_mode="spans"` bakes the offsets into the launches and re-captures whenever
the partition changes; it is kept for the A/B and is the slower of the two, measured once at each size on
the same lattice: 3.65 against 3.61 ms at N = 1e3 and 6.62 against 6.31 ms at N = 1e4. The bodies of the two sweeps are now warp functions (`f_gs_vel`, `f_gs_pos`)
called by both the launch-offset and the device-offset kernels, so the captured and the uncaptured path run
the same compiled arithmetic.

Identity to the path it replaces, asserted, not assumed: same colours and same `n_contacts_solved` at every
step and `max |delta xc| = 0` between the captured and the uncaptured engine on the K=8 stack (400 steps)
and on the 1e3 lattice (12 steps); the two graph modes agree the same way; and the uncaptured path is still
bit-identical to the step-3a code. (a) and (c) are therefore the numbers of the column before it.

```
PYTHONPATH=src .venv/bin/python -m motion_engine.contact_engine_gpu_colored   # contract selftest (CUDA)
.venv/bin/python scripts/colored_vs_jacobi.py            # -> reports/colored_vs_jacobi.json (796 s)
PYTHONPATH=src .venv/bin/python -m pytest tests/test_colored_solver.py   # 11 passed
```

Measured on this machine (RTX 5070, warp 1.13.0, CUDA 12.9), fresh, two runs per timing number, the same
script and scenes as the sections above.

(d) Wall time per step, lattice dropped onto the ground, each column at its default iteration counts:

| N bodies | Jacobi ms/step | coloured ms/step | + manifold on host | + manifold on device | + graph-captured | captured steps/s | captured / Jacobi |
|---|---|---|---|---|---|---|---|
| 1 000 | 2.947, 2.964 | 20.298, 18.899 | 18.494, 16.863 | 11.739, 12.249 | 3.591, 3.583 | 278.5, 279.1 | 1.21x |
| 10 000 | 5.742, 5.606 | 30.629, 31.607 | 53.084, 54.587 | 15.093, 15.316 | 6.349, 6.447 | 157.5, 155.1 | 1.13x |

The capture is 3.3x the uncaptured coloured step at N = 1e3 and 2.4x at N = 1e4, and it is still slower
than relaxed Jacobi: 6.40 ms against 5.67 ms at N = 1e4 (mean of the two runs), 1.13x, and 1.21x at
N = 1e3. It is not faster than Jacobi.

(e) Per-stage wall time of one step, manifold on device, uncaptured and captured, device-synchronised at
every stage boundary (the synchronisation is why the profiled total is a few per cent above (d)):

| stage | N = 1e3 uncaptured | N = 1e3 captured | N = 1e4 uncaptured | N = 1e4 captured |
|---|---|---|---|---|
| generation (gravity, world points, hash grid, contacts) | 0.22 | 0.22 | 1.45 | 1.43 |
| sort (two radix sorts, gather, segment flags) | 0.20 | 0.19 | 0.33 | 0.30 |
| select (warm start, E-optimal pick, scan, compaction, priority) | 0.28 | 0.30 | 0.86 | 0.91 |
| colour (Jones-Plassmann rounds + the colour spans) | 1.02 | 0.79 | 1.55 | 0.87 |
| solve (40 velocity + 20 position sweeps over 8 colours) | 9.80 | 1.97 | 9.78 | 2.31 |
| integrate (+ the contact-force readback) | 0.18 | 0.18 | 0.65 | 0.64 |
| warm-start store | 0.05 | 0.16 | 0.04 | 0.06 |
| total (profiled) | 11.86 | 3.90 | 14.80 | 6.64 |

Where the time is now, at N = 1e4: the solve is 2.31 ms of the 6.4 ms step (480 sweeps over 40 000
contacts, now genuinely on the device), contact generation 1.43 ms, the E-optimal selection 0.91 ms, the
colouring 0.87 ms (Jones-Plassmann rounds, each of which still reads one integer back to decide whether to
stop), integration 0.64 ms and the pair sorts 0.30 ms. No stage dominates any more. The host round trips left in a step
are the contact count after generation, the kept count of the selection, one integer per colouring round,
and the two velocity readbacks that produce the reported contact force; no contact data crosses.

(b) `engine_metrics.py --gpu-engine all --gpu-only` (four GPU rows; the new row `gpu_graph` is the manifold
engine with `graph_capture=True`): M1 5.210411e-04 PASS, M5 0.169374 bounded PASS, M6 0.0 bit-deterministic
-- the same digits as `gpu_manifold`, at 11.9 s instead of 36.1 s of harness wall time. (a) K=8 reaches the
M5 tolerance at 40 velocity iterations and K=16 at 80, (c) 8 colours and 32 of 72 contacts solved on the
K=8 stack and 4 000 of 8 160 on the 1e3 lattice: the fourth column's numbers, by the identity above.


## Body-pair chunks (2026-09-12)

The step above left a solve that needs 40 velocity iterations at K=8 and 80 at K=16, because after the
manifold reduction a body pair still carries up to 4 contacts, those 4 conflict with each other and are
therefore spread over up to 8 colours, so a pair converges only ACROSS colour sweeps. With
`pair_chunks=True` the colouring moves from the contact graph to the BODY-PAIR graph (a node per contacting
pair, an edge where two pairs share a dynamic body) and one thread takes a whole pair: its <= 4 contacts are
solved one after the other inside the thread. That is an exact sequential Gauss-Seidel sweep of the chunk.
The WY compression of the chunk is not applicable and was not used: each of the <= 4 updates is state
dependent through `max(0, .)` on the normal impulse and through the projection onto the friction cone, so
the chunk is not a fixed product of rank-1 projectors -- the sequential loop IS the exact form. The two ways
of running that loop were measured against each other: `chunk_mode="registers"` (default) holds the two
bodies' velocities in registers over the chunk, `chunk_mode="global"` calls the same per-contact warp
function once per contact so the velocities round-trip through global memory. They produce the same states
bit for bit and registers is the faster of the two on the same lattice, two runs each: 2.32, 2.41 against
2.88, 2.87 ms at N = 1e3 and 5.00, 4.93 against 5.60, 5.51 ms at N = 1e4.

Everything else is unchanged: the same manifold reduction and warm start, the same plain (non-atomic) writes
-- the bodies of a colour are still disjoint, now per pair instead of per contact -- and the whole chunk loop
is captured as one CUDA graph the same way, offsets and counts read from device arrays. The pair grouping
itself runs on the device (radix sort on the pair key, run-length flags, a scan, one thread per segment) and
costs 0.14 ms at N = 1e3 and 0.15 ms at N = 1e4.

```
PYTHONPATH=src .venv/bin/python -m motion_engine.contact_engine_gpu_colored   # contract selftest (CUDA)
.venv/bin/python scripts/colored_vs_jacobi.py            # -> reports/colored_vs_jacobi.json (1172 s)
.venv/bin/python scripts/engine_metrics.py --gpu-engine chunks --gpu-only     # the gpu_chunks row
PYTHONPATH=src .venv/bin/python -m pytest tests/test_colored_solver.py   # 16 passed
```

Measured on this machine (RTX 5070, warp 1.13.0, CUDA 12.9), fresh, two runs per timing number, the same
script and scenes as the sections above.

(c) Colours, and what a colour now is:

| scene | contacts | solved after the reduction | pairs | colours, per contact | colours, per pair |
|---|---|---|---|---|---|
| K=4 stack | 36 | 16 | 4 | 8 | 3 |
| K=8 stack | 72 | 32 | 8 | 8 | 3 |
| K=16 stack | 144 | 64 | 16 | 8 | 3 |
| 1e3 lattice | 8 160 | 4 000 | 1 000 | 8 | 2 |

(a) Velocity iterations to the M5 tolerance (pen/R < 0.5 and bounded drift), pos_iters = 10, 600 steps:

| engine | K = 8 | K = 16 |
|---|---|---|
| relaxed Jacobi | 6 | 40 |
| coloured, no reduction | 20 | 60 |
| coloured + manifold (steps 3a, 3b) | 40 | 80 |
| coloured + manifold + pair chunks | 40 | 160 |

The iteration count did NOT improve; at K = 16 it doubled. The chunk removes the split WITHIN a pair, but a
stack's coupling is BETWEEN pairs, and that coupling is carried by the colour sweep: with 8 colours the
per-contact colouring happened to order the pairs of the chain over many colours, so one iteration pushed
information several links up the stack, while 3 colours push it about two links. Fewer colours is cheaper
per iteration and weaker per iteration, and on the stacks the second effect wins. On the lattice, where the
bodies are wide and shallow rather than a long chain, it does not bite.

(d) Wall time per step, lattice dropped onto the ground, each column at its default iteration counts
(40 velocity + 20 position for the last three; the chunk column is measured at the same 40 as the column it
is compared with):

| N bodies | Jacobi ms/step | + graph-captured (3b) ms/step | + pair chunks ms/step | pair-chunk steps/s | pair chunks / Jacobi |
|---|---|---|---|---|---|
| 1 000 | 2.823, 2.796 | 3.644, 3.666 | 2.318, 2.306 | 431.5, 433.6 | 0.82x |
| 10 000 | 5.591, 5.586 | 6.396, 6.401 | 4.931, 4.944 | 202.8, 202.3 | 0.88x |

Pair chunks beat relaxed Jacobi at N = 1e4: 4.94 ms against 5.59 ms per step (means of the two runs each),
1.13x faster, 202 against 179 steps/s, with Gauss-Seidel convergence and atomics-free bit determinism; at
N = 1e3, 2.31 against 2.81 ms, 1.21x faster. This is the first column of the series that is faster than the
Jacobi engine.

(e) Per-stage wall time of one step, device-synchronised at every stage boundary (the synchronisation is why
the profiled total is a few per cent above (d)):

| stage | N = 1e3, 3b | N = 1e3, pair chunks | N = 1e4, 3b | N = 1e4, pair chunks |
|---|---|---|---|---|
| generation (gravity, world points, hash grid, contacts) | 0.23 | 0.20 | 1.43 | 1.47 |
| sort (two radix sorts, gather, segment flags) | 0.19 | 0.19 | 0.39 | 0.30 |
| select (warm start, E-optimal pick, scan, compaction) | 0.28 | 0.27 | 0.92 | 0.95 |
| pairs (pair-key sort, run lengths, segment build) | - | 0.14 | - | 0.15 |
| colour (Jones-Plassmann rounds + the colour spans) | 0.77 | 0.31 | 0.82 | 0.34 |
| solve (40 velocity + 20 position sweeps over the colours) | 1.95 | 1.02 | 2.24 | 1.13 |
| integrate (+ the contact-force readback) | 0.21 | 0.17 | 0.62 | 0.65 |
| warm-start store | 0.18 | 0.12 | 0.07 | 0.14 |
| total (profiled) | 3.89 | 2.50 | 6.61 | 5.25 |

Where the time went: the solve fell 2.24 -> 1.13 ms at N = 1e4 (60 sweeps over 2 colours instead of over 8,
and each thread now amortises the body load over 4 contacts) and the colouring 0.82 -> 0.34 ms (1 000 pair
nodes to colour instead of 4 000 contact nodes, in fewer rounds), against 0.15 ms for the new pair grouping.
What is left at N = 1e4 is contact generation 1.47 ms, the solve 1.13 ms, the E-optimal selection 0.95 ms,
integration 0.65 ms, colouring 0.34 ms, the sorts 0.30 ms: generation is now the largest single stage and
the solve is no longer the step.

(b) `engine_metrics.py --gpu-engine all --gpu-only` (five GPU rows; the new row `gpu_chunks`): M1
1.339110e-02 PASS, M5 0.170328 bounded PASS, M6 0.0 bit-deterministic PASS, harness wall 8.4 s. Fz = Mg on
the resting box is exact to 0.000 % in the selftest, and the K=8 stack keeps its 0.200 m spacing. On the
1000:1 mass-ratio scene of M4 it stays where the coloured columns are (tail-mean residual 0.143 against
Jacobi's 1.000) on geometry the validity gate still marks INVALID, so M4 is NOT REACHED for every engine.

## Innovation target 1: registered measurement

| module | status | evidence |
|---|---|---|

Before execution: K=16 over 600 steps, position iterations=10, velocity iterations<=40;
existing M5 penetration/R<0.5, bounded drift and geometry validity must all pass.
The N=10000 two-layer lattice uses 40 velocity and 20 position iterations, 10 warm-up
steps and 30 measured steps; each of two wall-time legs must be <=4 ms.
Two full K=16 trajectories (positions, rotations, linear and angular velocities)
must match byte for byte. Timing values are excluded from the trajectory digest.
Abort before each measurement if another non-desktop compute process is present.
Before the first measurement, the classifier was corrected to distinguish known desktop
G/C+G contexts from pure compute jobs; the original classifier aborted on the browser
and collected zero timing samples. Initial utilization is recorded. The whole command
must hold the machine-wide lock agreed by the concurrent sessions.
The frozen pair-chunk and relaxed solver source files are not edited.

Measurement attempt outcome: the initial desktop-process classifier and a duplicate
acquisition of two aliases of the same advisory lock both rejected before sampling.
After those preflight corrections, CUDA failed before the first simulation step:
`Failed to acquire primary context for device cuda:0` (errors 999 and 201).
The kernel log reports Xid 62 followed by Xid 45 and `GPU Reset Required`.
No convergence, determinism or performance verdict is available from this attempt.
GPU retries are paused. No candidate solver has been designed or implemented.

| target | fresh mechanism table | gates | measured outcome | remaining |
|---|---|---|---|---|
| 1, tall stacks | unavailable: context creation failed | K16, <=40 velocity iterations, existing M5 and validity, two byte-identical trajectories, both timing legs <=4 ms | 0 steps; 0 timing samples; CUDA errors 999/201 | Restore a usable CUDA context, measure baseline, write table, then design beside baseline. |

### Target 1 CPU mechanism probe: gates registered before execution

| module | status | evidence |
|---|---|---|

Pre-registered instrument gates: unit-mass K=4/8/16 contact Jacobians must map
closed-form impulses (K,...,1) to unit force with exactly zero error; the dense
solve must agree within 1e-12; contacts sharing a cyclic colour must share no body;
all measured residuals must be finite. Run the complete script twice and require
byte-identical JSON. Measure 1, 40 and 160 sweeps from zero impulses. A worst-body
force residual <0.10 at K=16 and 40 sweeps is the diagnostic target, not the engine's
M5 gate. The three cyclic colours are synthetic, not the frozen engine's hash
colouring. No friction, rotation, changing contacts, warm start or timing is modelled.

Measured CPU mechanism table, written before candidate design:

| K | first-sweep reached contacts | worst force error, 40 sweeps | worst force error, 160 sweeps |
|---|---|---|---|
| 4 | 3 | 0.002142759175 | 1.198563471e-11 |
| 8 | 3 | 0.2560596095 | 0.002432689560 |
| 16 | 3 | 0.8358915691 | 0.2656505763 |

Both complete executions produced byte-identical stdout and report JSON, including
all trajectory hashes. The exact force and dense-reference errors were both 0.0
on all nine rows; 4/4 instrument gates passed. The diagnostic K16 target fails:
0.8358915691 > 0.10 after 40 sweeps. This idealised system omits the engine's warm
start and must not be compared numerically with its M5 penetration metric.
Evidence: `reports/innovation_stack_linear_probe.json`.

Next experiment, registered before implementation/execution: compare the measured
three-colour schedule with an alternating bottom-up/top-down schedule at the SAME
number of contact updates (one direction per sweep), and a two-level variant using
adjacent-pair constant prolongation, a dense coarse residual solve, projection onto
nonnegative impulses and one unchanged three-colour sweep. Report the coarse-solve
count explicitly; no equal-work or speed claim for the two-level variant.
Pre-registered candidate gates: K16 at 40 sweeps worst-body force error <0.10;
nonnegative finite impulses; two full runs byte-identical. Test K4/8/16 and 1/40/160
sweeps, preserve the instrument reference gates, and record rejected schedules.

| module | status | evidence |
|---|---|---|

Measured candidate table (CPU, fixed normal-only K16 system):

| schedule | force residual, 40 sweeps | force residual, 160 sweeps | contact updates at 40 | extra coarse solves at 40 | diagnostic <0.10 at 40 |
|---|---|---|---|---|---|
| three cyclic colours, frozen baseline | 0.835891569071 | 0.265650576340 | 640 | 0 | FAIL |
| alternating bottom-up/top-down | 0.861427127555 | 0.299449766128 | 640 | 0 | FAIL |
| adjacent-pair coarse correction + baseline sweep | 1.77635683940e-15 | 1.77635683940e-15 | 640 | 40 | PASS |

The alternating schedule is an honest negative: at equal contact-update count its
40-sweep residual is 0.0255355584834 higher than the baseline. Reaching the whole
chain in one sweep does not by itself remove the slow force mode. The two-level
experiment passes this restricted diagnostic but adds 40 dense 8x8 solves; no speed
claim follows. All impulses remain nonnegative and finite, the independent dense
and analytic instrument references still pass, and both full stdout/report pairs
are byte-identical (including all K4/K8/K16 trajectories). The cell deliberately
returns failure when any schedule misses the registered diagnostic.
Evidence: `reports/innovation_stack_two_level_probe.json`.

Remaining for target 1: restore CUDA, obtain the frozen full-engine mechanism table,
then implement and measure a separate contact-engine candidate with the unchanged
M5, geometry, trajectory identity and two <=4 ms timing gates. The synthetic
normal-only result does not establish behaviour with friction, rotations, changing
contact sets, warm start, mass contrast or a projected coarse correction in those cases.

## Innovation target 2: contact order, CPU reference

| module | status | evidence |
|---|---|---|

Registered before execution: stack4, stack16 and two-layer lattice50, frozen box
voxel geometry, compressed spacing 0.195 and initial center height 0.099 to exercise
contacts without time integration. Exhaustive pairs use the source radius predicates.
Require unique strictly increasing feature keys, reversed emission to change raw order,
and sorting the reversed emission to reproduce the full reference pair array exactly.
Run twice with byte-identical stdout and report. This is a controlled emission-order
perturbation, not an observation of GPU nondeterminism. No timing claim is made.

Target 2 mechanism table, measured before candidate design:

| fixture | points | exhaustive point pairs | accepted contacts | body pairs |
|---|---|---|---|---|
| stack4 | 72 | 2556 | 36 | 4 |
| stack16 | 288 | 41328 | 144 | 16 |
| lattice50 | 900 | 404550 | 450 | 50 |

All three fixtures change raw order under reversed emission, while sorting restores
the exact reference pair array. All feature keys are unique and strictly increasing
in the reference. Two complete runs produced byte-identical stdout and JSON.
Evidence: `reports/innovation_contact_reference.json`. No GPU order or speed claim.

Post-reboot retry: the separately reported context checks had passed, but this
measurement again failed before step 1 with 999/201. The new boot's kernel log
contains a new Xid 62, browser channel errors, and `GPU Reset Required`.
Zero simulation or timing samples were collected; the CUDA-only status stands.

Target 2 candidate gates, registered after the table and before implementation:
a cell hash grid of width 2R, per-cell bitsets of stable point ids, ascending bit
extraction and feature/pair keys emitted together must reproduce all reference
contact endpoints and both key arrays exactly, in canonical feature order, without
a final sort. Reverse grid insertion and neighbour traversal; output must stay
byte-identical. Include negative coordinates, coincident points, and contacts on
both sides of the strict radius cutoff. Two complete reports must be byte-identical.
CPU distance comparisons must equal the reference predicate exactly. No GPU speed
or floating-point backend parity is inferred from these gates.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_generation_bitset.py` | SYNTHETIC-ONLY | 4/4 exact reference and order-perturbation gates; two full reports byte-identical; lattice50: 450 contacts, 3249 distance tests. GPU integration pending. |

Target 2 measured candidate table:

| fixture | contacts | hash-grid distance tests | exact endpoints and both key arrays | reversed insertion/traversal |
|---|---|---|---|---|
| stack4 | 36 | 147 | PASS | byte-identical |
| stack16 | 144 | 833 | PASS | byte-identical |
| lattice50 | 450 | 3249 | PASS | byte-identical |
| cutoff/coincident/negative-coordinate fixture | 6 | 10 | PASS | byte-identical |

Both full executions produced byte-identical stdout and report JSON. Canonical
feature order is emitted directly without a final sort; feature and body-pair keys
are generated at contact emission. Evidence: `reports/contact_generation_bitset.json`.
This is an executable CPU prototype, not a GPU integration. Python's arbitrary-size
bitsets index global point ids, so their memory grows with the largest id in each
cell; a bounded local bitset representation and a device prefix scan are still
needed before this can become a scalable GPU generator. Full contact geometry,
GPU arithmetic parity, overflow guards at device capacity and both idle timing
legs remain open. No performance target has been declared passed.

Recovery smoke gate registered before execution: `innovation_stack_probe.py smoke`
runs K4, 40 velocity/10 position iterations, 20 steps, twice; all four state arrays
must be finite and both full trajectory hashes identical. This short context/solver
check does not certify the K16 or timing targets. Light field queries occur only
before initialization in this mode; no GPU polling occurs inside the run.

### Target 1 recovered GPU baseline: measured before solver design

The patched light-query smoke passed: two K4/20-step full-state trajectories were
byte-identical and finite. The following full baseline ran under the shared lock
with no competing compute process, two repetitions. Historical context failures
above are preserved; they no longer block this measured baseline.

| metric | run 1 | run 2 | registered gate |
|---|---|---|---|
| K16, 40 velocity / 10 position, max pen/R | 1.412588655949 | 1.412588655949 | <0.5: FAIL; secular and INVALID spacing |
| K16, 160 velocity / 10 position, max pen/R | 0.276366770267 | 0.276366770267 | <0.5, bounded, valid: PASS |
| K16/40 minimum spacing | 0.134370565414 | 0.134370565414 | >=0.164: FAIL |
| K16/40 full-state trajectory SHA256 | 6e7d0a10cb1644fd7105440d85c7cf336473d2159bc36a7a6e9e8565703dcaaa | same | byte identity: PASS |
| N10000 wall time per step | 4.922 ms | 4.878 ms | both <=4 ms: FAIL |
| generation, profiled | 1.498 ms | 1.431 ms | diagnostic |
| selection, profiled | 0.890 ms | 1.104 ms | diagnostic |
| solve, profiled | 1.101 ms | 1.104 ms | diagnostic |
| total with stage synchronization | 5.047 ms | 5.782 ms | diagnostic; not the wall-time gate |

The full-cell exit is 1 because the baseline misses the innovation target, not
because execution failed. Evidence: `reports/innovation_stack_baseline.json` and
`reports/innovation_stack_smoke.json`. The per-leg utilization fields include this
process's immediately preceding simulation (67-68%) and must NOT be interpreted
as idle-background measurements. GPU ownership was exclusive via the agreed lock
and the foreign-process guard; these numbers establish no candidate speedup.
The generation stage remains the largest individual stage in both profiled runs.

### Target 1 coarse normal correction: design registered after the GPU table

The normal-only experiment removes the slow mode with a coarse pair correction;
the full baseline still needs 160 iterations. New separate candidate: average the
normal contact Jacobians inside each body pair, assemble their effective-mass
matrix, solve one bounded coarse impulse correction before the unchanged pair
sweeps, and apply the same correction to accumulated normal impulses and velocities.
The coarse solve runs on the host for at most 64 bodies; larger scenes explicitly
use the frozen baseline. This is a convergence prototype, not a scalable speed claim.
The >=10000-body target is still measured and still must satisfy <=4 ms to pass.

Before execution, unchanged primary gates: K16/600 steps, 40 velocity/10 position
iterations, pen/R<0.5, bounded drift and valid spacing; two full state trajectories
byte-identical; both N10000 timing legs <=4 ms. Additional contract gates: K4 tail
mean force residual <0.10 using the existing harness and a resting box force error
<1%. Coarse impulses are bounded below so no accumulated normal impulse becomes
negative. Singular coarse matrices skip correction explicitly; counts are reported.
No baseline engine file or tolerance is modified.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_coarse.py` | OWN-GATE-FAIL | Two trajectories bit-identical; K16/40 pen/R 1.185726225376 INVALID/secular; M1 1.60744430822e-5 PASS; target FAIL. |

Measured coarse-candidate table, two full executions:

| metric | run 1 | run 2 | verdict |
|---|---|---|---|
| K16/40 max pen/R | 1.185726225376 | 1.185726225376 | FAIL <0.5; secular drift |
| K16 minimum spacing | 0.148075103760 | 0.148075103760 | INVALID <0.164 |
| coarse corrections / singular skips | 597 / 0 | 597 / 0 | diagnostic |
| K4 tail force residual | 1.60744430822e-5 | 1.60744430822e-5 | PASS <0.10 |
| resting box force relative error | 1.5734168005e-6 | 1.5734168005e-6 | PASS <0.01 |
| N10000 wall time | 5.079 ms | 5.007 ms | FAIL <=4 ms |
| full K16 trajectory SHA256 | 274791dfc5cebcc5ca21f79803cbd5fd95cdafa2be4fc9de933341af4aa8d9ea | same | byte identity PASS |

The coarse correction improves K16 penetration from the frozen baseline's
1.412588655949 to 1.185726225376, but the target still fails on geometry and drift.
The normal-only synthetic result did not transfer into a successful full contact
solver. The large lattice uses the declared baseline fallback (37 contacted steps
skipped correction), so its timing is NOT a measurement of a scalable coarse solve.
The cell exits 1 and the candidate remains experimental, without changing any default.
Evidence: `reports/innovation_stack_coarse.json`. Next mechanism question: how much
of the remaining K16 error comes from angular/friction coupling versus the position
pass; measure that split before designing another correction. GPU generation and
kernel/physics targets remain open; no tolerances changed.

Timing qualification added after coordination: another session reported that a
CPU-only single-thread numerical job overlapped part of the coarse-candidate
window before it read the reservation. Exact overlap with the two timing legs was
not instrumented. The observed 5.079/5.007 ms values are retained, but are provisional
and NOT accepted as idle-certified performance evidence. The K16 geometry/drift
failure, force gates and trajectory byte identity remain valid; the target still
fails independently of timing. Obtain two uncontended timing legs before using
these values for a performance comparison. No tolerance is changed.

### Target 1 mechanism split: registered ablation, before another design

Measure the frozen baseline and frozen coarse candidate at K16, 40 velocity
iterations, 600 steps; cross position iterations {10,40} with friction {0.5,0.0}.
The 0.5/10 row is the original primary case. Other rows are labelled diagnostic
ablations and cannot promote the primary target. Reuse the existing M5/geometry
gates unchanged. Record peak angular speed, peak orientation angle from the initial
orientation, and two full observed-state trajectory hashes for each row. Finite
states and byte-identical repeats are instrument gates. No wall-time claim and no
candidate source edit in this experiment. A negative row remains in the table.

| module | status | evidence |
|---|---|---|

Measured ablation table (both repeats agree on every reported field):

| solver | friction | position passes | max pen/R | minimum spacing | peak angular speed | peak angle | geometry |
|---|---|---|---|---|---|---|---|
| baseline | 0.5 | 10 | 1.41258865595 | 0.134370565414 | 4.40995550156 | 0.342210918665 | INVALID |
| baseline | 0.5 | 40 | 1.29964500666 | 0.140017747879 | 6.57697868347 | 0.738199055195 | INVALID |
| baseline | 0.0 | 10 | 4.09999936819 | 1.6838312149e-06 | 23.4719333649 | 3.14159274101 | INVALID |
| baseline | 0.0 | 40 | 4.09999787807 | 1.16974115372e-06 | 22.6310195923 | 3.14159274101 | INVALID |
| coarse | 0.5 | 10 | 1.18572622538 | 0.14807510376 | 4.87547588348 | 0.274684488773 | INVALID |
| coarse | 0.5 | 40 | 1.04723602533 | 0.157837033272 | 4.81588220596 | 0.357561588287 | INVALID |
| coarse | 0.0 | 10 | 4.09999966621 | 3.42726707458e-07 | 22.7851600647 | 3.14159274101 | INVALID |
| coarse | 0.0 | 40 | 4.099996984 | 1.49011611938e-06 | 16.2857456207 | 3.14159274101 | INVALID |

Evidence: `reports/innovation_stack_ablation.json`, including both hashes and the
unchanged M5 thresholds/reasons. Instrument gate PASS (exit 0); physical target
FAIL in all eight cases. The original friction 0.5 / 10-position-pass case alone
is eligible for the original target; diagnostic settings do not replace it.
Increasing position passes to 40 leaves even the coarse case at pen/R
1.047236025333 and minimum spacing 0.157837033272, below the fixed 0.164 floor.
Zero-friction cases approach pen/R 4.1 and near-zero spacing despite a bounded
late drift classification: collapse is not convergence. Peak rotation reaches
3.141592741013. These interventions show sensitivity to friction and position
passes; they do not establish which internal contact mode causes the residual.
No performance samples were taken. Next measure the frozen contact Jacobian's
normal/angular response and coarse-space coverage before another solver design;
retain both engines unchanged. Target 2 GPU generation remains open.

### Target 1 normal-space measurement, registered before execution

Observe the frozen baseline at contact-colouring calls 1,20,100,300,590 in the
unchanged K16/40, friction0.5, position10, 600-step fixture. No state is modified.
Compare the full normal Jacobian with one averaged row per body pair in mass-
weighted coordinates. Fix relative SVD rank threshold at 1e-10. Report both ranks,
relative Frobenius projection loss, normal velocity component and the component
missed by the pair mean. This measures linear subspace coverage, not a feasible
unilateral/friction solve. No convergence or performance gate is inferred from it.
Instrument gates: all five samples, finite diagnostics, two bit-identical snapshot
hashes/results, and exact unchanged M5 dictionary against the recorded baseline.
Existing physical gates remain unchanged; baseline failure must remain visible.

| module | status | evidence |
|---|---|---|

Measured normal-space table; both repeats identical:

| contact call | contacts / pairs | full rank / mean rank | Jacobian projection loss | normal velocity norm | missed velocity norm |
|---|---|---|---|---|---|
| 1 | 4 / 1 | 3 / 1 | 0.584037429256 | 0.580367455707 | 0.369528396033 |
| 20 | 32 / 8 | 31 / 8 | 0.798798523265 | 3.36457224627 | 1.32468938981 |
| 100 | 64 / 16 | 63 / 16 | 0.814075679781 | 0.468858607059 | 0.156902968077 |
| 300 | 61 / 16 | 60 / 16 | 0.630835716108 | 0.585936325298 | 0.458104758908 |
| 590 | 64 / 16 | 63 / 16 | 0.724554939699 | 0.662102204555 | 0.537150898249 |

All four preregistered instrument gates PASS. Evidence:
`reports/innovation_stack_normal_space.json`. Baseline M5 remains exactly
pen/R=1.412588655948639, INVALID/secular. At call 590 the pair mean misses
0.537150898249 of a 0.662102204555 normal-space velocity norm; a rank-16 mean
cannot span the measured rank-63 contact space. This supports testing a richer
normal space, but does not prove convergence: inequality bounds, friction and
position correction are absent from the projection diagnostic. No timing claim.
The probe only observes normal rows and cannot attribute all losses specifically
to rotation rather than differing contact normals.

### Target 1 full-normal candidate: design and gates before execution

The measured rank63/rank16 loss motivates retaining all normal contact rows in a
separate bounded host prototype. Solve nonnegative least squares for absolute
normal impulses to minimise final mass-weighted velocity; then apply the actual
float32 impulse difference before the frozen 40 velocity / 10 position sweeps.
Tangential impulses remain fixed during the added solve; the unchanged sweeps
handle friction afterward. This is not a coupled friction-cone solution.
Limit to 64 dynamic bodies; larger scenes explicitly use the frozen fallback.
NNLS has a fixed 3*contact-count iteration cap, no regularisation; nonconvergence
raises. Additional host active-set iterations are disclosed, not counted as free
work or a scalable 40-iteration implementation. Fix raw negative-gradient and
complementarity residual <=1e-8 before float32 delivery. Positive inverse masses
and inertias are required. The numerical gate does not certify the rounded state.

Run two complete K16/600-step trajectories, K4/400-step load and box/200-step
force probes. Require exact repeat dictionaries/hashes, unchanged K16 pen/R<0.5,
bounded drift and valid geometry, unchanged M1<0.10 and box-force error<0.01.
No timing samples in this mechanism experiment; the original <=4ms/10000-body
target stays open regardless of these gates. No baseline/candidate source edited.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_normal_nnls.py` | OWN-GATE-FAIL | 3/4 own gates; two exact repeats, K16 pen/R1.748346388340 >=0.5 and spacing0.117582678795 <0.164 INVALID; KKT1.06686783327e-9 passes. |

Measured full-normal candidate table, both complete repetitions:

| metric | run 1 | run 2 | fixed gate / result |
|---|---|---|---|
| K16 max pen/R | 1.748346388340 | 1.748346388340 | <0.5 FAIL; secular |
| K16 minimum spacing | 0.117582678795 | 0.117582678795 | >=0.164 FAIL / INVALID |
| added normal solves | 597 | 597 | diagnostic |
| maximum raw KKT residual | 1.06686783327e-9 | 1.06686783327e-9 | <=1e-8 PASS |
| K4 tail load residual | 0.00769738094705 | 0.00769738094705 | <0.10 PASS |
| box force error | 5.70892194181e-7 | 5.70892194181e-7 | <0.01 PASS |
| full trajectory hash | 006d29cf240e9ecd18fcc089ff56eb151e91aab79c6666f88cc590883c788067 | same | byte identity PASS |

Evidence: `reports/innovation_stack_normal_nnls.json`. Exit 1 is the physical
failure, not a runtime error. The full-normal correction worsens K16 penetration
against the frozen baseline (1.412588655949) and pair-mean candidate
(1.185726225376). Recovering normal row-space coverage is insufficient with this
update order and frozen tangential impulses. Neither normal-only prototype is
promoted. No 10000-body timing samples were collected for this bounded prototype;
the original performance target remains open. A next solver experiment requires
measurement of normal/tangential update coupling before a new design.

### Target 2 frozen GPU contact measurement, registered before execution

Run the frozen feature-id generator on float32 stack4, stack16 and lattice50
fixtures, rebuilding the grid twice. Measure raw and canonical full-array hashes,
contact counts, raw feature ordering and endpoint differences against the existing
float64 CPU predicate applied to the same float32 inputs. Canonical comparison
includes body ids, both contact points, normals, penetration and feature ids.
Fixed instrument gates: two exact canonical repeats, unique feature keys and
finite geometry. Allocate the exhaustive pair-count-plus-ground upper bound;
reject overflow. Raw ordering and CPU/GPU predicate parity are measured outcomes,
not assumed gates. No timing samples or solver changes. The raw arrays are the
reference for a future GPU generator; no CPU arithmetic equivalence is implied.

| module | status | evidence |
|---|---|---|

Modal baseline registration: run the unchanged `innovation_stack_probe.py baseline`
on L4 with its existing two-leg physical, identity and <=4ms gates. Store the
new report separately from local evidence. L4 timings never certify local timing.

Modal L4 baseline attempt: exit1 before the first timing leg, 0 timing samples.
The idle guard reported another compute process after the probe created its own
context. PID namespace mismatch is a hypothesis, not established evidence.
No tolerance or guard changed. Previously uploaded reports copied back by the
runner are stale and are not new L4 results. A separate process-identity probe
records process ids before import and after synchronized context creation; no
queries during creation, no timing claim. Require two identical zero-array reads.

| module | status | evidence |
|---|---|---|

Measured target-2 GPU table, Modal L4, two runs:

| fixture | points | contacts each run | raw arrays identical | canonical arrays identical | CPU-only / GPU-only endpoints |
|---|---|---|---|---|---|
| stack4 | 72 | 36 | yes | yes | 0 / 0 |
| stack16 | 288 | 144 | no | yes | 0 / 0 |
| lattice50 | 900 | 450 | no | yes | 0 / 0 |

All three instrument gates PASS; raw feature order is noncanonical in all three
fixtures. Raw nondeterminism is observed in 2/3 fixtures, while sorting preserves
exact full geometry, not just endpoint sets. Evidence:
`reports/innovation_contact_gpu_reference_l4.json`. This is L4 correctness evidence,
not a speed claim or local-device result. It establishes the need for canonical
emission before replacing the generator. The CPU predicate agrees only on these
fixtures; near-threshold arithmetic parity remains open. Baseline sources frozen.

Cloud identity diagnostic observation, Modal L4 (one invocation, not a completed
two-run certification): before context creation, no compute-process rows; after
synchronization, the compute query returns PID1 while the Python process is PID35.
The visible process status supplies no NSpid mapping. Both zero-array readbacks
are identical. Evidence: `reports/innovation_cuda_identity_l4.json`. This reproduces
why the host-PID equality assumption rejects this cloud context, but does not
establish a general method of identifying its owner. Do not whitelist PID1 or
remove the idle guard on this evidence. Next: repeat the diagnostic and establish
a cloud-appropriate ownership check before fresh timing legs. Timing samples=0.

### Cloud baseline isolation, gates registered before execution

The observed PID35/PID1 mismatch motivates process isolation, not a whitelist.
A CPU-only supervisor invokes the unchanged idle guard before and after each of
 two sequential workers. Each worker creates and finishes its own context, reuses
frozen K16 metrics/trajectory and timing functions, and exits before the guard
runs again. No polling during context creation or timed sections. Any guard or
worker failure aborts. Same fixed physical gates: K16/40 pen/R<0.5, bounded drift,
valid geometry; two full trajectories and M5 dictionaries bit-identical; each
N10000 timing <=4ms. L4-only results, never local-device evidence. No baseline
source changed. Warm10/timed30 and stage profile remain unchanged.

| module | status | evidence |
|---|---|---|

### Target 2 GPU bitset design and gates before execution

Measured raw-order variation in 2/3 fixtures motivates count/scan/canonical emit.
New module beside the frozen generator: count valid neighbours per point, exclusive
scan for disjoint output spans, then emit ground first followed by ascending
32-id words and ascending occupied bits. Fused feature/body-pair keys are emitted
with the full contact geometry. Hash-grid traversal order only changes temporary
slots. Bound valid non-ground neighbours to32, reject overflow BEFORE emission;
no truncation/fallback masquerading as success. Two predicate passes and local
word selection are additional costs; no speed claim before timings. Prototype
allocates/transfers per invocation and is not integrated in the engine.

Fixed gates on stack4/stack16/lattice50 and an additional cutoff/coincidence case:
exact bytes for all eight geometry/id arrays and two fused-key arrays against
the frozen CUDA generator in canonical order; two byte-identical candidate runs;
strictly ascending features and finite arrays. Dense34-point scene must report
exactly33>32 overflow twice without emission. No tolerances or baseline edits.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_generation_gpu_bitset.py` | VERIFIED-FRESH | L4 four fixtures: all ten output arrays exactly match frozen reference; two canonical repeats identical; 33>32 overflow rejected. |

Cloud isolated-baseline table, L4, both complete runs:

| metric | run1 | run2 | gate/result |
|---|---|---|---|
| K16/40 pen/R | 1.341516077518 | 1.341516077518 | <0.5 FAIL, secular/INVALID |
| K16/160 pen/R | 0.273366272449 | 0.273366272449 | PASS, bounded/valid |
| N10000 wall time | 6.597ms | 6.362ms | <=4ms FAIL |
| generate stage | 1.570ms | 1.563ms | diagnostic |
| select stage | 1.134ms | 1.128ms | diagnostic |
| solve stage | 1.345ms | 1.343ms | diagnostic |
| full trajectory hash | 982505ffc79f86ab5a767f64660bffefd710b343e7a50254ef0789373f19abc3 | same | byte identity PASS |

Evidence: `reports/innovation_cloud_stack_l4.json`. All four frozen idle checks
passed outside worker contexts; no foreign-process exception or PID whitelist.
Two timing legs each retain10 warmup and30 timed steps. Exit1 records physical/
performance failure. These L4 values differ from the previously recorded local
values, including the trajectory; repeatability is established within L4 only,
not cross-architecture identity. No local performance claim or guard change.

First candidate probe on L4: 4/4 exact-reference checks failed; repeatability,
canonical order, finiteness and33>32 overflow rejection passed. Investigation
before rerun found a host-oracle defect: passing int32 endpoints/owners to the
existing CPU key helper produced int32 keys, while the declared fused CUDA keys
are int64. A CPU dtype check reproduced both erroneous int32 outputs and corrected
int64 outputs. Correct only the new probe's key inputs to int64, retaining the
byte/dtype gate and adding ten per-field identity diagnostics. Candidate and
frozen generator remain unchanged. Preserve first report as negative evidence.

Measured corrected candidate table, L4, two repeats each:

| fixture | contacts | maximum neighbours | ten reference arrays exact | raw candidate repeat exact |
|---|---|---|---|---|
| stack4 | 36 | 1 | yes | yes |
| stack16 | 144 | 1 | yes | yes |
| lattice50 | 450 | 1 | yes | yes |
| cutoff/coincidence | 7 | 2 | yes | yes |

Overflow33>32 rejected twice before emission; all five gates PASS. Candidate
hashes are unchanged from the first failed-oracle run, confirming no generator
change between the runs. Evidence: `reports/innovation_contact_gpu_bitset_l4.json`;
negative instrument evidence: `reports/innovation_contact_gpu_bitset_oracle_failure_l4.json`.
Correctness is established only for the measured bounded cases. Next measure the
preallocated generation stages at scale before integration; no default changes,
no engine-level determinism/performance claim and no local-device claim.

### Target 2 standalone stage cost: fixed gates before execution

Freeze both generators. At1000 and10000 bodies, obtain world-point snapshots
from step40 of the unchanged two-layer lattice baseline (40 velocity/20 position
passes). Preallocate both pipelines. Raw baseline includes grid build, counter
reset, frozen emission and count readback. Candidate includes grid build, maximum
reset, count/scan/summary, capacity readback and canonical emission with fused keys.
Synchronize each invocation;10 warmup +30 timed invocations, two isolated workers,
frozen idle checks before/after each worker. No in-context device queries.

Gates: exact ten-array reference bytes, strict canonical keys, two identical
fixture/output hashes and contact/max-neighbour counts; candidate wall time no
higher than raw baseline in each leg/size. This intentionally compares against
raw generation without giving credit for downstream sort/key elimination. A
failure is a stage-cost negative, not proof of full-engine regression. These are
preallocated stage timings, excluding allocations and initial simulation; they
cannot certify the original full-engine <=4ms target. No tolerances changed.

| module | status | evidence |
|---|---|---|

Measured preallocated stage table, L4:

| bodies | contacts | raw run1/run2 ms | canonical run1/run2 ms | exact geometry/repeats | cost gate |
|---|---|---|---|---|---|
| 1000 | 8772 | 0.227110 / 0.224634 | 0.387299 / 0.383635 | PASS | FAIL |
| 10000 | 85427 | 1.467272 / 1.477224 | 3.398754 / 3.385904 | PASS | FAIL |

Evidence: `reports/innovation_contact_gpu_stage_l4.json`. Two fixture and complete
candidate output hashes are identical, all ten arrays match the frozen reference,
maximum valid neighbours=1 in both sizes and runs. Canonical output costs about
2.3x raw generation at10000 bodies; the no-slower gate fails in all4 observations.
The prototype performs two neighbour traversals plus scan/summary/emission; that
cost is retained, with no full-engine speed claim. Baseline excludes downstream
sort/keying, so this measurement alone cannot quantify savings from integration.
Next motion mechanism: stage decomposition and storing first-pass neighbour ids
instead of traversing twice; measure traffic/cost before another generator design.

### Contact-stage mechanism split, gates before execution

The measured canonical stage costs3.398754/3.385904 ms at10000 bodies versus
raw1.467272/1.477224 ms. Before designing another generator, measure unchanged
canonical kernels separately: grid build, maximum reset, neighbour count,
exclusive scan, summary, host readback and canonical emission. Use the same
frozen40-step snapshots,1000/10000 bodies,10 warmups/30 samples, two isolated
workers and frozen idle guards before/after each context. Record event and wall
intervals for each stage; individually warmed/synchronized phases are not an
additive decomposition of an asynchronous production pipeline.

Fixed gates: every output byte matches frozen canonical reference, strictly
ordered feature keys, both full fixture/output hashes identical. Retain the
existing no-slower-than-raw comparison as a separate measured negative. Do not
change either generator or increase neighbour capacity32. Only this new observer
adds instrumentation; no candidate design before its measured table is written.

| module | status | evidence |
|---|---|---|

Measured canonical phase table, L4,10000 bodies/180000 points:

| phase | wall ms run1 | wall ms run2 |
|---|---|---|
| grid_build | 0.068903433 | 0.070351133 |
| maximum_reset | 0.004176933 | 0.004193700 |
| neighbour_count | 1.367176500 | 1.370592200 |
| exclusive_scan | 0.014534700 | 0.014800367 |
| summary | 0.023535800 | 0.022867467 |
| host_readback | 0.042715533 | 0.043523767 |
| canonical_emit | 1.668157933 | 1.674170233 |

All exact-output/ordering/repeat/instrumentation gates PASS. Raw-speed gate FAIL:
full canonical3.408874500/3.309195767 ms versus raw1.463357200/1.475127367 ms.
Evidence: `reports/innovation_contact_gpu_phase_l4.json`. Both traversal-bearing
kernels dominate; scan itself is only0.0145/0.0148 ms. The phase intervals include
submission gaps and separate warming, so they are not summed into an exact
production-pipeline cost. No baseline or capacity changed.

### Cached-neighbour generator, design and gates before execution

Store valid neighbour IDs during the first traversal in a separate32-by-point
int32 buffer, with contiguous writes across points for each slot. Keep the same
32-neighbour bound; record actual counts and reject overflow before emission.
The emission kernel loads those IDs and uses the existing ascending-word/bit
ordering and identical contact arithmetic; it performs no second grid query.
At180000 points, cache storage is23040000 bytes, explicit extra workspace.
No changes to either prior generator or their measurements.

Fixed correctness gates: all ten output arrays exactly match the frozen raw
canonicalized reference, exact repeats, strict feature ordering, finite values,
33-neighbour overflow rejected twice; additionally test exactly32 neighbours.
Then two isolated preallocated timing legs at1000/10000 bodies, same10/30 scheme
and frozen idle guard. Retain the original no-slower-than-raw speed gate; also
report reduction versus the prior canonical stage without relabelling old failures.
No end-to-end engine performance/integration claim.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_generation_gpu_cached.py` | VERIFIED-FRESH | L4 five fixtures all ten arrays exact twice;32-neighbour boundary528 contacts PASS,33-neighbour overflow rejected twice. |

Cached-generator correctness evidence: `reports/innovation_contact_gpu_cached_l4.json`.
All ten arrays/dtypes exact against frozen reference on all five scenes; exactly32
neighbours emit528 contacts and both33-neighbour cases fail before emission. All
five gates PASS; timing and engine integration are separate pending measurements.

### Tall-stack normal/tangent coupling observer, gates before execution

The full-normal NNLS candidate satisfied its KKT gate but worsened penetration;
normal-only optimality did not establish coupled contact convergence. Observe
unchanged K16/40 snapshots at calls1/20/100/300/590 before the frozen pair sweeps.
Construct mass-weighted normal and two tangent Jacobians (the tangent plane uses
the existing deterministic basis). Measure normalized cross-coupling, subspace
ranks and the tangential velocity change predicted by the same bounded normal
NNLS calculation. Apply no correction to the engine. Tangent-plane norms do not
depend on a rotation of the two orthonormal tangent axes.

Fixed instrument gates: two full M5/snapshot/measurement dictionaries identical;
M5 exactly matches the saved L4 baseline40 result; all five snapshots present;
finite metrics and orthonormal tangent basis error<=1e-6; NNLS KKT<=1e-8 unchanged.
The original K16 physical gate remains separately reported as a negative. No
new solver design or timing claim before the coupling table is written.

| module | status | evidence |
|---|---|---|

Measured cached stage table, L4, both full fixture/output hashes identical:

| bodies | cached ms1/2 | raw ms1/2 | cache bytes | original raw-speed gate |
|---|---|---|---|---|
| 1000 | 0.322612200/0.306080100 | 0.214089467/0.197749967 | 2304000 | FAIL |
| 10000 | 1.759377667/1.738346067 | 1.449987800/1.411383367 | 23040000 | FAIL |

All three exact-reference/ordering/repeat gates PASS; fixed no-slower-than-raw
FAIL at both sizes. Compared with the previously measured two-query canonical
stage3.408874500/3.309195767 ms, the cached10000-body stage is approximately half
the time, across separate cloud invocations. Same-run raw comparison still shows
21-23% overhead; no raw-speed or full-engine success is claimed. Evidence:
`reports/innovation_contact_gpu_cached_stage_l4.json`. Next target2 measurement:
compare complete canonical-generation/key-order pipelines, including the frozen
baseline's downstream work, before considering any engine integration.

Measured frozen normal/tangent table, L4, two full dictionaries identical:

| call | normal velocity before | after normal-only solve | induced tangent velocity norm | normalized cross max |
|---|---|---|---|---|
| 1 | 0.327000021935 | 0.0654800838163 | 0.0654800838163 | 0.334366924585 |
| 20 | 1.78411116896 | 3.61304171754e-08 | 0.0820049301163 | 0.21508663912 |
| 100 | 0.136596712346 | 0.00610514865263 | 0.335442747509 | 0.408120816699 |
| 300 | 0.348820175021 | 0.00930753879593 | 0.399872975825 | 0.897866113047 |
| 590 | 0.231878671 | 0.0347089401102 | 0.280669421648 | 0.918719031109 |

All five instrument gates PASS; original L4 M5 stays exactly pen/R1.341516077518,
INVALID/secular. At call100, reducing the normal velocity norm0.136597->0.006105
induces tangent norm0.335443 versus pre-existing0.021153. This is a strong measured
coupling, not proof of the cause of the complete failed NNLS trajectory. No
correction was applied. Evidence: `reports/innovation_stack_tangent_coupling_l4.json`.
Next solver design must include normal/tangential coupling and unilateral/friction
bounds; normal-only KKT success is insufficient. No full-engine timing claim.

### Complete canonical prefix comparison, gates before execution

The cached generator remains slower than raw generation, but produces canonical
geometry and keys that raw generation lacks. Measure the unchanged raw generator
plus the frozen device manifold's first fkey/radix-sort/gather prefix, stopping
before pair grouping/selection. Compare against the unchanged cached stage. Both
use the same snapshots,10/30 sampling, two isolated workers and idle guards.
Verify the six gathered baseline arrays and sorted feature keys against the full
CPU canonical reference; cached ten-array exactness and repeats remain required.

Fixed additional speed gate: cached time<=raw-plus-canonical-prefix time at both
sizes and legs. This is a distinct comparison; the original raw-only speed FAIL
remains recorded. No integration, default replacement, or skipped downstream
selection/solve is implied. Pair-key sorting after this prefix is not measured.

| module | status | evidence |
|---|---|---|

Measured canonical-prefix comparison, L4:

| bodies | frozen prefix ms1/2 | cached ms1/2 | speed gate |
|---|---|---|---|
| 1000 | 0.313094367/0.315747800 | 0.269045567/0.259141200 | PASS |
| 10000 | 1.566056200/1.604639467 | 1.762389500/1.801562400 | FAIL |

All four correctness/repeat/prefix/ordering gates PASS. Large cached case remains
12.27-12.54% slower even after the frozen feature-sort/gather work is included.
No integration or engine-speed success. Evidence: `reports/innovation_contact_gpu_prefix_l4.json`.

### Hash-grid candidate inflation, gates before execution

Both frozen engines use64x64x64 hash buckets. The10000-body lattice spans about
248 contact-grid cells along each horizontal axis, which may alias unrelated
cells into the same buckets. Observe actual hash-query visit counts versus valid
neighbour counts using new grids beside the frozen engine, dimensions64x64x64,
128x128x128 and256x256x8. Same1000/10000-body40-step point snapshots. No source,
point positions, query radius or acceptance predicate changes.

Fixed gates: two whole snapshot/count tables identical; every grid has identical
per-point valid counts; summed valid counts plus ground contacts reproduce the
previous exact canonical contact count; all visits>=valid counts. Record candidate
inflation, extrema and nominal bucket count. No timing claim or new generator
design until this table is written; anisotropic grid correctness is measured,
not assumed from world extents.

| module | status | evidence |
|---|---|---|

Measured hash-query visits, L4, all per-point counts/tables identical twice:

| bodies | grid | visits | max visits/point | buckets | contacts |
|---|---|---|---|---|---|
| 1000 | 64x64x64 | 577355 | 111 | 262144 | 8772 |
| 1000 | 128x128x128 | 309720 | 36 | 2097152 | 8772 |
| 1000 | 256x256x8 | 309720 | 36 | 524288 | 8772 |
| 10000 | 64x64x64 | 46262638 | 653 | 262144 | 85427 |
| 10000 | 128x128x128 | 11559570 | 156 | 2097152 | 85427 |
| 10000 | 256x256x8 | 3180004 | 36 | 524288 | 85427 |

At10000 bodies the wide/shallow grid reduces actual visits by14.55x with twice
as many buckets as64^3, and preserves every valid per-point count. Evidence:
`reports/innovation_contact_grid_visits_l4.json`. No speed claim from visit counts.

### Wide-grid engine variant, gates before execution

A separate subclass changes only the grid allocation after frozen construction
to256x256x8. Reuse the original raw generation, canonical sorting, manifold
selection, colour/chunk sweeps and graph capture. No cached generator is integrated.
This isolates the measured bucket aliasing; the allocation is specialized to the
measured wide lattice, not an adaptive or universal grid choice. Tall scenes may
alias vertically, which must affect work only, not accepted contacts.

Fixed gates: two entire small-trajectory hashes identical and equal to baseline;
K16 M5 at40/160 exactly equal to baseline;10000-body final full state hashes from
both timing and stage-profile runs exactly equal to baseline and repeated runs;
wide-grid10000 wall time<=baseline in each leg. Retain independent original gates
K16/40 physical pass and wall<=4 ms, without relaxation. Two isolated workers,
frozen idle guards before/after each,10 warm/30 timed and existing stage profiler.
State reads/hashes occur after timed intervals. Baseline sources untouched.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_wide_grid.py` | OWN-GATE-FAIL | L4 two-run physics/state hashes exact vs baseline; full-engine wall improved; K16/40 and original4ms gates remain FAIL. |

Measured full-engine wide-grid comparison, L4:

| run | baseline wall ms | wide wall ms | baseline generation ms | wide generation ms |
|---|---|---|---|---|
| 1 | 5.477 | 4.416 | 1.456 | 0.386 |
| 2 | 5.597 | 4.492 | 1.463 | 0.381 |

Both small full-trajectory hashes and both10000-body final state hashes are exact
between baseline/variant and across runs. Large state hash:
`8868c8e1b10dbf2c8081a887f9c8e39253548cc993f65a0e6fd2b8db556146c6`.
M5 is unchanged at40/160: pen/R1.341516077518 INVALID versus0.273366272449 PASS.
The grid-only engine is faster in both legs, but the original<=4ms and K16/40
physical gates still FAIL. Evidence: `reports/innovation_wide_grid_engine_l4.json`.
Stage-profiler intervals are synchronized and are not the quoted whole-step wall
measurement. No default replacement; fixed wide-grid performance on other scene
shapes is unverified. All contact/manifold/solve kernels are inherited unchanged.

### Coupled cone shadow solve, gates before execution

The measured normal-only correction at call100 induces tangent norm0.335443
against existing0.021153. On the same frozen five snapshots, test a separate
convex friction-cone energy surrogate over normal/two tangent impulses together.
Use mu0.5, nonnegative normal impulse and tangent magnitude<=mu*normal. Initialize
from the measured feasible normal-only NNLS solution, then use deterministic
projected accelerated gradient with objective restart and a spectral step bound.
This is a shadow calculation; apply no impulse to the engine. Cone energy
minimization is not by itself a validated Coulomb time-step solver.

Fixed gates: two complete snapshot/solution/M5 results identical; original M5
unchanged; finite solutions; cone violation<=1e-12; projected-gradient infinity
residual<=1e-8 within40000 iterations; objective no larger than the feasible
normal-only initial solution. Projection uses the exact Euclidean formula for
radius<=mu*normal; no tolerance or iteration cap changes after execution. Report
normal/tangent velocities and iteration counts, including any nonconvergence.
No K16 physical improvement or performance claim from a shadow solve.

| module | status | evidence |
|---|---|---|

Measured coupled cone shadow table, L4, two full results identical:

| call | iterations | projected residual | normal-only energy | coupled energy |
|---|---|---|---|---|
| 1 | 43 | 6.91153620447e-09 | 2.55636143372 | 2.55084371004 |
| 20 | 709 | 9.32210575399e-09 | 44.5575708192 | 44.5453104184 |
| 100 | 40000 | 2.55607301369e-06 | 1.51051364227 | 0.290896669397 |
| 300 | 40000 | 6.50179288186e-08 | 9.65327741127 | 3.02595919061 |
| 590 | 40000 | 3.45226707887e-07 | 7.7766999537 | 6.4123702892 |

Four/five gates PASS; projected residual FAIL on three snapshots at the40000
iteration cap (maximum2.556073013693e-6 versus1e-8). Cone violation is at most
7.105427357601e-15<=1e-12 and energies decrease, but those are not convergence.
At call100 tangent velocity norm falls0.350727401089->0.002986931174 while normal
velocity norm rises0.006105148653->0.025854629639: a measured surrogate tradeoff,
not a validated physical improvement. No correction applied; M5 remains exactly
INVALID. Evidence: `reports/innovation_stack_cone_shadow_l4.json`. No iteration
cap or tolerance raised. Next: conditioning/active-set mechanism before using a
coupled correction inside any engine; this shadow solver is not promoted.

### Cone operator conditioning measurement (preregistered)

The preceding shadow table records residuals 2.556073e-6, 6.501793e-8,
and 3.452267e-7 at the unchanged 40000 cap. Before another solver design,
measure the frozen weighted contact operator spectrum and contact-direction
diagonal scales. No correction or new optimizer is applied. Rank uses the
explicit diagnostic cutoff max(shape)*machine-epsilon*largest singular value;
this cutoff is not a physical acceptance tolerance. Report singular extrema,
positive-spectrum condition number, nullity, and normal/tangent diagonal ranges.

Fixed measurement gates: complete results bit-identical twice; original M5
exactly unchanged; finite diagnostics; Gram symmetry error<=1e-12 and minimum
eigenvalue>=-1e-10*maximum eigenvalue. No condition-number acceptance gate or
claim that rank deficiency alone explains constrained convergence.

| module | status | evidence |
|---|---|---|

Measured conditioning table (two identical results):

| call | columns | rank | nullity | positive Gram condition | smallest positive singular value |
|---|---|---|---|---|---|
| 1 | 12 | 6 | 6 | 31.4943868704 | 0.14626495802 |
| 20 | 96 | 48 | 48 | 3109.50474724 | 0.0242324560667 |
| 100 | 192 | 96 | 96 | 59480.5751212 | 0.00553265490527 |
| 300 | 180 | 96 | 84 | 4.24446839644e+17 | 2.21119352964e-09 |
| 590 | 192 | 96 | 96 | 16104.4669174 | 0.0112832392954 |

Gram symmetry error is zero; negative eigenvalues have magnitude<=5.973e-16,
within the preregistered roundoff gate. The condition estimate uses singular
values of the operator rather than dividing rounded Gram eigenvalues. At call300
the squared smallest singular value is below the Gram rounding scale: this
diagnostic does not certify accurate recovery of that mode from the Gram matrix.
Normal/tangent diagonal entries span only0.1009..0.5056 across snapshots; simple
diagonal scaling alone cannot be assumed to remove the measured near dependency.
No claim that this spectrum alone explains constrained convergence. Evidence:
`reports/innovation_stack_cone_conditioning_l4.json`. Original physical gate FAIL.

### Split cone shadow candidate (preregistered)

The conditioning table above precedes this design. Use a separate ADMM shadow
solver for the identical convex objective/cone, with fixed rho=0.1*largest Gram
eigenvalue. Solve (Gram+rho*I) using a Cholesky factor and project the split
variable onto the unchanged cone. The diagonal term is an algorithmic split,
not added to the reported physical objective. No adaptive rho, relaxed cone,
rank truncation, or engine impulse application.

Fixed gates remain: two complete results identical; original M5 exactly
unchanged; finite/cone violation<=1e-12; the original unscaled projected-gradient
infinity residual<=1e-8 within40000 iterations; energy no worse than normal-only
initialization. Record both success and failure at every snapshot. This convex
surrogate remains insufficient to establish a Coulomb time-step improvement.

| module | status | evidence |
|---|---|---|

Measured split cone shadow table (two full results identical):

| call | iterations | projected residual | normal-only energy | split energy |
|---|---|---|---|---|
| 1 | 40 | 6.66606073119e-09 | 2.55636143372 | 2.55084371004 |
| 20 | 6242 | 9.98508653538e-09 | 44.5575708192 | 44.5453104184 |
| 100 | 40000 | 3.28624861057e-05 | 1.51051364227 | 0.296983138347 |
| 300 | 40000 | 0.000111550293287 | 9.65327741127 | 3.03308953275 |
| 590 | 40000 | 2.17995289198e-05 | 7.7766999537 | 6.41545107331 |

The same three snapshots fail the unchanged1e-8 residual gate. At call300 the
residual worsens6.501792881863e-8->1.115502932869e-4 compared with the preceding
projected solver. No penalty retuning, iteration increase, tolerance relaxation,
or solver promotion. Cone violation<=1.776357e-15 and lower objective are not
convergence. All original M5 values remain exactly invalid. Evidence:
`reports/innovation_stack_cone_split_shadow_l4.json`.

This closes the two tested convex shadow algorithms with honest negatives.
Before another physical solver design, measure the frozen sweep's actual
normal complementarity and friction update residuals at40 versus160 iterations;
include the accumulated tangent impulse representation. The present snapshots
are before each current contact solve, not its converged output, and subtract
only normal accumulated impulses. They define the documented surrogate and
must not be relabelled as a full free-velocity Coulomb reconstruction.

### Actual sweep residual observer (preregistered)

The two preceding shadow algorithms failed3/5 snapshots each; their free state
omits accumulated tangent reconstruction. Before another engine change, capture
post-solve, pre-integration contact states for the frozen40/160 iteration engines.
Measure the largest isolated normal update and tangent-coordinate update using
the frozen slip-basis rule and sequential normal-then-friction arithmetic. Each
virtual contact reads the same captured velocities and never mutates the engine.
This is a float64 diagnostic of the update formula, not a GPU bitwise emulation.

Gates: two entire results identical; original40 and160 M5 exactly unchanged;
all diagnostics finite; accumulated normal impulse>=-1e-12 and circular friction
cone violation<=1e-6. Do not require residual monotonicity across distinct evolving
trajectories. Report all five sampled calls and both iteration budgets.

Require all five declared snapshot calls at both iteration budgets in both legs.

| module | status | evidence |
|---|---|---|

Measured isolated post-solve updates, two identical legs:

| iterations | call | max normal impulse update | max tangent coordinate update | max closing velocity |
|---|---|---|---|---|
| 40 | 1 | 1.58106114867e-08 | 9.83702477647e-09 | 2.62369460186e-09 |
| 40 | 20 | 0.0471628882736 | 0.0223071141161 | 0.0205992512733 |
| 40 | 100 | 0.0742793001406 | 0.0192080742917 | 0.0246353401918 |
| 40 | 300 | 0.120411298521 | 0.0623836886607 | 0.0334967517502 |
| 40 | 590 | 0.0684564235658 | 0.0677819323938 | 0.0172402095854 |
| 160 | 1 | 1.58106114867e-08 | 9.83702477647e-09 | 2.62369460186e-09 |
| 160 | 20 | 0.00254674101773 | 0.000796580805964 | 0.000698838013448 |
| 160 | 100 | 0.0119972917351 | 0.00394159651803 | 0.00483019910145 |
| 160 | 300 | 0.00924253578683 | 0.00391516656629 | 0.0039130903861 |
| 160 | 590 | 0.0162206553358 | 0.00741240994697 | 0.00702747156537 |

At call300 the40-iteration residuals are0.1204113 normal and0.06238369 tangent,
versus0.009242536 and0.003915167 at160. These are different evolving states,
not convergence traces for one frozen problem. Cone violation max9.097478e-7
passes the preregistered1e-6 diagnostic gate; no tolerance changed. Physical
K16/40 remains INVALID, pen/R1.3415160775184631. Evidence:
`reports/innovation_stack_sweep_residual_l4.json`. Next motion design requires
a coupled update and consistent tangent impulse representation, validated in
the full engine; the present observer changes no solver.

## C1 fixed tangent frame: mechanism and preregistered gates

Measured on Modal NVIDIA L4 by the frozen post-solve observer:

| iterations | sample call | normal update | tangent update | K16 penetration / radius |
|---|---|---|---|---|
| 40 | 300 | 0.120411298521 | 0.0623836886607 | 1.3415160775184631 (FAIL) |
| 160 | 300 | 0.00924253578683 | 0.00391516656629 | original M5 PASS |

The register chunk recomputes a slip-aligned tangent frame each update while
retaining two impulse coordinates without transporting them. The separate
fixed-frame candidate uses the existing normal-derived fallback frame for every
update. All other chunk operations and the frozen reference remain unchanged.
This isolates the representation change; it does not establish convergence.

Before execution: two independent processes must match every K16 trajectory
state byte and all M5/M1/force results; K16 M5 at 40 velocity / 10 position
iterations and 600 steps must pass its unchanged penetration, drift and geometry
gates; K4 M1 at 400 steps must pass; resting force relative error <0.01.
N10000 uses 40 velocity /20 position iterations,10 warm/30 timed steps.
The retained original speed gate is <=4.0 ms per step in each leg; the day-plan
<=4.9 ms criterion is reported separately, never substituted for the old gate.
No successful status before all own gates pass.

Fixed-frame result, Modal NVIDIA L4, Warp1.13.0: K16 penetration/radius
1.32211834192276, secular, spacing0.14306789636611938: FAIL in both legs.
Full trajectory hashes and physical results identical; M1 error0.0221256531
and force error5.89994845e-6 PASS. Wall6.353/6.334 ms fails both time bounds.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_fixed_frame.py` | OWN-GATE-FAIL | L4 exact repeats, K16 pen/R1.32211834192276 FAIL; wall6.353/6.334ms FAIL. |

### C1 cached impulse mechanism, gates before execution

Before another solver design, observe the frozen device manifold warm-start
seam at calls1,20,100,300,590 of the original K16/40 M5 trajectory. Measure
loaded normal/tangent impulse norms and the velocity/angular-velocity change
across that seam, without applying any update. Fixed observer gates: two full
reports identical, all five samples present, finite measurements and original
M5 exactly equal to the saved L4 result. A passing observer does not certify
the physical solver.

Measured frozen warm seam, Modal NVIDIA L4, two entire results identical,4/4
observer gates PASS, original M5 unchanged:

| call | cached normal maximum | cached normal sum | cached tangent norm | velocity change maximum |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 |
| 20 | 33.7031478882 | 272.907114655 | 7.83905778408 | 0 |
| 100 | 578.192565918 | 4872.69443429 | 125.550089047 | 0 |
| 300 | 950.385192871 | 4605.49167320 | 256.707579591 | 0 |
| 590 | 465.734649658 | 2387.64688709 | 35.3961843365 | 0 |

| module | status | evidence |
|---|---|---|

### C1 applied warm start: design after the mechanism table

A separate subclass applies the selected cached impulse once before the velocity
sweeps, in deterministic pair-colour/contact order, using the fixed normal-derived
tangent frame. The same stored impulse is then the initial accumulator for delta
updates. This is an initial guess application, not an extra velocity iteration.
The measured wide-grid allocation is composed explicitly; previous solvers stay
frozen. Test velocity budgets20 and40, both bounded by the original maximum40.
Each budget must pass the unchanged K16 M5, K4 M1, resting-force and full physical
repeat gates. A successful budget must also pass the retained <=4.0ms wall gate
in both legs at N10000; report <=4.9ms separately. Position budgets and fixtures
are unchanged. Also require finite full state and exact repeated large-scene
state bytes outside the timing region. No defaults are promoted by this test.

Applied-warm results, Modal NVIDIA L4, Warp1.13.0:

| velocity iterations | K16 penetration/radius | drift | M1 relative error | wall ms, legs1/2 |
|---|---|---|---|---|
| 20 | 0.699075758457 | secular | 9.74909610e-8 | 4.779/4.900 |
| 40 | 0.195636451244 | bounded | 1.02585609e-7 | 5.494/5.316 |

Both full K16 and large-scene hashes match across independent workers. At40 all
physical gates pass, but both speed bounds fail. At20 the geometry is valid but
penetration/drift fail. All negatives remain. Evidence:
`reports/innovation_stack_applied_warm_l4.json`.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_applied_warm.py` | OWN-GATE-FAIL | L4 K16/40 M5 PASS0.195636451244, full repeats; speed FAIL5.494/5.316ms. |

### C1 applied-warm cost observer, before optimization

Use the frozen per-stage synchronized profiler at N10000,40 velocity/20 position
iterations,10 warm/30 measured steps. Two isolated workers, empty-context guards,
finite nonnegative timing measurements and complete final state byte identity.
Report every stage before changing execution. This observer has no speed or
physical promotion gate; the candidate above remains OWN-GATE-FAIL.

Measured cost table, Modal NVIDIA L4, two exact final-state hashes:

| stage | ms, leg1 | ms, leg2 |
|---|---|---|
| generate | 0.473 | 0.472 |
| sort | 0.327 | 0.330 |
| select | 1.160 | 1.165 |
| pairs | 0.321 | 0.326 |
| colour | 0.678 | 0.684 |
| solve | 1.331 | 1.332 |
| warm store | 0.059 | 0.061 |
| integrate and force readback | 1.064 | 1.034 |

| module | status | evidence |
|---|---|---|

### C1 deferred force: design and gates before execution

The separate deferred-force subclass retains the post-gravity velocity on the
device and evaluates the unchanged host force formula when contact_forces() is
called. The step still synchronizes before returning, so timed GPU work cannot
escape the measured interval. Contact generation, manifold selection, colouring,
impulse updates and integration remain the same operations. No force request
is omitted from physical gates; time the same standard N10000 fixture.

Fixed gates: K16/40 M5, K4 M1 and resting force pass unchanged and equal the saved
applied-warm results exactly; both full K16 trajectories and large-scene state
hashes repeat; large-scene full force bytes match a same-worker frozen reference;
all states finite. Both candidate wall times must be <=4.0ms; report the day-plan
4.9ms bound independently. Two isolated workers and unchanged idle guards.

Deferred-force result on Modal NVIDIA L4:8/9 gates PASS. K16/40 M5 remains
exactly0.195636451244, all physical, full state and full force bytes identical
to the frozen reference in both legs. Candidate4.283/4.195ms versus
reference5.424/5.314ms. Day-plan4.9ms PASS; retained original4.0ms FAIL.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_deferred_force.py` | OWN-GATE-FAIL | L4 physical/state/force parity PASS, timing4.283/4.195ms fails original4.0ms. |

### C1 pair-graph persistence observer, before cache design

The measured colour stage costs0.678/0.684ms. Observe ordered pair endpoints
and dynamic masses across40 large-scene steps (10 warm plus30 timed-fixture
steps), and the600-step K16 M5 fixture. Do not change colouring. Fixed gates:
two entire reports identical; every expected step observed; original K16 M5
and complete large-scene state hash unchanged. Report repeated graph counts
before designing any reuse. The observer makes no performance claim.

Initial pair-graph instrument:3/4 gates PASS. It records597/600 stack and37/40
large-scene colour calls: each fixture has three initial contact-free steps.
All recorded graph/state sequences repeat and reference states/M5 are unchanged.
The all-step-count gate FAIL is retained in
`reports/innovation_pair_graph_reuse_initial_l4.json`. The observer now records
explicit no-contact steps as well; the original600/40 expected counts are
unchanged. No solver operation or physical gate changes.

| observed contact graphs | colour calls | equal successive graphs | unique graphs |
|---|---|---|---|
| K16 | 597 | 581 | 16 |
| N10000 | 37 | 35 | 2 |

### C1 exact colour cache: design after persistence measurements

A separate subclass compares current ordered pair endpoints and every inverse
mass to device copies of the previous coloured graph. Only an exact match with
the same pair count reuses the colour/permutation/count arrays. Any difference
recomputes the unchanged frozen colouring and refreshes the copies. The original
colour priorities depend on pair index; endpoint/mass identity therefore retains
the entire conflict graph and priority order, not just its colour count.

Before execution: preserve all nine deferred-force physical/parity/repeat/speed
gates, including <=4.0ms in both legs. Additionally require actual cache hits
and misses, and exact frozen-colouring parity after deliberately changing an
endpoint and an inverse mass in a separate colour-only fixture. Record hits and
misses for both main fixtures. The deferred-force reference remains unchanged.

Corrected observer:4/4 gates PASS in two isolated L4 workers. It now records
all600/40 simulation steps, including three contact-free steps in each fixture.
The contact-bearing graph reuse remains581/596 transitions for K16 and35/36
for N10000; the additional two equal transitions per fixture are explicitly
contact-free. Physical and complete large-scene state references remain exact.

| module | status | evidence |
|---|---|---|

Final C1 result, Modal NVIDIA L4, driver580.95.05, Warp1.13.0,2026-09-13:
all11/11 gates PASS in two independent processes. The40-iteration K16 fixture
has penetration/radius0.19563645124435425, bounded drift, minimum spacing
0.19521817564964294; M1 worst relative error1.02585608744e-7 and resting-force
error6.70780099516e-8. Both complete trajectories and large-scene state/force
arrays repeat byte-for-byte and match the corresponding applied-warm reference.

| case | candidate wall ms | frozen deferred-force wall ms | original <=4.0ms |
|---|---|---|---|
| independent leg1, N10000 | 3.684 | 4.186 | PASS |
| independent leg2, N10000 | 3.798 | 4.327 | PASS |

K16 cache hits/misses581/16; N10000 hits/misses35/2. Endpoint and inverse-mass
changes invalidate the cache, and recomputed arrays equal frozen colouring
exactly. Identical graphs hit the cache. All four controls pass twice.
Earlier fixed-frame,20-iteration,uncached-time and observation-coverage failures
remain recorded above; no threshold or frozen comparator was edited.

| module | status | evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_color_cache.py` | VERIFIED-FRESH | L4 11/11 own gates; K16/40 pen/R0.195636451244, N10000 3.684/3.798ms; full state/force repeats and cache invalidation pass. |

Reproduce with `PYTHONPATH=src:scripts python -m motion_engine.contact_engine_gpu_color_cache`
on an isolated CUDA worker. Evidence: `reports/innovation_stack_color_cache_l4.json`.
The timing fixture is40 velocity/20 position iterations,10 warm/30 measured
steps; K16 uses40 velocity/10 position iterations over600 steps.
Force evaluation is deferred until requested; step completion remains synchronized.
This is an opt-in module, with no default engine replacement or universal
performance claim.

### C0 current-engine re-score, gates before execution

Re-measure four frozen configurations on one isolated L4: fixed-point Jacobi,
colored device-manifold plus captured solve, captured pair chunks, and
ColorCacheContactEngine. Shared dimensions0.3/0.3/0.2, friction0.5, dt1/240;
40 velocity and10 position iterations except explicitly labeled M4 sweeps.
Run two independent processes. No engine source changes.

Use unchanged M1 K4/400 steps, M5 K8 andK16/600 steps, and M4 ratio1/1000,
200 steps with velocity counts2/4/6/8/10/20/40. Record validity separately at
every M4 count: no collapsed solution counts as reaching residual0.10.
Hash complete state arrays after every step of these physical experiments.
Measure N1000/N10000 at40 velocity/20 position iterations,10 warm/30 timed
steps; keep observations out of timing and synchronize at both timing bounds.
Store final state hashes and contact counts. No unmatched external speed win.

Also measure free-fall position error against the analytic solution at0.25s,
60/120 steps, and normalized mechanical-energy error. This detects whether a
second-order/symplectic claim actually belongs to these current engines.
An observed first-order method cannot inherit an unimplemented second-order
method's certificate. No existing rolling/SDF, planner or fluid certificate
is transferred to the box-contact benchmark.

Observer gates: complete four-engine/two-process coverage, exact physical
outputs and full-state hashes between runs, finite recorded states and positive
finite timing, active contacts in loaded scenes. Individual solver gates may
fail and remain recorded with their values. WIN/TIE/LOSE/UNFAIR in the separate
comparison table describes only the named gate and its comparability scope;
no unavailable external implementation is declared defeated.

C0 result:5/5 observer gates, four engines/two independent L4 processes, all
physical histories and final timing states exact. ColorCache K16 pen/R
0.195636451, large timing3.780337933/3.787189267ms. Current Jacobi also passes
K16 at40 with0.409035087. Mass ratio1000 is not reached by any tested engine;
ColorCache at40 has residual0.551028245 and invalid geometry. All variants
have free-fall order0.999483106, so no second-order claim is inherited.
Full gate comparison and next action: `OUTCLASS_2026_09_13.md`.

| module | status | evidence |
|---|---|---|

### Cross-hardware state comparison, gates before execution

Use fixed-point Jacobi, captured pair chunks and ColorCache with unchanged
solver sources, Warp1.13.0 and NumPy2.5.3. Shared box dimensions0.3/0.3/0.2,
friction0.5,40 velocity/10 position iterations,dt1/240. Fixtures: K8 stack
at0.205 vertical pitch for120 steps, and the existing two-layerN1000 lattice
for40 steps. These are contact-active determinism fixtures, not a new600-step
stack-stability certificate. No timing score is taken in this experiment.

For each GPU run every engine/fixture in two independent child processes.
Save complete xc,Rm,vc,om and contact-force arrays after every step, plus all
initial body/inertia/rest-point array hashes. Require finite outputs, positive
contact counts, all six cases and exact full arrays on same-GPU repeats.
Across the declared three GPU models require identical initial arrays and
max absolute position difference0.0; also require every other saved state and
force array byte-identical. Report first differing step/component, count,
maximum difference and unsigned bit-pattern distance for every differing array.
No tolerance is widened if another architecture differs. Trace a discovered
first divergence before assigning an operation as its cause.

The local leg must use the shared GPU lock in a coordinated window bounded
by180s. Cloud legs use isolated L4 andA10G allocations. Runtime/backend metadata
is recorded separately from numerical data. Any missing GPU or initialization
mismatch leaves the cross-hardware gate unpassed.

Cross-hardware measurement: all three backends pass their4/4 local capture
gates, and every initial array is identical. A10G andL4 contract outputs match
exactly for all three solvers. The local GPU matches Jacobi exactly but differs
for pair chunks (first completed step4) and ColorCache (first completed step9).
For the K8 trace the maximum position differences are0.000385263935 and
0.000912646297 respectively. The three-backend comparison is3/4 gates,
OWN-GATE-FAIL; the zero-difference target remains unchanged.

Before attributing an operation, capture ten K8 steps at the existing phase
boundaries: generation (canonicalize raw contact order by feature IDs), sort,
manifold selection, pair building/coloring, solve, warm storage and integration.
Record full valid state/contact/impulse arrays, excluding unused capacity.
Two isolated repetitions must match all records and full arrays. Instrumented
contract states must match the corresponding first ten steps of the unmodified
capture byte-for-byte on the same GPU. No timing claim. Compare L4 to the local
GPU to locate the first divergent phase; raw append order alone is not a
numerical divergence if the canonical contact set and later sorted arrays match.


Measured phase boundary (two exact observers on each backend, 3/3 own gates):

| Solver | First completed step | First differing phase | max velocity delta | max normal-impulse delta |
|---|---|---|---|---|
| Pair chunks | 4 | solve | 7.450580596923828e-9 | 1.1920928955078125e-7 |
| ColorCache | 9 | solve | 7.450580596923828e-9 | 3.5762786865234375e-7 |

All recorded earlier phase arrays match. Before selecting an arithmetic fix,
observe individual warm/velocity/position kernel launches at these two steps.
Disable graph capture only in this new observer and require complete ten-step
contract parity against the original captured solver on each backend. Gates:
two independent full snapshot repeats exact, unchanged contract exact, all
snapshots finite. A kernel-boundary result alone does not identify a scalar
instruction or establish whether compiler fusion or division is responsible.

The reusable contact entry point is `probes/cross_hardware_state_probe.py
--capture LABEL`. Run it in the pinned environment on each isolated backend;
local execution requires the shared lock and a bounded coordination window.
Collect both `cross_hardware_LABEL.json` and `.npz` in one report directory.
Then run `python probes/cross_hardware_suite.py --reports reports l4 local a10g`
on CPU. It validates archive checksums and every recorded full-array fingerprint,
constructs the matrix twice exactly, and exits1 if any contact solver differs.
Additional backend labels such as `h100` are supported once captured. This is
the reusable contact portion of the proposed nightly suite; reductions,
adjoints, photon transport, export and automatic scheduling are still pending.
No scheduler is installed by either script.


Individual-kernel observer: 3/3 gates on L4 and the local backend, two full
repeats and unmodified ten-step contract parity. First differing launches:

| Solver | Step (one-based) | Solve launch (zero-based) | Kernel | First differing outputs |
|---|---|---|---|---|
| Pair chunks | 4 | 1 | k_chunk_vel_r | w:2.7939677238464355e-9; jt1:7.450580596923828e-9; jt2:3.725290298461914e-9 |
| ColorCache | 9 | 3 | k_fixed_frame | v:1.4551915228366852e-11; jt1:2.3283064365386963e-10 |

Earlier observed launches match byte-for-byte. These identify differing kernel
operations, not an exact scalar instruction; fusion/division/normalization
remain hypotheses. The initial kernel observer failed with CUDA906 because
the deferred-force step always captures; the corrected observer intercepts
only its graph-dispatch seam and passes exact reference parity. No accepted
complete report came from the failed attempt. Existing solver sources unchanged.


### Sub-millisecond path: frozen ColorCache mechanism gates

First remeasure the current ColorCache engine atN10000,40 velocity/20 position
iterations,10 warm and30 timed steps on isolated L4. Compare unprofiled complete
step wall time with the existing synchronized phase observer. Do not subtract
phase costs from an unrelated unsynchronized total. Each mode runs in two
independent workers. Save complete final state/force arrays and valid sorted
contact/impulse arrays. Gates: all arrays byte-identical across repetitions and
between modes; all times finite/nonnegative; profiled phase sum no greater than
its enclosing wall time; positive contact/pair counts. Report phase overhead,
raw/selected/pair counts, cache hits/misses and actual captured solve-node count.
No candidate is designed from the earlier inconsistent4.6-versus3.7ms table.

Fresh L4 profile:4/4 observer gates, all16 complete output/contact/force arrays
match across both repetitions and both modes. Unprofiled3.685204/3.776263ms;
profiled3.889119/3.860173ms. The following phase costs belong to profiled totals:

| Phase | ms1 | ms2 |
|---|---|---|
| generate | 0.409697 | 0.409443 |
| sort | 0.307316 | 0.305290 |
| select | 1.157386 | 1.148626 |
| pairs | 0.320089 | 0.313636 |
| colour | 0.121065 | 0.119137 |
| solve | 1.336229 | 1.332376 |
| warm_store | 0.059217 | 0.058951 |
| integrate | 0.054282 | 0.052413 |
| outside named phases | 0.123838 | 0.120301 |

Final counts:87716 raw contacts,40000 selected,10000 pairs,2 colours,122 solve
graph nodes;35 cache hits/2 misses. The original selection launches one thread
per raw contact and returns except at each pair's first contact. That leaves
roughly one active thread per8.77 contacts in this fixture. Hypothesis: packing
pair leaders into adjacent threads improves utilization for the same expensive
float64 selection arithmetic. This is a bounded step toward the contact-path
target, not a claim of direct solid-box manifolds or sub-millisecond throughput.

Preregister separate packed-selection subclass: scan sorted pair-start flags,
pack their offsets, and remap selection threads only. Preserve every arithmetic
expression, tie break, impulse accumulation order and original overflow check.
Compare full selected contacts/impulses, state and force arrays to frozen
ColorCache on K16/600, K4/400, rest200 andN10000/40. Require exact two-process
repeats, original M1/M5/resting-force gates, and matched whole-step timing no
slower than baseline in both legs. Report the separate1ms goal as unpassed
unless both complete-step timings reach it; do not replace the4ms old bound.


The packed-selection source audit also finds identical ASTs for the complete
selection arithmetic after thread indexing and for the planar-coordinate helper.
This static check supplements, and cannot replace, the full-engine byte gates.

Packed-selection result:8/8 own gates on L4, two independent workers with
opposite baseline/candidate order. Whole-step time2.926/2.887ms versus matched
ColorCache3.415/3.439ms (14.3%/16.1% reduction). Every saved state, selected
contact, impulse and force byte contributes to exact matching complete traces
for K16/600,K4/400,rest/200; finalN10000 arrays also match. K16pen/R remains
0.19563645124435425, M1worst relative error1.0258560874445039e-7, resting-force
error6.707800995163706e-8. No arithmetic, original physics threshold or iteration
budget changed. The separate sub-millisecond goal remains unpassed.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_packed_select.py` | VERIFIED-FRESH | Own paired full-engine8/8; exact baseline/repeats;2.926/2.887ms, sub-ms goal not reached. |

### Independent-component solve: mechanism and gates before design

The two saved completeN10000 snapshots yield identical connectivity arrays:
10000 pairs form5041 independent dynamic-body components,4959 containing two
bodies and82 containing one. Maximum pairs/component2;5041 ground contacts are
excluded from dynamic connectivity. This measured fixture pays122 solve graph
nodes although components can advance independently. Ground or shared kinematic
bodies must not merge components because their states are read-only in the solve.

Separate candidate: one thread per independent component, execute the existing
warm/velocity/position pair arithmetic in original colour/order within that
component. Preserve40 velocity/20 position sweeps in timing, and40/10 for physical
gates. Recompute component packing only after the frozen exact endpoint/mass
cache invalidates; retain its checks and clear captured graphs on rebuild.
Components above64 pairs use the unchanged captured pair solver. All-static
pairs get private slots. No cross-component state may be written by two threads.

Require CPU partition controls for shared static endpoints, order, empty and
invalid inputs. Require the previous eight paired full-engine gates against
PackedSelect, including full per-step state/contact/impulse/force byte identity
and K16/600,K4/400,rest/200,N10000 timing, in two reversed-order workers. Fewer
launches or a faster microkernel alone cannot pass. The1ms complete-step goal
remains a separate unpassed target unless reached in both legs.

Independent-component result:8/8 full-engine gates on L4, two reversed-order
workers. Whole-step2.887/2.817ms against matched PackedSelect3.190/3.215ms
(9.5%/12.4% reduction). Complete K16/600,K4/400,rest/200 state/contact/impulse/
force traces and finalN10000 arrays are bit-identical to baseline and repeats.
K16pen/R remains0.19563645124435425 with bounded drift and valid geometry.
The source ASTs of all three pair arithmetic bodies are identical after their
indexing wrappers. CPU partition controls2/2 pass. This certificate covers the
measured dynamic-body fixtures; the >64-pair fallback and changing kinematic
state have not received separate GPU controls, and no cross-SKU claim is made.

The reduction is0.303/0.398ms, substantially below the earlier1.1ms extrapolation
from122 ordinary launch costs. That estimate did not represent this captured
execution protocol; fewer graph nodes alone do not establish the remaining
kernel's arithmetic/memory cost. The sub-millisecond goal remains unpassed.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_island_solve.py` | VERIFIED-FRESH | Own full-engine8/8, CPU partition2/2; exact baseline/repeated traces;2.887/2.817ms on the declared fixtures. |

### Remaining contact path: phase and layout-reuse observer

Use the sameN10000/40+20/10warm/30timed protocol on IslandSolve in two independent
workers per mode, with the four existing phase/accounting/full-array gates.
Add a separate untimed40-step replay observing ordered selected body endpoints
at every pair-build call. Its final full physical arrays must equal the unobserved
engine; report exact layout-repeat counts. This replay is excluded from all
timings. Measure whether the remaining pair grouping can reuse its permutation
before designing a cache. Source solvers and all thresholds remain frozen.

IslandSolve phase observer passes5/5 gates. Unprofiled2.872863/2.858812ms;
profiled3.032050/3.024985ms with identical16 full arrays. Untimed layout replay
is also exact to the physical reference;35 of36 consecutive pair-build inputs
are identical in both workers. Measured phase table:

| Phase | ms1 | ms2 |
|---|---|---|
| generate | 0.402653 | 0.407790 |
| sort | 0.297908 | 0.293500 |
| select | 0.702814 | 0.701042 |
| pairs | 0.308049 | 0.302818 |
| colour | 0.120004 | 0.120682 |
| solve | 0.975545 | 0.975885 |
| warm_store | 0.057195 | 0.056583 |
| integrate | 0.050798 | 0.050171 |
| outside phases | 0.117084 | 0.116514 |

Preregister a separate ordered-pair-layout cache. Reuse the pair permutation,
segment offsets/counts and endpoints only after exact selected endpoint-array
comparison with the same contact count. In the same validation kernel compare
all inverse masses to the frozen colour cache. Reuse colours/components only
when both the layout and mass checks pass. Positions, impulses and normals are
never cached by this step. Require the existing eight paired full-engine gates
against IslandSolve plus explicit count/permutation/mass invalidation controls
and cache hit/miss coverage. Every missed key follows the unchanged baseline.

Pair-layout cache passes10/10 own gates on L4 in two reversed-order workers.
Whole-step2.579/2.564ms versus matched IslandSolve2.871/2.942ms. Complete physical,
selected-contact, impulse and force traces remain byte-identical, including
K16pen/R0.19563645124435425. Five explicit controls reproduce their full pair
maps exactly: unchanged reuse; endpoint permutation invalidates; contact-count
change invalidates; inverse mass changed to zero invalidates colouring and
rebuilds one component as two; unchanged reuse resumes. The reuse token is
consumed once, so later direct colour calls retain the original validation.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_pair_layout_cache.py` | VERIFIED-FRESH | Own L4 10/10 gates; exact reference/repeats/invalidation controls;2.579/2.564ms. |

### Exact fixed-point opportunity: observer gates before execution

Observe all original40 velocity and10/20 position sweeps, without early exit.
After every pair update compare the exact32-bit patterns of all writable
velocity/pseudo-velocity and impulse fields, including signed zero. Record a
per-component/per-sweep changed flag. A complete sweep in which every pair
operator leaves its state unchanged is an exact fixed point while contact
geometry, inverse inertia and coefficients stay fixed within this solve.
No approximate residual or tolerance is introduced.

Fixtures: K16/600 at40+10, N10000/40 at40+20, all dynamic bodies. Two independent
workers compare the instrumented candidate against frozen PairLayoutCache after
every complete step, using full physical/contact/impulse/force hashes. Own gates:
exact baseline parity, exact repeated full change arrays and histories, finite
states, nonempty sweep coverage, and no changing sweep after an unchanged one
within either phase. Report exact fixed-point iteration histograms before an
early-exit candidate is designed. This observer has no performance gate.

Fixed-point observer result:5/5 gates on L4, full reference parity and complete
activity-array repeats. No phase resumes changing after an unchanged sweep.

| Fixture/phase | Component observations | Original sweeps | Mean first exact fixed point (capped) | Mean if checked every8 sweeps |
|---|---|---|---|---|
| K16 velocity | 597 | 40 | 39.835846 | 39.852596 |
| K16 position | 597 | 10 | 9.984925 | 9.996650 |
| N10000 velocity | 186517 | 40 | 26.963006 | 29.197081 |
| N10000 position | 186517 | 20 | 19.057464 | 19.427076 |

Preregister separate exact-stop candidate: check every eighth sweep and the
last sweep; stop only after every pair leaves all writable32-bit values
unchanged. Other sweeps retain the original update expressions. No residual
threshold is introduced. Native bit inspection is forward-only; no autodiff or
cross-hardware certificate is implied. Require the existing eight paired
full-engine/byte/physical/speed gates against PairLayoutCache, plus actual early
exit coverage and exact repeated iteration-count arrays. Report the full-step
1ms goal separately. Divergence between component loop lengths may limit speed;
iteration savings alone are not a throughput certificate.

Exact-stop result:10/10 own gates, full baseline/repeated physical/contact/
impulse/force traces identical. Whole-step2.506/2.562ms versus2.567/2.660ms.
FinalN10000 counts repeat exactly: mean velocity sweeps15.62388415, mean position
sweeps19.96111883;4918/5041 components stop velocity early,45 stop position early.
The observed timing benefit is only2.4%/3.7%; no proportional iteration-to-speed
claim or statistically established improvement beyond these paired runs.
All original40/10/20 iteration caps and physical thresholds are unchanged.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_fixed_point_observer.py` | VERIFIED-FRESH | Own L4 5/5: complete activity repeats, reference parity and no restart after exact fixed point. |
| `src/motion_engine/contact_engine_gpu_exact_stop.py` | VERIFIED-FRESH | Own L4 10/10; exact physical traces and repeated sweep counts;2.506/2.562ms. |

### Selection work reuse: mechanism and ablation gates

Fresh measured selection phase remains0.701–0.703ms. In the frozen four-point
selection, each L>4 pair evaluates the same immutable planar coordinates
8L-5 times (deep point, three greedy scans, nearest-kept redistribution).
Direct source-loop counts, not GPU timing estimates:

| Contacts in pair | Existing coordinate evaluations | One evaluation per contact |
|---|---|---|
| 5 | 35 | 5 |
| 9 | 67 | 9 |
| 18 | 139 | 18 |
| 32 | 251 | 32 |

A separate kernel will write float64 planar coordinates once per active raw
contact, then reuse those exact stored values within that selection call.
Inputs and arithmetic remain unchanged; no cross-step geometry cache. Pairs
with L<=4 retain the original direct branch. Overflow checks remain unchanged.
A second independent ablation removes the four kept-contact priority operations
whose output is read only by per-contact colouring, not the forced pair path.

Preregister three paired configurations: frozen PairLayoutCache; prepared
coordinates only; prepared coordinates plus unused-priority removal (default
candidate). Two workers reverse configuration order. Each candidate requires
all eight existing physical/full-trace/byte/timing gates against the matched
baseline. Report the coordinate-only result separately even if it fails; no
performance benefit may be assigned to one change without its ablation.

After the selection candidate gates, run the separate prepared-selection phase
observer with the frozen N10000 timing protocol. Preregister five gates: complete
plain/profiled final-array byte identity across two independent workers each;
finite states and positive finite wall times; nonnegative finite phases whose
sum does not exceed wall time; active contacts/pairs; and unchanged final state
under the untimed layout observer. Report synchronization overhead separately.

Selection ablation result: both configurations pass8/8 own gates on L4,
Warp1.13.0/NumPy2.5.3, two independent workers in reversed configuration order.

| Configuration | First worker | Reversed worker |
|---|---:|---:|
| Frozen pair-layout baseline |2.626ms|2.653ms|
| Prepared coordinates only |2.592ms|2.600ms|
| Prepared coordinates plus priority trimming |2.437ms|2.408ms|

Complete K16/600, K4/400 and resting/200 state/contact/impulse/force traces,
plus final N10000 arrays, are byte-identical to the reference and across workers.
K16/40 pen/R remains0.19563645124435425. Coordinate-only savings are0.034/0.053ms;
additional priority trimming saves0.155/0.192ms in this paired sample. These are
whole-step measurements, not isolated kernel times or statistical significance.
No exact-stop composition or cross-SKU certification is claimed. The <1ms target
remains OWN-GATE-FAIL at2.437/2.408ms, independently of the module's passed gates.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_prepared_planar.py` | VERIFIED-FRESH | Own8/8 in each ablation configuration; full reference/repeat bytes exact; default2.437/2.408ms. |

Prepared-selection phase observer passes5/5. Plain wall times2.496509/2.463660ms; synchronized profile times2.588130/2.617842ms. Complete arrays match across all four workers.

| Phase | First worker | Second worker |
|---|---:|---:|
|generate|0.408095ms|0.417729ms|
|sort|0.300834ms|0.306588ms|
|select|0.528978ms|0.531891ms|
|pairs|0.123671ms|0.127236ms|
|colour|0.005067ms|0.005281ms|
|solve|1.000338ms|1.002660ms|
|warm_store|0.058345ms|0.058709ms|
|integrate|0.050611ms|0.054506ms|
|outside phases|0.112193ms|0.113241ms|



### Solve geometry reuse: measured scope and preregistered gates

The fresh prepared-selection profile measures solve1.000338/1.002660ms, with
40000 selected contacts,40 velocity and20 position sweeps at the final N10000
state. The following is a source-level work count for that measured state,
not generated-instruction counts or a predicted runtime speedup:

| Invariant expression | Evaluations in velocity/position sweeps | Per-step preparation |
|---|---:|---:|
| Contact lever arms |2400000|40000|
| Normal effective inverse mass |2400000|40000|
| Normal-derived tangent basis |1600000|40000|
| Each tangent effective inverse mass |1600000 maximum|40000|
| Position bias |800000 maximum|40000|

Warm-start expressions are excluded and remain unchanged. Tangent/bias counts
are upper bounds because the original branches require effective mass>=1e-12.
Normals, points, centers, inverse inertia/mass and penetration are written before
solve and read throughout it; only velocities, split-position velocities and
impulses change inside solve. This establishes the reuse interval, not numerical
identity after compiler lowering. No values are reused across steps.

Next isolated candidate prepares these expressions once per pair before the
captured component solve, preserving every remaining update and division.
Preparation iterates current pair counts/indices so captured launches do not
assume a stale selected-contact count. Keep the original >64-pair fallback.
Compare against frozen PreparedPlanar with all eight existing gates: full
reference trace equality, full independent-repeat equality, K16, K4, resting
force, finiteness, no paired timing regression, and original<=4ms. Two workers
reverse baseline/candidate order, same fixtures/budgets and strict tolerances.
Report <1ms separately and retain any failed candidate without changing gates.


Prepared solve geometry result: OWN-GATE-FAIL7/8. Whole-step2.474/2.452ms
versus frozen PreparedPlanar2.415/2.396ms, respectively; the no-regression gate
fails in both reversed-order workers. Complete reference and independent-repeat
state/contact/impulse/force traces remain exact; K16/40 pen/R remains
0.19563645124435425 and all physical gates pass. The warm function AST is
unchanged, and the original velocity/position normal-mass source blocks match.
The additional persistent allocation is64bytes per selected-contact capacity,
25.6MB at capacity400000. Arithmetic reuse did not produce a measured speedup;
preparation cost, traffic and compiler optimization are not separately attributed.
The candidate is retained as a negative, with thresholds and baselines unchanged.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_prepared_solve.py` | OWN-GATE-FAIL |7/8:2.474/2.452ms exceeds matched2.415/2.396ms; full physical/repeat bytes exact. |


### Raw contact layout observer: preregistration

The accepted prepared-selection profile measures sort0.300834/0.306588ms at
N10000. Before designing sort reuse, observe sorted raw feature keys and pair
permutations after each manifold call in the unchanged40-step fixture. Record
all arrays, counts, equal transitions and feature-key uniqueness, with no cache
or solver changes. Three independent child processes: one unobserved reference,
two observed repeats. Four gates: exact final full state/contact/impulse/force
arrays against reference; every observed layout array identical across repeats;
all outputs finite; and one record for each contact-bearing step. Reuse frequency
is a measurement, not a pass threshold; duplicate keys must be reported.

Raw-layout observer result:4/4, full observed arrays exactly repeated and final
full physical arrays identical to the independent unobserved reference. There
are37 contact-bearing steps and36 transitions. Every raw key set has0duplicates.

| Layout observation | Equal transitions | Total transitions |
|---|---:|---:|
| Selected endpoints, earlier observer |35|36|
| Raw contact count |16|36|
| Raw feature keys and pair permutation |15|36|

Thus stable selected contacts do not establish stable raw contacts. In particular,
steps21->22 retain84740 raw contacts but change the raw key layout. Count-only
reuse would accept this invalid case. The final raw count is87716; the final
31->39 segment has8 exact transitions. All indices here are zero-based.

At the measured0.300834/0.306588ms sort cost, eliminating the entire sort phase
on15/36 transitions would save only about0.125/0.128ms averaged over those
transitions, before validation and any additional storage cost. This is a rough
opportunity estimate using the measured mean phase cost, not a speed measurement
or a bound across other trajectories. No raw-layout cache is implemented or
certified by this observer. The <1ms objective remains unpassed.



### Four-device motion extension: preregistration

Complete the existing three-solver/two-fixture matrix with H100. Reuse the frozen
capture and full-array checksum auditor; adapt the existing kernel-family
sequential runner orchestration from commit e9aa89d. Keep motion Warp1.13.0 and
NumPy2.5.3, not the kernel family's different version. Preserve retained L4,
A10 and RTX5070 capsules, original3-device matrix and known negative rows.
Each capture requires its original4/4 gates. Four-device acceptance requires
all12 module/backend rows, identical initial inputs, all complete output bytes
exact against L4, and identical independently constructed reports. Any differing
row remains OWN-GATE-FAIL with numeric position delta and named differing arrays.
Scalar instruction attribution remains a separate unresolved requirement.

Run `python probes/motion_matrix_modal_v1.py --capture h100` with
`MODAL_RUNNER_ROOT` set to the configured pinned runner. Omitting `--capture`
reruns the three cloud backends; an empty `--capture` audits existing capsules
without GPU execution. Local acquisition remains a separately locked operation.


Four-device extension result: H100 capture4/4, six fixture/solver cases with
two independent workers each; complete inputs/output arrays use the same pinned
environment as the other devices. Jacobi is exact on all four devices. Pair
chunks and ColorCache are exact across L4/A10/H100, but still differ on RTX5070:
maximum position differences0.00038526393473148346 and0.00091264629736542702.
The overall matrix is OWN-GATE-FAIL3/4, with10/12 exact module/backend rows.
Initial inputs, coverage and report repeats pass; cross-hardware exactness fails.
The H100 addition does not resolve scalar instruction provenance. No new speed
claim or certification of later solver variants is implied.

The runner now audits dated copies of all four input capsules, so recapturing a
backend cannot overwrite frozen comparison evidence. The storage correction was
checked with a CPU-only audit; no GPU repeat was needed. Existing original
three-device reports are retained alongside `reports/motion_matrix_v1/matrix.md`.



### Body-pair generation mechanism: preregistration

Observe frozen PreparedPlanar at zero-based steps19/199/599 of K16 and9/19/39
of N10000, with the existing40/10 and40/20 sweep budgets. Capture actual GPU
world points/raw sorted feature keys. Form per-body bounds from those exact
points, expanded by float32 bead radius plus8*float32_epsilon*max(1,abs(points)).
A CPU center-tree query and inclusive AABB test produce conservative body-pair
candidates. Compare against all raw contact body pairs and ground bodies.
Count actual hash-grid visits and inter-body distance tests on the unchanged
GPU grid. Report exhaustive P*P work per candidate pair separately; no GPU speed
is inferred from CPU broad-phase counts or a different solid-box geometry.

Three independent children: unobserved reference plus two complete observers.
Five gates: exact final full state/contact/impulse/force arrays against reference;
all observed arrays/receipts repeat exactly; all six snapshots present; zero
missed actual body pairs or ground bodies; and finite outputs with visits>=tests
and tests>=actual inter-body contacts. The finite fixture/padding test is not a
universal floating-point broad-phase proof. No solver/baseline edits.

Body-pair observer result:5/5; both complete observers exactly match each other
and final state/contact/impulse/force arrays match the independent reference.
Initial attempt had0complete receipts due to a JSON scalar serialization error;
the retry changed only the recorded scalar type, not arithmetic or gates.

| Fixture/step | Active/candidate pairs | Grid visits | Current distance tests | Exhaustive pair tests | Missed pairs/ground |
|---|---:|---:|---:|---:|---:|
|stack16/19|5/5|17372|7138|1620|0/0|
|stack16/199|15/15|17642|7276|4860|0/0|
|stack16/599|15/15|17820|7364|4860|0/0|
|lattice10000/9|4959/4959|3153700|576252|1606716|0/0|
|lattice10000/19|4959/4959|3194317|611001|1606716|0/0|
|lattice10000/39|4959/4959|3195174|612540|1606716|0/0|

The large fixture has exact body-pair candidates, but exhaustive P*P narrow-phase
work is2.623 times the final point-grid distance-test count. Fewer broad-phase
objects do not establish fewer distance tests or faster execution. Ground tests
could fall from180000 to90738 on these snapshots. CPU broad-phase time is not a
GPU timing result, and no direct manifold implementation is certified here.


Next preregistered CPU analysis: from the retained raw GPU world points, subtract
coordinates in float32 and require each absolute component<=nextafter(float32(0.1),+inf)
before an expensive norm test. Check every actual raw inter-body feature pair
survives, record all candidate bit masks, compare both captured legs exactly,
and require finite/complete counts bounded by the exhaustive P*P table. These
four gates certify only this finite-data filter observation. No GPU speedup or
universal floating-point error guarantee is inferred.

Coordinate-filter observation passes4/4 on both retained captures and two
independent CPU workers, with0missed actual raw contacts.

| Fixture/step | Exhaustive component tests | Norm candidates after filter | Current grid norm tests | Actual inter-body contacts |
|---|---:|---:|---:|---:|
|stack16/19|1620|120|7138|44|
|stack16/199|4860|265|7276|114|
|stack16/599|4860|276|7364|107|
|lattice10000/9|1606716|114088|576252|35805|
|lattice10000/19|1606716|116461|611001|43792|
|lattice10000/39|1606716|112203|612540|44631|

The last large snapshot reduces norm candidates612540->112203, but still needs
1606716 cheap component tests. GPU timing must decide whether this trade pays.



### Body-pair bitmask generation: preregistered prototype gates

Use one warp per conservative candidate body pair, float32 coordinate rejection
at the measured nextafter bound, then the unchanged bead distance/cutoff formula.
Warp ballots encode accepted point-pair IDs in fixed-order32-bit words. Scan
per-pair counts, then emit only accepted IDs in that order, recomputing the
unchanged contact normal/penetration. Ground uses the unchanged bead/plane rule.
Candidate bodies come from retained measured bounds; broad-phase construction
and input upload are outside this narrow-phase experiment. No direct manifold
or full-engine speed claim follows from this stage alone.

Compare all six retained snapshots plus cutoff/coincidence/ground controls to
the frozen point-grid generation kernel. Two independent workers reverse case
and implementation timing order. Six gates: all canonical raw arrays exactly
match the frozen generator; original saved feature IDs are reproduced; full
independent outputs repeat; finite values and all seven cases complete; capacity
zero refuses all nonempty output without writing; malformed pair lists are
rejected. Time matched captured generation calls only, including counters/scan,
excluding each broad phase, allocation and canonical sorting. Timings are
mechanism observations, not a pass gate or a full solver certificate. Keep the
original400000-contact capacity, and explicitly restrict this prototype to1..32
points per body and forward execution. No tolerance or frozen-source changes.

Pair-bitmask generator result:6/6 own prototype gates on L4, two independent
reversed-order workers. All raw body IDs, points, normals, penetrations and
feature IDs match the frozen point-grid generator bit-for-bit in all seven
cases. All six original snapshot key sets reproduce exactly. Capacity zero
refuses all13control contacts per worker without writing outputs; eight malformed inputs
are rejected in both workers.

| Snapshot | Frozen captured generation, two workers | Pair-bitmask captured generation, two workers |
|---|---:|---:|
|stack16__19|0.075455/0.075682ms|0.015482/0.015578ms|
|stack16__199|0.080551/0.080500ms|0.015713/0.015698ms|
|stack16__599|0.082110/0.082172ms|0.015695/0.015694ms|
|lattice10000__9|0.236834/0.236510ms|0.058995/0.059255ms|
|lattice10000__19|0.171641/0.171468ms|0.059290/0.059057ms|
|lattice10000__39|0.171635/0.171873ms|0.058831/0.058976ms|
|cutoff_control|0.021281/0.021215ms|0.009207/0.009266ms|

The final large snapshot is about2.9x faster for this isolated generation stage.
Both timings exclude their broad-phase construction, allocation/upload and
canonical output sorting. The candidate includes count, scan and emit; the
frozen path includes its counter reset and generation call. Candidate pairs
are supplied from the measured CPU bounds, so this is not an end-to-end GPU
broad-phase/manifold path or a whole-engine improvement. The original<1ms
whole-step target remains unpassed. Warp ballots/popcounts are forward-only;
no autodiff or cross-SKU certification is claimed.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_generation_pair_bitmask.py` | VERIFIED-FRESH |6/6 raw-generation prototype gates; complete reference/repeat arrays and capacity/input refusals; L4 only. |


### GPU body-pair search: preregistration

Use GPU reductions to reproduce the measured padded float64 bounds, then a
center hash-grid search, inclusive bounds tests and sorted integer pair keys.
Verify the supplied enclosing-radius bound and finite coordinate limit before
building the grid; refuse capacity overflow before pair writes. Ground pairs
remain explicit. The default radius bound0.35 and coordinate bound10000 restrict
this prototype; they do not replace the bead collision threshold.

On all six retained snapshots, require exact pairs and bounds against the CPU
observer, full repeats across two independent reversed-order workers, positive
finite timings, complete coverage and no stale pair count after refusal. Controls:
capacity zero, too-small enclosing radius, nonfinite points, coordinate limit,
empty-contact scene and changed point layout. Five gates: exact reference,
exact repeats, full finite coverage, all refusals, and empty-contact behavior.
Time complete rebuild including bounds/grid/sort/readbacks on GPU. Compare CPU
reference time descriptively; no speed acceptance or full-engine claim yet.


### GPU body-pair search result (2026-09-13)

Modal NVIDIA L4, driver580.95.05, Warp1.13.0 and NumPy2.5.3: all5/5
preregistered gates pass. Six snapshots in each of two independent workers
produce exact CPU-reference pairs and bounds and bit-identical full arrays.
All five refusal controls pass, including cleared result counts; rebuilding
a previously contacting scene as an empty-contact scene returns zero.

| Snapshot | Pairs including ground | Full rebuild, first/second worker |
|---|---:|---:|
|stack16__19|6|0.380754/0.374249ms|
|stack16__199|16|0.376952/0.377063ms|
|stack16__599|16|0.381096/0.375801ms|
|lattice10000__9|10000|0.500711/0.502015ms|
|lattice10000__19|10000|0.492701/0.469819ms|
|lattice10000__39|10000|0.483207/0.467520ms|

Timing includes bounds, grid construction, counts, scan, pair sorting and
host synchronization/readbacks; allocation and initial upload are excluded.
CPU timings in the report cover bounds only, not a competing full search.
The final large rebuild costs0.483207/0.467520ms before narrow generation
or manifold selection; this does not establish a whole-engine speedup.
Reports: `reports/innovation_body_pairs_gpu.json` and its complete NPZ arrays.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_body_pairs_gpu.py` | VERIFIED-FRESH |5/5 conservative search gates, six snapshots, full exact bounds/pairs/repeats, five refusals and empty transition; L4 only. |

### Direct body-pair manifold integration: preregistration

Measured complete GPU broad-phase rebuild is0.468-0.483ms on the final large
snapshot. Earlier baseline generation/sort/select phases are approximately
0.418/0.307/0.532ms; isolated pair-bitmask generation is0.059ms. These separate
measurements do not predict end-to-end speed. Test a separate engine that feeds
GPU pairs directly to bitmask generation, selects in canonical pair/feature
order, and sorts only the kept contacts into the unchanged solver order.
Preserve original bead geometry, warm lookup, selection arithmetic and solver.

Before promotion require eight unchanged full-engine gates against frozen
PreparedPlanar: exact full-state/contact/impulse/force traces, exact independent
repeats, K16 at40iterations, K4 load, resting force<1%, finite states, no paired
step-time regression and original step-time<=4ms. Use two independent workers
in reversed engine order, K16/600steps, K4/400, rest/200 and N10000 warm10+timed30
at40velocity/20position iterations. The separate<1ms objective stays explicit.
Raw overflow and spatial-bound violations must refuse, never truncate contacts.


### Direct body-pair manifold result (2026-09-13)

Modal NVIDIA L4, driver580.95.05, Warp1.13.0, NumPy2.5.3: **OWN-GATE-FAIL7/8**.
The complete physical-state, selected-contact, impulse and force histories are
bit-identical to PreparedPlanar and across the independent workers. K16 at40
iterations retains penetration/R0.19563645124435425; K4 worst relative load
error1.0258560874445039e-7 and resting-force error6.707800995163706e-8 pass.
Final large-case states are exact with87716raw/40000selected contacts and10000pairs.

| Full-step measurement | First worker | Reversed worker |
|---|---:|---:|
| Frozen PreparedPlanar |2.143ms|2.132ms|
| DirectPairs |2.413ms|2.376ms|

The candidate is slower by0.270/0.244ms and fails the unchanged no-regression
gate. The isolated narrow-generation improvement does not survive this complete
integration. No compiler, bandwidth or synchronization cause is established by
these aggregate timings. Retain the candidate as a negative; do not replace the
baseline or claim the<1ms target. Cross-SKU behavior is not certified.

Evidence: `reports/innovation_direct_pairs.json`, two reversed independent workers,
full trace digests and original gate values. Idle field queries show0%/4MiB
before/after both workers. The client stream showed two completed receipts under
one invocation stamp; the fetched capsule is the61.9s receipt tabulated above.
The other61.7s receipt also reported7/8, with candidate2.368/2.385ms against
baseline2.135/2.112ms. Its full capsule was not separately retained; the reason
for duplicate execution receipts is unresolved. Neither receipt is a speed win.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_direct_pairs.py` | OWN-GATE-FAIL |7/8; full reference/repeat physical bits exact, but2.413/2.376ms versus2.143/2.132ms fails paired speed. |

### Sampled-field contact mechanism: preregistration

Queue11 first requires a real grid query; the analytic SDF engine only tests
support points against a plane and cannot observe holes. Use the field repo's
retained synthetic bracket/plate/two-hole samples at pitches0.5/0.25, origin
(-2,-2,-4), and its documented negative-inside convention. Bind exact source
array hashes. These CSG values are level sets, not certified Euclidean distances.
Recompute the half-cell-corrected signed EDT of their occupancy as a separate
input, preserving the raw samples for comparison.

Before choosing the contact adapter, measure trilinear values and analytic
trilinear gradients on flat support, hole, wall and corner probes; report
normal lengths and errors to known flat faces. Two independent CPU workers
must reproduce all arrays exactly. Four observer gates: input/probe coverage,
independent repeats, affine interpolation/gradient control within1e-12, and
hole-versus-support classification. The table is observational: EDT normal or
surface errors are retained, not hidden by these observer gates. No dynamics,
RT execution, arbitrary moving-body or M1/M5/M6 claim follows from this probe.


### Sampled-field mechanism result (2026-09-13)

Observer OWN-GATE-FAIL3/4: input coverage, full independent array repeats and
affine interpolation/gradient control pass; support-versus-hole classification
fails on coarse EDT. At pitch0.5 the point0.2inside the analytic support face
has EDT value+0.05, so it is classified outside. The holes remain outside in
all cases. This is a half-cell surface shift from thresholded occupancy, not
a missing-hole error. No change to the predeclared sign check.

| Representation | Pitch | Maximum flat-face distance error | Corner gradient norm range |
|---|---:|---:|---:|
|levelset|0.5|4.4408921e-16|1.046320-1.414673|
|edt|0.5|0.25|1.004969-1.149561|
|levelset|0.25|4.4408921e-16|0.815497-1.134858|
|edt|0.25|0.125|0.945008-1.062769|

Fine EDT retains the tested support sign but still shifts the surface0.125.
Neither interpolated representation has unit gradients near corners; normal
normalization is required. The original CSG field is retained and not relabeled
as an exact SDF. This observer performs no CUDA or RT computation.


### Sampled static-part contact: preregistration

The fine bracket EDT has measured flat-face shift0.125 input units. Use uniform
scale0.05 (pitch0.0125, face shift0.00625), translation(-0.5,-0.45,-0.1), and
query the actual 3D field instead of extracting a plane. The bracket is static;
dynamic bodies retain the frozen bead model and body-body generation/solver.
This is a static arbitrary-part seam, not dynamic arbitrary STEP-body support.
Enclosed positive boundary padding must exceed the bead radius; reject invalid
fields and nonfinite/zero contact gradients, never clamp an out-of-domain point
onto the boundary. Normalize local gradients and preserve raw feature IDs.

Eight gates: CPU/GPU values and gradients within1e-12 on identical float32 probe
points; independent full state/contact/force trace repeats; unchanged M1<0.10
and validity; unchanged K16 M5 penetration/R<0.5, bounded drift and validity at
40iterations; resting-force error<1%; finite positive full-step timing;
malformed-field refusals; hole/wall controls showing the grid is actually used.
Run K4/400steps, K16/600, rest/200 and M6 four repeats/100steps. Require M6 exact
zero (stricter than its original1e-2), included in the repeat gate. Report paired
frozen-box metrics and wall time; physical equality means the original gates,
not identical trajectories for different geometry. No speed-win requirement.


### Sampled static-part contact result (2026-09-13)

Modal NVIDIA L4, driver580.95.05, Warp1.13.0, NumPy2.5.3, SciPy1.18.1:
**VERIFIED-FRESH8/8** for the static sampled bracket seam. Both independent
workers reproduce complete physical/contact/force trace digests and saved
arrays exactly. All24 CPU/GPU value/gradient queries agree exactly (error0).
Eight malformed-field controls pass per worker. Explicit contact IDs are
[0,2]: support and vertical wall collide; the hole and out-of-domain high
point do not. Normals match the two expected face directions within1e-6.

| Metric | Frozen bead boxes on plane | Same bead bodies on sampled bracket | Original gate |
|---|---:|---:|---|
| K4 mean tail load error |1.0258560874445039e-7|1.0780785055977747e-7|<0.10 and valid|
| K16 maximum penetration/R,40iterations |0.19563645124435425|0.20690709352493286|<0.5, bounded and valid|
| M6 maximum position spread,4repeats |0|0|<1e-2; this probe additionally requires exact0|
| Resting force relative error |6.707800995163706e-8|6.707800995163706e-8|<0.01|
| Warm K16 full step,first/reversed worker |4.420502/4.204961ms|4.180639/4.177904ms|Positive finite time; no speed-win gate|

Each dynamics worker runs K16/600steps, K4/400, rest/200, M6 four100step
repeats, then warm10/timed30 full K16 steps with40velocity/10position iterations.
These K16 timings are not comparable to the earlier N10000 lattice throughput
measurements. Idle guards detect no other compute process; field readings are
0%/3MiB except20%/3MiB before the second worker, retained in the report.

The measured EDT support shift0.00625 is retained. Physical gate outcomes match
the box baseline, but M1/M5 numbers and trajectories are not identical between
the two different surfaces. Do not claim literal numerical equality to the box
results. Independent repeats of each geometry are exact. The coarse-field
observer remains OWN-GATE-FAIL3/4; this success does not erase that negative.

The input is a dense negative-inside signed distance grid plus its origin and
uniform pitch, in the same coordinate units as the moving bodies. The field
must enclose its fixed part with positive boundary distance greater than the
bead radius. Caller geometry/provenance remains its responsibility. The adapter
uses normalized trilinear gradients, refuses invalid contacts/overflow, and
leaves original body-body generation and the solver unchanged. It does not
supply STEP import, mesh signing, RT execution, moving sampled-body contact or
sampled-body mass/inertia. Thus this closes the sampled static-part seam of
queue11, not its broadest arbitrary-moving-STEP-body interpretation.

Run with the pinned dependencies and `PYTHONPATH=src:scripts`:

```bash
python probes/innovation_sampled_field_contact.py
```

API: `SampledFieldContactEngine(field, origin, pitch, dims=..., vit=40, pit=10)`;
then the existing `add_body`, `step`, `get_state`, `contact_forces` contract.
Construction copies the input arrays. Reports are
`reports/innovation_sampled_field_contact.json` and its complete saved NPZ.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_sampled_field.py` | VERIFIED-FRESH |8/8 static bracket gates; K4error1.07808e-7, K16pen/R0.206907, M6zero, actual hole/wall controls, L4 only. |

### Cross-device scalar localization: preregistration

The retained unchanged-contract stage/kernel observers already locate the first
pair-chunk difference at zero-based step3, solve launch1 (`k_chunk_vel_r`):
angular velocity2.7939677238464355e-9, tangential impulses7.450580596923828e-9.
ColorCache first differs at step8, solve launch3 (`k_fixed_frame`): velocity
1.4551915228366852e-11, tangent impulse2.3283064365386963e-10.

Copy only those two kernels into a separate instrumented module. Record each
contact-loop assignment, with a write-presence mask, on the first differing
launch. Original launch order and all other kernels remain frozen. Instrumentation
can change compiler decisions: its own gate must reproduce the complete frozen
10-step state/force contract on each device before scalar attribution is trusted.
Four gates per device: unchanged contract, two exact independent repeats,
complete two-kernel scalar/presence coverage, finite contract arrays. Transient
unused tangent0/0 is retained in scalar logs, not silently replaced. Compare
L4 and local full traces only after these gates; no12/12 matrix claim follows
from localization alone. No timing result is taken from instrumented kernels.


### Cross-device assignment localization result (2026-09-13)

The separate scalar observer passes4/4 on NVIDIA L4 and RTX5070; both independent
legs preserve the original ten-step state/force contracts exactly. Comparison
passes4/4 with identical presence masks. The first differences are:

| Solver | Step / launch / contact (zero-based) | First differing assignment | L4 | RTX5070 |
|---|---|---|---:|---:|
| pair chunks |3 / 1 / 1|`a2 = o2 - dot(vrel,t2)/meft2`|-0.06181439384818077|-0.06181439757347107|
| ColorCache |8 / 3 / 2|`a1 = o1 - dot(vrel,t1)/meft1`|0.0032636946998536587|0.003263694467023015|

All previously recorded assignments on those threads match. This narrows the
failure to the tangential impulse update, not generation, sorting, coloring,
normalization or integer redistribution. Dot/division/subtraction still form
compound expressions; an individual instruction cause is not yet established.
There are33/6 downstream differing trace values respectively. Full scalar and
presence arrays are retained in `reports/cross_scalar_{l4,local}.npz`; the schema
binds expression/source-line labels. The original full matrix remains10/12exact.


### Friction expression replay: preregistration

Replay the two retained identical operand tuples. Compare default expression,
materialized dot/quotient/subtraction, disabled contraction, and explicit CUDA
round-to-nearest multiply/add/divide/subtract boundaries. Four gates perdevice:
independent exact repeats, finite complete arrays, whole-expression reproduction
of the original scalar witness, and explicit-boundary agreement with contraction
on/off. If witness reproduction fails, report compiler-context sensitivity and
do not attribute the frozen difference to a microkernel's first differing output.
This replay alone cannot certify the full solver or four-device matrix.

### Ordered friction-dot candidate: preregistration

With identical captured operands, CPU round-to-nearest arithmetic reproduces
both observed endpoints by changing which term of the three-term dot product
is fused. Pair chunks: separately rounded products yield-0.06181439384818077;
fusing the second product yields-0.06181439757347107. ColorCache: separate
products yield0.003263694467023015; fusing the first yields0.0032636946998536587.
This is a compatible arithmetic explanation, not a disassembly proof. The small
local replay does not reproduce the original compiler context and remains a
negative witness-reproduction result.

Test two separate solver classes that change only the two friction numerator
dot calls, using explicit CUDA round-to-nearest products and left-to-right sums.
Keep all other arithmetic, source baselines and physical thresholds unchanged.
New candidate captures must cover the same Jacobi/pair/ColorCache matrix and
fixtures, exact independent repeats and initial arrays, then exact output bytes
acrossL4/local before extending toA10/H100. Final target12/12 exact plus original
K16/40, K4 and resting force gates; no old matrix row is overwritten or promoted.


### Friction expression replay result

Replay passes4/4 onL4 but OWN-GATE-FAIL3/4 onRTX5070: independent arrays are
exact, finite and explicit-rounding outputs agree with contraction enabled or
disabled, but the small default-expression kernel reproduces the cloud values
on both devices. Thus the original local compiler context is not reproduced;
this microkernel cannot by itself identify the original machine instruction.
The explicit-boundary seven-column arrays match exactly between devices.

Initial captures on both devices failed2/4 because operand extraction selected
an unwritten conditional effective-mass slot, giving zero mass and infinities.
Those initial JSON/NPZ receipts are retained as `cross_impulse_initial_*`.
The extraction now honors the original write-presence mask; no solver input,
source arithmetic or tolerance changed. Corrected reports: `cross_impulse_*`.


### Ordered friction-dot result

Candidate OWN-GATE-FAIL3/4 on the L4/RTX5070 screening matrix: both captures4/4,
all initial arrays and independent repeats exact, but pair-chunk position delta
0.00033061392605304718 and ColorCache0.0000030258670449256897 remain nonzero.
Four of six module/backend rows are exact;12/12 is not established. Do not extend
this failed candidate to additional GPUs or replace the original solver.
The smaller ColorCache delta is not a relaxed determinism pass.

The exact-rational rounding model passes3/3: both recorded endpoints reproduced
by compatible fused-product choices, exactly repeated. It remains a numerical
counterfactual, not evidence of which instructions the original binary issued.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_ordered_friction.py` | OWN-GATE-FAIL |3/4 two-device matrix; pair0.000330614,ColorCache0.00000302587 nonzero. |

### Noncontracting solve-kernel candidate: preregistration

The narrow friction-dot fix leaves both solvers nonidentical. Next copy only
the two localized velocity-solve kernels with `fuse_fp=False`, keeping their
original expressions and all other baseline kernels unchanged. Same screening
gates, exact initial arrays/repeats/output bytes;12/12 plus physical gates still
required for promotion. Separate sources and capsules preserve the dot-only
negative. No tolerance, solver iteration or collision geometry changes.

Noncontracting ColorCache physical preregistration: independently repeat the
original K16/600steps at40velocity/10position iterations, K4/400steps, resting
body/200steps and N10000 warm10/timed30 at40/20. Seven gates: exact independent
full physical/contact/force traces; K16 original0.5 and validity/boundedness;
K4 original0.10 and validity; rest<1%; finite; full fixture coverage; original
N10000step<=4ms. Compare frozen ColorCache numbers and times descriptively;
changed rounding is not required to reproduce its old trajectory. Matrix
identity and physical quality remain separate necessary checks.

### Noncontracting velocity result and next localization

OWN-GATE-FAIL3/4 in the L4/local matrix: each capture4/4 and initial arrays exact,
but pair-chunk max position delta0.0010018395259976387 and ColorCache
0.00023790961131453514 remain. Earliest stack contract differences are orientation
at zero-based step29(pair) and11(ColorCache); velocity/force differences first
appear at30/81 respectively. ColorCache lattice also differs only in orientation
from21, maximum2.9103830456733704e-11. Original/dot-only negatives remain retained.

Next reuse the original phase observer for30steps of this separate candidate.
Three gates perdevice: original candidate-contract parity, full independent
repeats and finite complete arrays. Include split-position velocities (`pv`,`po`)
after solve, and quaternion before/after integration. This is localization only;
no further kernel is disabled based only on the orientation symptom.

The phase comparison passes3/3 with both observers3/3. Earliest divergence is
in the split-position solve, before integration; both independent legs agree.

| Engine | Step (zero-based) | First phase | max delta jp | max delta pv | max delta po |
|---|---|---|---|---|---|
| Pair chunk |29|solve|5.960464477539063e-8|7.450580596923828e-9|6.705522537231445e-8|
| ColorCache |11|solve|5.587935447692871e-9|5.820766091346741e-10|6.51925802230835e-9|

The independent physical harness passes7/7: K16 penetration/R0.1913776993751526,
K4 relative error1.0486407375448883e-7, rest error6.707800995163706e-8,
N10000step3.831/3.780ms. Complete traces repeat exactly. This does not promote
the cross-device candidate or meet the submillisecond objective.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_unfused_solve.py` | OWN-GATE-FAIL |Matrix3/4,4/6 exact rows; physical7/7 separately. |

### Noncontracting position-solve candidate: preregistration

The measured position-solve divergence above motivates one additional copied
kernel: register chunk position sweep with `fuse_fp=False`. Retain original
expressions, constants, integration, ordering and all previous candidates.
Screen the same six matrix cases on L4/local with exact initial arrays, two
independent full traces and all-byte equality. Four gates remain unchanged.
Only an exact screen warrants A10G/H100 extension; physical seven-gate harness
remains required independently. No tolerance or iteration changes.

The position-sweep candidate passes the two-device screen4/4: all6matrix rows
exact, including all state/force arrays and both independent legs. Its physical
harness passes7/7: K16penetration/R0.19117027521133423 at40velocity iterations,
K4error1.0552938554973902e-7, rest6.707800995163706e-8, N10000step3.724/3.683ms.
The position-kernel AST is identical to the frozen source except its name and
contraction decorator. Extend the unchanged candidate to A10G/H100, retaining
same fixtures/pins/two independent traces; require12/12 exact rows for completion.

### Noncontracting position-solve result

`reports/position_solve_matrix/matrix.json` passes4/4, all12/12 rows exact across
L4, A10G, H100 and the local architecture. All initial arrays match; all archived
output bytes (position, rotation, linear/angular velocity, contact force) match
across both independent runs for stack8/120steps and lattice1000/40steps.
Archive and per-array checksums are verified before comparison. Static source
parity records all3copied kernel expressions unchanged. Only contraction is
disabled in the2velocity kernels and the shared register position sweep.

This is a scoped cross-device result on the recorded fixtures, pins and drivers;
it does not certify arbitrary scenes, different toolchains or future devices.
Original10/12, ordered-dot3/4 and velocity-only3/4 negatives remain unchanged.
K16 physical7/7 applies to the ColorCache branch; no mass1000 or submillisecond
claim follows. The first original differences are the recorded tangential
impulse expressions; remaining velocity-only differences first arise in split
position. No exact machine-instruction attribution is claimed without disassembly.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_unfused_position.py` | VERIFIED-FRESH |Four-device matrix4/4,12/12 exact rows; ColorCache physical7/7, K16pen/R0.1911702752 at40iterations. |

### Existing midpoint free-fall fixture: preregistration

Queue30 uses the unchanged AV-SIGMA start(0,0,10), box0.3/0.3/0.2, gravity9.81,
T0.25,60/120steps and one substep, now through the existing CPU midpoint engine.
No integration code changes. Record errors, energy, full state/force hashes and
raw log2(error60/error120). Five own gates: exact independent repeats, finite,
complete coverage, zero contact forces, requested observed order>=1.8. The old
GPU row's>=2criterion is also reported unchanged; CPU evidence cannot promote
those GPU engines. Zero errors make the order undefined, not infinite. Constant
acceleration is analytically exact for midpoint, so this fixture may be dominated
by roundoff and cannot by itself establish general second-order convergence.

The midpoint fixture closes OWN-GATE-FAIL4/5. Both full traces repeat exactly;
position errors6.179945444273471e-12/1.9399593043090135e-11 give raw order
-1.6503603810689598, below both requested1.8 and original2. Contact forces are
exactly zero and energy relative errors6.179766178904357e-13/-1.939977230845925e-12.
No second-order promotion is supported by this roundoff-dominated constant-
acceleration fixture; the GPU AV-SIGMA row remains LOSE. The integrator is unchanged.

### Mass-ratio baseline mechanism: preregistration

Archived M4 at ratio1000: Jacobi residual1.0 and final spacing237290560 despite
legacy validity=true; that legacy check excludes collapse, not explosion.
Other engines collapse (spacing0.04996--0.05194). Preserve those receipts.
Before correction, observe unchanged Jacobi ratio1/1000 at40velocity/10position
iterations for200steps, original start/geometry/density/timestep. Record every
apply-stage state norm and accumulator magnitude, canonical first coupled-contact
snapshot, and full state hashes. Four observer gates: exact independent repeats,
finite diagnostic values, full200step/2ratio coverage, and full-state parity with
the archived same-GPU reference. Physical failure is not an observer pass failure.
No claim that legacy validity establishes physical stability.

The analytic two-normal control uses the dual operator
H=[[1/m,-1/m],[-1/m,1/m+1/M]],m12.6. It omits rotation, friction,
contact multiplicity and changing geometry; it is not an engine measurement.

| Mass ratio | Eigenvalues of diagonal-preconditioned H | Slow-mode remainder after40 relaxed0.2steps | Complement Schur |
|---|---|---|---|
|1|0.29289322,1.70710678|0.08940500748|0.03968253968|
|1000|0.000499625312,1.999500375|0.99601077597|0.00007928579357|

[JGS2](https://arxiv.org/html/2506.06494v1) builds a local perturbation subspace
from complementary Hessian blocks (equations8--12). The paper concerns elastic
variational solves and includes a precomputed approximation. Applying that idea
to unilateral rigid contact requires separate treatment of active inequalities,
friction and redundant constraints. A renamed scalar relaxation is not JGS2,
and its elastodynamic convergence result cannot certify this contact engine.

The unchanged baseline observer passes4/4, exact archived state parity and two
identical complete diagnostic runs. At ratio1000 the first joint contacts occur
at step8 (9ground+9body). Spacing first drops below0.164 at step32. At step57
contact counts jump to27ground+33body and velocity grows within that solve from
0.722868 to400081216; peak recorded accumulator7.088869346e18. This observation
establishes growth/contact multiplication, not the exact first overflow instruction.

| Ratio | First coupled normal operator rank/rows | Positive diagonally scaled eigenvalue range | Dense normal-only projection max speed |
|---|---|---|---|
|1|6/18|0.70730983--7.11895342|5.218048216e-15|
|1000|6/18|0.00120654858--8.33828919|6.821210263e-13|

The independent least-squares projection on the captured step8 normal operator
has nonnegative impulses (minimum0.5150250795/515.0251184). It removes the initial
closing velocity, but is not yet an engine or a friction/trajectory certificate.
Next test a fixed-order active-set normal quadratic solve, using dense Cholesky
for each independent active block, before integration into int64 Jacobi. It is
an exact complementary-coupling control for small contact systems, not a claim
to implement the paper's cubature/precomputed elastodynamic method.
Preregister: KKT residual<=1e-9 on captured operators; projected velocity within
1e-9 of independent least squares; independent exact repeats; duplicate-row and
inactive controls; infeasible/range/budget refusal; at most40active linear solves.
Original engine M4 residual<0.10 and validity remain required later.

The static normal quadratic passes6/6: captured ratio1/1000 require5/3active
linear solves, KKT1.665334537e-16/1.705302566e-13 and independent projected-velocity
agreement5.218048216e-15/6.821210263e-13. Duplicate/inactive controls and4refusals
pass in both exact repeats. This certifies only the small static normal problem.

### Complement-normal Jacobi candidate: preregistration

Next add that bounded normal correction beside the frozen int64 Jacobi engine.
Canonical contact ordering and explicit double-precision scalar arithmetic build
the small dense operator on CPU; rounded contributions enter int64 accumulators
and the original GPU apply kernel. This is a hybrid small-system experiment,
not a GPU throughput result or full JGS2 implementation. Contact generation,
friction/cone kernels, integration and position sweeps remain frozen. Each normal
active linear solve consumes one of the40velocity iterations; remaining rounds
use the original Jacobi kernel, reserving at least one friction round. Refuse
budget/rank/range failures; cap32bodies/256contacts rather than claim scalability.

Original M4 fixtures at mass ratios1/1000,200steps,40velocity/10position budget,
original residual<0.10 and validity are mandatory. Add complete-trace stability
(max body-center magnitude<2 and max speed<5) because legacy validity misses
explosion. Require finite, exact independent full-state/force traces, measured
normal+Jacobi rounds<=40, and explicit capacity/budget refusal. Cross-device
identity is required separately after same-device physical screening passes.
Hybrid physical screen passes6/6 on L4: ratio1 and1000 residual6.707800995e-8,
spacing0.1998999417/0.1998999715, peak center0.3058296741 and peak speed0.3270000517.
Maximum normal active solves5; normal+remaining Jacobi rounds40. Both complete
state/force traces repeat exactly. Capacity/range/budget controls pass4/4.
Next capture complete200step traces, both ratios, two isolated workers/device,
with pinned packages and initial-state fingerprints. Screen L4/local, then
A10G/H100 only if exact. Require all output bytes and repeated audits, retaining
hardware labels, archive and per-array checksums. No four-device claim yet.

The hybrid's L4/local screen is exact2/2rows, both ratios and both full200step
state/force runs, archive checksums and initial arrays verified. Extend the
unchanged capsule recipe to A10G/H100 before recording a four-device result.

### Hybrid mass-ratio result

The original ratio1/1000 fixtures now pass on the separate hybrid candidate:
physical6/6, refusal controls4/4, and4/4exact GPU-backend rows in
`reports/complement_normal_matrix.json`. Both200step traces match byte-for-byte
on L4/A10G/H100/local, including initial fingerprints, position, rotation,
linear/angular velocity and contact forces. Repeated matrix audits agree.
At ratio1000 residual6.707800995e-8 is below the unchanged0.10 gate; final spacing
0.1998999715 passes original validity. Peak center0.3058296741 and speed0.3270000517
pass the additional explosion controls. At most5active solves consume part of
the40round budget; original Jacobi rounds consume the remainder.

The result is a numerical small-system mass-ratio win. CPU dense normal coupling
and GPU int64 application are both part of it; it is not fully GPU-resident JGS2,
its cubature/precomputation, a convergence-rate proof or a throughput win. Capacity
is32bodies/256contacts with explicit refusal. Frictional sliding, larger stacks,
arbitrary moving contacts and throughput still require their own measurements.
The frozen four engines retain their mass-ratio failures; their numerical source
and recorded trajectories are unchanged.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_complement_normal.py` | VERIFIED-FRESH |Physical6/6, guards4/4, four-device4/4 exact; ratio1000 residual6.707801e-8 at40rounds, bounded hybrid scope. |


| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_normal_schur.py` | VERIFIED-FRESH |6/6 static normal quadratic controls; captured rank6/18, KKT<=1.71e-13, no dynamics claim. |

The fixed-point observer engine remains a solver component: its arrays and kernels are used by ExactStop. The assignment-trace-only kernel module is in probes.

Publication hygiene:70diagnostic files moved,77diagnostic status rows grouped
below, no missing or duplicate current module rows, scrub0. AST comparison
confirms only import/entry-point relocation for70moved files and19solver files.
Five CPU numerical reports remain byte-identical. The relocated scalar observer
passes4/4 on local/L4 with9224arrays/backend and complete JSON byte-identical to
its pre-move evidence; each backend includes two independent original-contract
runs. `reports/probe_hygiene_verification.json` records the path map and checks.

### Persistent-context100step measurement: preregistration

Measure the existing PreparedPlanar engine, unchanged, on the frozen N10000
2-layer lattice,40velocity/20position rounds,10warm+100measured steps. Separate
plain and synchronized phase-profile runs (two independent workers each) from
an observed replica. The replica records each pair-set hash and symmetric-
difference fraction, layout/color hits/misses, island rebuilds, graph captures,
and a full state/contact/force trace hash. Require observer final full-array
parity with each unobserved run and exact repeated observed traces/counters.
Count captures within the100step window separately from warmup.

Own gates: original final-state/finite/phase-accounting/active-contact checks;
100step counter coverage; pair churn<5%; no recapture without pair-set change;
exact repeated observer traces/counts; plain step<=4ms. Submillisecond status is
reported separately. This is a measurement of existing caches, not a new solver
or a transfer of another engine's cross-device certificate. The frozen workload
may have zero churn; that case does not certify arbitrary changing topology.

Persistent-context probe result:10/10. Plain N10000 step2.48736614/2.47570147ms;
profiled2.62381068/2.61166958ms. All final state/contact/force arrays match across
four workers and each observer replica; full observed traces/counts match.
The100step window has100pair-layout hits,100color hits,0misses,0island rebuilds,
0graph captures and0pair churn. Warmup separately had2misses/rebuilds/captures.
This closes the existing-cache measurement on a zero-churn fixture; it does not
certify mutation/invalidation behavior at nonzero churn or reach submillisecond.

| Profiled phase | Run1 ms | Run2 ms |
|---|---|---|
| generate |0.41704717|0.41494149|
| sort |0.31254801|0.30689862|
| select |0.53773490|0.53651894|
| pairs |0.12403841|0.12565208|
| colour |0.00534218|0.00481445|
| solve |0.99771631|0.99723180|
| warm store |0.06087802|0.05927373|
| integrate |0.05279375|0.05311778|

Another persistence cache is not supported by these counts. Eliminating the
measured generation phase alone would still leave over2ms in the profiled run;
that is a cost decomposition, not a predicted measured RT speedup.

## Bounded RT body-pair search

See `docs/RT_BODY_PAIR_EXPERIMENT.md` for preregistration, external build and complete timings.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_body_pairs_rt.py` | OWN-GATE-FAIL |4/5; exact bounds/pairs/repeats and11refusals pass; fullL4 rebuild0.866/0.870ms atN10000 and1.367/1.379ms atN100000 is slower than the unchanged baseline. Native files are implementation components of this module. |

## Explicit verification

See `docs/VERIFICATION.md` for the complete-coverage refusal and isolated selected recipes.

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/verification_contract.py` | VERIFIED-FRESH |8/8 own hostile-input and execution controls, two independent complete reports exact; no implicit coverage of numerical modules. |
| `scripts/verify_motion.py` | OWN-GATE-FAIL |Full verification remains unaccepted. Source pins predate the new compute sibling and mode routing; previous selected8/8 on10/161 declarations is historical, not current full coverage. |


### Prepared noncontracting candidate: measured mechanism and gates

Fresh paired stage profiles use the original N10000 lattice, 40 velocity/20
position rounds and 10 warm/30 measured steps. Five unchanged paths pass all
four observation gates on both hosts: complete repeat hashes, plain/profiled
state-contact-force hash parity, finite outputs and fixture coverage.

| Backend | Path | Plain ms, two workers | Selection ms | Pair-build ms | Solve ms |
|---|---|---|---|---|---|
| l4 | applied_warm | 5.501/5.341 | 1.18/1.159 | 0.351/0.319 | 1.351/1.361 |
| l4 | deferred_force | 4.253/4.25 | 1.174/1.16 | 0.348/0.315 | 1.317/1.339 |
| l4 | color_cache | 3.673/3.673 | 1.176/1.154 | 0.343/0.311 | 1.349/1.332 |
| l4 | noncontracting | 3.717/3.709 | 1.16/1.173 | 0.319/0.317 | 1.386/1.352 |
| l4 | prepared_planar | 2.453/2.418 | 0.536/0.529 | 0.131/0.125 | 0.991/0.985 |
| h100 | applied_warm | 4.267/4.078 | 0.446/0.479 | 0.293/0.314 | 1.361/1.388 |
| h100 | deferred_force | 3.523/2.792 | 0.39/0.394 | 0.257/0.259 | 1.417/1.363 |
| h100 | color_cache | 2.934/2.49 | 0.349/0.293 | 0.244/0.207 | 1.388/1.352 |
| h100 | noncontracting | 2.594/2.441 | 0.377/0.281 | 0.254/0.208 | 1.392/1.421 |
| h100 | prepared_planar | 1.772/1.809 | 0.266/0.297 | 0.08/0.082 | 0.859/0.903 |

Profile timings include extra synchronization and are not the plain wall time.
Evidence: `reports/night_color_profile_l4.json` and
`reports/night_color_profile_h100.json`. The noncontracting solver retains
about 1.17 ms selection cost on the first host; the prepared path spends about
0.53 ms. This motivates a separate composition of prepared selection/pair
caching with the unchanged noncontracting velocity and position kernels.
Bypass the prepared island-solve specialization: its contraction policy is not
the certified baseline policy. Retain all baselines and physical thresholds.

Before execution, require the existing seven physical harness gates, full
physical/contact/force trajectory hash parity against the noncontracting
baseline, and lower paired complete N10000 step time in both workers. The
original 4 ms bound is retained. Two independent workers run opposite engine
orders. A pass does not inherit unmeasured cross-device fixture coverage.


### Prepared noncontracting result

The candidate passes 9/9 physical/parity/speed gates on both measured devices.
The original K16 penetration/R remains 0.19117027521133423; K4 load error is
1.0552938554973902e-7 and resting-force error is 6.707800995163706e-8.
All complete physical/contact/force trace hashes and final large-scene hashes
match the noncontracting baseline in both independent, reversed-order workers.

| Backend / invocation | Baseline ms, two workers | Prepared ms, two workers |
|---|---|---|
| L4 constructor-substitution check |3.749 /3.726|2.850 /2.872|
| H100 constructor-substitution check |2.611 /2.699|2.309 /2.289|
| L4 public prepared mode |3.715 /3.735|2.820 /2.820|

The existing matrix auditor passes 4/4: all 12/12 solver/backend rows are exact
across L4/A10G/H100/local, including two independent runs per case. Jacobi and
pair-chunk controls are unchanged; the ColorCache entry uses the new sibling.
All 240 captured arrays (94,371,840 bytes) also match the corresponding retained
noncontracting baseline arrays exactly. No tolerance or iteration changes,
submillisecond claim or original-default adoption follows.

Evidence: `reports/prepared_noncontracting_l4.json`,
`reports/prepared_noncontracting_h100.json`, `reports/prepared_noncontracting.json`,
`reports/prepared_solve_matrix/matrix.json` and
`reports/prepared_reference_parity.json`. Initial missing-output-directory
capture attempts are retained in `reports/prepared_capture_setup.json`; complete
captures were rerun after creating the directory. A first matrix CLI invocation
refused its unregistered prepared flag before reading a capture; corrected
routing reuses the unchanged auditor and preserves all original matrix paths.

```sh
python probes/innovation_position_physics.py --prepared
python probes/cross_position_state_probe.py --capture prepared_l4 --prepared
python probes/cross_ordered_compare.py --prepared prepared_l4 prepared_a10g prepared_h100 prepared_local
```

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_prepared_noncontracting.py` | VERIFIED-FRESH |9/9 physical/parity/speed gates on L4/H100; public L4 3.715/3.735 ->2.820/2.820ms, all full traces exact; four-device matrix4/4 and12/12exact,240arrays94371840bytes also exact versus frozen references. |


### Bounded noncontracting island solve: measured mechanism and gates

The prepared candidate's fresh second profile leaves solve as the largest
measured stage. Existing profiling and complete-repeat checks pass 4/4 on
both hosts. The five measured controls are recorded below before implementation.

| Backend | Path | Plain ms | Selection ms | Solve ms |
|---|---|---|---|---|
|l4|applied_warm|5.473/5.393|1.176/1.169|1.341/1.366|
|l4|deferred_force|4.239/4.278|1.167/1.167|1.317/1.331|
|l4|color_cache|3.735/3.791|1.188/1.16|1.353/1.326|
|l4|noncontracting|3.818/3.775|1.186/1.153|1.383/1.348|
|l4|prepared_noncontracting|2.84/2.824|0.539/0.542|1.38/1.366|
|h100|applied_warm|3.084/3.099|0.247/0.247|1.342/1.367|
|h100|deferred_force|2.52/2.537|0.222/0.233|1.407/1.35|
|h100|color_cache|2.248/2.229|0.219/0.228|1.378/1.347|
|h100|noncontracting|2.282/2.237|0.229/0.212|1.383/1.415|
|h100|prepared_noncontracting|2.047/2.051|0.191/0.189|1.382/1.385|

An actual graph-build observer records 122 calls in the final two-color graph:
2 unchanged warm applications, 80 velocity calls and 40 position calls. Both
independent observers find 5041 dynamic components with at most 2 pairs each;
full final-state/contact/force hash matches the reference. The allocation was
H200 despite an H100 request; this count is not an H100 latency measurement.
Evidence: `reports/prepared_launch_count_h200.json` and the two
`reports/night_color_profile_prepared_*.json` receipts.

Test a separate bounded component kernel with noncontracting velocity and
position functions. Apply warm impulses through the original kernels first;
do not change their arithmetic or ordering. Components over 64 pairs use the
unchanged prepared baseline. Preserve pair order inside each component and
reuse the existing partition invalidation rules. This combines fewer launches
and different device locality; do not attribute a future gain uniquely to either.

Require the same seven physical gates plus complete trace parity against the
prepared noncontracting baseline and lower paired full-step time in both
independent reverse-order workers. Retain the 4 ms limit and four-device
capture requirements; no tolerance, budget, iteration or default changes.

Additional boundary tests use actual 64- and 65-body touching stacks, three
steps each, unchanged 40/10 rounds. Require all state/contact/force arrays to
match the prepared baseline, finite outputs, measured component sizes 64/65,
fewer captured calls at 64 and the same call counts as baseline at 65. Run the
existing test module twice in independent processes and retain its full arrays.



### Bounded noncontracting island result

Physical/parity/speed gates pass 9/9 on L4 and H100, with full trajectory,
contact and force hashes exactly matching the prepared noncontracting baseline.
The K16 penetration/R remains 0.19117027521133423 with the original 40-round
velocity budget. The two copied inner register-body ASTs are also identical to
the certified noncontracting velocity/position bodies; warm application remains
outside the new kernel through the original function.

| Backend | Prepared baseline ms, two workers | Island candidate ms, two workers |
|---|---|---|
| L4 |2.798 /2.791|2.442 /2.413|
| H100 |2.192 /2.239|1.758 /1.778|

The existing four-device matrix passes 4/4 with 12/12 exact rows. All 240 full
arrays (94,371,840 bytes) additionally match the prior prepared captures on
the corresponding devices. The original matrix and all earlier candidates
remain unchanged. The submillisecond target is still open.

The existing partition test module passes 4/4 in each of two independent H100
processes. Actual 64-pair stacks use 6 captured calls versus 255 in the
baseline; at 65 pairs both use the same 255-call fallback. Complete boundary
JSON and 147,456/149,760 array bytes repeat exactly for the two cases, with
full reference parity and finite values over all three observed steps.
The initial missing test-output parent failed fixture setup before either GPU
case; the complete reruns passed after creating the directory.

Evidence: `reports/noncontracting_island_l4.json`,
`reports/noncontracting_island_h100.json`, `reports/island_solve_matrix/matrix.json`,
`reports/island_reference_parity.json` and `reports/noncontracting_island_boundary/`.
The setup failure is retained in `reports/island_boundary_setup.json`.

```sh
python probes/innovation_position_physics.py --island
python probes/cross_position_state_probe.py --capture island_l4 --island
python probes/cross_ordered_compare.py --island island_l4 island_a10g island_h100 island_local
python -m pytest -q tests/test_island_partition.py
```

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_noncontracting_island.py` | VERIFIED-FRESH |9/9 L4/H100 physical/parity/speed; L4 2.798/2.791 ->2.442/2.413ms, H100 2.192/2.239 ->1.758/1.778ms; four-device4/4,12/12exact; 64/65-pair boundaries and all full reference arrays exact. |

### Component boundary tests

| Module | Status | Evidence |
|---|---|---|
| `tests/test_island_partition.py` | VERIFIED-FRESH |4/4 cases in two independent H100 processes; static partition/refusal controls and actual64/65-pair GPU boundary arrays exact versus baseline and repeats. |


### Post-fusion stage measurement

The unchanged five-path profile passes all four observer gates on L4 and H100
in independent workers with reversed constructor order. Profile and plain-step
state/contact/force hashes match exactly; synchronization adds timing overhead.

| Backend | Path | Plain ms, two workers | Selection ms | Solve ms |
|---|---|---|---|---|
| L4 | ColorCache |3.647/3.772|1.155/1.168|1.335/1.326|
| L4 | Noncontracting |3.715/3.785|1.160/1.159|1.351/1.356|
| L4 | Prepared noncontracting |2.773/2.815|0.536/0.532|1.355/1.359|
| L4 | Noncontracting island |2.492/2.621|0.544/0.546|1.024/1.017|
| L4 | Prepared planar |2.429/2.513|0.540/0.611|0.990/1.069|
| H100 | ColorCache |2.300/2.290|0.257/0.264|1.341/1.346|
| H100 | Noncontracting |2.363/2.336|0.274/0.258|1.414/1.381|
| H100 | Prepared noncontracting |2.135/2.177|0.235/0.228|1.380/1.380|
| H100 | Noncontracting island |1.749/1.636|0.243/0.227|0.899/0.901|
| H100 | Prepared planar |1.691/1.619|0.252/0.224|0.900/0.890|

Evidence: `reports/night_color_profile_island_l4.json` and
`reports/night_color_profile_island_h100.json`. Solve remains the largest
measured stage. Next measure the identical component kernel at 32, 64, 128,
256 and 512 threads per block before designing a launch sibling. Require
identical full-state hashes across sizes, profile/plain parity, finite values,
complete coverage and exact independent repeats. Retain slower observations.

The 512-thread configuration fails CUDA graph creation with error 701
(resource request exceeds launch capacity) on both L4 and H100. Neither
initial sweep completes; no numerical or speed pass is inferred. Evidence:
`reports/island_blocks_512_failure.json`. A separate complete sweep measures
16, 32, 64, 128 and 256 threads with the same observer requirements.


H100 completes the supported sweep with 5/5 observer gates. All five block
sizes have identical state/contact/force hashes in both independent workers.
These observer results do not certify a new solver module.

| Threads per block | H100 plain ms, two workers | Solve ms |
|---|---|---|
|16|1.361/1.347|0.676/0.67|
|32|1.385/1.355|0.695/0.692|
|64|1.37/1.38|0.742/0.704|
|128|1.397/1.386|0.751/0.763|
|256|1.696/1.697|0.909/0.909|

Evidence: `reports/night_color_profile_island_blocks_supported_h100.json`.

L4 completes the same five-gate sweep, all full-state hashes exact.

| Threads per block | L4 plain ms, two workers | Solve ms |
|---|---|---|
|16|2.331/2.229|0.809/0.822|
|32|2.27/2.244|0.778/0.833|
|64|2.307/2.227|0.808/0.79|
|128|2.277/2.189|0.764/0.788|
|256|2.494/2.424|1.031/1.032|

Choose a separate fixed128 launch sibling: it has the lowest worst L4 full-step
time among measured sizes and improves both H100 legs versus256. No claim of
universal optimality. Reuse the identical kernel, original warm launches and
greater-than64-pair fallback. Require the same nine physical/parity/speed gates
against NoncontractingIsland, plus the existing four-device exact matrix.
Evidence: `reports/night_color_profile_island_blocks_supported_l4.json`.


### Fixed128 component result

The separate CompactIsland module passes 9/9 physical/parity/speed gates on
both L4 and H100 in two independent reversed-order workers. Complete physical
traces, final contact/state/force hashes and numerical records equal the
NoncontractingIsland baseline. Only its component launch block dimension
changes; the chunk method AST is identical after removing that one keyword.

| Backend | Default256 baseline ms | Fixed128 candidate ms |
|---|---|---|
| L4 |2.429/2.436|2.362/2.238|
| H100 |1.696/1.725|1.550/1.537|

Four-device capture comparison passes 4/4 and all12 rows exactly. All240
arrays,94371840bytes, additionally match the prior island captures per device.
The existing component-boundary test module, with CompactIsland substituted
as the candidate constructor, passes4/4 in each independent H100 process
(25.49s cold,1.56s warm). Full boundary JSON and arrays repeat exactly;
64pairs retains255->6calls,65pairs retains255/255fallback. The original
numerical budgets and tolerances remain unchanged; no submillisecond claim.

Evidence: `reports/compact_island_l4.json`, `reports/compact_island_h100.json`,
`reports/compact_solve_matrix/matrix.json`, `reports/compact_reference_parity.json`
and `reports/compact_island_boundary/`. The rejected512-thread resource limit
and all slower block observations remain recorded above.

```sh
python probes/innovation_position_physics.py --compact
python probes/cross_position_state_probe.py --capture compact_l4 --compact
python probes/cross_ordered_compare.py --compact compact_l4 compact_a10g compact_h100 compact_local
```

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_compact_island.py` | VERIFIED-FRESH |9/9 physical/parity/speed L4/H100; L4 2.429/2.436->2.362/2.238ms,H1001.696/1.725->1.550/1.537ms; four-device4/4,12/12exact;64/65pair boundaries exact twice. |



### CompactIsland fresh stage table

|GPU|Path|Plain ms|Select ms|Solve ms|
|---|---|---|---|---|
|l4|color_cache|3.645/3.692|1.164/1.16|1.337/1.332|
|l4|noncontracting|3.72/3.702|1.154/1.158|1.351/1.358|
|l4|prepared_noncontracting|2.775/2.775|0.537/0.531|1.359/1.364|
|l4|noncontracting_island|2.466/2.447|0.539/0.534|1.024/1.016|
|l4|compact_island|2.266/2.168|0.536/0.532|0.765/0.818|
|h100|color_cache|2.311/2.317|0.257/0.274|1.348/1.359|
|h100|noncontracting|2.371/2.365|0.267/0.265|1.425/1.387|
|h100|prepared_noncontracting|2.181/2.207|0.233/0.229|1.389/1.391|
|h100|noncontracting_island|1.7/1.701|0.241/0.23|0.903/0.906|
|h100|compact_island|1.614/1.51|0.243/0.228|0.831/0.752|

Both profiles pass4/4: independent reverse-order workers, full-state/contact/
force hashes repeat and agree between plain and synchronized profiling modes.
The five paths retain the original N10000,40V/20P,10warm/30measured fixture.
Evidence: `reports/night_color_profile_compact_l4.json` and
`reports/night_color_profile_compact_h100.json`.

The next selector experiment compares local vector capacity8/block256 with
capacity4/block256,4/block128,4/block64 and8/block128. The requested manifold
size remains4 and the same builder retains arithmetic and selection order.
Require five configurations, exact reference and repeated full-state hashes,
profile/plain parity and finite outputs. This experiment selects a candidate;
it does not certify physical gates. If a smaller vector is adopted later,
requested manifold sizes above4 must keep the existing capacity8 path.


### Selector capacity experiment result

|Backend|Capacity/block|Plain ms, two workers|Select ms|
|---|---|---|---|
|l4|8_256|2.214/2.255|0.544/0.551|
|l4|4_256|2.25/2.555|0.549/0.561|
|l4|4_128|2.145/2.148|0.43/0.427|
|l4|4_64|2.095/2.135|0.429/0.416|
|l4|8_128|2.134/2.086|0.428/0.42|
|h100|8_256|1.584/1.731|0.269/0.262|
|h100|4_256|1.667/1.595|0.289/0.262|
|h100|4_128|1.785/1.739|0.294/0.263|
|h100|4_64|1.699/1.611|0.277/0.262|
|h100|8_128|1.628/1.667|0.265/0.28|

All five observer gates pass on both devices, including exact full-state
reference/repeat hashes. No alternative beats the reference in both H100
workers; no public solver sibling is adopted. The smaller local vector does
not establish a speed gain. These are negative optimization results, not
failed numerical parity. Evidence: `reports/night_color_profile_selection_capacity_l4.json`
and `reports/night_color_profile_selection_capacity_h100.json`.

### Pair-layout readback measurement

|Backend|Readback|Calls per30steps|Time ms, two workers|
|---|---|---|---|
|l4|cnt|30/30|5.819635/5.759490|
|l4|dflags|30/30|5.757139/6.098614|
|l4|_layout_changed|30/30|1.802796/1.871984|
|l4|_graph_mismatch|0/0|0.000000/0.000000|
|l4|other|0/0|0.000000/0.000000|
|h100|cnt|30/30|3.899357/3.602739|
|h100|dflags|30/30|1.222851/0.983369|
|h100|_layout_changed|30/30|0.860007/0.876477|
|h100|_graph_mismatch|0/0|0.000000/0.000000|
|h100|other|0/0|0.000000/0.000000|

Both observers pass4/4, including exact plain/observed full output hashes and
independent repeats. Readback duration includes preceding queued device work;
it is not a measurement of isolated transfer latency. The pair-layout flags
add30separate readbacks after30selection flag readbacks in each worker.

Design a separate sibling which computes ordered endpoint/mass comparisons
after compaction and returns their flags in the existing selection readback.
Keep all cache predicates and the direct-call fallback; consume the saved
flags once. Require unchanged9physical/parity/speed gates, full four-device
identity, existing invalidation controls with the new device check, and an
observed reduction from90to60readbacks in the original30step fixture. No
speed gain is assumed from readback counts alone.


### Shared selection/layout readback result

The separate SharedLayout module passes9/9 physical/parity/speed gates on
L4 and H100 in two independent reversed-order workers. Complete trajectory,
contact and force hashes match CompactIsland. The selection method AST is
unchanged except for scheduling the integer layout check and storing its
returned flags; all original cache predicates and direct-call behavior remain.

| Backend | CompactIsland baseline ms | SharedLayout candidate ms |
|---|---|---|
| L4 |2.270/2.239|2.120/2.112|
| H100 |1.945/1.626|1.691/1.548|

The H100 timing variation is retained; these are paired fixture measurements,
not a universal speedup or isolated transfer-cost attribution. K16 remains
within the original40round budget, with unchanged penetration and rest forces.
The submillisecond target remains open.

Readback observers pass5/5 on each GPU:90calls become60over30steps while total
returned data remains600bytes. Plain/observed and baseline/candidate complete
output hashes agree exactly. Both existing five-case cache-control sequences
run through the direct-call fallback and the shared device check, twice in
independent processes on each GPU. Count changes and ordered endpoint changes
invalidate layout; mass changes invalidate color; unchanged inputs reuse.
The two control gates pass on both hosts with all20case observations perhost.

The existing four-device matrix passes4/4 with12/12exact rows. All240full
arrays,94371840bytes, also equal the previous CompactIsland captures on their
corresponding devices. Baseline archives are retained unchanged.

Evidence: `reports/shared_layout_l4.json`, `reports/shared_layout_h100.json`,
`reports/readback_profile_shared_l4.json`, `reports/readback_profile_shared_h100.json`,
`reports/shared_layout_controls_shared_l4.json`,
`reports/shared_layout_controls_shared_h100.json`,
`reports/shared_solve_matrix/matrix.json` and
`reports/shared_layout_reference_parity.json`.

```sh
python probes/innovation_position_physics.py --shared
python probes/cross_position_state_probe.py --capture shared_l4 --shared
python probes/cross_ordered_compare.py --shared shared_l4 shared_a10g shared_h100 shared_local
```

| Module | Status | Evidence |
|---|---|---|
| `src/motion_engine/contact_engine_gpu_shared_layout.py` | VERIFIED-FRESH |9/9 physical/parity/speed L4/H100; L4 2.270/2.239->2.120/2.112ms,H1001.945/1.626->1.691/1.548ms; readbacks90->60 with600bytes unchanged; invalidation2/2 perhost; matrix4/4,12/12exact. |



### Whole-step CUDA graph attempt: closed negative

One direct capture design wraps the unchanged SharedLayout step on a dedicated
CUDA stream after10warm steps. Two independent L4 processes reproduce the same
failure at the first host contact-count readback, `cnt.numpy()`, in
`contact_engine_gpu_deferred_force.py:31`. CUDA error906 reports a dependency
between the legacy stream and the capturing blocking stream during `wp_memcpy_d2h`.
The existing solver-loop graphs were already captured successfully in warmup.

| Measurement | First worker | Second worker |
|---|---|---|
| Baseline complete step, ms |2.065|2.137|
| Existing solver graph captures |2|2|
| Whole-step capture succeeded |false|false|
| Failed capture attempt, ms |6.625538|7.446708|
| Whole-step replay time / parity |not reached|not reached|

Baseline full-state/contact/force hashes are identical and finite. Failure
message and first failing call chain repeat exactly. No complete graph exists,
so concurrent island-stream replay and candidate output parity are unmeasured.
This is a negative for this capture attempt, not proof that a redesigned
fully device-resident pipeline is impossible. Stop after this first measured
refusal; no follow-up implementation or tangential-impulse experiment.
Evidence: `reports/whole_step_graph_attempt_l4.json`.

| Attempt | Status | Evidence |
|---|---|---|
| Whole-step graph capture of SharedLayout | OWN-GATE-FAIL |0/1 capture gate; same CUDA906 at contact-count host readback in2independent L4 workers. Baseline2.065/2.137ms, exact full-output hashes; no replay timing or concurrent-stream claim. |



### Final stop-directive check

Final isolated L4 pytest stopped at its first failure:27passed,1failed,1skipped
in442.170s. `test_gpu_module_imports_clean[motion_engine.torch_gravity]` exits1
when writing `data/ur10e_gravparams.npz`; the source-only test payload omitted
the parent directory. The compose test skips because sibling repositories
are absent. Remaining tests were not run. This is not a green complete suite.
No source fix or retry follows the stop directive. The raw JUnit receipt stays
with the isolated run; sanitized summary: `reports/session_c_final_check.json`.
The graph attempt remains closed0/1, with no new numerical implementation.

| Final check | Status | Evidence |
|---|---|---|
| Isolated repository pytest | OWN-GATE-FAIL |27passed/1failed/1skipped; missing data/ output directory, exit1; stopped at first failure, remaining tests unrun. |


## Diagnostic probes

Diagnostic entry points live in `probes/`. Their numerical gates and retained failures are unchanged by relocation. Launch them from the repository root; imports include the solver and legacy helper directories.

| Module | Status | Evidence |
|---|---|---|
| `probes/complement_normal_cross_compare.py` | VERIFIED-FRESH |4/4 exact backend rows, verified archive/array checksums, repeated audit identical. |
| `probes/complement_normal_guards_probe.py` | VERIFIED-FRESH |4/4 capacity/range/budget refusals, exact repeats, no GPU context. |
| `probes/complement_normal_physics_probe.py` | VERIFIED-FRESH |6/6 original ratio1/1000 trajectories plus stability/budget, full state/force repeats exact. |
| `probes/complement_normal_state_probe.py` | VERIFIED-FRESH |4/4 capture gates perdevice, both ratios x2full200step runs on4devices. |
| `probes/contact_normal_schur_probe.py` | VERIFIED-FRESH |6/6, two exact independent runs, projected velocity agrees with independent oracle<=6.83e-13. |
| `probes/contact_solver_scalar_observer.py` | VERIFIED-FRESH |4/4 on both devices, two complete original-contract repeats each; instrumentation only. |
| `probes/cross_hardware_kernel_probe.py` | VERIFIED-FRESH | 3/3 on L4/local, complete kernel repeats and original-contract parity. |
| `probes/cross_hardware_phase_probe.py` | VERIFIED-FRESH | 3/3 on L4/local, complete phase repeats and original-contract parity. |
| `probes/cross_hardware_scalar_probe.py` | VERIFIED-FRESH |4/4 per device, complete two-kernel assignment/presence capture. |
| `probes/cross_hardware_state_probe.py` | OWN-GATE-FAIL | Cross-hardware3/4; max position delta0.00091264629736542702; local capture4/4 on each backend. |
| `probes/cross_hardware_suite.py` | OWN-GATE-FAIL | Reusable matrix7/9 exact solver/backend rows; integrity and repeated rendering pass, two cross-hardware rows fail. |
| `probes/cross_impulse_expression_probe.py` | OWN-GATE-FAIL |Local3/4,L4 4/4; original local witness not reproduced in smaller compiler context. Explicit rounding controls repeat and match across devices. |
| `probes/cross_impulse_rounding_model.py` | VERIFIED-FRESH |3/3 exact-rational counterfactuals reproduce both scalar endpoints, no binary-instruction attribution. |
| `probes/cross_ordered_compare.py` | VERIFIED-FRESH |Position, prepared, island, compact and shared candidates4/4,12/12 exact four-device rows. Ordered-dot and velocity-only matrices remain3/4 negatives in their own directories. |
| `probes/cross_ordered_state_probe.py` | VERIFIED-FRESH |4/4 own capture gates on each device, six cases with independent exact repeats; cross-device solver failures retained. |
| `probes/cross_position_state_probe.py` | VERIFIED-FRESH |Original, prepared, island, compact and shared captures4/4 perdevice;6cases x2independent runs on each of4devices, full arrays retained. |
| `probes/cross_scalar_compare.py` | VERIFIED-FRESH |4/4, identical first-expression attribution in both independent legs. |
| `probes/cross_unfused_phase_compare.py` | VERIFIED-FRESH |3/3; position-solve first divergence repeated for both engines. |
| `probes/cross_unfused_phase_probe.py` | VERIFIED-FRESH |3/3 per device; exact candidate-contract parity and repeats. |
| `probes/cross_unfused_state_probe.py` | VERIFIED-FRESH |4/4 capture gates on each device; cross-device differences retained. |
| `probes/honest_plannability_probe.py` | VERIFIED-FRESH |
| `probes/innovation_applied_warm_profile.py` | VERIFIED-FRESH | L4 2/2 observer gates; final-state hashes exact, integrate/force1.064/1.034ms. |
| `probes/innovation_body_pair_observer.py` | VERIFIED-FRESH |5/5, six snapshots twice; no missing pairs/ground, full reference/observer bytes exact. |
| `probes/innovation_body_pairs_gpu.py` | VERIFIED-FRESH |5/5 complete rebuild audit in two independent reversed-order workers. |
| `probes/innovation_cloud_stack_probe.py` | OWN-GATE-FAIL | Modal L4: two exact trajectories/M5 dictionaries; four idle checks passed; K16/40 pen/R1.341516077518 FAIL, timings6.597/6.362ms FAIL. |
| `probes/innovation_color_cache_profile_v2.py` | VERIFIED-FRESH | L4 4/4 own gates;16 complete arrays equal across both modes and both independent workers; unprofiled3.685204/3.776263ms. |
| `probes/innovation_contact_gpu_bitset_probe.py` | VERIFIED-FRESH | L4 corrected int64 oracle: all five fixed gates pass; no performance samples. |
| `probes/innovation_contact_gpu_cached_probe.py` | VERIFIED-FRESH | L4 all five correctness gates PASS on36/144/450/7/528 contacts; two full repeats exact and overflow rejected. |
| `probes/innovation_contact_gpu_cached_stage.py` | OWN-GATE-FAIL | L4 full outputs exact twice;10000 cached1.759378/1.738346ms vs raw1.449988/1.411383ms, speed gate FAIL;23.04 MB extra workspace. |
| `probes/innovation_contact_gpu_phase_probe.py` | OWN-GATE-FAIL | L4 exact repeated outputs preserved;10000-body count1.367177/1.370592ms and emit1.668158/1.674170ms dominate; raw-speed gate remains FAIL. |
| `probes/innovation_contact_gpu_prefix_probe.py` | OWN-GATE-FAIL | L4 exact outputs/repeats;1000 cached0.269046/0.259141ms beats prefix0.313094/0.315748;10000 cached1.762389/1.801562 loses1.566056/1.604639, combined speed FAIL. |
| `probes/innovation_contact_gpu_reference.py` | VERIFIED-FRESH | Modal L4: three fixtures twice, exact canonical geometry/keys; raw order differs in 2/3 fixtures. |
| `probes/innovation_contact_gpu_stage_probe.py` | OWN-GATE-FAIL | L4 exact geometry/repeats at1000/10000 bodies; candidate3.399/3.386ms vs raw1.467/1.477ms at10000: cost gate FAIL. |
| `probes/innovation_contact_grid_visits.py` | VERIFIED-FRESH | L4 all four count/repeat gates PASS;10000 visits46262638 at64^3 vs3180004 at256x256x8; all85427 contacts preserved. |
| `probes/innovation_contact_reference.py` | SYNTHETIC-ONLY | 3/3 fixtures pass exact sorted identity and unique feature keys; two complete reports byte-identical; 36/144/450 contacts. |
| `probes/innovation_cuda_identity_probe.py` | CUDA-ONLY | Modal L4 single diagnostic completed: Python PID35, reported compute PID1; two zero readbacks identical, full-run repeat pending, 0 timing samples. |
| `probes/innovation_direct_pairs.py` | OWN-GATE-FAIL |7/8 full integration gates; negative retained, no timing threshold change. |
| `probes/innovation_exact_stop.py` | VERIFIED-FRESH | 10/10 paired full-engine, early-exit and count-repeat gates. |
| `probes/innovation_fixed_point_observer.py` | VERIFIED-FRESH | 5/5 two-worker K16/N10000 observer gates. |
| `probes/innovation_island_profile_v1.py` | VERIFIED-FRESH | L4 5/5 gates, complete physical parity and35/36 identical layout transitions. |
| `probes/innovation_island_solve.py` | VERIFIED-FRESH | Matched eight-gate physical/byte/timing audit, two reversed-order workers. |
| `probes/innovation_packed_select.py` | VERIFIED-FRESH | 8/8 matched physical, complete-trace, timing and repeat gates. |
| `probes/innovation_pair_axis_filter.py` | VERIFIED-FRESH |4/4 finite-data gates; full mask repeats exact;0missed raw contacts in all six snapshots. |
| `probes/innovation_pair_bitmask_probe.py` | VERIFIED-FRESH |6/6 on six retained snapshots plus cutoff/coincidence control, two reversed-order workers. |
| `probes/innovation_pair_graph_reuse.py` | VERIFIED-FRESH | L4 4/4 gates after explicit empty-step observations;600/40 records exact twice,35/36 large contact-graph transitions reusable. Initial597/37 coverage failure retained. |
| `probes/innovation_pair_layout_cache.py` | VERIFIED-FRESH |Fresh default10/10 on L4 with exact full reference/repeat traces; optional shared-constructor controls2/2 on L4/H100,20case observations perhost. Evidence: reports/pair_layout_default_after_shared.json and shared_layout_controls_shared_*.json. |
| `probes/innovation_position_physics.py` | VERIFIED-FRESH |Original7/7 receipt retained; prepared/island/compact/shared modes9/9 each on L4/H100, full baseline traces and independent repeats exact; paired timings in the corresponding result tables. |
| `probes/innovation_prepared_planar.py` | VERIFIED-FRESH | Two independent reversed-order workers,8/8 unchanged gates per candidate. |
| `probes/innovation_prepared_profile_v1.py` | VERIFIED-FRESH |5/5 full-array repeat, observer parity and synchronized accounting gates. |
| `probes/innovation_prepared_solve.py` | OWN-GATE-FAIL |7/8 unchanged paired gates; no-regression failure in both workers. |
| `probes/innovation_raw_layout_observer.py` | VERIFIED-FRESH |4/4, full raw arrays repeat, independent reference parity;15/36 exact transitions and0duplicate keys. |
| `probes/innovation_sampled_field_contact.py` | VERIFIED-FRESH |8/8 in two reversed independent workers; query error0, full repeats exact,8field refusals per worker. |
| `probes/innovation_sampled_field_mechanism.py` | OWN-GATE-FAIL |3/4; two full repeats exact, affine error<1e-12, coarse EDT support sign wrong at+0.05. |
| `probes/innovation_stack_ablation.py` | OWN-GATE-FAIL | Eight cases, two observed-state trajectories each: finite and byte-identical; all eight unchanged K16 geometry gates FAIL. |
| `probes/innovation_stack_applied_warm.py` | OWN-GATE-FAIL | L4 40-budget5/6 gates PASS; original speed FAIL.20-budget also fails M5. |
| `probes/innovation_stack_coarse_probe.py` | OWN-GATE-FAIL | Two-run candidate measurement; K16 geometry FAIL, wall 5.079/5.007 ms FAIL; negative results retained. |
| `probes/innovation_stack_color_cache.py` | VERIFIED-FRESH | L4 11/11 gates across two independent workers; all original physical/parity/speed bounds unchanged. |
| `probes/innovation_stack_cone_conditioning.py` | VERIFIED-FRESH | L4 four measurement gates PASS; exact operator/result repeats and unchanged invalid M5; positive-spectrum Gram condition reaches4.2444684e17. |
| `probes/innovation_stack_cone_shadow.py` | OWN-GATE-FAIL | L4 exact repeats/unchanged M5 and cone/energy gates PASS; three/five snapshots fail1e-8 residual at40000 iterations, max2.556073e-6. |
| `probes/innovation_stack_cone_split_shadow.py` | OWN-GATE-FAIL | L4 exact repeats and unchanged M5; four/five gates PASS, projected residual FAIL3/5, max1.115502932869e-4 at40000 iterations. |
| `probes/innovation_stack_deferred_force.py` | OWN-GATE-FAIL | L4 8/9 gates PASS; original speed fails, day-plan4.9ms passes. |
| `probes/innovation_stack_fixed_frame.py` | OWN-GATE-FAIL | L4: 3/5 gates pass; K16 and original speed fail with values above. |
| `probes/innovation_stack_linear_probe.py` | SYNTHETIC-ONLY | 4/4 instrument gates; two reports byte-identical; K16 force residual 0.8358915691 at 40 sweeps (diagnostic target fails). Does not certify the contact engine. |
| `probes/innovation_stack_normal_nnls_probe.py` | OWN-GATE-FAIL | Two complete result dictionaries identical; M5 FAIL, M1 and box force PASS; no performance samples. |
| `probes/innovation_stack_normal_space.py` | VERIFIED-FRESH | Five snapshots twice bit-identical, finite, exact baseline M5; full/mean rank 63/16 at calls 100 and 590. |
| `probes/innovation_stack_probe.py` | OWN-GATE-FAIL | Recovered baseline: two trajectories byte-identical; K16 pen/R 1.412588655949 at40 FAIL, 0.276366770267 at160 PASS; wall 4.922/4.878 ms FAIL <=4. Historical runtime blockers retained below. |
| `probes/innovation_stack_sweep_residual.py` | VERIFIED-FRESH | L4 observer5/5, exact repeated post-solve states and frozen M5 parity. Original40iteration failure/160iteration pass are observed baseline results, not this module certifying a new solver. |
| `probes/innovation_stack_tangent_coupling.py` | VERIFIED-FRESH | L4 observer5/5 with exact repeats and frozen M5 parity; call590 coupling0.918719, induced tangent norm0.280669. Baseline K16 physical failure is retained, not a failed observer gate. |
| `probes/innovation_stack_two_level_probe.py` | SYNTHETIC-ONLY | Two full runs byte-identical; K16/40 residual: alternating 0.861427127555 FAIL, two-level 1.77635683940e-15 PASS with 40 extra coarse solves. |
| `probes/innovation_stack_warm_mechanism.py` | VERIFIED-FRESH | L4: 4/4 observer gates; cached normal sum4872.69443429 with exactly zero velocity change at call100; two full repeats. |
| `probes/innovation_unfused_physics.py` | VERIFIED-FRESH |7/7 original physical gates; no cross-device or submillisecond claim. |
| `probes/innovation_wide_grid_engine_probe.py` | OWN-GATE-FAIL | L4 two-run physics/state hashes exact vs baseline; full-engine wall improved; K16/40 and original4ms gates remain FAIL. |
| `probes/massratio_baseline_probe.py` | VERIFIED-FRESH |4/4; both200step ratios reproduce archived full states exactly; ratio1000 collapse precedes step57 explosion. |
| `probes/midpoint_sigma_probe.py` | OWN-GATE-FAIL |4/5; free-fall raw order-1.6503603811, two complete traces exact; near-exact errors do not establish convergence order. |
| `probes/motion_matrix_modal_v1.py` | OWN-GATE-FAIL |3/4 four-device gates;10/12 exact rows; RTX5070 pair/ColorCache deltas retained. |
| `probes/outclass_rescore_probe.py` | VERIFIED-FRESH | 5/5 observer gates, two exact complete physical histories per configuration; individual mass-ratio/order failures retained in reports/outclass_rescore_probe_l4.json. |
| `probes/probe_friction_grip.py` | VERIFIED-FRESH |
| `probes/persistent_context_probe.py` | VERIFIED-FRESH |10/10;100step cache hits100/100, rebuilds/captures0, churn0; plain2.48736614/2.47570147ms, exact complete repeats. |
| `probes/body_pair_scale_probe.py` | VERIFIED-FRESH |4/4; nonemptyN10000/100000 pair lists exact versusCPU and repeated full arrays; L4 rebuild0.492/0.462 and0.762/0.768ms. |
| `probes/body_pair_rt_probe.py` | OWN-GATE-FAIL |4/5; full-rebuild speed gate fails, both isolated public-output captures exact; no solver adoption. |
| `probes/body_pair_rt_boundary_probe.py` | VERIFIED-FRESH |3/3;12exact padded-boundary cases per worker;120arrays/3224bytes and fullJSON identical under memory check, exit0. |
| `probes/substep_pareto_probe.py` | VERIFIED-FRESH |6/6;20configurations exact twice and archivedS1/V40 trace exact;4ms selectsK16 S2/V16 pen/R0.140916407,worst3.958035ms; N10000 selectsS1/V40,worst3.602970ms; no valid row<=2ms. |
| `probes/verification_contract_probe.py` | VERIFIED-FRESH |8/8;24deliberate metadata/report/process controls plus positive and repeated isolation cases; full reports exact. |
