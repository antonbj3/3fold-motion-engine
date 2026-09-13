"""motion_engine.manip — the manipulation-suite library.

Modules:
  report   — run_meta() / write_report(): reproducibility metadata.
  profiles — data/robot_profiles.yaml -> RobotProfile (yaml + numpy).
  rplant   — plant randomisation over a replicated world (numpy).
  scene    — scene builder (arm + end-effector sphere + pedestal + cube + replicate + plant
             randomisation; grasp weld / build_pick_scene for the contact-pick cell).
  grip     — gripper open/close and the weld/release transitions.
  servo    — the servo loop (torch + warp kernels).
  sensors  — sensor readout for the manipulation cells.

scene, grip, servo and sensors require the `newton` physics package and `warp`; report, profiles and
rplant are pure numpy/yaml.
"""
