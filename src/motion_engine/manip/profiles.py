"""Robot profiles: reads data/robot_profiles.yaml into a RobotProfile (joint names, limits, gains,
end-effector link). Pure yaml + numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
PROFILES_YAML = ROOT / "data" / "robot_profiles.yaml"

# Keys that become named fields on RobotProfile; everything else (depth_win, n_desc,
# obj, maxk, settle_park, kp_y/kd_y, provenance ...) hamnar i .extras.
_CORE_KEYS = ("arm_pattern", "home", "tau_max", "kp_p", "kd_p", "ki_p",
              "kp_o", "kd_o", "fmax", "mmax", "ierr_clip")


@dataclass(frozen=True)
class RobotProfile:
    """Effective servo/scene constants for ONE (script, robot) cell.

    home/tau_max are np.ndarray (full length from the YAML; the consumer truncates
    with [:NJ]). ierr_clip is a float OR an np.ndarray
    (push-abb is a vector [x, y, z]); np.clip
    handles both forms unchanged.
    """
    script: str
    robot: str
    arm_pattern: str
    home: np.ndarray
    tau_max: np.ndarray
    kp_p: float
    kd_p: float
    ki_p: float
    kp_o: float
    kd_o: float
    fmax: float
    mmax: float
    ierr_clip: float | np.ndarray
    phases: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


def _load_yaml(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or PROFILES_YAML).read_text())


def load_profile(script: str, robot: str, path: Path | None = None) -> RobotProfile:
    """-> RobotProfile for (script, robot) from data/robot_profiles.yaml.

    `script` is the YAML key under scripts: (for example "newton_push", the chain's
    identitet, oberoende av vilket skriptfilnamn som konsumerar profilen).
    """
    data = _load_yaml(path)
    try:
        sd = data["scripts"][script]
    except KeyError:
        raise KeyError(f"unknown script {script!r} in {PROFILES_YAML.name} "
                       f"(finns: {sorted(data['scripts'])})") from None
    cell = sd.get("robots", {}).get(robot)
    if cell is None:
        cov = sd.get("coverage", {}).get(robot, "unknown")
        raise KeyError(
            f"{script} har ingen {robot}-profil i {PROFILES_YAML.name} "
            f"(coverage: {cov}) — legacy-fallbacken (*iiwa*) var latent "
            f"broken and is NOT ported; build and measure the branch first.")
    core = {}
    for k in _CORE_KEYS:
        if k not in cell:
            raise KeyError(f"{script}/{robot}: obligatorisk nyckel {k!r} "
                           f"saknas i {PROFILES_YAML.name}")
        core[k] = cell[k]
    extras = {k: v for k, v in cell.items() if k not in _CORE_KEYS}
    ierr = core["ierr_clip"]
    return RobotProfile(
        script=script, robot=robot,
        arm_pattern=core["arm_pattern"],
        home=np.array(core["home"], dtype=float),
        tau_max=np.array(core["tau_max"], dtype=float),
        kp_p=float(core["kp_p"]), kd_p=float(core["kd_p"]),
        ki_p=float(core["ki_p"]),
        kp_o=float(core["kp_o"]), kd_o=float(core["kd_o"]),
        fmax=float(core["fmax"]), mmax=float(core["mmax"]),
        ierr_clip=(np.array(ierr, dtype=float) if isinstance(ierr, list)
                   else float(ierr)),
        phases=dict(sd.get("phases", {})),
        extras=extras,
    )
