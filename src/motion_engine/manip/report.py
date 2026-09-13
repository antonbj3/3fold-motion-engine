"""Run metadata and report writing for the manipulation probes.

run_meta() is side-effect free and stdlib-only: it reads the date, the git revision and dirty flag, the
environment overrides and the simulator version, and never raises into the measuring script (every
external call is guarded). Scripts simply put `meta=run_meta()` into their existing report dict.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Env-variabler som styr manipulationssviten (auditens lista + de faktiskt
# grep-belagda i scripts/newton_*.py). Prefix-poster matchar FG_*/PICK_*/PUSH_*.
_ENV_VARS = ("IMPRATIO", "OBJ_NOISE_MM", "SENSOR_NOISE", "ROBOT",
             "PLANT_SEED", "MU_FINGER", "PAYLOAD_KG")
_ENV_PREFIXES = ("FG_", "PICK_", "PUSH_")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _newton_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("newton")
    except Exception:
        return None


def env_overrides() -> dict:
    """All suite-controlling environment variables that are actually set (an unset variable means the
    script default applied and is not recorded)."""
    ov = {k: v for k, v in os.environ.items()
          if k in _ENV_VARS or k.startswith(_ENV_PREFIXES)}
    return dict(sorted(ov.items()))


def run_meta(**extra) -> dict:
    """Reproducibility metadata for a measurement run.

    Contents: date (UTC ISO), script, git revision and dirty flag, robot/W/plant_seed (from the
    environment when set; scripts may override via kwargs), environment overrides and the simulator
    version. **extra is merged on top."""
    env = env_overrides()
    meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": Path(sys.argv[0]).name if sys.argv else None,
        "git_rev": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "robot": env.get("ROBOT") or env.get("FG_ROBOT"),
        "W": next((int(env[k]) for k in ("PICK_W", "FG_W", "PUSH_W")
                   if k in env), None),
        "plant_seed": int(env["PLANT_SEED"]) if "PLANT_SEED" in env else None,
        "env_overrides": env,
        "newton_version": _newton_version(),
        "python": sys.version.split()[0],
    }
    meta.update(extra)
    return meta


def write_report(path, criteria: dict, metrics: dict, meta: dict | None = None) -> Path:
    """Write a standardised report: {criteria, metrics, meta}.

    `path` may be relative (resolved against the repository root). Returns the written Path."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {"criteria": criteria, "metrics": metrics,
         "meta": meta if meta is not None else run_meta()}, indent=1))
    return p
