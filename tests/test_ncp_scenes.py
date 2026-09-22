"""The canonical contact-scene package loads and matches its HDF5 reference.

    python -m pytest -q tests/test_ncp_scenes.py

This is the lightweight check the P1 manuscript names: `data/ncp/loader.py`
rebuilds each operator from the scene JSON alone and compares it with the
packed HDF5 container. It reads only the repository paths, imports no project
build tree, and needs NumPy and h5py.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NCP = ROOT / "data/ncp"
LOADER = NCP / "loader.py"
SCENES = NCP / "scenes"
N_SCENES = 21


def _load_loader():
    pytest.importorskip("h5py")
    spec = importlib.util.spec_from_file_location("ncp_scene_loader", LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_package_rebuilds_every_operator():
    loader = _load_loader()
    manifest = json.loads((SCENES / "manifest.json").read_text())
    assert manifest["count"] == N_SCENES
    assert len(manifest["scenes"]) == N_SCENES

    names = []
    for i in range(N_SCENES):
        result = loader.verify(SCENES / f"{i:02d}.json", SCENES / f"{i:02d}.hdf5")
        assert result["G_rel_fro"] <= 1e-15, f"scene {i}: {result['G_rel_fro']}"
        assert result["b_abs"] <= 1e-15, f"scene {i}: {result['b_abs']}"
        assert result["mu_abs"] == 0.0, f"scene {i}: {result['mu_abs']}"
        names.append(result["scene"])

    assert len(set(names)) == N_SCENES, "scene aliases are not unique"
