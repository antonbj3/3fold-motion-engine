"""Pytest wrapper for the cross-repo compose example (field SDF -> contacts -> graph node).

Skips when the sibling repositories are not on THREEFOLD_ROOT (default: the parent of this repository),
so the suite of this repository alone is unaffected.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
THREEFOLD_ROOT = Path(os.environ.get("THREEFOLD_ROOT", ROOT.parent)).resolve()
EXAMPLE = ROOT / "examples" / "compose" / "field_sdf_to_contacts_to_graph.py"
SIBLINGS = [THREEFOLD_ROOT / "3fold-field-engine", THREEFOLD_ROOT / "3fold-graph-engine"]


@pytest.mark.skipif(not all(p.is_dir() for p in SIBLINGS),
                    reason="sibling repositories (field, graph) not present under THREEFOLD_ROOT")
def test_compose_field_sdf_to_contacts_to_graph(tmp_path):
    r = subprocess.run([sys.executable, str(EXAMPLE), "--out", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "stage results: field=PASS  motion=PASS  graph=PASS" in out, out[-3000:]
