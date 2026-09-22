"""Isolated reproduction run for the `contact-scene-v1` package.

Under a file-open audit hook that forbids reads outside this package and the
interpreter/system paths, run the standalone loader and the full verification.
Prints every import that was resolved from a file, so a dependence on an internal
build module would be visible.

    python iso_run.py
"""
from __future__ import annotations

import json
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ALLOW = tuple(dict.fromkeys([
    ROOT, sys.prefix, sys.base_prefix,
    os.path.dirname(os.__file__),
    "/usr", "/lib", "/lib64", "/proc", "/sys", "/dev", "/etc", "/run",
    "/tmp",
]))
blocked = []


def hook(event, args):
    if event != "open":
        return
    path = args[0]
    if not isinstance(path, (str, bytes)):
        return
    p = os.path.abspath(os.fsdecode(path))
    if any(p == a or p.startswith(a + os.sep) or p.startswith(a + "/")
           for a in ALLOW):
        return
    blocked.append(p)
    raise PermissionError(f"blocked read outside package: {p}")


def imported_files():
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f:
            out[name] = os.path.abspath(f)
    return out


sys.addaudithook(hook)
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def run_main(path):
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise


run_main(os.path.join(ROOT, "loader.py"))
sys.argv = ["verify.py", "--all"]
run_main(os.path.join(ROOT, "verify.py"))

outside = {k: v for k, v in imported_files().items()
           if not (v.startswith(ROOT + os.sep))}
print("ISOLATION OK; blocked reads:", len(blocked), blocked)
print("IMPORTED_OUTSIDE_PACKAGE_JSON", json.dumps(outside, sort_keys=True))
