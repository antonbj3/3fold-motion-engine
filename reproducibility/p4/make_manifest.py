#!/usr/bin/env python3
"""Regenerate the public release SHA-256 manifest after reproduction."""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.sha256"


def main() -> None:
    entries = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            raise RuntimeError(f"cache in public release: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  ./{path.relative_to(HERE).as_posix()}")
    MANIFEST.write_text("\n".join(entries) + "\n")
    print(f"manifest entries: {len(entries)}")


if __name__ == "__main__":
    main()
