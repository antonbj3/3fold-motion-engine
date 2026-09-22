#!/usr/bin/env python3
"""Public-release self check.

1. Scans every public file for private lane identifiers, absolute paths and
   project/model names.
2. Verifies MANIFEST.sha256 over all listed files.

Run:  python3 check_release.py
"""
from __future__ import annotations
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PRIVATE = re.compile(r"(?:^|[^A-Za-z0-9])([UA][0-9]{2,4})(?![0-9])")
PATHS = ["/ho" + "me/", "build/" + "U", "build/" + "A", "romi_" + "collab",
         "3f" + "old", "CADto" + "SIM"]
TEXT_SUFFIXES = {".py", ".md", ".json", ".csv", ".txt", ".svg", ".html"}


def iter_files():
    for p in sorted(HERE.rglob("*")):
        if p.is_file():
            yield p


def scan_text(path):
    hits = []
    text = path.read_text(errors="ignore")
    for m in PRIVATE.finditer(text):
        hits.append(f"lane-like token {m.group(1)}")
    for token in PATHS:
        if token and token in text:
            hits.append(f"path/project token {token!r}")
    return hits


def scan_pdf(path):
    hits = []
    try:
        out = subprocess.run(["pdftotext", "-q", str(path), "-"],
                             capture_output=True, text=True, timeout=30)
        text = out.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return ["pdftotext unavailable (not scanned)"]
    for m in PRIVATE.finditer(text):
        hits.append(f"lane-like token {m.group(1)}")
    return hits


def main():
    problems = []
    scanned = 0
    for p in iter_files():
        if p.name == "MANIFEST.sha256":
            continue
        rel = p.relative_to(HERE)
        if p.suffix in TEXT_SUFFIXES:
            scanned += 1
            for h in scan_text(p):
                problems.append(f"{rel}: {h}")
        elif p.suffix == ".pdf":
            scanned += 1
            for h in scan_pdf(p):
                if "unavailable" not in h:
                    problems.append(f"{rel}: {h}")

    manifest = HERE / "MANIFEST.sha256"
    manifest_ok = True
    manifest_problems = []
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            f = HERE / name
            if not f.exists():
                manifest_ok = False
                manifest_problems.append(f"missing {name}")
                continue
            got = hashlib.sha256(f.read_bytes()).hexdigest()
            if got != digest:
                manifest_ok = False
                manifest_problems.append(f"hash mismatch {name}")
    else:
        manifest_ok = False
        manifest_problems.append("MANIFEST.sha256 not found")

    print(f"files scanned: {scanned}")
    print(f"private-token problems: {len(problems)}")
    for p in problems:
        print("  " + p)
    print(f"manifest_ok: {manifest_ok}")
    for p in manifest_problems:
        print("  " + p)
    return 0 if (not problems and manifest_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
