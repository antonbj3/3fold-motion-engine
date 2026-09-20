"""The two papers, byte for byte against their own checksum file.

    python -m pytest -q tests/test_papers_integrity.py

A PDF is a build artefact of text this repository does not carry, so the only thing
that can be locked here is that the file committed is the file that was measured. The
checksum file is the one the build wrote.
"""
import hashlib
from pathlib import Path

import pytest

PAPERS = Path(__file__).resolve().parents[1] / "docs/papers"
SUMS = [l.split() for l in (PAPERS / "SHA256SUMS").read_text().split("\n") if l.strip()]


def test_the_checksum_file_lists_both_papers():
    names = sorted(name for _, name in SUMS)
    assert names == ["P1.pdf", "P2.pdf"]


@pytest.mark.parametrize("digest,name", SUMS, ids=[n for _, n in SUMS])
def test_pdf_matches_its_recorded_digest(digest, name):
    path = PAPERS / name
    assert path.is_file(), f"{name} is listed in SHA256SUMS but not committed"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert path.read_bytes()[:5] == b"%PDF-"
    assert path.stat().st_size > 1_000_000
