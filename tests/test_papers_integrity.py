"""The four papers and the reproduction-bundle manifests, byte for byte.

    python -m pytest -q tests/test_papers_integrity.py

A PDF is a build artefact of text this repository does not carry, so the only
thing that can be locked here is that the committed file is the file that was
measured. The checksum file is the one the build wrote. Each reproduction bundle
ships its own manifest; this test confirms every listed member is present and
matches its recorded full SHA-256, so a missing or altered bundle file fails
here rather than at reproduction time.
"""
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "docs/papers"
SUMS = [l.split() for l in (PAPERS / "SHA256SUMS").read_text().split("\n") if l.strip()]

BUNDLES = {
    "p1": "SHA256SUMS",
    "p2": "SHA256SUMS",
    "p3": "MANIFEST.sha256",
    "p4": "MANIFEST.sha256",
}


def _manifest_entries(bundle, name):
    text = (ROOT / "reproducibility" / bundle / name).read_text()
    entries = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(None, 1)
        entries.append((digest, rel.strip()))
    return entries


def test_the_checksum_file_lists_four_papers():
    names = sorted(name for _, name in SUMS)
    assert names == ["P1.pdf", "P2.pdf", "P3.pdf", "P4.pdf"]


@pytest.mark.parametrize("digest,name", SUMS, ids=[n for _, n in SUMS])
def test_pdf_matches_its_recorded_digest(digest, name):
    path = PAPERS / name
    assert path.is_file(), f"{name} is listed in SHA256SUMS but not committed"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert path.read_bytes()[:5] == b"%PDF-"
    assert path.stat().st_size > 100_000


@pytest.mark.parametrize("bundle,name", sorted(BUNDLES.items()))
def test_bundle_manifest_is_complete_and_correct(bundle, name):
    root = (ROOT / "reproducibility" / bundle).resolve()
    entries = _manifest_entries(bundle, name)
    assert entries, f"{bundle}/{name} lists no files"
    for digest, rel in entries:
        assert len(digest) == 64, f"{bundle}: malformed digest for {rel}"
        target = (root / rel).resolve()
        assert root in target.parents or target == root, f"{bundle}: unsafe path {rel}"
        assert target.is_file(), f"{bundle}: {rel} is listed but missing"
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, (
            f"{bundle}: {rel} does not match its recorded digest"
        )
