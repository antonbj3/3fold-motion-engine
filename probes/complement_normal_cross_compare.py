"""Render a reusable, integrity-checked contact determinism matrix on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from complement_normal_state_probe import ENGINES, SCENES, fingerprint


def matrix(directory, labels):
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError('Require at least two distinct backend labels')
    saved = {}
    hardware = {}
    for label in labels:
        if not label.replace('_', '').isalnum():
            raise ValueError('Simple artifact labels required')
        report = json.loads((directory / f'cross_complement_{label}.json').read_text())
        archive = directory / f'cross_complement_{label}.npz'
        if hashlib.sha256(archive.read_bytes()).hexdigest() != report['archive_sha256']:
            raise ValueError('Archive checksum mismatch: ' + label)
        if not all(report['gates'].values()):
            raise ValueError('Capture gates failed: ' + label)
        expected = {(e, s) for e in ENGINES for s in SCENES}
        if {(r['engine'], r['scene']) for r in report['records']} != expected:
            raise ValueError('Incomplete case coverage')
        with np.load(archive, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        for row in report['records']:
            for leg, receipt in enumerate(row['legs']):
                for field, digest in receipt['arrays'].items():
                    key = f"{row['engine']}__{row['scene']}__{leg}__{field}"
                    if fingerprint(arrays[key]) != digest:
                        raise ValueError('Array fingerprint mismatch: ' + key)
        saved[label] = (report, arrays)
        hardware[label] = report['hardware']
    rows = []
    reference_report, reference = saved[labels[0]]
    for engine in ENGINES:
        for label in labels:
            report, arrays = saved[label]
            differences = []
            maximum = 0.0
            for key, a in reference.items():
                if not key.startswith(engine + '__'):
                    continue
                b = arrays[key]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError('Incompatible contracts')
                if a.tobytes() != b.tobytes():
                    differences.append(key)
                if key.endswith('__xc'):
                    maximum = max(maximum, float(np.max(np.abs(a.astype(float) - b.astype(float)))))
            initial_exact = all(a['legs'][0]['initial'] == b['legs'][0]['initial']
                                for a, b in zip(reference_report['records'], report['records'])
                                if a['engine'] == engine)
            rows.append(dict(module=engine, backend=label, reference=labels[0],
                             max_position_delta=maximum, differing_arrays=differences,
                             initial_exact=initial_exact,
                             status='VERIFIED-FRESH' if initial_exact and not differences else 'OWN-GATE-FAIL'))
    return dict(scope='Hybrid normal correction, original ratio1/1000 fixtures only', hardware=hardware, rows=rows)


def render(result):
    lines = ['# Contact cross-hardware matrix', '', result['scope'], '',
             '| Module | Backend | Reference | max position delta | Status |',
             '|---|---|---|---|---|']
    for row in result['rows']:
        lines.append(f"| {row['module']} | {row['backend']} | {row['reference']} | {row['max_position_delta']:.17g} | {row['status']} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=Path('reports'))
    parser.add_argument('labels', nargs='+')
    args = parser.parse_args()
    first = matrix(args.reports, args.labels)
    second = matrix(args.reports, args.labels)
    if first != second:
        raise RuntimeError('Repeated matrix construction differs')
    (args.reports / 'complement_normal_matrix.json').write_text(json.dumps(first, indent=2) + '\n')
    (args.reports / 'complement_normal_matrix.md').write_text(render(first))
    print(render(first), end='')
    return 0 if all(row['status'] == 'VERIFIED-FRESH' for row in first['rows']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
