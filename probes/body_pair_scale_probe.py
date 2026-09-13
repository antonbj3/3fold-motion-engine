"""Measure the unchanged broad phase on nonempty fixtures before RT design."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'probes'), str(ROOT/'scripts'), str(ROOT/'src')]
import json
import subprocess
import tempfile
import time
import numpy as np
from scipy.spatial import cKDTree
from cross_hardware_state_probe import fingerprint
from motion_engine.contact_engine_gpu import _voxbox
from motion_engine.contact_body_pairs_gpu import BodyPairSearch


def fixtures():
    beads = _voxbox(.3, .3, .2).astype(np.float32)
    for n in (10000, 100000):
        side = int(np.ceil(np.sqrt(n/2)))
        index = np.arange(n)
        centers = np.column_stack(((index % side)*.35,
                                   ((index//side) % side)*.35,
                                   .0999+(index//(side*side))*.1999)).astype(np.float32)
        points = (centers[:, None, :]+beads[None, :, :]).reshape(-1, 3)
        yield str(n), points, len(beads), side*side


def reference(lower, upper):
    center = (lower+upper)*.5
    radius = float(np.max(np.linalg.norm((upper-lower)*.5, axis=1)))
    candidates = cKDTree(center).query_pairs(np.nextafter(2*radius, np.inf), output_type='ndarray')
    a, b = candidates.T
    pairs = candidates[np.all(lower[a]<=upper[b], axis=1) & np.all(lower[b]<=upper[a], axis=1)]
    ground = np.flatnonzero(lower[:, 2]<0)
    pairs = np.concatenate((pairs, np.column_stack((ground, np.full(len(ground), -1)))))
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))].astype(np.int32), len(candidates)


def worker(reverse, output):
    import warp as wp
    wp.init()
    if wp.config.version!='1.13.0' or np.__version__!='2.5.3':
        raise RuntimeError('Pinned environment required')
    rows = {}; saved = {}
    cases = list(fixtures())
    for name, points, width, ground_count in (cases[::-1] if reverse else cases):
        n = len(points)//width
        gpu = wp.array(points, dtype=wp.vec3, device='cuda:0')
        search = BodyPairSearch(wp, n, width)
        for _ in range(3): search.rebuild(gpu)
        wp.synchronize()
        begin = time.perf_counter()
        for _ in range(20): search.rebuild(gpu)
        wp.synchronize()
        elapsed = (time.perf_counter()-begin)*1000/20
        lo, hi = search.lower.numpy(), search.upper.numpy()
        actual = search.pairs.numpy()[:search.number].copy()
        oracle, visits = reference(lo, hi)
        arrays = dict(points=points, lower=lo, upper=hi, pairs=actual, oracle=oracle)
        saved.update({name+'__'+k:v for k,v in arrays.items()})
        rows[name] = dict(bodies=n, width=width, points=len(points), pairs=len(actual),
            ground_pairs=int(np.sum(actual[:, 1]<0)), expected_ground=ground_count,
            expected_body_pairs=n-ground_count, sphere_candidates=visits,
            exact_reference=actual.tobytes()==oracle.tobytes() and actual.shape==oracle.shape,
            finite=all(bool(np.isfinite(v).all()) for v in arrays.values()),
            hashes={k:fingerprint(v) for k,v in arrays.items()}, rebuild_ms=elapsed)
        print(name+' complete', flush=True)
    np.savez_compressed(output, **saved)
    print('RESULT '+json.dumps(rows), flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        worker(sys.argv[2]=='1', sys.argv[3]); return 0
    rows=[]; captures=[]
    with tempfile.TemporaryDirectory() as temp:
        for leg in range(2):
            output=Path(temp)/f'{leg}.npz'
            p=subprocess.run([sys.executable, __file__, '--worker', str(leg), str(output)],
                             cwd=ROOT, capture_output=True, text=True, timeout=240)
            if p.returncode: raise RuntimeError(p.stdout+p.stderr)
            rows.append(json.loads(next(line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT '))))
            with np.load(output) as data: captures.append({k:data[k].copy() for k in data.files})
            print('leg '+str(leg)+' complete', flush=True)
    def fixed(row):
        return {k:{a:b for a,b in v.items() if a!='rebuild_ms'} for k,v in row.items()}
    values=[v for r in rows for v in r.values()]
    gates=dict(exact_reference=all(v['exact_reference'] for v in values),
        exact_repeats=fixed(rows[0])==fixed(rows[1]) and captures[0].keys()==captures[1].keys() and all(captures[0][k].tobytes()==captures[1][k].tobytes() for k in captures[0]),
        nonempty_expected_pairs=all(v['pairs']==v['bodies'] and v['ground_pairs']==v['expected_ground'] and v['expected_body_pairs']>0 for v in values),
        finite_complete=len(values)==4 and all(v['finite'] and np.isfinite(v['rebuild_ms']) and v['rebuild_ms']>0 for v in values))
    np.savez_compressed(ROOT/'reports/body_pair_scale_probe.npz', **{str(i)+'__'+k:v for i,c in enumerate(captures) for k,v in c.items()})
    report=dict(gates=gates, results=rows, scope='Full unchanged hash-grid rebuild; synthetic contacted two-layer fixtures; no dynamics or RT speed claim.')
    (ROOT/'reports/body_pair_scale_probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__': raise SystemExit(main())
