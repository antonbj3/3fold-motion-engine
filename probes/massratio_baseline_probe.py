"""Read-only launch observer for the unchanged high-mass-ratio baseline."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def worker():
    from outclass_rescore_probe import factories, state_hash
    import warp as wp
    reference = json.loads((ROOT/'reports/outclass_rescore_probe_l4.json').read_text())['legs'][0]['jacobi']
    rows = []
    for ratio, reference_index in ((1, 9), (1000, 16)):
        e = factories()['jacobi'](vit=40, pit=10)
        if e.dev != 'cuda:0':
            raise RuntimeError('CUDA required')
        e.add_body([0, 0, .101], density=700.)
        e.add_body([0, 0, .306], density=700.*ratio)
        e._build()
        records = []
        frames = []
        snapshot = None
        digest = hashlib.sha256()
        launch = wp.launch
        step = 0
        iteration = 0

        def observed(kernel, *args, **kwargs):
            nonlocal snapshot, iteration
            if kernel is e.K['jac_vel'] and snapshot is None:
                count = int(e.cnt.numpy()[0])
                bi, bj = e.cbi.numpy()[:count], e.cbj.numpy()[:count]
                if np.any(bj < 0) and np.any(bj >= 0):
                    fields = {key: getattr(e, key).numpy()[:count].copy()
                              for key in ('cbi', 'cbj', 'cpA', 'cpB', 'cn', 'cpen')}
                    order = sorted(range(count), key=lambda k: tuple(np.concatenate([
                        fields[key][k].reshape(-1) for key in fields])))
                    snapshot = dict(step=step, contacts={key: value[order].tolist() for key, value in fields.items()},
                                    bodies={key: getattr(e, key).numpy().tolist()
                                            for key in ('xc', 'v', 'w', 'invM', 'invIw')})
            peak = None
            if kernel is e.K['apply']:
                peak = max(float(np.max(np.abs(a.numpy().astype(np.float64)))) for a in e.acc)
            result = launch(kernel, *args, **kwargs)
            if kernel is e.K['apply']:
                inputs = kwargs['inputs']
                kind = 'velocity' if inputs[0] is e.v else 'position'
                linear, angular = inputs[0].numpy(), inputs[1].numpy()
                records.append(dict(step=step, iteration=iteration, kind=kind,
                                    linear_max=float(np.max(np.abs(linear))),
                                    angular_max=float(np.max(np.abs(angular))), accumulator_max=peak))
                iteration += 1
            return result

        wp.launch = observed
        try:
            for step in range(200):
                iteration = 0
                e.step(1/240, substeps=1)
                value, finite = state_hash(e)
                digest.update(bytes.fromhex(value))
                force = e.contact_forces()
                state = e.get_state()
                count = int(e.cnt.numpy()[0])
                bj = e.cbj.numpy()[:count]
                frames.append(dict(step=step, xyz=state.xc.tolist(), finite=finite,
                                   velocity_max=float(np.max(np.abs(state.vc))),
                                   force_residual=float(np.max(np.abs(force[:, 2]-e._M*9.81)/(e._M*9.81))),
                                   ground_contacts=int(np.count_nonzero(bj < 0)),
                                   body_contacts=int(np.count_nonzero(bj >= 0))))
        finally:
            wp.launch = launch
        trace = digest.hexdigest()
        rows.append(dict(ratio=ratio, trace_sha256=trace,
                         reference_parity=trace == reference['observations'][reference_index]['sha256'],
                         frames=frames, iterations=records, first_coupled=snapshot))
    print('RESULT '+json.dumps(rows, allow_nan=False))


def main():
    if sys.argv[1:] == ['--worker']:
        worker()
        return 0
    if sys.argv[1:]:
        raise ValueError('Expected no arguments or --worker')
    legs = []
    for _ in range(2):
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                           capture_output=True, text=True)
        if p.returncode:
            print(p.stdout); print(p.stderr)
            return p.returncode
        lines = [line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT ')]
        if len(lines) != 1:
            raise RuntimeError('Expected one worker receipt')
        legs.append(json.loads(lines[0]))
        print('Completed independent leg '+str(len(legs)), flush=True)
    gates = dict(exact_repeats=legs[0] == legs[1],
                 finite=all(frame['finite'] for leg in legs for row in leg for frame in row['frames']),
                 coverage=all([row['ratio'] for row in leg] == [1, 1000] and
                              all(len(row['frames']) == 200 and row['first_coupled'] is not None for row in leg)
                              for leg in legs),
                 reference_parity=all(row['reference_parity'] for leg in legs for row in leg))
    report = dict(legs=legs, gates=gates, status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL')
    (ROOT/'reports/massratio_baseline_probe.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(gates))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
