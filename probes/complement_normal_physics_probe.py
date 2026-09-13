"""Original M4 fixtures for the bounded hybrid normal-coupling candidate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


class Observed:
    def __init__(self, engine):
        self.engine = engine
        self.sha = hashlib.sha256()
        self.finite = True
        self.peak_center = 0.
        self.peak_speed = 0.
        self.rounds = []
        self.steps = 0

    def __getattr__(self, key):
        return getattr(self.engine, key)

    def step(self, *args, **kwargs):
        self.engine.step(*args, **kwargs)
        state = self.engine.get_state()
        for value in [getattr(state, key) for key in ('xc', 'Rm', 'vc', 'om')]+[self.engine.contact_forces()]:
            value = np.ascontiguousarray(value)
            self.sha.update(value.tobytes())
            self.finite &= bool(np.isfinite(value).all())
        self.peak_center = max(self.peak_center, float(np.max(np.abs(state.xc))))
        self.peak_speed = max(self.peak_speed, float(np.max(np.abs(state.vc))))
        self.rounds.append([self.engine.normal_solves, self.engine.jacobi_rounds])
        self.steps += 1

    def receipt(self):
        return dict(trace_sha256=self.sha.hexdigest(), finite=self.finite,
                    peak_center=self.peak_center, peak_speed=self.peak_speed,
                    rounds=self.rounds, steps=self.steps)


def worker():
    from motion_engine.contact_engine_gpu_complement_normal import ComplementNormalContactEngine
    from engine_metrics import m4_iters_to_tol, DIMS
    rows = []
    for ratio in (1, 1000):
        observed = []
        def factory(**kwargs):
            engine = ComplementNormalContactEngine(dims=DIMS, pit=10, mu=.5, **kwargs)
            if engine.dev != 'cuda:0':
                raise RuntimeError('CUDA required')
            result = Observed(engine)
            observed.append(result)
            return result
        try:
            metric = m4_iters_to_tol(factory, 'gpu', ratio, [40], 200)
            error = None
        except Exception as exception:
            metric = None
            error = type(exception).__name__+': '+str(exception)
        rows.append(dict(ratio=ratio, metric=metric, error=error,
                         observations=[e.receipt() for e in observed]))
    print('RESULT '+json.dumps(rows, allow_nan=False))


def main():
    if sys.argv[1:] == ['--worker']:
        worker()
        return 0
    legs = []
    for _ in range(2):
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                           capture_output=True, text=True, check=True)
        lines = [line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT ')]
        if len(lines) != 1:
            raise RuntimeError('Expected one receipt')
        legs.append(json.loads(lines[0]))
        print('Completed independent leg '+str(len(legs)), flush=True)
    rows = [row for leg in legs for row in leg]
    observations = [o for row in rows for o in row['observations']]
    gates = dict(exact_repeats=legs[0] == legs[1],
                 coverage=len(rows) == 4 and all(row['error'] is None and len(row['observations']) == 1
                                                and row['observations'][0]['steps'] == 200 for row in rows),
                 original_m4=all(row['metric'] is not None and row['metric']['valid'] and
                                 row['metric']['residual_PRIMARY'] < .10 and row['metric']['iters_to_tol_PRIMARY'] == 40
                                 for row in rows),
                 finite=bool(observations) and all(o['finite'] for o in observations),
                 stable=bool(observations) and all(o['peak_center'] < 2 and o['peak_speed'] < 5 for o in observations),
                 budget=bool(observations) and all(sum(rounds) <= 40 for o in observations for rounds in o['rounds']))
    report = dict(legs=legs, gates=gates, status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
                  passed=sum(gates.values()), total=len(gates), scope='Hybrid CPU normal coupling and GPU Jacobi; same-device M4 only.')
    (ROOT/'reports/complement_normal_physics_probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(gates=gates, rows=[dict(ratio=row['ratio'], metric=row['metric'], error=row['error']) for row in rows]), indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
