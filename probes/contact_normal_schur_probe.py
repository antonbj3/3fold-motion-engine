"""Check a bounded normal quadratic solve on retained contact snapshots."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from motion_engine.contact_normal_schur import solve_normal


def system(snapshot):
    bodies = {k: np.asarray(v) for k, v in snapshot['bodies'].items()}
    contacts = {k: np.asarray(v) for k, v in snapshot['contacts'].items()}
    count = len(bodies['xc'])
    jacobian = np.zeros((len(contacts['cbi']), 6*count))
    inverse = np.zeros((6*count, 6*count))
    for i in range(count):
        inverse[6*i:6*i+3, 6*i:6*i+3] = np.eye(3)*bodies['invM'][i]
        inverse[6*i+3:6*i+6, 6*i+3:6*i+6] = bodies['invIw'][i]
    for k, (i, j) in enumerate(zip(contacts['cbi'], contacts['cbj'])):
        normal = contacts['cn'][k]
        jacobian[k, 6*i:6*i+3] = normal
        jacobian[k, 6*i+3:6*i+6] = np.cross(contacts['cpA'][k]-bodies['xc'][i], normal)
        if j >= 0:
            jacobian[k, 6*j:6*j+3] = -normal
            jacobian[k, 6*j+3:6*j+6] = -np.cross(contacts['cpB'][k]-bodies['xc'][j], normal)
    velocity = np.concatenate([np.r_[bodies['v'][i], bodies['w'][i]] for i in range(count)])
    operator = jacobian@inverse@jacobian.T
    operator = .5*(operator+operator.T)
    return jacobian, inverse, velocity, operator


def worker():
    capture = json.loads((ROOT/'reports/massratio_baseline_probe.json').read_text())
    if not all(capture['gates'].values()):
        raise RuntimeError('Unverified contact observer')
    rows = []
    for row in capture['legs'][0]:
        j, inverse, velocity, operator = system(row['first_coupled'])
        b = j@velocity
        solution = solve_normal(operator, b)
        impulse = np.asarray(solution['impulses'])
        oracle = np.linalg.lstsq(operator, -b, rcond=None)[0]
        projected = velocity+inverse@j.T@impulse
        reference = velocity+inverse@j.T@oracle
        rows.append(dict(ratio=row['ratio'], solution=solution, projected=projected.tolist(),
                         oracle_impulse_min=float(np.min(oracle)),
                         oracle_velocity_delta=float(np.max(np.abs(projected-reference))),
                         rank=int(np.linalg.matrix_rank(operator)), rows=len(b)))
    duplicate = solve_normal([[1., 1.], [1., 1.]], [-1., -1.])
    inactive = solve_normal([[1.]], [1.])
    refusals = []
    for a, b, limit in (([[1., -1.], [-1., 1.]], [-1., -1.], 40),
                        ([[1., 0.], [0., 1.]], [-1., -1.], 1),
                        ([[float('nan')]], [-1.], 40),
                        ([[1., 2.], [0., 1.]], [-1., -1.], 40)):
        try:
            solve_normal(a, b, max_solves=limit)
        except ValueError:
            refusals.append(True)
        else:
            refusals.append(False)
    print(json.dumps(dict(rows=rows, duplicate=duplicate, inactive=inactive, refusals=refusals), allow_nan=False))


def main():
    if sys.argv[1:] == ['--worker']:
        worker()
        return 0
    legs = []
    for _ in range(2):
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                           cwd=ROOT, capture_output=True, text=True, check=True)
        legs.append(json.loads(p.stdout))
    gates = dict(exact_repeats=legs[0] == legs[1],
                 kkt=all(r['solution']['kkt'] <= 1e-9 for leg in legs for r in leg['rows']),
                 independent_oracle=all(r['oracle_impulse_min'] >= 0 and r['oracle_velocity_delta'] <= 1e-9
                                        for leg in legs for r in leg['rows']),
                 controls=all(leg['duplicate']['kkt'] == 0 and leg['inactive']['impulses'] == [0.] for leg in legs),
                 refusals=all(all(leg['refusals']) for leg in legs),
                 budget=all(r['solution']['solves'] <= 40 for leg in legs for r in leg['rows']))
    report = dict(legs=legs, gates=gates, status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
                  scope='Static normal-contact snapshots only; no dynamics, friction or GPU convergence certificate.')
    (ROOT/'reports/contact_normal_schur_probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
