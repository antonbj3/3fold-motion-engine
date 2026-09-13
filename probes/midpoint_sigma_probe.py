"""Measure the retained free-fall fixture on the existing midpoint backend."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def worker():
    from motion_engine.midpoint_vbd_engine import MidpointVBDEngine
    rows = []
    for steps in (60, 120):
        engine = MidpointVBDEngine(ramp_deg=0, mu=.5)
        engine.add_body(.3, .3, .2, [0, 0, 10.])
        digest = hashlib.sha256()
        finite = True
        force_max = 0.
        for _ in range(steps):
            engine.step(.25 / steps, substeps=1)
            state = engine.get_state()
            force = engine.contact_forces()
            arrays = [getattr(state, key) for key in ('xc', 'Rm', 'vc', 'om')] + [force]
            for array in arrays:
                array = np.ascontiguousarray(array)
                digest.update(array.tobytes())
                finite &= bool(np.isfinite(array).all())
            force_max = max(force_max, float(np.max(np.abs(force))))
        z = float(state.xc[0, 2])
        v = float(state.vc[0, 2])
        rows.append(dict(steps=steps, z=z, v=v,
                         position_error=abs(z - (10 - .5 * 9.81 * .25**2)),
                         energy_relative=(9.81*z + .5*v*v - 9.81*10)/(9.81*10),
                         force_max=force_max, finite=finite,
                         full_trace_sha256=digest.hexdigest()))
    errors = [row['position_error'] for row in rows]
    order = float(np.log2(errors[0] / errors[1])) if min(errors) > 0 else None
    print(json.dumps(dict(rows=rows, observed_order=order), allow_nan=False))


def main():
    if sys.argv[1:] == ['--worker']:
        worker()
        return 0
    if sys.argv[1:]:
        raise ValueError('Expected no arguments or --worker')
    legs = []
    for _ in range(2):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        legs.append(json.loads(result.stdout))
    gates = dict(exact_repeats=legs[0] == legs[1],
                 finite=all(row['finite'] for leg in legs for row in leg['rows']),
                 coverage=all([row['steps'] for row in leg['rows']] == [60, 120] for leg in legs),
                 no_contact=all(row['force_max'] == 0 for leg in legs for row in leg['rows']),
                 requested_order=all(leg['observed_order'] is not None and leg['observed_order'] >= 1.8 for leg in legs))
    report = dict(legs=legs, gates=gates, passed=sum(gates.values()), total=len(gates),
                  original_order_two=all(leg['observed_order'] is not None and leg['observed_order'] >= 2 for leg in legs),
                  status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
                  scope='Existing CPU midpoint only; constant-acceleration exactness does not measure general convergence order.')
    (ROOT / 'reports/midpoint_sigma_probe.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
