"""Exercise hybrid-correction refusals without creating a GPU context."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from motion_engine.contact_engine_gpu_complement_normal import ComplementNormalContactEngine
from motion_engine.contact_normal_schur import solve_normal


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


def refused(call, text):
    try:
        call()
    except ValueError as error:
        return text in str(error)
    return False


def run():
    e = object.__new__(ComplementNormalContactEngine)
    e.N = 33
    body_capacity = refused(lambda: e._normal_project(0), 'capacity')
    e.N = 1
    contact_capacity = refused(lambda: e._normal_project(257), 'capacity')
    e.vit = 40
    for key, value in dict(cbi=[0], cbj=[-1], cpA=[[0., 0., 0.]], cpB=[[0., 0., 0.]],
                           cn=[[0., 0., 1.]], xc=[[0., 0., 0.]], v=[[0., 0., -1e10]],
                           w=[[0., 0., 0.]], invM=[1.], invIw=[np.eye(3).tolist()]).items():
        setattr(e, key, Array(value))
    fixed_point_range = refused(lambda: e._normal_project(1), 'fixed-point contribution range')
    budget = refused(lambda: solve_normal([[1., 0.], [0., 1.]], [-1., -1.], max_solves=1), 'budget')
    return dict(body_capacity=body_capacity, contact_capacity=contact_capacity,
                fixed_point_range=fixed_point_range, budget=budget)


def main():
    first, second = run(), run()
    report = dict(gates=first, exact_repeats=first == second,
                  status='VERIFIED-FRESH' if all(first.values()) and first == second else 'OWN-GATE-FAIL')
    (ROOT/'reports/complement_normal_guards_probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return 0 if all(first.values()) and first == second else 1


if __name__ == '__main__':
    raise SystemExit(main())
