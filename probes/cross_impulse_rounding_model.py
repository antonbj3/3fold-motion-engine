"""Exact rational counterfactuals for the retained two friction dot witnesses."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
from fractions import Fraction
import json
import numpy as np
from cross_impulse_expression_probe import ROOT,fixtures


def rounded(value):
    center=np.float32(float(value))
    candidates=[center,np.nextafter(center,np.float32(-np.inf)),np.nextafter(center,np.float32(np.inf))]
    def score(x):
        distance=abs(Fraction(float(x))-value)
        return distance,int(x.view(np.uint32))&1
    return Fraction(float(min(candidates,key=score)))


def calculate():
    rows=[]
    for inputs,expected in zip(*fixtures()):
        p=[Fraction(float(inputs[i]))*Fraction(float(inputs[i+3])) for i in range(3)]
        r=rounded
        dots=dict(separate=r(r(r(p[0])+r(p[1]))+r(p[2])),fuse_first=r(r(p[0]+r(p[1]))+r(p[2])),fuse_second=r(r(r(p[0])+p[1])+r(p[2])),fuse_third=r(r(r(p[0])+r(p[1]))+p[2]))
        mass=Fraction(float(inputs[7]));old=Fraction(float(inputs[6]))
        values={k:float(r(old-r(d/mass))) for k,d in dots.items()}
        rows.append(dict(expected=expected.tolist(),dots={k:float(d) for k,d in dots.items()},impulses=values))
    return rows


def main():
    first=calculate();second=calculate()
    gates=dict(repeat=first==second,both_endpoints_explained=all(all(x in row['impulses'].values() for x in row['expected']) for row in first),
        distinct_rounding_outcomes=all(len(set(row['impulses'].values()))==2 for row in first))
    report=dict(rows=first,gates=gates,scope='Compatible fused/nonfused arithmetic explanations, not actual machine instruction attribution. Fraction arithmetic with nearest-even float32 rounding at explicit boundaries.')
    (ROOT/'reports/cross_impulse_rounding_model.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
