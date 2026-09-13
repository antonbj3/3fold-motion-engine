"""Locate remaining candidate divergence without inferring an unobserved cause."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
import numpy as np
from cross_hardware_state_probe import ROOT


def main():
    reports={label:json.loads((ROOT/f'reports/cross_unfused_phase_{label}.json').read_text()) for label in ('l4','local')}
    rows=[]
    with np.load(ROOT/'reports/cross_unfused_phase_l4.npz') as a,np.load(ROOT/'reports/cross_unfused_phase_local.npz') as b:
        for index,engine in enumerate(('pair_chunks','color_cache')):
            for leg in range(2):
                left=reports['l4']['rows'][index]['legs'][leg]['records'];right=reports['local']['rows'][index]['legs'][leg]['records']
                if [(r['step'],r['phase']) for r in left]!=[(r['step'],r['phase']) for r in right]:raise RuntimeError('Different phase coverage')
                for l,r in zip(left,right):
                    differences=[]
                    for field in l['fields']:
                        key=f"{engine}__{leg}__{l['step']}__{l['phase']}__{field}";x=a[key];y=b[key]
                        if x.tobytes()!=y.tobytes():
                            changed=(np.ascontiguousarray(x).view(np.uint8).reshape(x.shape+(x.dtype.itemsize,))!=np.ascontiguousarray(y).view(np.uint8).reshape(y.shape+(y.dtype.itemsize,))).any(axis=-1)
                            differences.append(dict(field=field,count=int(changed.sum()),max_abs=float(np.max(abs(x.astype(float)-y.astype(float)))),first_index=np.argwhere(changed)[0].tolist()))
                    if differences:
                        rows.append(dict(engine=engine,leg=leg,step=l['step'],phase=l['phase'],differences=differences));break
    fixed=lambda r:{k:v for k,v in r.items() if k!='leg'}
    gates=dict(observer_gates=all(all(r['gates'].values()) for r in reports.values()),coverage=len(rows)==4,repeat=len(rows)==4 and fixed(rows[0])==fixed(rows[1]) and fixed(rows[2])==fixed(rows[3]))
    result=dict(rows=rows,gates=gates,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL')
    (ROOT/'reports/cross_unfused_phase_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
