"""Compare certified scalar observers without reclassifying matrix failures."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    schema=json.loads((ROOT/'reports/cross_scalar_schema.json').read_text())
    inputs=[json.loads((ROOT/f'reports/cross_scalar_{label}.json').read_text()) for label in ('l4','local')]
    rows=[];same_masks=True
    with np.load(ROOT/'reports/cross_scalar_l4.npz') as cloud,np.load(ROOT/'reports/cross_scalar_local.npz') as local:
        for leg in range(2):
            for engine,key in [('pair_chunks','pair_trace'),('color_cache','fixed_trace')]:
                prefix=f'{engine}__{leg}__';a=cloud[prefix+'scalar_trace'];b=local[prefix+'scalar_trace']
                ma=cloud[prefix+'scalar_seen'];mb=local[prefix+'scalar_seen'];mask_equal=ma.tobytes()==mb.tobytes();same_masks &= mask_equal
                if a.shape!=b.shape or a.shape!=ma.shape or a.shape[2]!=len(schema[key]):raise RuntimeError('Trace shape mismatch')
                difference=(a.view(np.uint64)!=b.view(np.uint64)) & ((ma!=0)|(mb!=0))
                first=[]
                for thread in range(a.shape[0]):
                    where=np.argwhere(difference[thread])
                    if len(where):
                        contact,slot=map(int,where[0]);description=schema[key][slot]
                        first.append(dict(thread=thread,contact=contact,**description,cloud=float(a[thread,contact,slot]),local=float(b[thread,contact,slot]),delta=float(b[thread,contact,slot]-a[thread,contact,slot])))
                rows.append(dict(leg=leg,engine=engine,presence_exact=mask_equal,differing_values=int(difference.sum()),first_per_thread=first))
    fixed=lambda r:{k:v for k,v in r.items() if k!='leg'}
    gates=dict(observers_pass=all(all(r['gates'].values()) for r in inputs),presence_exact=same_masks,coverage=len(rows)==4 and all(r['first_per_thread'] for r in rows),repeat=all(fixed(rows[i])==fixed(rows[i+2]) for i in range(2)))
    report=dict(rows=rows,gates=gates,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',scope='First differing assignment on each active pair thread. Compound expressions are not yet individual machine instructions; frozen matrix remains10/12exact.')
    (ROOT/'reports/cross_scalar_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
