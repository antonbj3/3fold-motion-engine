"""Adapt the existing sequential kernel-matrix orchestration to motion captures.

Orchestration source: kernel runner at commit e9aa89dc509ee748eb009be550823096fe39d657.
Motion keeps its frozen capture, array auditor and pinned numeric environment.
"""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from cross_hardware_state_probe import ROOT
from cross_hardware_suite import matrix, render

DEVICES={'l4':'L4','a10g':'A10G','h100':'H100'}
EXPECTED={'NVIDIA L4','NVIDIA A10','NVIDIA H100 80GB HBM3','NVIDIA GeForce RTX 5070'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner-root',type=Path,default=os.environ.get('MODAL_RUNNER_ROOT'))
    parser.add_argument('--capture',nargs='*',choices=tuple(DEVICES),default=list(DEVICES))
    args=parser.parse_args()
    if args.capture and (args.runner_root is None or not (args.runner_root/'session_c.py').is_file()):
        parser.error('Set MODAL_RUNNER_ROOT or --runner-root to the configured pinned motion runner')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=ROOT/'reports/motion_matrix_v1/runs'/stamp;destination.mkdir(parents=True)
    labels=['l4','a10g','h100','local']
    inputs=destination/'inputs';inputs.mkdir()
    for label in labels:
        for suffix in ('json','npz'):
            name=f'cross_hardware_{label}.{suffix}'
            saved=ROOT/'reports'/name
            if saved.is_file():shutil.copyfile(saved,inputs/name)
    transport=[]
    for label in args.capture:
        gpu=DEVICES[label]
        command='python probes/cross_hardware_state_probe.py --capture '+label
        proc=subprocess.run(['modal','run','session_c.py','--repo','3fold-motion-engine',
            '--cmd',command,'--gpu',gpu,'--timeout-s','600','--packages','warp-lang==1.13.0 numpy==2.5.3'],
            cwd=args.runner_root,capture_output=True,text=True,timeout=900)
        output=proc.stdout+proc.stderr
        found=re.findall(r'^results:\s*(.+?)\s*$',output,re.MULTILINE)
        if len(found)!=1:raise RuntimeError('Missing unique fetched result directory: '+output[-1500:])
        source=Path(found[0])/'reports'
        for suffix in ('json','npz'):
            name=f'cross_hardware_{label}.{suffix}'
            if not (source/name).is_file():raise RuntimeError('Capture incomplete on '+gpu)
            shutil.copyfile(source/name,inputs/name)
        transport.append(dict(gpu=gpu,returncode=proc.returncode))
        print(label+' capture fetched',flush=True)
    # Local capture is supplied only through an independently locked GPU window.
    # Audit saved capsules rather than accepting a successful transport as proof.
    first=matrix(inputs,labels);second=matrix(inputs,labels)
    repeated_report=first==second
    if not repeated_report:raise RuntimeError('Repeated matrix differs')
    first['scope']='Frozen motion contact fixtures on four devices; no universal-input or other-family claim'
    gates=dict(full_coverage=len(first['rows'])==12 and {h.split(',')[0] for h in first['hardware'].values()}==EXPECTED,
        initial_exact=all(r['initial_exact'] for r in first['rows']),
        cross_hardware_exact=all(r['status']=='VERIFIED-FRESH' for r in first['rows']),
        repeated_report=repeated_report)
    first['gates']=gates
    first['status']='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL'
    code=int(any(r['returncode']!=0 for r in transport) or not all(gates.values()))
    for target in (destination,ROOT/'reports/motion_matrix_v1'):
        (target/'matrix.json').write_text(json.dumps(first,indent=2)+'\n')
        (target/'matrix.md').write_text(render(first))
    (destination/'run.json').write_text(json.dumps(dict(timestamp=stamp,transport=transport,
        labels=labels,returncode=code,warp='1.13.0',numpy='2.5.3'),indent=2)+'\n')
    print(render(first),end='')
    return code


if __name__=='__main__':raise SystemExit(main())
