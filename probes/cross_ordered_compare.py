"""Audit separate ordered-friction solver capsules using the frozen matrix auditor."""

import sys as _probe_sys
from pathlib import Path as _ProbePath
_probe_root = _ProbePath(__file__).resolve().parents[1]
_probe_sys.path[:0] = [str(_probe_root/"probes"), str(_probe_root/"scripts"), str(_probe_root/"src")]
import json
from pathlib import Path
import shutil
import sys
from cross_hardware_suite import matrix,render
ROOT=Path(__file__).resolve().parents[1]


def main(labels,kind="ordered"):
    if kind not in ("ordered","unfused","position","prepared","island","compact","shared"):raise ValueError("Unknown candidate")
    destination=ROOT/('reports/ordered_friction_matrix' if kind=='ordered' else f'reports/{kind}_solve_matrix');inputs=destination/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    source_kind='position' if kind in ('prepared','island','compact','shared') else kind
    for label in labels:
        for ext in ('json','npz'):
            shutil.copy2(ROOT/f'reports/cross_{source_kind}_{label}.{ext}',inputs/f'cross_hardware_{label}.{ext}')
    report=matrix(inputs,labels)
    repeat=report==matrix(inputs,labels)
    report['variant']='ordered_friction_dot' if kind=='ordered' else ('unfused_velocity_solve' if kind=='unfused' else 'unfused_velocity_position');report['scope']='Candidate matrix only; original solver captures remain unchanged.'
    report['gates']=dict(capture_gates=all(all(json.loads((inputs/f'cross_hardware_{label}.json').read_text())['gates'].values()) for label in labels),initial_exact=all(r['initial_exact'] for r in report['rows']),all_exact=all(r['status']=='VERIFIED-FRESH' for r in report['rows']),repeat=repeat)
    if kind=='prepared':report['variant']='prepared_noncontracting'
    if kind=='island':report['variant']='noncontracting_island'
    if kind=='compact':report['variant']='compact_island'
    if kind=='shared':report['variant']='shared_layout'
    report['status']='VERIFIED-FRESH' if all(report['gates'].values()) else 'OWN-GATE-FAIL'
    (destination/'matrix.json').write_text(json.dumps(report,indent=2)+'\n');(destination/'matrix.md').write_text(render(report))
    print(render(report));print(json.dumps(report['gates']));return 0 if all(report['gates'].values()) else 1


if __name__=='__main__':
    args=sys.argv[1:];kind='ordered'
    if args and args[0] in ('--unfused','--position','--prepared','--island','--compact','--shared'):kind=args[0][2:];args=args[1:]
    raise SystemExit(main(args,kind))
