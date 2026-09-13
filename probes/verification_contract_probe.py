"""Falsify verification contracts with invalid metadata and dishonest children."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import copy
import json
import subprocess
import tempfile
from motion_engine.verification_contract import Refusal,check_report,declarations,digest,execute,load_json,run,safe,tree_digest,validate

GOOD=b'{"gates":{"a":true,"b":true},"value":7}\n'


def fixture(root,body=None):
    for folder in ('src','scripts','reports','docs'): (root/folder).mkdir(parents=True)
    (root/'src/sample.py').write_text('value=7\n');(root/'src/other.py').write_text('value=9\n')
    (root/'docs/RUNNING.md').write_text('| Module | Status | Evidence |\n|---|---|---|\n| `src/sample.py` | VERIFIED-FRESH | two gates |\n| `src/other.py` | VERIFIED-FRESH | separate row |\n')
    (root/'scripts/check.py').write_text(body if body is not None else "from pathlib import Path\nPath('reports/result.json').write_bytes("+repr(GOOD)+")\n")
    (root/'reports/result.json').write_bytes(GOOD)
    rows=declarations(root)
    recipe=dict(id='sample',declarations={'src/sample.py':rows['src/sample.py']['note_sha256']},argv=['scripts/check.py'],reports=[dict(path='reports/result.json',sha256=digest(GOOD),gate_pointer='/gates',gates=['a','b'])],timeout=2)
    manifest=dict(schema=1,source_tree_sha256=tree_digest(root,('src','scripts','probes','tests','examples','Makefile')),
        input_tree_sha256=tree_digest(root,('reports',),['reports/result.json']),recipes=[recipe])
    return manifest


def refusal(fn):
    try:fn()
    except (Refusal,ValueError,KeyError,IndexError,TypeError):return True
    return False


def controls():
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp);root=base/'good';m=fixture(root)
        first=run(root,m);second=run(root,m)
        valid=first['status']=='VERIFIED-FRESH' and len(first['executions'])==1 and first['executions'][0]['reports'][0]['sha256']==digest(GOOD)
        full=run(root,m,True)
        isolation=first==second and not first['complete'] and full['status']=='OWN-GATE-FAIL' and not full['executions'] and (root/'reports/result.json').read_bytes()==GOOD
        complete=copy.deepcopy(m);complete['recipes'][0]['declarations']['src/other.py']=declarations(root)['src/other.py']['note_sha256']
        isolation &= run(root,complete,True)['complete']
        (root/'examples').mkdir();(root/'examples/demo.py').write_text('value=1\n')
        with (root/'docs/RUNNING.md').open('a') as f:f.write('| `examples/demo.py` | VERIFIED-FRESH | example row |\n')
        complete['recipes'][0]['declarations']['examples/demo.py']=declarations(root)['examples/demo.py']['note_sha256']
        complete['source_tree_sha256']=tree_digest(root,('src','scripts','probes','tests','examples','Makefile'))
        isolation &= run(root,complete,True)['complete']
        m['source_tree_sha256']=complete['source_tree_sha256']
        declared=[]
        text=(root/'docs/RUNNING.md').read_text()
        for change in (text+'| `src/sample.py` | VERIFIED-FRESH | duplicate |\n',text.replace('src/other.py','src/missing.py'),text.replace('two gates','changed note')):
            (root/'docs/RUNNING.md').write_text(change)
            declared.append(refusal(lambda:validate(root,m)))
        (root/'docs/RUNNING.md').write_text(text)
        (root/'src/sample.py').write_text('value=8\n');declared.append(refusal(lambda:validate(root,m)))
        (root/'src/sample.py').write_text('value=7\n')
        bad_manifest=[]
        changes=[lambda x:x.update(schema=True),lambda x:x['recipes'].append(copy.deepcopy(x['recipes'][0])),
                 lambda x:x['recipes'][0].update(argv=['../outside.py']),lambda x:x['recipes'][0].update(timeout=True),
                 lambda x:x['recipes'][0]['reports'][0].update(path='src/sample.py'),lambda x:x['recipes'][0]['reports'][0].update(gates=[])]
        for change in changes:
            q=copy.deepcopy(m);change(q);bad_manifest.append(refusal(lambda:validate(root,q)))
        bad_manifest.append(refusal(lambda:safe(root,'/outside')))
        bad_manifest.append(refusal(lambda:safe(root,'reports/../outside',output=True)))
        spec=m['recipes'][0]['reports'][0]
        malformed=[b'{"gates":{"a":false,"b":true}}',b'{"gates":{"a":true}}',b'{"gates":{"a":1,"b":true}}',b'{"gates":{"a":true,"b":true,"hidden":false}}',b'{"gates":{"a":true,"a":true,"b":true}}',b'{"gates":{"a":true,"b":true},"x":NaN}',b'{"gates":{"a":true,"b":true},"x":1e999}']
        typed=[refusal(lambda data=data:check_report(data,dict(spec,sha256=digest(data)))) for data in malformed]
        changed=GOOD.replace(b'7',b'8')
        integrity=refusal(lambda:check_report(changed,spec))
        stale_root=base/'stale';stale=fixture(stale_root,'pass\n');stale_result=run(stale_root,stale)
        fresh=stale_result['status']=='OWN-GATE-FAIL' and not stale_result['executions'][0]['passed'] and (stale_root/'reports/result.json').read_bytes()==GOOD
        exit_root=base/'exit';exit_m=fixture(exit_root,"from pathlib import Path\nPath('reports/result.json').write_bytes("+repr(GOOD)+")\nraise SystemExit(3)\n")
        exit_result=run(exit_root,exit_m)
        timeout_root=base/'timeout';timeout_m=fixture(timeout_root,'import time\ntime.sleep(10)\n');timeout_m['recipes'][0]['timeout']=1
        timeout_result=run(timeout_root,timeout_m)
        false_root=base/'false';false_data=GOOD.replace(b'"a":true',b'"a":false')
        false_m=fixture(false_root,"from pathlib import Path\nPath('reports/result.json').write_bytes("+repr(false_data)+")\n")
        false_result=run(false_root,false_m)
        evidence=false_result['executions'][0]['reports'][0]['failure_evidence']
        retained=(false_root/evidence).read_bytes()==false_data and not false_result['executions'][0]['passed']
        process=retained and exit_result['executions'][0]['returncode']==3 and not exit_result['executions'][0]['passed'] and timeout_result['executions'][0]['returncode']==124 and not timeout_result['executions'][0]['passed']
        return dict(gates=dict(valid_receipt=valid,declaration_refusals=all(declared),manifest_refusals=all(bad_manifest),typed_gate_refusals=all(typed),full_report_integrity=integrity,fresh_output=fresh,process_refusal=process,independent_isolation=isolation),
                    controls=dict(declarations=len(declared),manifest=len(bad_manifest),typed=len(typed),integrity=1,stale=1,process=3))


def main():
    if sys.argv[1:]==['--worker']:print(json.dumps(controls()));return 0
    records=[]
    for _ in range(2):
        p=subprocess.run([sys.executable,__file__,'--worker'],capture_output=True,text=True,check=True,timeout=30)
        records.append(load_json(p.stdout))
    report=dict(gates=records[0]['gates'],records=records,exact_repeat=records[0]==records[1])
    (ROOT/'reports/verification_contract_probe.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(gates=report['gates'],exact_repeat=report['exact_repeat'])))
    return 0 if all(report['gates'].values()) and report['exact_repeat'] else 1


if __name__=='__main__':raise SystemExit(main())
