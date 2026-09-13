"""Explicit isolated verification of declared numerical receipts."""
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import shutil
import signal
import subprocess
import sys
import tempfile


class Refusal(ValueError):
    pass


def digest(data):return hashlib.sha256(data).hexdigest()


def load_json(data):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise Refusal('duplicate_json_key')
            result[key]=value
        return result
    def number(text):
        value=float(text)
        if not math.isfinite(value):raise Refusal('nonfinite_json')
        return value
    def invalid(text):raise Refusal('nonfinite_json')
    return json.loads(data,object_pairs_hook=pairs,parse_float=number,parse_constant=invalid)


def safe(root,name,output=False):
    if not isinstance(name,str) or not name or '\\' in name:raise Refusal('invalid_path')
    pure=PurePosixPath(name)
    if pure.is_absolute() or any(p in ('.','..') for p in name.split('/')):raise Refusal('invalid_path')
    path=(root/name).resolve()
    boundary=(root/'reports').resolve() if output else root.resolve()
    if not path.is_relative_to(boundary) or path==boundary:raise Refusal('path_escape')
    return path


def declarations(root):
    rows={}
    for line,raw in enumerate((root/'docs/RUNNING.md').read_text().splitlines(),1):
        cells=raw.split('|')
        if len(cells)<4 or not raw.startswith('|') or cells[2].strip()!='VERIFIED-FRESH':continue
        label=cells[1].strip()
        if not label.startswith('`') or not label.endswith('`') or label.count('`')!=2:raise Refusal('noncanonical_declaration')
        name=label[1:-1];path=safe(root,name)
        if name in rows:raise Refusal('duplicate_declaration')
        if not path.is_file():raise Refusal('missing_source')
        note='|'.join(cells[3:-1]).strip()
        rows[name]=dict(line=line,note_sha256=digest(note.encode()),declaration_sha256=digest(name.encode()))
    if not rows:raise Refusal('empty_declarations')
    return rows


def tree_digest(root,folders,excluded=()):
    h=hashlib.sha256();excluded=set(excluded)
    for folder in folders:
        for path in ([root/folder] if (root/folder).is_file() else sorted((root/folder).rglob('*'))):
            if not path.is_file() or '__pycache__' in path.parts:continue
            name=path.relative_to(root).as_posix()
            if name in excluded or (folder=='reports' and path.name.startswith('verification_')):continue
            safe(root,name)
            h.update(name.encode()+b'\0');h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def pointer(value,path):
    if not isinstance(path,str) or not path.startswith('/'):raise Refusal('invalid_pointer')
    for token in path[1:].split('/'):
        token=token.replace('~1','/').replace('~0','~')
        if isinstance(value,list):
            if not token.isdecimal():raise Refusal('invalid_index')
            value=value[int(token)]
        elif isinstance(value,dict):value=value[token]
        else:raise Refusal('invalid_pointer_target')
    return value


def check_report(data,spec):
    parsed=load_json(data)
    actual=pointer(parsed,spec['gate_pointer']);expected=spec['gates']
    if not isinstance(actual,dict) or set(actual)!=set(expected) or not expected:raise Refusal('gate_set_mismatch')
    if any(type(value) is not bool or not value for value in actual.values()):raise Refusal('false_or_untyped_gate')
    if digest(data)!=spec['sha256']:raise Refusal('report_bytes_changed')
    return digest(data)


def validate(root,manifest):
    if not isinstance(manifest,dict) or set(manifest)!={'schema','source_tree_sha256','input_tree_sha256','recipes'} or type(manifest['schema']) is not int or manifest['schema']!=1:raise Refusal('manifest_schema')
    rows=declarations(root);recipes=manifest['recipes']
    if not isinstance(recipes,list) or not recipes:raise Refusal('empty_recipes')
    seen=set();covered=set();outputs=set()
    for recipe in recipes:
        if not isinstance(recipe,dict) or set(recipe)!={'id','declarations','argv','reports','timeout'}:raise Refusal('recipe_schema')
        ident=recipe['id']
        if not isinstance(ident,str) or not ident or ident in seen:raise Refusal('duplicate_recipe')
        seen.add(ident)
        if not isinstance(recipe['declarations'],dict) or not recipe['declarations']:raise Refusal('empty_recipe_declarations')
        for name,pin in recipe['declarations'].items():
            if name in covered or name not in rows or rows[name]['note_sha256']!=pin:raise Refusal('declaration_pin')
            covered.add(name)
        args=recipe['argv']
        if not isinstance(args,list) or not args or not all(isinstance(x,str) and x for x in args):raise Refusal('argv_schema')
        command=safe(root,args[0])
        if not command.is_file() or command.suffix!='.py':raise Refusal('python_entry_required')
        if any('\x00' in x for x in args):raise Refusal('argv_schema')
        if type(recipe['timeout']) is not int or not 1<=recipe['timeout']<=900:raise Refusal('timeout_range')
        if not isinstance(recipe['reports'],list) or not recipe['reports']:raise Refusal('empty_reports')
        for spec in recipe['reports']:
            if not isinstance(spec,dict) or set(spec)!={'path','sha256','gate_pointer','gates'}:raise Refusal('report_schema')
            safe(root,spec['path'],output=True)
            if spec['path'] in outputs:raise Refusal('duplicate_output')
            outputs.add(spec['path'])
            if not isinstance(spec['sha256'],str) or len(spec['sha256'])!=64 or any(c not in '0123456789abcdef' for c in spec['sha256']):raise Refusal('invalid_hash')
            if not isinstance(spec['gates'],list) or not spec['gates'] or not all(isinstance(x,str) and x for x in spec['gates']) or len(set(spec['gates']))!=len(spec['gates']):raise Refusal('gate_schema')
            if not isinstance(spec['gate_pointer'],str) or not spec['gate_pointer'].startswith('/'):raise Refusal('invalid_pointer')
    if tree_digest(root,('src','scripts','probes','tests','examples','Makefile'))!=manifest['source_tree_sha256']:raise Refusal('source_tree_changed')
    if tree_digest(root,('reports',),outputs)!=manifest['input_tree_sha256']:raise Refusal('input_tree_changed')
    return rows,covered


def execute(root,recipe):
    for spec in recipe['reports']:safe(root,spec['path'],output=True).unlink(missing_ok=True)
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_CORETYPE='Haswell',
        PYTHONPATH=os.pathsep.join(str(root/p) for p in ('src','scripts','probes')))
    process=subprocess.Popen([sys.executable,*recipe['argv']],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:process.communicate(timeout=recipe['timeout']);code=process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.communicate();code=124
    result=dict(recipe=recipe['id'],returncode=code,reports=[],passed=False)
    valid_reports=True
    for spec in recipe['reports']:
        entry=dict(path=spec['path']);data=None;valid=False
        try:
            data=safe(root,spec['path'],output=True).read_bytes()
            entry['sha256']=digest(data)
            check_report(data,spec);valid=True
        except (ValueError,KeyError,IndexError,OSError,TypeError) as error:
            entry['error']=str(error) if isinstance(error,Refusal) else type(error).__name__
        entry['contract_passed']=valid
        if data is not None and (code or not valid):
            name='reports/verification_failed_'+digest(recipe['id'].encode())[:12]+'_'+Path(spec['path']).name
            safe(root,name,output=True).write_bytes(data);entry['failure_evidence']=name
        valid_reports &= valid;result['reports'].append(entry)
    result['passed']=code==0 and valid_reports
    return result


def run(root,manifest,require_complete=False):
    rows,covered=validate(root,manifest)
    unmapped=[dict(line=value['line'],declaration_sha256=value['declaration_sha256']) for name,value in rows.items() if name not in covered]
    result=dict(manifest_sha256=digest(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()),source_tree_sha256=manifest['source_tree_sha256'],input_tree_sha256=manifest['input_tree_sha256'],declared=len(rows),covered=len(covered),unmapped=unmapped,executions=[],complete=False,
                scope='Explicit CPU recipes with per-child Haswell profile; no fresh GPU execution.')
    if require_complete and unmapped:
        result.update(status='OWN-GATE-FAIL',reason='incomplete_coverage');return result
    with tempfile.TemporaryDirectory(prefix='motion-verify-') as temp:
        copy=Path(temp)/'repo';copy.mkdir()
        for folder in ('src','scripts','probes','tests','examples','reports','docs'):
            if (root/folder).is_dir():shutil.copytree(root/folder,copy/folder,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        if (root/'Makefile').is_file():shutil.copy2(root/'Makefile',copy/'Makefile')
        validate(copy,manifest)
        for recipe in manifest['recipes']:
            result['executions'].append(execute(copy,recipe))
        for execution in result['executions']:
            for report in execution['reports']:
                if 'failure_evidence' in report:
                    name=report['failure_evidence'];shutil.copy2(copy/name,root/name)
        if tree_digest(copy,('src','scripts','probes','tests','examples','Makefile'))!=manifest['source_tree_sha256']:raise Refusal('child_source_changed')
    passed=bool(result['executions']) and all(x['passed'] for x in result['executions'])
    result.update(status='VERIFIED-FRESH' if passed else 'OWN-GATE-FAIL',complete=passed and not unmapped)
    return result
