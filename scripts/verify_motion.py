"""Run explicitly declared motion verification recipes in an isolated copy."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from motion_engine.verification_contract import load_json,run,Refusal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    try:result=run(ROOT,load_json((ROOT/'docs/VERIFY_RECIPES.json').read_bytes()),args.require_complete)
    except (ValueError,KeyError,IndexError,OSError,TypeError) as error:
        result=dict(status='OWN-GATE-FAIL',complete=False,executions=[],reason=str(error) if isinstance(error,Refusal) else type(error).__name__)
    name='verification_complete.json' if args.require_complete else 'verification_selected.json'
    (ROOT/'reports'/name).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],declared=result.get('declared'),covered=result.get('covered'),executions=len(result['executions']),complete=result['complete'])))
    return 0 if result['status']=='VERIFIED-FRESH' else 2


if __name__=='__main__':raise SystemExit(main())
