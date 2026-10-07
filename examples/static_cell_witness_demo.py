#!/usr/bin/env python3
"""Three exact static witnesses using P4 source geometry and a NEW envelope.

Run from the repository: PYTHONPATH=src python examples/static_cell_witness_demo.py
No historical areas, pelvis coordinates, leaf list or weight grid are reproduced.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import json

from motion_engine.ncp.static_cell_witness import certify_static_cell


def main():
    source = Path(__file__).resolve().parents[1]/'reproducibility/p4/raw/certified_regions.json'
    expected = 'e86baabb47dac6ef09f5d884b7c620379743a20c856d20810aaec6f1b6380118'
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('P4 source geometry hash changed; rebind the example explicitly')
    raw = json.loads(source.read_text())['direction_cover']
    rows = []
    for name,pair in [('g2',[2,5]),('g8',[5,7]),('g9',[2,4])]:
        b = raw[name]
        if b['yaw_deg'] not in (20,60,-70):
            raise ValueError('unexpected source yaw')
        result = certify_static_cell(
            base_contacts_xy=[[Fraction(v) for v in p] for p in b['base_contacts_xy']],
            moving_ids=b['moving_ids'],yaw_deg=int(b['yaw_deg']),
            com_xy=[Fraction(v) for v in b['com_xy']],cell_center_xy=['1/100','1/100'],
            cell_side='1/50',pair=pair,weight='1/26',level='1/20',
            height_max='11/10',pelvis_offset_max='1/10')
        rows.append({'geometry':name,**result})
    print(json.dumps({'historical_region_reproduction':False,
                      'input_semantics':'exact binary64 source constants, ideal real yaw, exact rational gravity and new load envelope',
                      'source_sha256':expected,'witnesses':rows},indent=2))


if __name__ == '__main__':
    main()
