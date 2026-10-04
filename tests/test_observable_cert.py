import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('observable_cert', ROOT/'src/motion_engine/ncp/observable_cert.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
certify,verify=mod.certify_observable,mod.verify_observable

@pytest.mark.parametrize('h,C,status', [([1,2],[[1,1]],'UNIQUE'),([1,2],[[1,0]],'AMBIGUOUS'),([1,3],[[1,1]],'INCOMPATIBLE')])
def test_affine_fiber(h,C,status):
 A=[[1,1],[2,2]];r=certify(A,h,C)
 assert r['status']==status and verify(A,h,C,r)
 if status=='UNIQUE':assert r['value']==[1]

@pytest.mark.parametrize('bad', [0.1,True,float('nan')])
def test_reject_inexact(bad):
 with pytest.raises(TypeError):certify([[bad]],[1],[[1]])

@pytest.mark.parametrize('A,h,C', [([],[],[[1]]),([[1]],[1,2],[[1]]),([[1]],[1],[[1,2]]),([[1],[]],[1,2],[[1]])])
def test_dimensions(A,h,C):
 with pytest.raises(ValueError):certify(A,h,C)

@pytest.mark.parametrize('status,h,C,key', [('UNIQUE',[1,2],[[1,1]],'value'),('AMBIGUOUS',[1,2],[[1,0]],'right_null'),('INCOMPATIBLE',[1,3],[[1,1]],'left_null')])
def test_corrupted_witness(status,h,C,key):
 A=[[1,1],[2,2]];r=certify(A,h,C);assert verify(A,h,C,r);bad=copy.deepcopy(r)
 bad[key][0]+=1
 assert not verify(A,h,C,bad)

def test_wrong_input_invalidates_certificate():
 r=certify([[1,1]],[1],[[1,1]])
 assert verify([[1,1]],[1],[[1,1]],r)
 assert not verify([[1,2]],[1],[[1,1]],r)
 assert not verify([[1,1]],[2],[[1,1]],r)
 assert not verify([[1,1]],[1],[[1,0]],r)

def test_zero_rank():
 assert certify([[0,0]],[0],[[0,0]])['status']=='UNIQUE'
 assert certify([[0,0]],[0],[[1,0]])['status']=='AMBIGUOUS'
 assert certify([[0,0]],[1],[[0,0]])['status']=='INCOMPATIBLE'
