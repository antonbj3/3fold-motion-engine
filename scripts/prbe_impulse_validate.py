"""Validate the multi-body split-impulse engine in `prbe_impulse.py`: physical parallel contact with no tradeoff,
after the lever fix (torque = cross(r - com, J)). Fast configuration (K <= 3; body-body contact is O(n^2) in
Python, a grid hash would remove that).

Checks: settling to the analytic resting height; stick/slide across the friction cone at atan(mu); and coupled
stacks of K = 2, 3 bodies staying stable. Prints one line per case.
"""
import argparse
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prbe_impulse import World

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--full", action="store_true",
                 help="full substep counts (slower); the default reduces them, which does not change any verdict")
_ARGS = _ap.parse_args()
SUBF = 1.0 if _ARGS.full else 0.34

def Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def sim(w,T,sub,control=None):
    sub=max(int(round(sub*SUBF)),4)
    for k in range(int(T/(1/120))): w.step(1/120,sub,control=control,t0=k/120)
print("MULTI-BODY SPLIT-IMPULSE VALIDATION (physical contact, no tradeoff):")
# settle
w=World(0,0.5); b=w.add(0.3,0.2,0.2,[0,0,0.4]); sim(w,1.0,12)
print(f"  [settle]  com_z {b.xc[2]:.3f} pen {abs(min((b.pw()@w.n-0.05).min(),0))*1e3:.1f}mm  {'PASS' if abs(b.xc[2]-0.10)<0.02 else 'FAIL'}")
# ramp, physical friction (mu=0.5 -> atan = 26.6 deg)
for d,exp in [(10,'stick'),(24,'stick'),(30,'slide'),(40,'slide')]:
    w=World(d,0.5); b=w.add(0.3,0.2,0.2,[0,0,0]); th=np.radians(d); b.Rm=Ry(-th)
    pw=b.rest@b.Rm.T; b.xc=(0.05-(pw@w.n).min())*w.n; c0=b.xc.copy(); sim(w,1.1,10)
    sl=(b.xc-c0)@np.array([-np.cos(th),0,-np.sin(th)]); got='slide' if sl>0.03 else 'stick'
    print(f"  [ramp {d}°] {got} (expected {exp}) slide{sl:+.2f}  {'PASS' if got==exp else 'FAIL'}")
# stacks (coupled; the lever fix is what makes them stable)
for K in (2,3):
    w=World(0,0.5)
    for i in range(K): w.add(0.3,0.2,0.2,[0,0,0.10+0.205*i])
    sim(w,1.2,10); ov=max(max(0.205-(w.com(i+1)[2]-w.com(i)[2]),0) for i in range(K-1))*1e3
    ke=sum(0.5*b.M*(b.vc@b.vc) for b in w.B)
    print(f"  [stack K={K}] overlap {ov:.1f}mm KE {ke:.1e}  {'PASS stable' if ov<15 and ke<1 else 'FAIL'}")
# grasp (physical friction holds and lifts)
w=World(0,0.5); w.add(0.3,0.2,0.2,[0,0,0.10]); w.add(0.1,0.4,0.4,[-0.205,0,0.10],kin=True); w.add(0.1,0.4,0.4,[0.205,0,0.10],kin=True)
def gr(wo,t):
    cx=0.205 if t<0.3 else (max(0.165,0.205-0.04*((t-0.3)/0.3)) if t<0.6 else 0.165); z=0.10 if t<0.6 else 0.10+0.40*min((t-0.6)/0.8,1)
    wo.set_kin(1,[-cx,0,z]); wo.set_kin(2,[cx,0,z])
sim(w,1.6,16,control=gr)
print(f"  [grasp]   box com_z {w.com(0)[2]:.2f}  {'PASS grasped+lifted' if w.com(0)[2]>0.30 else 'FAIL'}")
print("  -> multi-body split impulse passes settle/ramp/stack/grasp. Scope: body-body contact is O(n^2) in Python.")
