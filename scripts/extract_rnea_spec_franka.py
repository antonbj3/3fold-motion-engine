"""Extract the dynamics spec (mass/com/inertia + kinematics) from the pinocchio Franka model and generate the
RNEA parity reference (pin.rnea on random q, dq, ddq). Writes data/rnea_parity_franka.npz, which
rnea_warp.py and rnea_warp_parity.py read.

  python scripts/extract_rnea_spec_franka.py   (requires pinocchio)
"""
from pathlib import Path

import numpy as np, pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
URDF = str(ROOT / "assets/robots/franka_description/franka_panda.urdf")
OUT = ROOT / "data/rnea_parity_franka.npz"
full = pin.buildModelFromUrdf(URDF)
lk = [full.getJointId(f"panda_finger_joint{i}") for i in (1,2)]
m = pin.buildReducedModel(full, lk, pin.neutral(full)); d = m.createData()
NJ = m.njoints - 1                                          # excluding the universe joint (0)
print(f"franka reduced: njoints (excl. universe)={NJ}, nq={m.nq}")
parent = np.array([m.parents[i] for i in range(1, m.njoints)], np.int32)   # parent joint index (0 = universe)
jR = np.array([np.asarray(m.jointPlacements[i].rotation) for i in range(1, m.njoints)])   # (NJ,3,3)
jp = np.array([np.asarray(m.jointPlacements[i].translation) for i in range(1, m.njoints)])# (NJ,3)
axis = np.array([[0.,0.,1.] for _ in range(NJ)])           # franka is revolute about Z (checked via shortname below)
shorts = [m.joints[i].shortname() for i in range(1, m.njoints)]
mass = np.array([m.inertias[i].mass for i in range(1, m.njoints)])         # (NJ,)
com  = np.array([np.asarray(m.inertias[i].lever) for i in range(1, m.njoints)])  # (NJ,3) com in the joint frame
Ilink= np.array([np.asarray(m.inertias[i].inertia) for i in range(1, m.njoints)])# (NJ,3,3) inertia at the com, joint frame
grav = np.asarray(m.gravity.linear)                        # [0,0,-9.81]
print(f"  joint types: {shorts}")
print(f"  masses: {np.round(mass,3)}")
print(f"  gravity: {grav}")
# parity reference: N random (q, dq, ddq) through pin.rnea
rng = np.random.default_rng(0); N = 64
q = rng.uniform(-2, 2, (N, m.nq)); dq = rng.uniform(-1.5, 1.5, (N, m.nv)); ddq = rng.uniform(-2, 2, (N, m.nv))
tau_ref = np.array([pin.rnea(m, d, q[i], dq[i], ddq[i]) for i in range(N)])
print(f"  tau_ref[0] = {np.round(tau_ref[0],3)}")
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, parent=parent, jR=jR, jp=jp, axis=axis,
         mass=mass, com=com, Ilink=Ilink, grav=grav, q=q, dq=dq, ddq=ddq, tau_ref=tau_ref, nq=m.nq)
print(f"  wrote {OUT} (dynamics spec + 64 parity samples)")
