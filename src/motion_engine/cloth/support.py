"""Rigid support, diagonal nodal mass NCP specialization.
Normal is exact closest-surface normal for a finite cylinder (Cusick) or slab edge.
Coulomb disk impulse is the diagonal Delassus solution; contact_check compares ncp_gpu.
This is an operator split AFTER elasticity, not a coupled implicit contact solve.
"""
import numpy as np
def apply(x0,xfree,v,mass,dt,h,mu=.5,kind='cylinder'):
 if kind=='cylinder':
  r=np.linalg.norm(x0[:,[0,2]],axis=1);a=r-.09;b=np.abs(x0[:,1]-.08)-.08
  outside=np.sqrt(np.maximum(a,0)**2+np.maximum(b,0)**2)
  dist=outside+np.minimum(np.maximum(a,b),0)
  n=np.zeros_like(x0);nr=np.zeros_like(x0);nr[:,0]=x0[:,0]/np.maximum(r,1e-30);nr[:,2]=x0[:,2]/np.maximum(r,1e-30)
  ny=np.zeros_like(x0);ny[:,1]=np.sign(x0[:,1]-.08)
  corner=(a>0)&(b>0);n[corner]=(a[corner,None]*nr[corner]+b[corner,None]*ny[corner])/outside[corner,None]
  side=(a>b)&~corner;n[side]=nr[side];top=~(side|corner);n[top]=ny[top]
 else:
  # Infinite half-space below y=0 for x<=.008; rounded by the cloth thickness.
  a=x0[:,0]-.008;b=x0[:,1]
  outside=np.sqrt(np.maximum(a,0)**2+np.maximum(b,0)**2);dist=outside+np.minimum(np.maximum(a,b),0)
  n=np.zeros_like(x0);corner=(a>0)&(b>0);n[corner,0]=a[corner]/outside[corner];n[corner,1]=b[corner]/outside[corner]
  side=(a>b)&~corner;n[side,0]=1;top=~(side|corner);n[top,1]=1
 gap=dist-h/2;vn=np.sum(v*n,1);jn=mass*np.maximum(0,-vn-gap/dt)
 vt=v-vn[:,None]*n;speed=np.linalg.norm(vt,axis=1);jt=np.minimum(mu*jn,mass*speed)
 impulse=jn[:,None]*n-jt[:,None]*vt/np.maximum(speed[:,None],1e-30)
 vc=v+impulse/mass[:,None];x=x0+dt*vc
 return x,vc,dict(contacts=int(np.count_nonzero(jn)),max_initial_penetration_m=float(np.maximum(-gap,0).max()),normal_impulse=float(jn.sum()))
