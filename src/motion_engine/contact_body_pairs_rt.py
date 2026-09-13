"""Bounded OptiX broad phase preserving the conservative baseline pair predicate."""
import ctypes as ct
import os
import threading
import numpy as np
from motion_engine.contact_body_pairs_gpu import BodyPairSearch


def kernels(wp):
    @wp.kernel(module='unique', module_options={'fuse_fp':False})
    def boxes(lower:wp.array(dtype=wp.vec3d),upper:wp.array(dtype=wp.vec3d),
              radius:float,output:wp.array(dtype=float)):
        i=wp.tid()
        for k in range(3):
            lo=lower[i][k]-wp.float64(radius);hi=upper[i][k]+wp.float64(radius)
            scale=wp.max(wp.float64(1.0),wp.max(wp.abs(lo),wp.abs(hi)))
            pad=wp.float64(0.00000762939453125)*scale
            output[6*i+k]=float(lo-pad);output[6*i+3+k]=float(hi+pad)

    @wp.kernel
    def total(counts:wp.array(dtype=int),flags:wp.array(dtype=int)):
        wp.atomic_add(flags,1,counts[wp.tid()])

    @wp.kernel
    def emit(slots:wp.array(dtype=int),counts:wp.array(dtype=int),offsets:wp.array(dtype=int),
             cap:int,stride:wp.int64,keys:wp.array(dtype=wp.int64),values:wp.array(dtype=int)):
        a=wp.tid()
        for j in range(counts[a]):
            i=offsets[a]+j
            keys[i]=wp.int64(a)*stride+wp.int64(slots[a*cap+j])+wp.int64(1)
            values[i]=i
    return boxes,total,emit


class RTBodyPairSearch(BodyPairSearch):
    def __init__(self,wp,bodies,width,capacity=None,radius_limit=.35,coordinate_limit=10000.,slot_capacity=64):
        if not isinstance(bodies,int) or not 1<=bodies<=100000:
            raise ValueError('RT body capacity1..100000 required')
        if not isinstance(slot_capacity,int) or not 1<=slot_capacity<=256:
            raise ValueError('RT slot capacity1..256 required')
        if not np.isfinite(radius_limit) or radius_limit>.35 or not np.isfinite(coordinate_limit) or coordinate_limit>10000.:
            raise ValueError('RT spatial contract exceeded')
        self.handle=ct.c_void_p();self.owner=threading.get_ident()
        self.lib=ct.CDLL(os.environ['MOTION_RT_LIBRARY'])
        self.lib.pairs_create.argtypes=[ct.c_char_p,ct.c_uint,ct.c_uint,ct.POINTER(ct.c_void_p)]
        self.lib.pairs_run.argtypes=[ct.c_void_p]*9
        self.lib.pairs_destroy.argtypes=[ct.POINTER(ct.c_void_p)]
        for name in ('pairs_create','pairs_run','pairs_destroy'):getattr(self.lib,name).restype=ct.c_int
        super().__init__(wp,bodies,width,capacity,radius_limit,coordinate_limit)
        self.slot_capacity=slot_capacity
        self.slots=wp.zeros(bodies*slot_capacity,dtype=int,device='cuda:0')
        self.aabbs=wp.zeros(bodies*6,dtype=float,device='cuda:0')
        self.overflow=wp.zeros(1,dtype=int,device='cuda:0')
        self.make_boxes,self.sum_counts,self.emit_slots=kernels(wp)
        self.check(self.lib.pairs_create(os.fsencode(os.environ['MOTION_RT_PTX']),bodies,slot_capacity,ct.byref(self.handle)))

    @staticmethod
    def check(code):
        if code:raise RuntimeError('RT native refusal '+str(code))

    def close(self):
        if threading.get_ident()!=self.owner:raise RuntimeError('RT owner thread required')
        if self.handle.value:
            self.wp.synchronize_device('cuda:0')
            self.check(self.lib.pairs_destroy(ct.byref(self.handle)))
        self.number=0

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def rebuild(self,points):
        self.number=0
        if not self.handle.value or threading.get_ident()!=self.owner:raise RuntimeError('Live RT owner required')
        w=self.wp
        if len(points)!=self.bodies*self.width or points.dtype!=w.vec3 or points.device!=w.get_device('cuda:0'):
            raise ValueError('Point layout/device mismatch')
        self.maximum.zero_();self.flags.zero_();self.overflow.zero_()
        w.launch(self.scale,self.bodies,inputs=[points,self.width,self.maximum],device='cuda:0')
        w.launch(self.bounds,self.bodies,inputs=[points,self.width,self.maximum,self.radius_limit,self.coordinate_limit,self.lower,self.upper,self.centers,self.flags],device='cuda:0')
        if int(self.flags.numpy()[0]):raise RuntimeError('Spatial bound exceeded or nonfinite point')
        w.launch(self.make_boxes,self.bodies,inputs=[self.lower,self.upper,self.radius_limit,self.aabbs],device='cuda:0')
        self.check(self.lib.pairs_run(self.handle,w.get_stream('cuda:0').cuda_stream,self.aabbs.ptr,
            self.lower.ptr,self.upper.ptr,self.centers.ptr,self.slots.ptr,self.counts.ptr,self.overflow.ptr))
        if int(self.overflow.numpy()[0]):raise RuntimeError('RT per-ray capacity exceeded')
        w.launch(self.sum_counts,self.bodies,inputs=[self.counts,self.flags],device='cuda:0')
        total=int(self.flags.numpy()[1])
        if total>self.capacity:raise RuntimeError('Body-pair capacity exceeded')
        if total:
            w.utils.array_scan(self.counts,self.offsets,inclusive=False)
            w.launch(self.emit_slots,self.bodies,inputs=[self.slots,self.counts,self.offsets,self.slot_capacity,self.bodies+1,self.keys,self.values],device='cuda:0')
            w.utils.radix_sort_pairs(self.keys,self.values,total)
            w.launch(self.decode,total,inputs=[self.keys,self.bodies+1,self.pairs],device='cuda:0')
        self.number=total
        return total
