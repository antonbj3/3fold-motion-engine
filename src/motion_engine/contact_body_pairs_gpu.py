"""Conservative GPU body-pair search for uniform-width bead clouds."""
import numpy as np
from motion_engine.contact_engine_gpu import _R,_MAXC


def build_kernels(wp):
    @wp.kernel
    def scale(points:wp.array(dtype=wp.vec3),width:int,maximum:wp.array(dtype=float)):
        body=wp.tid();value=float(1.0)
        for k in range(width):
            p=points[body*width+k]
            value=wp.max(value,wp.max(wp.abs(p[0]),wp.max(wp.abs(p[1]),wp.abs(p[2]))))
        wp.atomic_max(maximum,0,value)

    @wp.kernel(module='unique',module_options={'fuse_fp':False})
    def bounds(points:wp.array(dtype=wp.vec3),width:int,maximum:wp.array(dtype=float),radius_limit:float,
               coordinate_limit:float,lower:wp.array(dtype=wp.vec3d),upper:wp.array(dtype=wp.vec3d),
               centers:wp.array(dtype=wp.vec3),flags:wp.array(dtype=int)):
        body=wp.tid();first=points[body*width]
        lo=wp.vec3d(wp.float64(first[0]),wp.float64(first[1]),wp.float64(first[2]));hi=lo
        for k in range(width):
            p=points[body*width+k]
            for axis in range(3):
                if not wp.isfinite(p[axis]):wp.atomic_max(flags,0,1)
                lo[axis]=wp.min(lo[axis],wp.float64(p[axis]));hi[axis]=wp.max(hi[axis],wp.float64(p[axis]))
        padding=float(0.00000095367431640625)*maximum[0]
        radius=_R+padding
        for axis in range(3):lo[axis]-=wp.float64(radius);hi[axis]+=wp.float64(radius)
        middle=(lo+hi)*wp.float64(.5)
        center=wp.vec3(float(middle[0]),float(middle[1]),float(middle[2]))
        extent=(hi-lo)*wp.float64(.5)
        for axis in range(3):extent[axis]+=wp.abs(middle[axis]-wp.float64(center[axis]))
        if wp.length(extent)>wp.float64(radius_limit) or maximum[0]>coordinate_limit:
            wp.atomic_max(flags,0,1)
        lower[body]=lo;upper[body]=hi;centers[body]=center

    @wp.func
    def overlaps(a:int,b:int,lower:wp.array(dtype=wp.vec3d),upper:wp.array(dtype=wp.vec3d)):
        return lower[a][0]<=upper[b][0] and lower[b][0]<=upper[a][0] and lower[a][1]<=upper[b][1] and lower[b][1]<=upper[a][1] and lower[a][2]<=upper[b][2] and lower[b][2]<=upper[a][2]

    @wp.kernel
    def count(centers:wp.array(dtype=wp.vec3),lower:wp.array(dtype=wp.vec3d),upper:wp.array(dtype=wp.vec3d),
              grid:wp.uint64,query_radius:float,counts:wp.array(dtype=int),flags:wp.array(dtype=int)):
        a=wp.tid();total=int(0)
        if lower[a][2]<wp.float64(0.0):total=1
        query=wp.hash_grid_query(grid,centers[a],query_radius);b=int(0)
        while wp.hash_grid_query_next(query,b):
            if b>a and overlaps(a,b,lower,upper):total+=1
        counts[a]=total;wp.atomic_add(flags,1,total)

    @wp.kernel
    def emit(centers:wp.array(dtype=wp.vec3),lower:wp.array(dtype=wp.vec3d),upper:wp.array(dtype=wp.vec3d),
             grid:wp.uint64,query_radius:float,stride:wp.int64,offsets:wp.array(dtype=int),
             keys:wp.array(dtype=wp.int64),values:wp.array(dtype=int)):
        a=wp.tid();index=offsets[a]
        if lower[a][2]<wp.float64(0.0):
            keys[index]=wp.int64(a)*stride;values[index]=index;index+=1
        query=wp.hash_grid_query(grid,centers[a],query_radius);b=int(0)
        while wp.hash_grid_query_next(query,b):
            if b>a and overlaps(a,b,lower,upper):
                keys[index]=wp.int64(a)*stride+wp.int64(b)+wp.int64(1);values[index]=index;index+=1

    @wp.kernel
    def decode(keys:wp.array(dtype=wp.int64),stride:wp.int64,pairs:wp.array(dtype=wp.vec2i)):
        i=wp.tid();key=keys[i]
        pairs[i]=wp.vec2i(int(key//stride),int(key%stride)-1)
    return scale,bounds,count,emit,decode


class BodyPairSearch:
    def __init__(self,wp,bodies,width,capacity=None,radius_limit=.35,coordinate_limit=10000.):
        if not isinstance(bodies,int) or bodies<=0 or not isinstance(width,int) or not 1<=width<=32:
            raise ValueError('Positive body count and width1..32 required')
        if capacity is None:capacity=min(_MAXC,64*bodies)
        if not isinstance(capacity,int) or not 0<=capacity<=_MAXC:raise ValueError('Invalid pair capacity')
        if not np.isfinite(radius_limit) or not np.isfinite(coordinate_limit) or radius_limit<=0 or coordinate_limit<=0:
            raise ValueError('Positive finite spatial bounds required')
        self.wp=wp;self.bodies=bodies;self.width=width;self.capacity=capacity
        self.radius_limit=float(radius_limit);self.coordinate_limit=float(coordinate_limit)
        self.query_radius=float(np.nextafter(np.float32(2*radius_limit+8*np.finfo(np.float32).eps*coordinate_limit),np.float32(np.inf)))
        self.maximum=wp.zeros(1,dtype=float,device='cuda:0');self.flags=wp.zeros(2,dtype=int,device='cuda:0')
        self.lower=wp.empty(bodies,dtype=wp.vec3d,device='cuda:0');self.upper=wp.empty_like(self.lower)
        self.centers=wp.empty(bodies,dtype=wp.vec3,device='cuda:0');self.counts=wp.zeros(bodies,dtype=int,device='cuda:0')
        self.offsets=wp.zeros_like(self.counts);self.keys=wp.zeros(2*max(1,capacity),dtype=wp.int64,device='cuda:0')
        self.values=wp.zeros(2*max(1,capacity),dtype=int,device='cuda:0');self.pairs=wp.zeros(max(1,capacity),dtype=wp.vec2i,device='cuda:0')
        self.grid=wp.HashGrid(256,256,8,device='cuda:0')
        self.scale,self.bounds,self.count,self.emit,self.decode=build_kernels(wp)
        self.number=0

    def rebuild(self,points):
        self.number=0
        if len(points)!=self.bodies*self.width or points.dtype!=self.wp.vec3:raise ValueError('Point layout mismatch')
        self.number=0;w=self.wp;self.maximum.zero_();self.flags.zero_()
        w.launch(self.scale,self.bodies,inputs=[points,self.width,self.maximum],device='cuda:0')
        w.launch(self.bounds,self.bodies,inputs=[points,self.width,self.maximum,self.radius_limit,self.coordinate_limit,self.lower,self.upper,self.centers,self.flags],device='cuda:0')
        if int(self.flags.numpy()[0]):raise RuntimeError('Spatial bound exceeded or nonfinite point')
        self.grid.build(self.centers,self.query_radius)
        w.launch(self.count,self.bodies,inputs=[self.centers,self.lower,self.upper,self.grid.id,self.query_radius,self.counts,self.flags],device='cuda:0')
        total=int(self.flags.numpy()[1])
        if total>self.capacity:raise RuntimeError('Body-pair capacity exceeded')
        if total:
            w.utils.array_scan(self.counts,self.offsets,inclusive=False)
            w.launch(self.emit,self.bodies,inputs=[self.centers,self.lower,self.upper,self.grid.id,self.query_radius,self.bodies+1,self.offsets,self.keys,self.values],device='cuda:0')
            w.utils.radix_sort_pairs(self.keys,self.values,total)
            w.launch(self.decode,total,inputs=[self.keys,self.bodies+1,self.pairs],device='cuda:0')
        self.number=total
        return total
