"""Fixed-order raw bead contacts from supplied conservative body pairs.

Forward-only CUDA prototype; broad-phase construction is a caller responsibility.
"""
import numpy as np
from motion_engine.contact_engine_gpu import _R,_MAXC


def build_kernels(wp):
    @wp.func_native('return __ballot_sync(0xffffffffu, hit);')
    def ballot(hit:int)->wp.uint32: ...

    @wp.func_native('return __popc(word);')
    def popcount(word:wp.uint32)->int: ...

    @wp.kernel(module='unique',module_options={'enable_backward':False})
    def count(points:wp.array(dtype=wp.vec3),pairs:wp.array(dtype=wp.vec2i),width:int,words:int,
              masks:wp.array2d(dtype=wp.uint32),counts:wp.array(dtype=int)):
        tid=wp.tid();pair=tid//32;lane=tid%32
        a=pairs[pair][0];b=pairs[pair][1];total=int(0)
        for word in range(words):
            slot=word*32+lane;hit=int(0)
            if b<0:
                if slot<width:
                    if points[a*width+slot][2]-_R<0.0:hit=1
            else:
                if slot<width*width:
                    i=a*width+slot//width;j=b*width+slot%width
                    delta=points[i]-points[j]
                    bound=float(0.10000000894069672)
                    if wp.abs(delta[0])<=bound and wp.abs(delta[1])<=bound and wp.abs(delta[2])<=bound:
                        distance=wp.length(delta)
                        if distance<2.0*_R and distance>1e-9:hit=1
            mask=ballot(hit)
            if lane==0:
                masks[pair,word]=mask
                total+=popcount(mask)
        if lane==0:counts[pair]=total

    @wp.kernel(module='unique',module_options={'enable_backward':False})
    def emit(points:wp.array(dtype=wp.vec3),pairs:wp.array(dtype=wp.vec2i),width:int,words:int,
             masks:wp.array2d(dtype=wp.uint32),offsets:wp.array(dtype=int),capacity:int,overflow:wp.array(dtype=int),
             bi:wp.array(dtype=int),bj:wp.array(dtype=int),pa:wp.array(dtype=wp.vec3),pb:wp.array(dtype=wp.vec3),
             normal:wp.array(dtype=wp.vec3),pen:wp.array(dtype=float),fa:wp.array(dtype=int),fb:wp.array(dtype=int)):
        tid=wp.tid();pair=tid//32;lane=tid%32
        a=pairs[pair][0];b=pairs[pair][1];base=offsets[pair]
        for word in range(words):
            mask=masks[pair,word]
            bit=wp.uint32(1)<<wp.uint32(lane)
            if (mask&bit)!=wp.uint32(0):
                target=base+popcount(mask&(bit-wp.uint32(1)))
                if target>=capacity:
                    wp.atomic_add(overflow,0,1)
                else:
                    slot=word*32+lane
                    i=a*width+slot//width;j=b*width+slot%width
                    if b<0:i=a*width+slot;j=-1
                    xi=points[i]
                    bi[target]=a;bj[target]=b;pa[target]=xi;fa[target]=i;fb[target]=j
                    if b<0:
                        pb[target]=wp.vec3(0.0,0.0,0.0);normal[target]=wp.vec3(0.0,0.0,1.0)
                        pen[target]=-(xi[2]-_R)
                    else:
                        delta=xi-points[j];distance=wp.length(delta)
                        pb[target]=points[j];normal[target]=delta/distance;pen[target]=2.0*_R-distance
            base+=popcount(mask)
    return count,emit


class PairBitmaskGenerator:
    def __init__(self,wp,points,width,pairs,capacity=_MAXC):
        points=np.asarray(points);pairs=np.asarray(pairs)
        if points.ndim!=2 or points.shape[1]!=3 or points.dtype!=np.float32 or not np.isfinite(points).all():
            raise ValueError('Finite float32 point triples required')
        if not isinstance(width,int) or not 1<=width<=32 or len(points)%width:raise ValueError('Invalid body width')
        bodies=len(points)//width
        if pairs.ndim!=2 or pairs.shape[1]!=2 or not np.issubdtype(pairs.dtype,np.integer) or len(pairs)==0:
            raise ValueError('Nonempty integer body pairs required')
        if np.any(pairs[:,0]<0) or np.any(pairs[:,0]>=bodies) or np.any(pairs[:,1]<-1) or np.any(pairs[:,1]>=bodies):
            raise ValueError('Pair endpoint outside body set')
        if np.any((pairs[:,1]>=0)&(pairs[:,1]<=pairs[:,0])) or len(np.unique(pairs,axis=0))!=len(pairs):
            raise ValueError('Unique forward body pairs required')
        if not isinstance(capacity,int) or not 0<=capacity<=_MAXC:raise ValueError('Invalid capacity')
        self.wp=wp;self.width=width;self.words=(width*width+31)//32;self.capacity=capacity;self.number=len(pairs)
        self.points=wp.array(points,dtype=wp.vec3,device='cuda:0')
        self.pairs=wp.array(pairs.astype(np.int32),dtype=wp.vec2i,device='cuda:0')
        self.masks=wp.zeros((self.number,self.words),dtype=wp.uint32,device='cuda:0')
        self.counts=wp.zeros(self.number,dtype=int,device='cuda:0')
        self.offsets=wp.zeros_like(self.counts);self.overflow=wp.zeros(1,dtype=int,device='cuda:0')
        self.outputs=[wp.zeros(max(1,capacity),dtype=t,device='cuda:0') for t in (int,int,wp.vec3,wp.vec3,wp.vec3,float,int,int)]
        self.count_kernel,self.emit_kernel=build_kernels(wp)

    def launch(self):
        w=self.wp
        self.overflow.zero_()
        w.launch(self.count_kernel,self.number*32,inputs=[self.points,self.pairs,self.width,self.words,self.masks,self.counts],device='cuda:0',block_dim=128)
        w.utils.array_scan(self.counts,self.offsets,inclusive=False)
        w.launch(self.emit_kernel,self.number*32,inputs=[self.points,self.pairs,self.width,self.words,self.masks,self.offsets,self.capacity,self.overflow,*self.outputs],device='cuda:0',block_dim=128)

    def result(self):
        count=int(self.counts.numpy().astype(np.int64).sum())
        overflow=int(self.overflow.numpy()[0])
        if count>self.capacity or overflow:raise RuntimeError('Contact capacity exceeded')
        return [a.numpy()[:count].copy() for a in self.outputs]
