"""Sampled static-part contacts with the frozen bead body-body solver.

The caller supplies a negative-inside distance grid enclosing a fixed part.
Dynamic bodies retain the original bead geometry. No STEP parser or dynamic
sampled-body inertia is supplied by this module.
"""
import numpy as np
from motion_engine.contact_engine_gpu_prepared_planar import PreparedPlanarContactEngine
from motion_engine.contact_engine_gpu_colored import _R,_MAXC


def build_kernels(wp):
    @wp.func
    def sample(field:wp.array3d(dtype=float),origin:wp.vec3d,pitch:wp.float64,p:wp.vec3):
        u=(wp.vec3d(wp.float64(p[0]),wp.float64(p[1]),wp.float64(p[2]))-origin)/pitch
        ix=int(wp.floor(u[0]));iy=int(wp.floor(u[1]));iz=int(wp.floor(u[2]))
        result=wp.vec4d(wp.float64(1.0e30),wp.float64(0.),wp.float64(0.),wp.float64(0.))
        if ix>=0 and iy>=0 and iz>=0 and ix+1<field.shape[0] and iy+1<field.shape[1] and iz+1<field.shape[2]:
            f=u-wp.vec3d(wp.float64(ix),wp.float64(iy),wp.float64(iz));result=wp.vec4d()
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        wx=wp.float64(1.)-f[0];wy=wp.float64(1.)-f[1];wz=wp.float64(1.)-f[2]
                        sx=wp.float64(-1.);sy=wp.float64(-1.);sz=wp.float64(-1.)
                        if x==1:wx=f[0];sx=wp.float64(1.)
                        if y==1:wy=f[1];sy=wp.float64(1.)
                        if z==1:wz=f[2];sz=wp.float64(1.)
                        v=wp.float64(field[ix+x,iy+y,iz+z])
                        result[0]+=v*((wx*wy)*wz)
                        result[1]+=v*sx*(wy*wz)/pitch
                        result[2]+=v*sy*(wx*wz)/pitch
                        result[3]+=v*sz*(wx*wy)/pitch
        return result

    @wp.kernel(module='unique',module_options={'fuse_fp':False})
    def query(field:wp.array3d(dtype=float),origin:wp.vec3d,pitch:wp.float64,points:wp.array(dtype=wp.vec3),out:wp.array(dtype=wp.vec4d)):
        i=wp.tid();out[i]=sample(field,origin,pitch,points[i])

    @wp.kernel(module='unique',module_options={'fuse_fp':False})
    def ground(field:wp.array3d(dtype=float),origin:wp.vec3d,pitch:wp.float64,
               points:wp.array(dtype=wp.vec3),owner:wp.array(dtype=int),cnt:wp.array(dtype=int),invalid:wp.array(dtype=int),
               bi:wp.array(dtype=int),bj:wp.array(dtype=int),pa:wp.array(dtype=wp.vec3),pb:wp.array(dtype=wp.vec3),normal:wp.array(dtype=wp.vec3),pen:wp.array(dtype=float),fa:wp.array(dtype=int),fb:wp.array(dtype=int)):
        i=wp.tid();p=points[i];value=sample(field,origin,pitch,p)
        distance=float(value[0]);g=wp.vec3(float(value[1]),float(value[2]),float(value[3]));length=wp.length(g)
        if not wp.isfinite(distance) or not wp.isfinite(p[0]) or not wp.isfinite(p[1]) or not wp.isfinite(p[2]):wp.atomic_add(invalid,0,1)
        elif distance<_R:
            if not wp.isfinite(length) or length<1.0e-12:wp.atomic_add(invalid,0,1)
            else:
                j=wp.atomic_add(cnt,0,1)
                if j<_MAXC:
                    bi[j]=owner[i];bj[j]=-1;pa[j]=p;pb[j]=wp.vec3();normal[j]=g/length
                    pen[j]=_R-distance;fa[j]=i;fb[j]=-1

    @wp.kernel
    def body(points:wp.array(dtype=wp.vec3),owner:wp.array(dtype=int),grid:wp.uint64,cnt:wp.array(dtype=int),
             bi:wp.array(dtype=int),bj:wp.array(dtype=int),pa:wp.array(dtype=wp.vec3),pb:wp.array(dtype=wp.vec3),normal:wp.array(dtype=wp.vec3),pen:wp.array(dtype=float),fa:wp.array(dtype=int),fb:wp.array(dtype=int)):
        gid=wp.tid();xi=points[gid];a=owner[gid]
        qy=wp.hash_grid_query(grid,xi,2.0*_R);j=int(0)
        while wp.hash_grid_query_next(qy,j):
            if j>gid and owner[j]!=a:
                dvec=xi-points[j];dist=wp.length(dvec)
                if dist<2.0*_R and dist>1e-9:
                    idx=wp.atomic_add(cnt,0,1)
                    if idx<_MAXC:
                        bi[idx]=a;bj[idx]=owner[j];pa[idx]=xi;pb[idx]=points[j]
                        normal[idx]=dvec/dist;pen[idx]=2.0*_R-dist;fa[idx]=gid;fb[idx]=j
    return query,ground,body


def validate_field(field,origin,pitch):
    a=np.asarray(field);o=np.asarray(origin,dtype=np.float64)
    if a.ndim!=3 or min(a.shape)<2 or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Finite float32 volume required')
    if o.shape!=(3,) or not np.isfinite(o).all():raise ValueError('Finite origin triple required')
    if isinstance(pitch,(bool,np.bool_)) or not np.isscalar(pitch) or not np.isfinite(pitch) or pitch<=0:raise ValueError('Positive finite pitch required')
    if not np.any(a<0):raise ValueError('Field must enclose a nonempty part')
    faces=[np.take(a,k,axis=axis) for axis in range(3) for k in (0,a.shape[axis]-1)]
    if min(float(f.min()) for f in faces)<=_R:raise ValueError('Positive distance padding must exceed bead radius')
    return np.array(a,copy=True,order='C'),o.copy(),float(pitch)


class SampledFieldContactEngine(PreparedPlanarContactEngine):
    def __init__(self,field,origin,pitch,**kwargs):
        self._field_np,self._origin_np,self._pitch=validate_field(field,origin,pitch)
        super().__init__(**kwargs)
        self._query,self._ground,self._body=build_kernels(self.wp)

    def _build(self):
        super()._build();w=self.wp
        self._field=w.array(self._field_np,dtype=float,device=self.dev)
        self._origin=w.vec3d(*self._origin_np)
        self._invalid=w.zeros(1,dtype=int,device=self.dev)

    def _generate_field(self):
        w=self.wp;self.grid.build(self.allp,2.0*_R);self.cnt.zero_();self._invalid.zero_()
        outputs=[self.cbi,self.cbj,self.cpA,self.cpB,self.cn,self.cpen,self.cfa,self.cfb]
        w.launch(self._body,self.N*self.P,inputs=[self.allp,self.owner,self.grid.id,self.cnt,*outputs],device=self.dev)
        w.launch(self._ground,self.N*self.P,inputs=[self._field,self._origin,self._pitch,self.allp,self.owner,self.cnt,self._invalid,*outputs],device=self.dev)
        count=int(self.cnt.numpy()[0])
        if int(self._invalid.numpy()[0]):raise RuntimeError('Invalid field contact or point')
        if count>_MAXC:raise RuntimeError('Raw contact capacity exceeded')
        return count

    def step(s, dt, substeps=1):
        if not s._built:
            s._build()
        wp = s.wp
        K = s.K
        KC = s.KC
        for _ in range(substeps):
            sdt = dt / substeps
            with s._stage("generate"):
                wp.launch(K["grav"], s.N, inputs=[s.v, s.invM, sdt], device=s.dev)
                wp.launch(K["invIw"], s.N, inputs=[s.q, s.IbInv, s._kin_wp, s.invIw], device=s.dev)
                wp.launch(K["worldp"], s.N * s.P,
                          inputs=[s.xc, s.q, s.rest, s.P, s.allp, s.owner], device=s.dev)
                C = s._generate_field()
            s.n_contacts = C
            wp.copy(s._force_before, s.v)
            if C > 0:
                C = s._device_manifold(C)
                s.n_contacts_solved = C
                s.pv.zero_()
                s.po.zero_()
                with s._stage("pairs"):
                    P = s._build_pairs(C)
                s.n_pairs = P
                with s._stage("colour"):
                    ncol = s._color_pairs(P)
                s.n_colors = ncol
                with s._stage("solve"):
                    wp.capture_launch(s._solve_chunk_graph(P, ncol, sdt))
                if s.warm_start:
                    with s._stage("warm_store"):
                        s._store_warm_device(C)
            else:
                s.n_colors = 0
                s.n_contacts_solved = 0
                s.n_pairs = 0
                s.pv.zero_()
                s.po.zero_()
                s._ws_keys = np.zeros(0, np.int64)
                s._ws_imp = np.zeros((0, 3))
                s._ws_n = 0
            with s._stage("integrate"):
                wp.launch(K["integ"], s.N,
                          inputs=[s.xc, s.q, s.v, s.w, s.pv, s.po, s.invM, sdt], device=s.dev)
            s._force_dt = sdt
            s._prof_steps += 1
        wp.synchronize_device(s.dev)


if __name__=='__main__':
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'probes/innovation_sampled_field_contact.py'),run_name='__main__')
