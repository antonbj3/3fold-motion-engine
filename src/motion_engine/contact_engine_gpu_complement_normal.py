"""Hybrid small-system normal coupling beside frozen int64 Jacobi.

CPU fixed-order active linear solves feed integer GPU accumulation. No throughput
or full JGS2 implementation claim. Unsupported capacity/rank/budget is refused.
"""
import math
import numpy as np
from motion_engine.contact_engine_gpu import RelaxedJacobiContactEngine, _S, _R, _MAXC
from motion_engine.contact_normal_schur import dot, solve_normal


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


class ComplementNormalContactEngine(RelaxedJacobiContactEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.vit < 2:
            raise ValueError('At least two velocity rounds required')
        self.normal_solves = 0
        self.jacobi_rounds = 0
        self.normal_kkt = 0.

    def _normal_project(self, count):
        if self.N > 32 or count > 256:
            raise ValueError('Complement normal capacity exceeded')
        fields = {key: getattr(self, key).numpy()[:count] for key in ('cbi', 'cbj', 'cpA', 'cpB', 'cn')}
        bodies = {key: getattr(self, key).numpy().tolist() for key in ('xc', 'v', 'w', 'invM', 'invIw')}
        order = sorted(range(count), key=lambda k: (int(fields['cbi'][k]), int(fields['cbj'][k]),
                       *fields['cpA'][k].tolist(), *fields['cpB'][k].tolist(), *fields['cn'][k].tolist()))
        jacobian = []
        weighted = []
        for k in order:
            row = [0.]*(6*self.N)
            normal = fields['cn'][k].tolist()
            for body, point, sign in ((int(fields['cbi'][k]), fields['cpA'][k].tolist(), 1.),
                                      (int(fields['cbj'][k]), fields['cpB'][k].tolist(), -1.)):
                if body < 0:
                    continue
                arm = [point[d]-bodies['xc'][body][d] for d in range(3)]
                angular = cross(arm, normal)
                for d in range(3):
                    row[6*body+d] = sign*normal[d]
                    row[6*body+3+d] = sign*angular[d]
            mass_row = [0.]*(6*self.N)
            for body in range(self.N):
                for d in range(3):
                    mass_row[6*body+d] = bodies['invM'][body]*row[6*body+d]
                    mass_row[6*body+3+d] = dot(bodies['invIw'][body][d], row[6*body+3:6*body+6])
            jacobian.append(row)
            weighted.append(mass_row)
        matrix = [[0.]*count for _ in range(count)]
        for i in range(count):
            for j in range(i+1):
                matrix[i][j] = matrix[j][i] = .5*(dot(jacobian[i], weighted[j])+dot(jacobian[j], weighted[i]))
        velocity = [value for body in range(self.N) for value in bodies['v'][body]+bodies['w'][body]]
        result = solve_normal(matrix, [dot(row, velocity) for row in jacobian], max_solves=self.vit-1)
        sums = [0]*(6*self.N)
        impulses = np.zeros(count, dtype=np.float32)
        for local, impulse in enumerate(result['impulses']):
            if not math.isfinite(impulse) or impulse > np.finfo(np.float32).max:
                raise ValueError('Normal impulse range exceeded')
            impulses[order[local]] = impulse
            for d, weight in enumerate(weighted[local]):
                contribution = weight*impulse*_S
                if not math.isfinite(contribution) or abs(contribution) >= 2**62:
                    raise ValueError('Normal fixed-point contribution range exceeded')
                sums[d] += round(contribution)
                if abs(sums[d]) >= 2**62:
                    raise ValueError('Normal fixed-point accumulator range exceeded')
        for d in range(6):
            self.acc[d].assign(np.array([sums[6*body+d] for body in range(self.N)], dtype=np.int64))
        self.wp.copy(self.jn, self.wp.array(impulses, dtype=float, device=self.dev), count=count)
        self.wp.launch(self.K['apply'], self.N, inputs=[self.v, self.w, 1., _S, *self.acc], device=self.dev)
        self.normal_solves = result['solves']
        self.jacobi_rounds = self.vit-result['solves']
        self.normal_kkt = result['kkt']
        return result['solves']

    def step(s, dt, substeps=1):
        if not s._built:
            s._build()
        wp = s.wp
        K = s.K
        for _ in range(substeps):
            sdt = dt / substeps
            v_pre = s.v.numpy().copy()
            wp.launch(K['grav'], s.N, inputs=[s.v, s.invM, sdt], device=s.dev)
            wp.launch(K['invIw'], s.N, inputs=[s.q, s.IbInv, wp.array(s._kinarr, dtype=int, device=s.dev), s.invIw], device=s.dev)
            wp.launch(K['worldp'], s.N * s.P, inputs=[s.xc, s.q, s.rest, s.P, s.allp, s.owner], device=s.dev)
            s.grid.build(s.allp, 2.0 * _R)
            s.cnt.zero_()
            wp.launch(K['gen'], s.N * s.P, inputs=[s.allp, s.owner, s.grid.id, s.cnt, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.cpen], device=s.dev)
            C = int(s.cnt.numpy()[0])
            C = min(C, _MAXC)
            v_grav = s.v.numpy().copy()
            if C > 0:
                s.jn.zero_()
                s.jt1.zero_()
                s.jt2.zero_()
                s.jp.zero_()
                s.pv.zero_()
                s.po.zero_()
                for a in s.acc:
                    a.zero_()
                for _ in range(s.vit - s._normal_project(C)):
                    wp.launch(K['jac_vel'], C, inputs=[C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.jn, s.jt1, s.jt2, s.xc, s.v, s.w, s.invIw, s.invM, s.mu, _S, *s.acc], device=s.dev)
                    wp.launch(K['apply'], s.N, inputs=[s.v, s.w, s.relax, _S, *s.acc], device=s.dev)
                for _ in range(s.pit):
                    wp.launch(K['jac_pos'], C, inputs=[C, s.cbi, s.cbj, s.cpA, s.cpB, s.cn, s.cpen, s.jp, s.xc, s.pv, s.po, s.invIw, s.invM, sdt, _S, *s.acc], device=s.dev)
                    wp.launch(K['apply'], s.N, inputs=[s.pv, s.po, s.relax, _S, *s.acc], device=s.dev)
            wp.launch(K['integ'], s.N, inputs=[s.xc, s.q, s.v, s.w, s.pv, s.po, s.invM, sdt], device=s.dev)
            v_post = s.v.numpy()
            fin = np.zeros((s.N, 3))
            nk = s._kinarr == 0
            fin[nk] = s._M[nk, None] * (v_post[nk] - v_grav[nk]) / sdt
            s._cforce = fin
