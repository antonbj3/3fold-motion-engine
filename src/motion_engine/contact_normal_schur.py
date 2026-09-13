"""Fixed-order dense normal-contact quadratic control for small systems.

Solve min 0.5*x.T*A*x + b.T*x, x >= 0, with a bounded active set.
This is a classical active-set solve, not the JGS2 elastodynamic algorithm.
"""
import math


def dot(a, b):
    result = 0.0
    for x, y in zip(a, b):
        result = result + x*y
    return result


def linear_solve(matrix, rhs):
    n = len(rhs)
    lower = [[0.0]*n for _ in range(n)]
    scale = max(matrix[i][i] for i in range(n))
    for i in range(n):
        for j in range(i+1):
            value = matrix[i][j]
            for k in range(j):
                value = value - lower[i][k]*lower[j][k]
            if i == j:
                if value <= 1e-13*scale:
                    raise ValueError('Dependent or ill-conditioned active constraints')
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value/lower[j][j]
    y = [0.0]*n
    for i in range(n):
        value = rhs[i]
        for j in range(i):
            value = value - lower[i][j]*y[j]
        y[i] = value/lower[i][i]
    out = [0.0]*n
    for i in range(n-1, -1, -1):
        value = y[i]
        for j in range(i+1, n):
            value = value - lower[j][i]*out[j]
        out[i] = value/lower[i][i]
    return out


def solve_normal(matrix, rhs, max_solves=40):
    a = [[float(value) for value in row] for row in matrix]
    b = [float(value) for value in rhs]
    n = len(b)
    if n > 256 or len(a) != n or any(len(row) != n for row in a):
        raise ValueError('Expected square contact operator with at most256rows')
    if isinstance(max_solves, bool) or not isinstance(max_solves, int) or max_solves < 1:
        raise ValueError('Positive integer solve budget required')
    if not all(math.isfinite(v) for row in a for v in row) or not all(math.isfinite(v) for v in b):
        raise ValueError('Nonfinite quadratic input')
    if any(a[i][i] <= 0 for i in range(n)) or any(a[i][j] != a[j][i] for i in range(n) for j in range(i)):
        raise ValueError('Symmetric positive-diagonal operator required')
    x = [0.0]*n
    active = []
    solves = 0
    tolerance = 1e-10
    while True:
        gradient = [dot(row, x)+value for row, value in zip(a, b)]
        inactive = [i for i in range(n) if i not in active]
        entering = min(inactive, key=lambda i: (gradient[i], i)) if inactive else None
        if entering is None or gradient[entering] >= -tolerance:
            kkt = max([0.0]+[abs(gradient[i]) if x[i] > 0 else max(0., -gradient[i]) for i in range(n)])
            if kkt > 1e-9:
                raise ValueError('Normal KKT residual failed')
            return dict(impulses=x, solves=solves, kkt=kkt, active=active)
        active.append(entering)
        active.sort()
        while True:
            if solves >= max_solves:
                raise ValueError('Normal solve budget exhausted')
            values = linear_solve([[a[i][j] for j in active] for i in active], [-b[i] for i in active])
            solves += 1
            z = [0.0]*n
            for i, value in zip(active, values):
                z[i] = value
            nonpositive = [i for i in active if z[i] <= 0]
            if not nonpositive:
                x = z
                break
            ratios = [(x[i]/(x[i]-z[i]) if x[i] != z[i] else 0., i) for i in nonpositive]
            alpha, leaving = min(ratios)
            x = [old+alpha*(new-old) for old, new in zip(x, z)]
            x[leaving] = 0.
            active.remove(leaving)
            if not active:
                break
