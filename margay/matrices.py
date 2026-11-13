"""The horizontal basis: all matrices, with their conventions.

Each entry of CATALOG is a dict with
  name, N, stage, params (list of parameter names, empty if isolated),
  build(*params) -> numpy complex matrix, source (where the formula comes from).
Parameters are points of the unit circle, passed as complex numbers.
"""
import itertools
import numpy as np

E = lambda t: np.exp(2j * np.pi * t)          # e(t) = exp(2 pi i t)
w3 = E(1 / 3)


# ---------------------------------------------------------------- basics
def fourier(N):
    w = E(1 / N)
    return np.array([[w ** (i * j) for j in range(N)] for i in range(N)])


def kron(*Ms):
    out = np.array([[1.0 + 0j]])
    for M in Ms:
        out = np.kron(out, M)
    return out


def dita(H, K, Q):
    """Book convention (hm.tex, Thm 4.x): (H x_Q K)_{ia,jb} = Q_{ib} H_{ij} K_{ab}."""
    M, N = H.shape[0], K.shape[0]
    Q = np.asarray(Q)
    out = np.zeros((M * N, M * N), complex)
    for i, a, j, b in itertools.product(range(M), range(N), range(M), range(N)):
        out[i * N + a, j * N + b] = Q[i, b] * H[i, j] * K[a, b]
    return out


def dephased_Q(M, N, qs):
    """Q in M_{MxN}(T), first row and column 1, the rest filled row by row from qs."""
    Q = np.ones((M, N), complex)
    it = iter(qs)
    for i in range(1, M):
        for b in range(1, N):
            Q[i, b] = next(it)
    return Q


def from_log(L, root):
    """Logarithmic form: H_ij = omega^{L_ij}, omega = e^{2 pi i / root}."""
    return E(np.array(L, float) / root)


def from_hex(line):
    """McKay's format: one hex number per row, binary digit 0 -> -1."""
    rows = line.split()
    n = len(rows)
    M = [[1 if c == '1' else -1 for c in bin(int(r, 16))[2:].zfill(n)] for r in rows]
    return np.array(M, complex)


# ---------------------------------------------------------------- finite fields, for Paley
def gf(p, r):
    """Elements of GF(p^r) as tuples (coefficient vectors), with add, mul. Brute-force irreducible."""
    if r == 1:
        els = [(a,) for a in range(p)]
        return els, (lambda x, y: ((x[0] + y[0]) % p,)), (lambda x, y: ((x[0] * y[0]) % p,))
    # find monic irreducible of degree r
    for tail in itertools.product(range(p), repeat=r):
        poly = list(tail) + [1]       # coefficients c0..c_{r-1}, 1
        if all(sum(poly[k] * x ** k for k in range(r + 1)) % p for x in range(p)) and r <= 3:
            break
    def mul(x, y):
        prod = [0] * (2 * r - 1)
        for i in range(r):
            for j in range(r):
                prod[i + j] = (prod[i + j] + x[i] * y[j]) % p
        for d in range(2 * r - 2, r - 1, -1):
            c = prod[d]
            if c:
                for k in range(r + 1):
                    prod[d - r + k] = (prod[d - r + k] - c * poly[k]) % p
        return tuple(prod[:r])
    add = lambda x, y: tuple((a + b) % p for a, b in zip(x, y))
    els = list(itertools.product(range(p), repeat=r))
    return els, add, mul


def jacobsthal(p, r):
    els, add, mul = gf(p, r)
    zero = tuple([0] * r)
    squares = {mul(x, x) for x in els if x != zero}
    neg = lambda x: tuple((-a) % p for a in x)
    chi = lambda x: 0 if x == zero else (1 if x in squares else -1)
    return np.array([[chi(add(b, neg(a))) for b in els] for a in els])


def paley1(p, r=1):
    q = p ** r
    Q = jacobsthal(p, r)
    n = q + 1
    S = np.zeros((n, n))
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = Q
    return (np.eye(n) + S).astype(complex)


def paley2(p, r=1):
    q = p ** r
    Q = jacobsthal(p, r)
    C = np.zeros((q + 1, q + 1))
    C[0, 1:] = 1
    C[1:, 0] = 1
    C[1:, 1:] = Q
    G0 = np.array([[1, -1], [-1, -1]])
    F1 = np.array([[1, 1], [1, -1]])
    return (np.kron(C, F1) + np.kron(np.eye(q + 1), G0)).astype(complex)


def williamson(K):
    """First Williamson quadruple found by brute force (symmetric circulant, first entries +1)."""
    half = (K - 1) // 2
    vecs = []
    for bits in itertools.product([1, -1], repeat=half):
        v = [1] + list(bits) + list(bits[::-1])
        vecs.append(np.array(v))
    circ = lambda v: np.array([np.roll(v, i) for i in range(K)])
    sq = [(v, circ(v) @ circ(v)) for v in vecs]
    for (a, A2), (b, B2), (c, C2), (d, D2) in itertools.combinations_with_replacement(sq, 4):
        if np.array_equal(A2 + B2 + C2 + D2, 4 * K * np.eye(K)):
            A, B, C, D = map(circ, (a, b, c, d))
            return np.block([[A, B, C, D], [-B, A, -D, C], [-C, D, A, -B], [-D, -C, B, A]]).astype(complex)
    raise ValueError("no quadruple")


# ---------------------------------------------------------------- N = 4, 6 (book hm.tex)
def F4s(s):
    return np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, s, -1, -s], [1, -s, -1, s]], complex)


def T6():
    w = w3
    return np.array([[1, 1, 1, 1, 1, 1], [1, 1, w, w, w**2, w**2], [1, w, 1, w**2, w**2, w],
                     [1, w, w**2, 1, w, w**2], [1, w**2, w**2, w, 1, w], [1, w**2, w, w**2, w, 1]])


def H6(q):
    i, qb = 1j, np.conj(q)
    return np.array([[1, 1, 1, 1, 1, 1], [1, -1, i, i, -i, -i], [1, i, -1, -i, q, -q],
                     [1, i, -i, -1, -q, q], [1, -i, qb, -qb, i, -1], [1, -i, -qb, qb, -1, i]])


def F6rs(r, s):
    w = w3
    return np.array([[1, 1, 1, 1, 1, 1], [1, w, w**2, 1, w, w**2], [1, w**2, w, 1, w**2, w],
                     [1, r, s, -1, -r, -s], [1, w*r, w**2*s, -1, -w*r, -w**2*s],
                     [1, w**2*r, w*s, -1, -w**2*r, -w*s]])


def F6rs_col(r, s):
    w = w3
    return np.array([[1, 1, 1, 1, 1, 1], [1, -1, 1, -1, 1, -1], [1, r, w, w*r, w**2, w**2*r],
                     [1, -r, w, -w*r, w**2, -w**2*r], [1, s, w**2, w**2*s, w, w*s],
                     [1, -s, w**2, -w**2*s, w, -w*s]])


def BF6(branch=0):
    """Bjorck-Froberg circulant; a root of a^2 + (sqrt3 - 1) a + 1 = 0."""
    a = np.roots([1, np.sqrt(3) - 1, 1])[branch]
    i, ab = 1j, np.conj(a)
    row = np.array([1, i*a, -a, -i, -ab, i*ab])
    return np.array([np.roll(row, k) for k in range(6)])


def BN6(q):
    """Beauchamp-Nicoara self-adjoint family, book hm.tex Thm 5.x, principal branch of sqrt."""
    root = np.sqrt(2) * np.sqrt(1 + 2*q + 2*q**3 + q**4)
    x = (1 + 2*q + q**2 - root) / (1 + 2*q - q**2)
    y = q
    z = (1 + 2*q - q**2) / (q * (-1 + 2*q + q**2))
    t = (1 + 2*q + q**2 - root) / (-1 + 2*q + q**2)
    c = np.conj
    return np.array([[1, 1, 1, 1, 1, 1], [1, -1, c(x), -y, -c(x), y], [1, x, -1, t, -t, -x],
                     [1, -c(y), c(t), -1, c(y), -c(t)], [1, -x, -c(t), y, 1, c(z)],
                     [1, c(y), -c(x), -t, z, 1]])


def K6(theta, phi, psi1, signs=(1, 1, 1)):
    """Karlsson three-parameter family (arXiv 1003.4177), real parameters theta, phi, psi1."""
    F2 = np.array([[1, 1], [1, -1]], complex)
    Lam = np.array([[np.cos(theta), np.exp(1j*phi)*np.sin(theta)],
                    [np.exp(-1j*phi)*np.sin(theta), -np.cos(theta)]])
    A = F2 @ (-0.5*np.eye(2) + 1j*np.sqrt(3)/2*Lam)
    B = -F2 - A
    def mob(X, z):
        al, be = X[0, 1]**2, X[0, 0]**2
        return (al*z - be) / (np.conj(be)*z - np.conj(al))
    def mobinv(X, w):
        al, be = X[0, 1]**2, X[0, 0]**2
        return (np.conj(al)*w - be) / (np.conj(be)*w - al)
    z1 = np.exp(1j*psi1)
    z3 = signs[0]*np.sqrt(mob(A, z1**2))
    z4 = signs[1]*np.sqrt(mob(B, z1**2))
    z2 = signs[2]*np.sqrt(mobinv(A, mob(B, z1**2)))
    Zr = lambda z: np.array([[1, 1], [z, -z]])
    Zc = lambda z: np.array([[1, z], [1, -z]])
    Z1, Z2, Z3, Z4 = Zr(z1), Zr(z2), Zc(z3), Zc(z4)
    return np.block([[F2, Z1, Z2], [Z3, Z3@A@Z1/2, Z3@B@Z2/2], [Z4, Z4@B@Z1/2, Z4@A@Z2/2]])


# ---------------------------------------------------------------- N = 7 (paper 002)
def P7(q):
    w, qb = w3, np.conj(q)
    return np.array([[-q, q, w, 1, w, 1, w], [q, -q, w, 1, 1, w, w], [w, w, -w, 1, w, w, 1],
                     [1, 1, 1, -1, w, w, w], [w, 1, w, w, -qb*w, qb*w, 1],
                     [1, w, w, w, qb*w, -qb*w, 1], [w, w, 1, w, 1, 1, -1]])


def R7(q):
    w = w3
    return np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, w, -1, -1, w**2],
                     [1, -w**2, w**2, w**2, q, -q, w], [1, w, -1, 1, q*w**2, -q*w**2, w**2],
                     [1, -1, -w, -1, -q, q, -w**2], [1, w**2, w, q*w, q, -q*w, -q],
                     [1, w**2, w, -q*w, q*w, -q, q]])


# ---------------------------------------------------------------- N = 8 (paper 002)
def F85a(a, b, c, d, e):
    return np.array([[1]*8, [1, -1, -b*d, b*d, -b*d, b*d, 1, -1],
        [1, -1/c, -b*d/a, d, b*d/a, -d, -1, 1/c], [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1/c, b*d/a, -d, -b*d/a, d, -1, 1/c], [1, -1, b*d, -b*d, b*d, -b*d, 1, -1],
        [1, 1/c, e, a*e/b, -e, -a*e/b, -1, -1/c], [1, 1/c, -e, -a*e/b, e, a*e/b, -1, -1/c]], complex)


def F85b(a, b, c, d, e):
    return np.array([[1]*8, [1, -1, -b*d, b*d, -b*d, b*d, 1, -1],
        [1, -1/c, -b*d/a, d, b*d/a, -d, -1, 1/c], [1, 1/c, -b*d/a, -d, b*d/a, d, -1, -1/c],
        [1, 1, -1, -1, -1, -1, 1, 1], [1, -1, b*d, -b*d, b*d, -b*d, 1, -1],
        [1, e, b*d/a, c*d*e, -b*d/a, -c*d*e, -1, -e], [1, -e, b*d/a, -c*d*e, -b*d/a, c*d*e, -1, e]], complex)


def F85c(a, b, c, d, e):
    return np.array([[1]*8, [1, -1, a*d, -a*d, -b*d, b*d, 1, -1],
        [1, -1/c, d, d, -d, -d, -1, 1/c], [1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1/c, -d, -d, d, d, -1, 1/c], [1, -1, -a*d, a*d, b*d, -b*d, 1, -1],
        [1, 1/c, e, -e, b*e/a, -b*e/a, -1, -1/c], [1, 1/c, -e, e, -b*e/a, b*e/a, -1, -1/c]], complex)


def F84(a, b, c, d):
    return np.array([[1]*8, [1, -1, a*d, -a*d, -b*d, b*d, 1, -1],
        [1, -1/c, d, d, -d, -d, -1, 1/c], [1, 1/(c*d), -a*d, a*d, -1, -1, 1, -1/(c*d)],
        [1, 1/c, a*d**2, -a*d**2, b*d**2, -b*d**2, -1, -1/c], [1, d, -d, -d, -b*d**2, b*d**2, -1, d],
        [1, -1/(c*d), -1, -1, b*d, -b*d, 1, 1/(c*d)], [1, -d, -a*d**2, a*d**2, d, d, -1, -d]], complex)


def F84t(a, b, c, d):
    return np.array([[1]*8, [1, -1, -1/c, 1/(c*d), 1/c, d, -1/(c*d), -d],
        [1, a*d, d, -a*d, a*d**2, -d, -1, -a*d**2], [1, -a*d, d, a*d, -a*d**2, -d, -1, a*d**2],
        [1, -b*d, -d, -1, b*d**2, -b*d**2, b*d, d], [1, b*d, -d, -1, -b*d**2, b*d**2, -b*d, d],
        [1, 1, -1, 1, -1, -1, 1, -1], [1, -1, 1/c, -1/(c*d), -1/c, d, 1/(c*d), -d]], complex)


def F82(a, b):
    return np.array([[1]*8, [1, -1, b, -b, -b/a, b/a, a, -a],
        [1, -a, b/a, b, -b/a, -b, -1, a], [1, 1/a, b/a**2, -b/a, b/a, -b/a**2, -1/a, -1],
        [1, 1, -1, -1, -1, -1, 1, 1], [1, a, -b, b/a, -b/a, b, -a, -1],
        [1, -1/a, -b/a, -b/a**2, b/a, b/a**2, -1, 1/a], [1, -1, -b/a**2, b/a**2, b/a, -b/a, 1/a, -1/a]], complex)


def F82t(a, b):
    return np.array([[1]*8, [1, -1, -a, 1/a, 1, a, -1/a, -1],
        [1, b, b/a, b/a**2, -1, -b, -b/a, -b/a**2], [1, -b, b, -b/a, -1, b/a, -b/a**2, b/a**2],
        [1, -b/a, -b/a, b/a, -1, -b/a, b/a, b/a], [1, b/a, -b, -b/a**2, -1, b, b/a**2, -b/a],
        [1, a, -1, -1/a, 1, -a, -1, 1/a], [1, -a, a, -1, 1, -1, 1/a, -1/a]], complex)


def K82(a, b):
    return np.array([[1]*8, [1, a, -b**2, -b**2/a, -1, -a, b**2/a, b**2],
        [1, a/b, -1, b/a, 1, -a/b, -b/a, -1], [1, -b, b, b**2/a, -b/a, -1, b/a, -b**2/a],
        [1, -a, -b, b, -a/b, a/b, -1, a], [1, -1, b, -b/a, b/a, -b, -b**2/a, b**2/a],
        [1, -a/b, -b, -1, a/b, a, b, -a], [1, b, b**2, -b, -1, b, -b, -b**2]], complex)


def M81(a):
    w = w3
    return np.array([[1]*8, [1, -w*a**2, a, w*a, -1, w*a**2, -a, -w*a],
        [1, a, -a, a**3, -a**3, -a**4, a**4, -1], [1, w*a, -w/a, -w*a, w/a, w**2, -1, -w**2],
        [1, -1, w/a, -a**3, w*a**2, -w*a**2, a**3, -w/a],
        [1, w*a**2, w, -w*a**4, -w*a**2, -w**2*a**4, -a**4, w**2],
        [1, -a, -w, -1, -w/a, w, a, w/a], [1, -w*a, -1, w*a**4, a**3, -w*a**4, -a**3, w*a]], complex)


I812_L = [[0]*8, [0, 0, 0, 3, 6, 6, 6, 9], [0, 1, 5, 9, 1, 5, 9, 6], [0, 4, 8, 6, 10, 2, 3, 9],
          [0, 6, 2, 11, 9, 8, 3, 5], [0, 6, 11, 5, 3, 11, 9, 5], [0, 7, 6, 1, 7, 3, 9, 1],
          [0, 10, 6, 7, 4, 9, 3, 1]]
K830_L = [[0]*8, [0, 0, 6, 10, 12, 18, 20, 24], [0, 6, 13, 28, 27, 21, 12, 15],
          [0, 10, 28, 20, 16, 4, 10, 22], [0, 12, 27, 16, 1, 15, 24, 9], [0, 18, 21, 4, 15, 19, 6, 3],
          [0, 20, 12, 10, 24, 6, 0, 18], [0, 24, 15, 22, 9, 3, 18, 7]]
L830_L = [[0]*8, [0, 0, 6, 10, 12, 18, 20, 24], [0, 6, 21, 2, 19, 17, 4, 15],
          [0, 10, 17, 25, 8, 2, 15, 23], [0, 12, 4, 23, 27, 15, 19, 8], [0, 18, 2, 17, 15, 3, 1, 16],
          [0, 20, 19, 15, 4, 16, 5, 1], [0, 24, 15, 8, 23, 1, 16, 9]]
L830t_L = [[0]*8, [0, 0, 6, 10, 12, 18, 20, 24], [0, 6, 21, 17, 4, 2, 19, 15],
           [0, 10, 2, 25, 23, 17, 15, 8], [0, 12, 19, 8, 27, 15, 4, 23], [0, 18, 17, 2, 15, 3, 16, 1],
           [0, 20, 4, 15, 19, 1, 5, 16], [0, 24, 15, 23, 8, 16, 1, 9]]


# ---------------------------------------------------------------- N = 9, 10, 11 (Beauchamp-Nicoara Sec. 3)
def H9():
    y = (1 - 1j*np.sqrt(15)) / 4
    yi = 1 / y
    return np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, -1, y, yi, -y**-3, y**-3, y, yi],
        [1, y, y**3, -y**2, y**3, y, -1, -y, y],
        [1, y, y, -1, yi, y**-3, -y**-3, -1, yi],
        [1, -y, -y**2, y**3, y**3, -1, y, y, y],
        [1, y, y**3, y**3, y, yi, yi, y, y**2],
        [1, y**3, y, y, y**2, yi, yi, y**3, y],
        [1, -y**2, -y, y, y, -1, y, y**3, y**3],
        [1, y**3, y, -y, y, y, -1, -y**2, y**3]])


BN9_L = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 5, 3, 3, 5, 9, 8, 7, 1], [0, 4, 5, 7, 1, 3, 5, 9, 9],
         [0, 3, 7, 5, 1, 8, 9, 3, 5], [0, 9, 1, 5, 5, 3, 7, 2, 7], [0, 9, 5, 1, 3, 5, 1, 7, 6],
         [0, 1, 7, 9, 6, 1, 5, 5, 3], [0, 7, 9, 4, 9, 5, 3, 5, 1], [0, 5, 2, 9, 7, 7, 3, 1, 5]]


def BN10():
    a = (-1 + 1j*np.sqrt(15)) / 4
    A = lambda k: a ** k
    L = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, -2, -1, -2, 2, 0, 2, 0, 1],
         [0, 2, 0, -2, -2, 0, 1, 2, 0, -1],
         [0, 0, 0, 0, -2, 2, 2, 1, -2, -1],
         [0, 2, -2, 0, 0, 1, 0, 2, -2, -1],
         [0, 0, -1, -2, 0, 0, 2, 2, -2, 1],
         [0, 2, 0, -2, -1, 2, 0, 0, -2, 1],
         [0, 2, -2, 0, -2, 0, 2, 0, -1, 1],
         [0, 1, -2, -2, 0, 2, 2, 0, 0, -1],
         [0, 1, -1, -1, -1, 1, 1, 1, -1, 0]]
    return np.array([[A(k) for k in row] for row in L])


def N11(branch):
    x = 3/4 - 1j*np.sqrt(7)/4
    if branch == 0:
        a, b, y = -x, x, -x**2
    else:
        a, b, y = 1, -1, np.conj(x)
    return np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, y, y, a, b, b, -1, x, -1, x, -1],
        [1, y, b, b, a, y, x, x, -1, -1, -1],
        [1, a, b, y, y, b, -1, -1, x, -1, x],
        [1, b, a, y, b, y, x, -1, -1, -1, x],
        [1, b, y, b, y, a, -1, -1, x, x, -1],
        [1, -1, x, -1, x, -1, -x/a, -x/b, -x/y, -x/y, -x/b],
        [1, x, x, -1, -1, -1, -x/b, -x/y, -x/b, -x/y, -x/a],
        [1, -1, -1, x, -1, x, -x/y, -x/b, -x/b, -x/a, -x/y],
        [1, x, -1, -1, -1, x, -x/y, -x/y, -x/a, -x/b, -x/b],
        [1, -1, -1, x, x, -1, -x/b, -x/a, -x/y, -x/b, -x/y]])


# ---------------------------------------------------------------- real matrices
HAD16 = """FFFF FF00 F0F0 F00F CCCC CC33 C3C3 C33C AAAA AA55 A5A5 A55A 9999 9966 9696 9669
FFFF FF00 F0F0 F00F CCCC CC33 C3C3 C33C AAAA AA55 A5A5 A55A 9996 9969 9699 9666
FFFF FF00 F0F0 F00F CCCC CC33 C3C3 C33C AAAA AA55 A599 A566 9996 9969 96A5 965A
FFFF FF00 F0F0 F00F CCCC CC33 C3C3 C33C AAAA A995 A659 A566 9A56 9969 96A5 959A
FFFF FF00 F0F0 F00F CCCC CC33 C3AA C355 AAC3 AA3C A599 A566 9996 9969 96A5 965A""".splitlines()
HAD20 = """FFFFF FFC00 F83E0 F421E E3199 D3067 CC987 CC479 C2F54 C1EAA B0D4B AAA33 A94D6 A68EC A5725 9A68D 9993C 965B2 95AD1 8F34A
FFFFF FFC00 F83E0 F421E E3199 D3067 CC987 CC479 C2F54 C1EAA B0D33 AAA4B A94D6 A66A5 A596C 9A8BC 9970D 965CA 95AD1 8F332
FFFFF FFC00 F83E0 F421E E3199 D2947 CD0A7 CA47C C4F31 C1ECA B1473 AAA2B A8D96 A66C5 A596C 9C8D9 9970D 965AA 93AB4 8F352""".splitlines()


# ---------------------------------------------------------------- the catalog
def entry(name, N, stage, params, build, source):
    return dict(name=name, N=N, stage=stage, params=params, build=build, source=source)


CATALOG = []
add = CATALOG.append
# Stage 0
for n in range(2, 9):
    add(entry(f"F_{n}", n, 0, [], (lambda n=n: fourier(n)), "Fourier"))
add(entry("F_2^2 (Walsh W_4)", 4, 0, [], lambda: kron(fourier(2), fourier(2)), "Fourier"))
add(entry("F_2^3 (Walsh W_8)", 8, 0, [], lambda: kron(fourier(2), fourier(2), fourier(2)), "Fourier"))
add(entry("F_2 x F_4", 8, 0, [], lambda: kron(fourier(2), fourier(4)), "Fourier"))
add(entry("F_4^s", 4, 0, ["s"], F4s, "hm.tex Thm 5.13"))
# Stage 1, real
add(entry("P_12 (Paley I, q=11)", 12, 1, [], lambda: paley1(11), "hm.tex Thm 1.x"))
add(entry("P_12^II (Paley II, q=5)", 12, 1, [], lambda: paley2(5), "hm.tex Thm 1.x"))
add(entry("W_12 (Williamson, K=3)", 12, 1, [], lambda: williamson(3), "hm.tex Thm 1.x"))
for k, line in enumerate(HAD16):
    add(entry(f"had.16.{k}", 16, 1, [], (lambda line=line: from_hex(line)), "McKay library"))
for k, line in enumerate(HAD20):
    add(entry(f"had.20.{k}", 20, 1, [], (lambda line=line: from_hex(line)), "McKay library"))
add(entry("P_20 (Paley I, q=19)", 20, 1, [], lambda: paley1(19), "hm.tex"))
add(entry("P_20^II (Paley II, q=9)", 20, 1, [], lambda: paley2(3, 2), "hm.tex"))
add(entry("W_20 (Williamson, K=5)", 20, 1, [], lambda: williamson(5), "hm.tex"))
add(entry("P_24 (Paley I, q=23)", 24, 1, [], lambda: paley1(23), "hm.tex"))
add(entry("P_28 (Paley I, q=27)", 28, 1, [], lambda: paley1(3, 3), "hm.tex"))
add(entry("P_28^II (Paley II, q=13)", 28, 1, [], lambda: paley2(13), "hm.tex"))
add(entry("W_28 (Williamson, K=7)", 28, 1, [], lambda: williamson(7), "hm.tex"))
# Stage 2, Dita
add(entry("F_6^(rs) = F_2 x_Q F_3", 6, 2, ["r", "s"], F6rs, "hm.tex Prop 6.19"))
add(entry("F_6^(r/s) = F_3 x_Q F_2", 6, 2, ["r", "s"], F6rs_col, "hm.tex Prop 6.19"))
add(entry("F_2 x_Q F_4", 8, 2, ["q1", "q2", "q3"],
          lambda *q: dita(fourier(2), fourier(4), dephased_Q(2, 4, q)), "hm.tex Def 4.x"))
add(entry("F_4 x_Q F_2", 8, 2, ["q1", "q2", "q3"],
          lambda *q: dita(fourier(4), fourier(2), dephased_Q(4, 2, q)), "hm.tex Def 4.x"))
add(entry("W_4 x_Q F_2", 8, 2, ["q1", "q2", "q3"],
          lambda *q: dita(kron(fourier(2), fourier(2)), fourier(2), dephased_Q(4, 2, q)), "hm.tex Def 4.x"))
add(entry("F_2 x_Q W_4", 8, 2, ["q1", "q2", "q3"],
          lambda *q: dita(fourier(2), kron(fourier(2), fourier(2)), dephased_Q(2, 4, q)), "hm.tex Def 4.x"))
add(entry("F_3 x_Q F_3", 9, 2, ["q1", "q2", "q3", "q4"],
          lambda *q: dita(fourier(3), fourier(3), dephased_Q(3, 3, q)), "hm.tex Def 4.x"))
for nm, f, k in [("F_{8,5a}", F85a, 5), ("F_{8,5b}", F85b, 5), ("F_{8,5c}", F85c, 5),
                 ("F_{8,4}", F84, 4), ("F_{8,4}^t", F84t, 4), ("F_{8,2}", F82, 2), ("F_{8,2}^t", F82t, 2)]:
    add(entry(nm, 8, 2, list("abcde")[:k], f, "paper 002"))
# Stage 3, N = 6 wild candidates
add(entry("T_6 (Tao)", 6, 3, [], T6, "hm.tex Prop 6.16"))
add(entry("H_6^q (Haagerup)", 6, 3, ["q"], H6, "hm.tex Prop 6.17"))
add(entry("BF_6 (Bjorck-Froberg)", 6, 3, [], BF6, "hm.tex Prop 5.x"))
add(entry("BN_6^q (Beauchamp-Nicoara)", 6, 3, ["q"], BN6, "hm.tex Thm 5.x"))
add(entry("K_6 (Karlsson, 3 real params)", 6, 3, ["theta", "phi", "psi"],
          lambda th, ph, ps: K6(np.angle(th), np.angle(ph), np.angle(ps)), "arXiv 1003.4177"))
# Stage 4, N = 7
add(entry("P_7^q (Petrescu)", 7, 4, ["q"], P7, "paper 002 / hm.tex"))
add(entry("R_7^q", 7, 4, ["q"], R7, "paper 002"))
# Stage 5, N = 8 non-Dita
add(entry("K_{8,2}", 8, 5, ["a", "b"], K82, "paper 002"))
add(entry("M_{8,1}", 8, 5, ["a"], M81, "paper 002"))
add(entry("I_{8,12}", 8, 5, [], lambda: from_log(I812_L, 12), "paper 002"))
add(entry("K_{8,30}", 8, 5, [], lambda: from_log(K830_L, 30), "paper 002"))
add(entry("L_{8,30}", 8, 5, [], lambda: from_log(L830_L, 30), "paper 002"))
add(entry("L_{8,30}^t", 8, 5, [], lambda: from_log(L830t_L, 30), "paper 002"))
# Stage 6, N = 9, 10, 11
add(entry("H_9 (Beauchamp-Nicoara)", 9, 6, [], H9, "Beauchamp-Nicoara Sec. 3"))
add(entry("BN_9 (Beauchamp-Nicoara)", 9, 6, [], lambda: from_log(BN9_L, 10), "Beauchamp-Nicoara Sec. 3"))
add(entry("BN_10 (Beauchamp-Nicoara)", 10, 6, [], BN10, "Beauchamp-Nicoara Sec. 3"))
add(entry("N_11 (Beauchamp-Nicoara)", 11, 6, [], lambda: N11(0), "Beauchamp-Nicoara Sec. 3"))
add(entry("N'_11 (Beauchamp-Nicoara)", 11, 6, [], lambda: N11(1), "Beauchamp-Nicoara Sec. 3"))


def hadamard_defect(H):
    N = H.shape[0]
    return max(np.abs(np.abs(H) - 1).max(), np.abs(H @ H.conj().T - N * np.eye(N)).max())


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for e in CATALOG:
        errs = []
        trials = 5 if e["params"] else 1
        for _ in range(trials):
            ps = [E(rng.random()) for _ in e["params"]]
            try:
                errs.append(hadamard_defect(e["build"](*ps)))
            except Exception as ex:
                errs.append(f"ERR {ex}")
        bad = [x for x in errs if isinstance(x, str) or x > 1e-9]
        print(f"{'OK ' if not bad else 'BAD'} N={e['N']:2d}  {e['name']:34s} {bad[:2] if bad else ''}")
