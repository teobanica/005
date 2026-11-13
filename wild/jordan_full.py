"""Jordan test, full junction version. For a rotation eigenvector x in Fix(G_0) cap Lambda_k, and y = x* or y = x,
the nonzero planar contractions of x (x) y are: r nested caps across the junction, optionally followed by 1 merge.
They land at level 2(k-r), resp. 2(k-r)-1, and must be in TL there when the level is < k."""
import sys, numpy as np, scipy.linalg as sl, scipy.optimize as so
sys.path.insert(0, "/home/claude/day")


def nc_partitions(n):
    def rec(i, blocks):
        if i == n:
            yield [list(b) for b in blocks]; return
        for j, b in enumerate(blocks):        # add i to block j if noncrossing
            ok = True
            for c in blocks[j + 1:]:
                if c and c[0] > b[-1] or True:
                    pass
            # i joins b: crossing if some later-opened block c (opened after b's last element) is still open, i.e. has
            # elements between b[-1] and i and is not closed -> since we add in order, crossing iff exists block c != b
            # with min(c) > b[-1]  (c lies entirely after b's last element, so joining would enclose it: fine) ...
            # use explicit check instead
            nb = [list(x) for x in blocks]; nb[j].append(i)
            if noncrossing(nb):
                yield from rec(i + 1, nb)
        yield from rec(i + 1, [list(x) for x in blocks] + [[i]])
    seen = set()
    for p in rec(0, []):
        key = tuple(sorted(tuple(b) for b in p))
        if key not in seen:
            seen.add(key); yield p


def noncrossing(blocks):
    for a in range(len(blocks)):
        for b in range(len(blocks)):
            if a == b: continue
            A, B = blocks[a], blocks[b]
            for i in A:
                for j in A:
                    if i < j:
                        for p in B:
                            for q in B:
                                if i < p < j < q:
                                    return False
    return True


_TL = {}
def tl_basis(N, j):
    if (N, j) in _TL: return _TL[(N, j)]
    if j == 0:
        Q = np.ones((1, 1))
    else:
        n = N ** j; idx = np.array(np.unravel_index(np.arange(n), (N,) * j)).T; vecs = []
        for p in nc_partitions(j):
            ok = np.ones(n, bool)
            for bl in p:
                for a in bl[1:]: ok &= idx[:, a] == idx[:, bl[0]]
            vecs.append(ok.astype(float))
        Q = sl.orth(np.array(vecs).T)
    _TL[(N, j)] = Q; return Q


def forms(V, N, k, max_level=None):
    """V: n x f basis (rotation eigenvectors in Lambda_k). Returns list of (kind, B) with B shape (f, f, M), projected on TL^perp."""
    f = V.shape[1]; X = V.T.reshape((f,) + (N,) * k); out = []
    max_level = k - 1 if max_level is None else max_level
    for r in range(1, k + 1):
        for merge in (False, True):
            if merge and r >= k: continue
            lev = 2 * (k - r) - (1 if merge else 0)
            if lev >= k or lev > max_level or lev < 1: continue
            for kind in ("sesq", "bil"):
                if kind == "sesq":
                    # y = x*: pair x's last r legs with conj(x)'s last r legs; output x-front then conj(x)-front reversed
                    if not merge:
                        A = X.reshape(f, N ** (k - r), N ** r)
                        Z = np.einsum('apc,bqc->abpq', A, A.conj())                       # (f,f,front,front)
                        Z = Z.reshape((f, f) + (N,) * (k - r) + (N,) * (k - r))
                        perm = list(range(2 + k - r)) + [2 + k - r + (k - r - 1 - t) for t in range(k - r)]
                    else:
                        A = X.reshape(f, N ** (k - r - 1), N, N ** r)
                        Z = np.einsum('apmc,bqmc->abpmq', A, A.conj())
                        Z = Z.reshape((f, f) + (N,) * (k - r - 1) + (N,) + (N,) * (k - r - 1))
                        m = k - r - 1
                        perm = list(range(2 + m + 1)) + [2 + m + 1 + (m - 1 - t) for t in range(m)]
                else:
                    # y = x: pair first copy's last r legs (c_1..c_r) with second copy's first r legs reversed
                    if not merge:
                        A = X.reshape(f, N ** (k - r), N ** r)
                        Xr = X.reshape((f,) + (N,) * k)
                        Bm = np.flip(Xr, axis=tuple(range(1, r + 1))) if False else Xr
                        # second copy legs 1..r reversed to match c_1..c_r: y_t pairs with x_{k-t+1}
                        Y = np.transpose(Xr, [0] + [r - t for t in range(r)] + list(range(r + 1, k + 1)))  # legs (y_r..y_1, rest)
                        Y = Y.reshape(f, N ** r, N ** (k - r))
                        Z = np.einsum('apc,bcq->abpq', A, Y)
                        Z = Z.reshape((f, f) + (N,) * (2 * (k - r)))
                        perm = list(range(2 + 2 * (k - r)))
                    else:
                        Xr = X.reshape((f,) + (N,) * k)
                        A = X.reshape(f, N ** (k - r - 1), N, N ** r)                      # first copy: front, merged leg, junction
                        Y = np.transpose(Xr, [0] + [r - t for t in range(r)] + list(range(r + 1, k + 1)))
                        Y = Y.reshape(f, N ** r, N, N ** (k - r - 1))                       # second copy: junction (reversed), merged leg, back
                        Z = np.einsum('apmc,bcmq->abpmq', A, Y)
                        Z = Z.reshape((f, f) + (N,) * (2 * (k - r) - 1))
                        perm = list(range(2 + 2 * (k - r) - 1))
                Z = np.transpose(Z, perm).reshape(f, f, -1)
                Q = tl_basis(N, lev)
                Z = Z - (Z @ Q) @ Q.T
                if np.abs(Z).max() > 1e-12:
                    out.append((kind, Z))
    return out


def min_defect(F, f, starts=30, seed=0):
    def fg(x):
        c = x[:f] + 1j * x[f:]; nc = np.vdot(c, c).real; val = 0.0; g = np.zeros(f, complex)
        for kind, B in F:
            if kind == "sesq":
                z = np.einsum('a,b,abn->n', c, c.conj(), B)
                g += np.einsum('n,a,aen->e', z.conj(), c, B) + np.einsum('n,b,ebn->e', z, c, B.conj())
            else:
                z = np.einsum('a,b,abn->n', c, c, B)
                g += np.einsum('n,b,ebn->e', z, c.conj(), B.conj()) + np.einsum('n,a,aen->e', z, c.conj(), B.conj())
            val += np.vdot(z, z).real
        F_ = val; gc = g / nc ** 2 - 2 * F_ / nc ** 3 * c
        return F_ / nc ** 2, np.concatenate([2 * gc.real, 2 * gc.imag])
    rng = np.random.default_rng(seed); best = np.inf; bestx = None
    for _ in range(starts):
        r = so.minimize(fg, rng.standard_normal(2 * f), jac=True, method="L-BFGS-B", options={"maxiter": 10000, "gtol": 1e-13})
        if r.fun < best: best, bestx = r.fun, r.x
    return best, fg
