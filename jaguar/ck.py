"""Layer 1 computations: c_k = dim Fix(P^{otimes k}) for the magic matrix P_ij = Proj(H_i/H_j).

T_k is the N^k x N^k matrix (T_k)_{i,j} = tr(P_{i1 j1} ... P_{ik jk}), tr the normalized trace.
With xi_{ij} = (H_i/H_j)/sqrt(N) the unit vector spanning P_ij, and g(a,b) = <xi_b, xi_a>,
we have tr(P_{a1}...P_{ak}) = (1/N) g(a1,a2) g(a2,a3) ... g(ak,a1)   (cyclic product).
A vector is fixed by u^{otimes k} iff it is fixed by T_k (T_k is the conditional expectation
of the unitary u^{otimes k}), so c_k = dim ker(T_k - 1), computed from the singular values.

Direct method: memory N^{2k}, fine for N^k <= ~5000.
"""
import itertools
import numpy as np


def vectors(H):
    N = H.shape[0]
    X = np.zeros((N, N, N), complex)          # X[i, j] = xi_{ij}
    for i in range(N):
        for j in range(N):
            X[i, j] = H[i] / H[j] / np.sqrt(N)
    return X


def gram(H):
    """g[(i,j),(k,l)] = <xi_kl, xi_ij> = sum_x xi_ij(x) conj(xi_kl(x))  (linear at left in the first slot)"""
    N = H.shape[0]
    X = vectors(H).reshape(N * N, N)
    return X @ X.conj().T                     # g[a,b] = <xi_a, xi_b> linear at left: sum xi_a conj(xi_b)


def T_matrix(H, k):
    """(T_k)_{i,j} with multi-indices i=(i1..ik), j=(j1..jk), pair a_s=(i_s,j_s).

    tr(P_{a1} ... P_{ak}) = (1/N) prod_s <xi_{a_{s+1}}, xi_{a_s}> ... computed as a cyclic product of
    M[a_s, a_{s+1}] with M[a,b] = xi_a^* xi_b = <xi_b, xi_a>.
    """
    N = H.shape[0]
    g = gram(H)                 # g[a,b] = sum xi_a conj(xi_b) = xi_b^* xi_a
    M = g.T                     # M[a,b] = xi_a^* xi_b
    # T as tensor over (i1,j1,...,ik,jk): contract a cyclic chain of M's.
    # Build A[a1, ak] path products then close the cycle.
    Mt = M.reshape(N, N, N, N)  # Mt[i,j,k,l] = M[(i,j),(k,l)]
    if k == 1:
        T = np.einsum('ijij->ij', Mt) / N
        return T
    # chain: C[a1, a_s] tensor over all intermediate indices kept
    # we keep all indices; total size N^{2k}
    letters = 'abcdefghpqrstuvw'
    I = [letters[2 * s] for s in range(k)]
    J = [letters[2 * s + 1] for s in range(k)]
    terms = []
    for s in range(k):
        t = (s + 1) % k
        terms.append(I[s] + J[s] + I[t] + J[t])
    expr = ','.join(terms) + '->' + ''.join(I) + ''.join(J)
    T = np.einsum(expr, *([Mt] * k), optimize=True) / N
    return T.reshape(N ** k, N ** k)


def ck(H, k, tol=1e-7):
    T = T_matrix(H, k)
    n = T.shape[0]
    s = np.linalg.svd(T - np.eye(n), compute_uv=False)
    s = np.sort(s)
    dim = int(np.sum(s < tol))
    gap = (s[dim - 1] if dim > 0 else 0.0, s[dim] if dim < n else np.inf)
    return dim, gap


def catalan(k):
    from math import comb
    return comb(2 * k, k) // (k + 1)


# ---------------------------------------------------------------- iterative method, large N
# T_k = N^{-k-1} V V^*,  V_{i,x} = prod_s H_{i_s x_{s-1}} conj(H_{i_s x_s}),  x_0 = x_k.
# So T_k is Hermitian PSD with norm <= 1, and c_k = multiplicity of the eigenvalue 1.
import scipy.sparse.linalg as sla

_L = 'abcdefghij'                   # up to level 10 (was 'abcdefgh', level 8)
_X = 'pqrstuvwxy'


def _exprs(k):
    I = _L[:k]
    X = _X[:k]
    facs = [I[s] + X[(s - 1) % k] + X[s] for s in range(k)]
    vstar = I + ',' + ','.join(facs) + '->' + X          # (V^* xi)_x, with B = conj(H) H
    v = X + ',' + ','.join(facs) + '->' + I               # (V eta)_i, with conj(B)
    return vstar, v


def T_operator(H, k):
    N = H.shape[0]
    real = np.allclose(H.imag, 0)
    Hc = H.real if real else H
    B = np.einsum('iy,iz->iyz', Hc.conj(), Hc)           # B[i,y,z] = conj(H_iy) H_iz
    Bc = B.conj()
    e1, e2 = _exprs(k)
    shape = (N,) * k
    dtype = float if real else complex
    p1 = np.einsum_path(e1, np.zeros(shape, dtype), *([B] * k), optimize='optimal')[0]
    p2 = np.einsum_path(e2, np.zeros(shape, dtype), *([Bc] * k), optimize='optimal')[0]
    scale = 1.0 / N ** (k + 1)

    def mv(xi):
        t = np.einsum(e1, xi.reshape(shape), *([B] * k), optimize=p1)
        return (np.einsum(e2, t, *([Bc] * k), optimize=p2) * scale).reshape(-1)
    return sla.LinearOperator((N ** k, N ** k), matvec=mv, dtype=dtype)


def ck_iterative(H, k, nev=None, tol_one=1e-8, seed=0):
    """Multiplicity of eigenvalue 1 of T_k, via Lanczos (largest eigenvalues).
    Returns (c_k, top eigenvalues found, first eigenvalue below 1)."""
    N = H.shape[0]
    T = T_operator(H, k)
    n = N ** k
    if nev is None:
        nev = catalan(k) + 8
    rng = np.random.default_rng(seed)
    while True:
        nev = min(nev, n - 2)
        v0 = rng.standard_normal(n)
        vals = sla.eigsh(T, k=nev, which='LA', v0=v0, tol=1e-12, return_eigenvectors=False)
        vals = np.sort(vals)[::-1]
        c = int(np.sum(vals > 1 - tol_one))
        if c < nev or nev >= n - 2:
            below = float(vals[c]) if c < len(vals) else None
            return c, vals, below
        nev *= 2


def ck_deflated(H, k, batch=None, tol_one=1e-8, seed=0, maxit=50):
    """Robust multiplicity of eigenvalue 1: repeated Lanczos with deflation of the found eigenvectors.
    Stops when the top eigenvalue of the deflated operator is below 1 - tol_one.
    Returns (c_k, next eigenvalue below 1, number of eigsh calls)."""
    N = H.shape[0]
    if N ** k <= 1500:                       # small: dense SVD
        d, gap = ck(H, k)
        return d, float(1 - gap[1]) if np.isfinite(gap[1]) else 0.0, 0
    T = T_operator_blas(H, k) if k >= 3 else T_operator_fast(H, k)
    n = N ** k
    if batch is None:
        batch = min(catalan(k) + 8, n - 2)
    U = np.zeros((n, 0), dtype=T.dtype)
    rng = np.random.default_rng(seed)
    below = None
    for it in range(maxit):
        def mv(x, U=U):
            y = T.matvec(x)
            if U.shape[1]:
                y = y - U @ (U.conj().T @ x)
            return y
        Op = sla.LinearOperator((n, n), matvec=mv, dtype=T.dtype)
        nb = min(batch, n - 2 - U.shape[1])
        vals, vecs = sla.eigsh(Op, k=nb, which='LA', v0=rng.standard_normal(n), tol=1e-12)
        ones = vals > 1 - tol_one
        if not ones.any():
            below = float(vals.max())
            return U.shape[1], below, it + 1
        Q = vecs[:, ones]
        Q = Q - U @ (U.conj().T @ Q)
        Q, _ = np.linalg.qr(Q)
        U = np.hstack([U, Q])
        batch = max(batch, min(2 * U.shape[1], 600))     # large fixed spaces: grow the batch
        if not (~ones).any():
            continue
        below_candidate = float(vals[~ones].max())
    raise RuntimeError("no convergence")


def _chain(t, F, k, N, inn, outn):
    """Contract the cyclic chain prod_s F[in_s, out_{s-1}, out_s] against tensor t over the 'in' letters,
    one factor at a time; returns the tensor over out_1..out_k."""
    cur = inn
    t = t.reshape((N,) * k)
    for s in range(k):
        f = inn[s] + outn[(s - 1) % k] + outn[s]
        keep = [c for c in cur if c != inn[s]]
        new = [c for c in (outn[(s - 1) % k], outn[s]) if c not in keep]
        out = ''.join(keep + new)
        t = np.einsum(cur + ',' + f + '->' + out, t, F)
        cur = out
    return np.einsum(cur + '->' + outn, t)


def T_operator_fast(H, k):
    """Same operator as T_operator, contracted factor by factor: cost ~ k N^{k+2} per product."""
    N = H.shape[0]
    real = np.allclose(H.imag, 0)
    Hc = H.real if real else H
    B = np.einsum('iy,iz->iyz', Hc.conj(), Hc)       # B[i, y, z]
    C = np.ascontiguousarray(np.transpose(B.conj(), (1, 2, 0)))  # C[y, z, i] = conj B[i,y,z]
    dtype = float if real else complex
    scale = 1.0 / N ** (k + 1)
    I, X = _L[:k], _X[:k]

    def V(eta):
        # (V eta)_i = sum_x eta_x prod_s conj(B)[i_s, x_{s-1}, x_s]; contract x's one at a time.
        t = eta.reshape((N,) * k); cur = X
        for s in range(k):
            f = X[(s - 1) % k] + X[s] + I[s]
            needed = set(I[:s + 1])
            for u in range(s + 1, k):
                needed |= {X[(u - 1) % k], X[u]}
            out = ''.join([c for c in cur if c in needed] + [I[s]])
            t = np.einsum(cur + ',' + f + '->' + out, t, C)
            cur = out
        return np.einsum(cur + '->' + I, t)

    def mv(xi):
        y = _chain(xi, B, k, N, I, X)
        return (V(y) * scale).reshape(-1)
    return sla.LinearOperator((N ** k, N ** k), matvec=mv, dtype=dtype)


def _bmm_step(t, axes, F, a_in, a_b, a_out_new, out_present):
    """Generic step: t has named axes 'axes' (list of letters). Contract letter a_in with F[a_in, a_b, a_o],
    F given as array with axes (a_in, a_b, a_o). If a_b in axes it is a batch letter, else it becomes new.
    If out_present, a_o is also a batch letter (already in t), else new. Uses matmul."""
    # bring t to shape (batch..., rest, a_in)
    batch = [c for c in (a_b, a_out_new) if c in axes and c != a_in]
    rest = [c for c in axes if c not in batch and c != a_in]
    perm = [axes.index(c) for c in batch + rest + [a_in]]
    tt = np.transpose(t, perm)
    N = t.shape[0]
    nb, nr = len(batch), len(rest)
    tt = tt.reshape((N,) * nb + (N ** nr, N))
    if batch == [a_b]:                       # F as (a_b, a_in, a_o): batched over a_b
        Fm = np.transpose(F, (1, 0, 2))
        r = np.matmul(tt, Fm)                # (a_b, rest, a_o)
        new_axes = [a_b] + rest + [a_out_new]
        r = r.reshape((N,) * (1 + nr + 1))
    elif batch == [a_b, a_out_new]:          # both batch: elementwise in (a_b,a_o), contract a_in
        Fm = np.transpose(F, (1, 2, 0))      # (a_b, a_o, a_in)
        r = np.einsum('yzri,yzi->yzr', tt, Fm)
        new_axes = [a_b, a_out_new] + rest
        r = r.reshape((N,) * (2 + nr))
    else:                                     # no batch: plain matmul
        Fm = F.reshape(N, N * N)             # (a_in, a_b a_o)
        r = tt @ Fm                          # (rest, a_b a_o)
        new_axes = rest + [a_b, a_out_new]
        r = r.reshape((N,) * (nr + 2))
    return r, new_axes


def _cyc_contract(xi, F, k, N, inn, outn):
    """sum over inn of xi[inn] prod_s F[inn_s, outn_{s-1}, outn_s]; result indexed by outn."""
    t = xi.reshape((N,) * k); axes = list(inn)
    for s in range(k):
        a_b, a_o = outn[(s - 1) % k], outn[s]
        t, axes = _bmm_step(t, axes, F, inn[s], a_b, a_o, a_o in axes)
    perm = [axes.index(c) for c in outn]
    return np.ascontiguousarray(np.transpose(t, perm))



def _V_contract(eta, Bc, k, N):
    """(V eta)_i = sum_x eta_x prod_s Bc[i_s, x_{s-1}, x_s], Bc = conj(B). Factor by factor with matmul.
    Axis bookkeeping: t has axes listed in 'ax'."""
    t = eta.reshape((N,) * k)
    ax = ['x%d' % s for s in range(k)]                  # x_1..x_k stored as x0..x_{k-1}
    # factor s (s = 0..k-1) involves i_s, x_{s-1}, x_s (indices mod k)
    # step 0: no summation; multiply by Bc[i0, x_{k-1}, x_0] elementwise in (x_{k-1}, x_0), new i0
    perm = [ax.index('x%d' % (k - 1)), ax.index('x0')] + [j for j, c in enumerate(ax) if c not in ('x%d' % (k - 1), 'x0')]
    rest = [ax[j] for j in perm[2:]]
    tt = np.transpose(t, perm).reshape(N, N, N ** len(rest))          # (x_{k-1}, x_0, rest)
    F = np.transpose(Bc, (1, 2, 0))                                  # (x_{k-1}, x_0, i0)
    t = tt[:, :, :, None] * F[:, :, None, :]                         # (x_{k-1}, x_0, rest, i0)
    ax = ['x%d' % (k - 1), 'x0'] + rest + ['i0']
    t = t.reshape((N,) * len(ax))
    for s in range(1, k - 1):
        # sum x_{s-1}, batch x_s, new i_s ; factor Bc[i_s, x_{s-1}, x_s] -> (x_s, x_{s-1}, i_s)
        b, sm = 'x%d' % s, 'x%d' % (s - 1)
        others = [c for c in ax if c not in (b, sm)]
        perm = [ax.index(b)] + [ax.index(c) for c in others] + [ax.index(sm)]
        tt = np.transpose(t, perm).reshape(N, N ** len(others), N)
        F = np.transpose(Bc, (2, 1, 0))                              # (x_s, x_{s-1}, i_s)
        t = np.matmul(tt, F)                                         # (x_s, others, i_s)
        ax = [b] + others + ['i%d' % s]
        t = t.reshape((N,) * len(ax))
    # last factor k-1: Bc[i_{k-1}, x_{k-2}, x_{k-1}], sum both x's
    a1, a2 = 'x%d' % (k - 2), 'x%d' % (k - 1)
    others = [c for c in ax if c not in (a1, a2)]
    perm = [ax.index(c) for c in others] + [ax.index(a1), ax.index(a2)]
    tt = np.transpose(t, perm).reshape(N ** len(others), N * N)
    F = np.transpose(Bc, (1, 2, 0)).reshape(N * N, N)                # (x_{k-2} x_{k-1}, i_{k-1})
    t = tt @ F
    ax = others + ['i%d' % (k - 1)]
    t = t.reshape((N,) * k)
    perm = [ax.index('i%d' % s) for s in range(k)]
    return np.ascontiguousarray(np.transpose(t, perm))


def _cyc_contract_blk(xi, F, k, N, inn, outn):
    """sum over inn of xi[inn] prod_s F[inn_s, outn_{s-1}, outn_s]; result indexed by outn."""
    t = xi.reshape((N,) * (k + 1)); axes = list(inn) + ['z']
    for s in range(k):
        a_b, a_o = outn[(s - 1) % k], outn[s]
        t, axes = _bmm_step(t, axes, F, inn[s], a_b, a_o, a_o in axes)
    perm = [axes.index(c) for c in list(outn) + ['z']]
    return np.ascontiguousarray(np.transpose(t, perm))



def _V_contract_blk(eta, Bc, k, N):
    """(V eta)_i = sum_x eta_x prod_s Bc[i_s, x_{s-1}, x_s], Bc = conj(B). Factor by factor with matmul.
    Axis bookkeeping: t has axes listed in 'ax'."""
    t = eta.reshape((N,) * (k + 1))
    ax = ['x%d' % s for s in range(k)] + ['z']                  # x_1..x_k stored as x0..x_{k-1}
    # factor s (s = 0..k-1) involves i_s, x_{s-1}, x_s (indices mod k)
    # step 0: no summation; multiply by Bc[i0, x_{k-1}, x_0] elementwise in (x_{k-1}, x_0), new i0
    perm = [ax.index('x%d' % (k - 1)), ax.index('x0')] + [j for j, c in enumerate(ax) if c not in ('x%d' % (k - 1), 'x0')]
    rest = [ax[j] for j in perm[2:]]
    tt = np.transpose(t, perm).reshape(N, N, N ** len(rest))          # (x_{k-1}, x_0, rest)
    F = np.transpose(Bc, (1, 2, 0))                                  # (x_{k-1}, x_0, i0)
    t = tt[:, :, :, None] * F[:, :, None, :]                         # (x_{k-1}, x_0, rest, i0)
    ax = ['x%d' % (k - 1), 'x0'] + rest + ['i0']
    t = t.reshape((N,) * len(ax))
    for s in range(1, k - 1):
        # sum x_{s-1}, batch x_s, new i_s ; factor Bc[i_s, x_{s-1}, x_s] -> (x_s, x_{s-1}, i_s)
        b, sm = 'x%d' % s, 'x%d' % (s - 1)
        others = [c for c in ax if c not in (b, sm)]
        perm = [ax.index(b)] + [ax.index(c) for c in others] + [ax.index(sm)]
        tt = np.transpose(t, perm).reshape(N, N ** len(others), N)
        F = np.transpose(Bc, (2, 1, 0))                              # (x_s, x_{s-1}, i_s)
        t = np.matmul(tt, F)                                         # (x_s, others, i_s)
        ax = [b] + others + ['i%d' % s]
        t = t.reshape((N,) * len(ax))
    # last factor k-1: Bc[i_{k-1}, x_{k-2}, x_{k-1}], sum both x's
    a1, a2 = 'x%d' % (k - 2), 'x%d' % (k - 1)
    others = [c for c in ax if c not in (a1, a2)]
    perm = [ax.index(c) for c in others] + [ax.index(a1), ax.index(a2)]
    tt = np.transpose(t, perm).reshape(N ** len(others), N * N)
    F = np.transpose(Bc, (1, 2, 0)).reshape(N * N, N)                # (x_{k-2} x_{k-1}, i_{k-1})
    t = tt @ F
    ax = others + ['i%d' % (k - 1)]
    t = t.reshape((N,) * (k + 1))
    perm = [ax.index('i%d' % s) for s in range(k)] + [ax.index('z')]
    return np.ascontiguousarray(np.transpose(t, perm))



def T_block_blas(H, k):
    """Returns f(X) = T_k X for X of shape (N^k, m), processed N columns at a time."""
    N = H.shape[0]
    real = np.allclose(H.imag, 0)
    Hc = H.real if real else H
    B = np.ascontiguousarray(np.einsum('iy,iz->iyz', Hc.conj(), Hc))
    Bc = np.ascontiguousarray(B.conj())
    I, X_ = _L[:k], _X[:k]
    scale = 1.0 / N ** (k + 1)
    def f(X):
        n, m = X.shape
        out = np.empty_like(X)
        for j0 in range(0, m, N):
            blk = X[:, j0:j0 + N]
            w = blk.shape[1]
            if w < N:
                blk = np.concatenate([blk, np.zeros((n, N - w), X.dtype)], axis=1)
            y = _cyc_contract_blk(np.ascontiguousarray(blk), B, k, N, I, X_)
            out[:, j0:j0 + w] = (_V_contract_blk(y, Bc, k, N) * scale).reshape(n, N)[:, :w]
        return out
    return f


def T_operator_blas(H, k):
    """T_k xi = N^{-k-1} V (V^* xi), both halves with BLAS matmuls. Requires k >= 3."""
    N = H.shape[0]
    real = np.allclose(H.imag, 0)
    Hc = H.real if real else H
    B = np.ascontiguousarray(np.einsum('iy,iz->iyz', Hc.conj(), Hc))
    Bc = np.ascontiguousarray(B.conj())
    I, X = _L[:k], _X[:k]
    dtype = float if real else complex
    scale = 1.0 / N ** (k + 1)

    def mv(xi):
        y = _cyc_contract(xi, B, k, N, I, X)
        return (_V_contract(y, Bc, k, N) * scale).reshape(-1)
    return sla.LinearOperator((N ** k, N ** k), matvec=mv, dtype=dtype)


# ---------------------------------------------------------------- seeded computation (large fixed spaces)
def fixed_basis(H, k, U0=None, tol_one=1e-8, seed=0, batch=None):
    """Orthonormal basis of Fix at level k: deflated Lanczos, starting from known fixed vectors U0
    (which are re-verified). Returns (U, next eigenvalue below 1)."""
    N = H.shape[0]
    n = N ** k
    if n <= 1500:
        T = T_matrix(H, k)
        T = (T + T.conj().T) / 2
        if np.allclose(H.imag, 0):
            T = T.real
        w, v = np.linalg.eigh(T)
        ones = w > 1 - tol_one
        below = float(w[~ones].max()) if (~ones).any() else 0.0
        return v[:, ones], below
    T = T_operator_blas(H, k) if k >= 3 else T_operator_fast(H, k)
    dtype = T.dtype
    U = np.zeros((n, 0), dtype=dtype)
    if U0 is not None and U0.shape[1]:
        if dtype == float and np.iscomplexobj(U0):
            U0 = np.hstack([U0.real, U0.imag])          # real operator: real and imaginary parts are both fixed
        Q, _ = np.linalg.qr(U0.astype(dtype))
        R = np.column_stack([T.matvec(Q[:, j]) for j in range(Q.shape[1])])
        err = np.linalg.norm(R - Q, axis=0)
        Q = Q[:, err < 1e-8]
        U = Q
    if batch is None:
        batch = min(catalan(k) + 8, n - 2)
    rng = np.random.default_rng(seed)
    for it in range(60):
        def mv(x, U=U):
            y = T.matvec(x)
            if U.shape[1]:
                y = y - U @ (U.conj().T @ x)
            return y
        Op = sla.LinearOperator((n, n), matvec=mv, dtype=dtype)
        nb = min(batch, n - 2 - U.shape[1])
        vals, vecs = sla.eigsh(Op, k=nb, which='LA', v0=rng.standard_normal(n), tol=1e-12)
        ones = vals > 1 - tol_one
        if not ones.any():
            return U, float(vals.max())
        Q = vecs[:, ones]
        Q = Q - U @ (U.conj().T @ Q)
        Q, _ = np.linalg.qr(Q)
        U = np.hstack([U, Q])
        batch = max(batch, min(2 * U.shape[1], 600))
    raise RuntimeError("no convergence")


def rotate(v, N, k, r=1):
    return np.moveaxis(v.reshape((N,) * k), list(range(k)), [(j + r) % k for j in range(k)]).reshape(-1)


def generate_level(N, k, bases):
    """Candidate fixed vectors at level k from bases[j] (orthonormal columns, level j < k):
    tensor products, leg doublings of level k-1, all cyclic rotations, then products in End(u^{(k/2)})."""
    cands = []
    ones1 = np.ones(N)
    # doubling of level k-1
    if k - 1 in bases:
        Bk1 = bases[k - 1]
        for c in range(Bk1.shape[1]):
            t = Bk1[:, c].reshape((N,) * (k - 1))
            for p in range(k - 1):
                v = np.zeros((N,) * k, dtype=Bk1.dtype)
                # v[..., i_p, i_{p+1}=i_p, ...] = t[..., i_p, ...]
                idx = np.indices((N,) * (k - 1)).reshape(k - 1, -1)
                full = np.insert(idx, p + 1, idx[p], axis=0)
                v[tuple(full)] = t.reshape(-1)
                cands.append(v.reshape(-1))
    # tensor products of lower levels
    for a in range(1, k):
        b = k - a
        if a in bases and b in bases:
            for x in bases[a].T:
                for y in bases[b].T:
                    cands.append(np.kron(x, y))
    C = np.array(cands).T if cands else np.zeros((N ** k, 0))
    # rotations
    rots = [C] + [np.column_stack([rotate(C[:, j], N, k, r) for j in range(C.shape[1])]) for r in range(1, k)]
    C = np.hstack(rots)
    U, s, _ = np.linalg.svd(C, full_matrices=False)
    r = int(np.sum(s > 1e-9 * s[0]))
    U = U[:, :r]
    if k % 2 == 0:
        # close under products in End(u^{otimes k/2}): X_{(i),(j)} = xi_{i_1..i_m j_m..j_1}
        m_ = k // 2
        def tomat(v):
            t = v.reshape((N,) * k)
            t = np.transpose(t, list(range(m_)) + list(range(k - 1, m_ - 1, -1)))
            return t.reshape(N ** m_, N ** m_)
        def tovec(X):
            t = X.reshape((N,) * k)
            t = np.transpose(t, list(range(m_)) + list(range(k - 1, m_ - 1, -1)))
            return t.reshape(-1)
        rng = np.random.default_rng(1)
        for _ in range(4):
            mats = [tomat(U[:, j]) for j in range(U.shape[1])]
            dim = U.shape[1]
            new = []
            for r in range(6):
                c = rng.standard_normal(dim)
                X = sum(ci * Mi for ci, Mi in zip(c, mats))
                for Mi in mats:
                    new.append(tovec(X @ Mi)); new.append(tovec(Mi @ X))
            C2 = np.hstack([U, np.array(new).T])
            U2, s2, _ = np.linalg.svd(C2, full_matrices=False)
            r2 = int(np.sum(s2 > 1e-9 * s2[0]))
            if r2 == dim:
                break
            U = U2[:, :r2]
    return U


def ck_seeded(H, kmax, log=None):
    """c_1..c_kmax, each level seeded by vectors generated from the lower ones."""
    N = H.shape[0]
    bases, res = {}, {}
    for k in range(1, kmax + 1):
        U0 = generate_level_lowmem(N, k, bases, dtype=float if np.allclose(H.imag, 0) else complex, chunk=(24 if N ** k > 300000 else 128)) if k >= 2 else None
        U, below = fixed_basis(H, k, U0=U0)
        bases[k] = U
        res[k] = (U.shape[1], below, 0 if U0 is None else U0.shape[1])
        if log:
            log(k, res[k])
    return res


class Span:
    """Incrementally built orthonormal basis (columns), fed in chunks; memory ~ final basis size."""
    def __init__(self, n, dtype=float):
        self.U = np.zeros((n, 0), dtype=dtype)

    def add(self, C, tol=1e-9):
        if C.shape[1] == 0:
            return 0
        C = C.astype(self.U.dtype, copy=False)
        if self.U.shape[1]:
            C = C - self.U @ (self.U.conj().T @ C)
            C = C - self.U @ (self.U.conj().T @ C)
        nrm = np.linalg.norm(C, axis=0)
        C = C[:, nrm > tol]
        if C.shape[1] == 0:
            return 0
        Q, s, _ = np.linalg.svd(C, full_matrices=False)
        r = int(np.sum(s > tol * max(1.0, s[0]) if s.size else 0))
        if r:
            self.U = np.hstack([self.U, Q[:, :r]])
        return r


def generate_level_lowmem(N, k, bases, dtype=float, chunk=128):
    S = Span(N ** k, dtype)
    buf = []
    def flush():
        if buf:
            C = np.array(buf).T
            for r in range(k):
                S.add(np.column_stack([rotate(C[:, j], N, k, r) for j in range(C.shape[1])]) if r else C)
            buf.clear()
    def push(v):
        buf.append(v)
        if len(buf) >= chunk:
            flush()
    if k - 1 in bases:
        Bk1 = bases[k - 1]
        idx = np.indices((N,) * (k - 1)).reshape(k - 1, -1)
        for c in range(Bk1.shape[1]):
            t = Bk1[:, c]
            for p in range(k - 1):
                v = np.zeros((N,) * k, dtype=Bk1.dtype)
                v[tuple(np.insert(idx, p + 1, idx[p], axis=0))] = t
                push(v.reshape(-1))
    for a in range(1, k):
        b = k - a
        if a in bases and b in bases:
            for x in bases[a].T:
                for y in bases[b].T:
                    push(np.kron(x, y))
    flush()
    if k % 2 == 0:
        m_ = k // 2
        perm = list(range(m_)) + list(range(k - 1, m_ - 1, -1))
        tomat = lambda v: np.transpose(v.reshape((N,) * k), perm).reshape(N ** m_, N ** m_)
        tovec = lambda X: np.transpose(X.reshape((N,) * k), perm).reshape(-1)
        rng = np.random.default_rng(1)
        for _ in range(4):
            dim = S.U.shape[1]
            Us = S.U[:, :dim]
            for r in range(4):
                c = rng.standard_normal(dim)
                X = tomat(Us @ c)
                for j0 in range(0, dim, chunk):
                    cols = []
                    for j in range(j0, min(dim, j0 + chunk)):
                        Mj = tomat(Us[:, j])
                        cols.append(tovec(X @ Mj)); cols.append(tovec(Mj @ X))
                    S.add(np.array(cols).T)
            if S.U.shape[1] == dim:
                break
    return S.U


# ---------------------------------------------------------------- low-memory variant (huge spaces)
class SpanLM:
    """Orthonormal basis stored in one preallocated array (n x cap); no full copies."""
    def __init__(self, n, dtype=float, cap=400):
        self.B = np.empty((n, cap), dtype=dtype)
        self.r = 0

    @property
    def U(self):
        return self.B[:, :self.r]

    def _grow(self, need):
        if self.r + need <= self.B.shape[1]:
            return
        step = self.B.shape[1] if self.B.shape[0] * self.B.shape[1] < 5e7 else 128
        cap = max(self.B.shape[1] + step, self.r + need)
        nb = np.empty((self.B.shape[0], cap), dtype=self.B.dtype)
        nb[:, :self.r] = self.B[:, :self.r]
        self.B = nb

    def add_vec(self, v, tol=1e-9):
        v = v.astype(self.B.dtype, copy=True)
        for _ in range(2):
            if self.r:
                v -= self.U @ (self.U.conj().T @ v)
        nv = np.linalg.norm(v)
        if nv < tol:
            return False
        self._grow(1)
        self.B[:, self.r] = v / nv
        self.r += 1
        return True

    def add(self, C, tol=1e-9):
        return sum(self.add_vec(C[:, j], tol) for j in range(C.shape[1]))


def fixed_basis_lm(H, k, cand_iter, tol_one=1e-8, seed=0, batch=None, progress=None):
    """Like fixed_basis, low memory: candidates come one by one, each is verified
    (T v = v after orthogonalization) before being added. Then deflated Lanczos."""
    N = H.shape[0]
    n = N ** k
    T = T_operator_blas(H, k) if k >= 3 else T_operator_fast(H, k)
    S = SpanLM(n, T.dtype)
    nseed = 0
    for v in cand_iter:
        v = v.astype(T.dtype, copy=True)
        if S.r:
            v -= S.U @ (S.U.conj().T @ v)
            v -= S.U @ (S.U.conj().T @ v)
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            continue
        v /= nv
        if np.linalg.norm(T.matvec(v) - v) < 1e-8:
            S._grow(1); S.B[:, S.r] = v; S.r += 1; nseed += 1
            if progress and nseed % 10 == 0:
                progress("  seeds verified so far: %d" % nseed)
    if progress:
        progress("  seeding done: %d fixed vectors; now checking the complement with Lanczos" % nseed)
    if batch is None:
        batch = min(max(catalan(k) + 8, 60), n - 2)
    rng = np.random.default_rng(seed)
    for it in range(60):
        U = S.U
        def mv(x, U=U):
            y = T.matvec(x)
            if U.shape[1]:
                y = y - U @ (U.conj().T @ x)
            return y
        Op = sla.LinearOperator((n, n), matvec=mv, dtype=T.dtype)
        vals, vecs = sla.eigsh(Op, k=batch, which='LA', v0=rng.standard_normal(n), tol=1e-12)
        ones = vals > 1 - tol_one
        if progress:
            progress("  Lanczos round %d: %d more fixed vectors (total %d), largest other eigenvalue %.4f"
                     % (it + 1, int(ones.sum()), S.r + int(ones.sum()), float(vals[~ones].max()) if (~ones).any() else float('nan')))
        if not ones.any():
            return S.U, float(vals.max()), nseed
        for j in np.where(ones)[0]:
            S.add_vec(vecs[:, j])
    raise RuntimeError("no convergence")


def candidates(N, k, bases):
    """Generator of candidate fixed vectors at level k (doublings, tensor products, all rotations)."""
    if k - 1 in bases:
        Bk1 = bases[k - 1]
        idx = np.indices((N,) * (k - 1)).reshape(k - 1, -1)
        for c in range(Bk1.shape[1]):
            for p in range(k - 1):
                v = np.zeros((N,) * k, dtype=Bk1.dtype)
                v[tuple(np.insert(idx, p + 1, idx[p], axis=0))] = Bk1[:, c]
                v = v.reshape(-1)
                for r in range(k):
                    yield rotate(v, N, k, r)
    for a in range(1, k):
        b = k - a
        if a in bases and b in bases:
            for x in bases[a].T:
                for y in bases[b].T:
                    v = np.kron(x, y)
                    for r in range(k):
                        yield rotate(v, N, k, r)


def ck_seeded_lm(H, kmax, kstart_bases=None, log=None, progress=None):
    """Seeded computation; levels with N^k > 300000 use the low-memory path."""
    N = H.shape[0]
    bases = {}
    for k in range(1, kmax + 1):
        if N ** k <= 300000:
            U0 = generate_level_lowmem(N, k, bases, dtype=float if np.allclose(H.imag, 0) else complex) if k >= 2 else None
            U, below = fixed_basis(H, k, U0=U0)
            seeds = 0 if U0 is None else U0.shape[1]
        else:
            U, below, seeds = fixed_basis_lm(H, k, candidates(N, k, bases), progress=progress)
        bases[k] = U
        if log:
            log(k, (U.shape[1], below, seeds))
    return bases


# ---------------------------------------------------------------- approximate (probabilistic) count
def count_filtered(H, k, S=None, b=40, deg=30, a=0.95, seed=0, progress=None, tol=1e-6):
    """Count fixed vectors of T_k orthogonal to the columns of S (orthonormal, already fixed).
    Random block of b vectors, Chebyshev filter of degree deg damping [0,a] and blowing up 1,
    then Rayleigh-Ritz. Returns (count, ritz values sorted desc, residuals of counted vectors).
    Counted vectors are verified (residual < tol); the only failure mode is missing some,
    which needs count close to b (then rerun with larger b) or a gap above a."""
    N = H.shape[0]; n = N ** k
    T = T_operator_blas(H, k) if k >= 3 else sla.aslinearoperator(T_matrix(H, k))
    real = np.allclose(H.imag, 0)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, b)) if real else rng.standard_normal((n, b)) + 1j * rng.standard_normal((n, b))
    def proj(Y):
        if S is not None and S.shape[1]:
            Y -= S @ (S.conj().T @ Y)
        return Y
    TB = None
    def TT(Y):
        if TB is not None:
            return TB(Y)
        return np.column_stack([T.matvec(Y[:, j]) for j in range(Y.shape[1])])
    def L(Y):
        return (2.0 / a) * TT(Y) - Y
    X = proj(X); X /= np.linalg.norm(X, axis=0)
    t0, t1 = X, L(X)
    for j in range(2, deg + 1):
        t2 = L(t1); t2 *= 2; t2 -= t0; t0 = t1; t1 = t2; del t2
        sc = np.linalg.norm(t1, axis=0).max()
        t0 /= sc; t1 /= sc
        if j % 5 == 0:
            proj(t0); proj(t1)
            if progress:
                progress("  filter step %d/%d" % (j, deg))
    del t0
    proj(t1)
    # orthonormalize in place (Cholesky QR twice), avoiding the copies made by np.linalg.qr
    for _ in range(2):
        G = t1.conj().T @ t1
        G = (G + G.conj().T) / 2
        ev, EV = np.linalg.eigh(G)
        keep = ev > ev.max() * 1e-20
        t1 = t1 @ (EV[:, keep] / np.sqrt(ev[keep]))
    Q = t1; del t1
    M = np.empty((Q.shape[1], Q.shape[1]), dtype=Q.dtype)
    for j in range(Q.shape[1]):                      # M = Q* T Q, one column at a time
        M[:, j] = Q.conj().T @ T.matvec(Q[:, j])
    M = (M + M.conj().T) / 2
    w, Z = np.linalg.eigh(M)
    idx = np.argsort(w)[::-1]; w = w[idx]; Z = Z[:, idx]
    near = w > 1 - 1e-4
    V = Q @ Z[:, near]; del Q
    TV = np.column_stack([T.matvec(V[:, j]) for j in range(V.shape[1])]) if V.shape[1] else V
    res = np.linalg.norm(TV - V, axis=0)
    cnt = int(near.sum())
    return cnt, w, res, V


def seed_lm(H, k, cand_iter, progress=None):
    """Verified seeds only (no Lanczos). Returns SpanLM."""
    N = H.shape[0]; n = N ** k
    T = T_operator_blas(H, k) if k >= 3 else T_operator_fast(H, k)
    S = SpanLM(n, T.dtype, cap=400 if n < 500000 else 240)
    ncand = 0
    for v in cand_iter:
        ncand += 1
        v = v.astype(T.dtype, copy=True)
        if S.r:
            v -= S.U @ (S.U.conj().T @ v)
            v -= S.U @ (S.U.conj().T @ v)
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            continue
        v /= nv
        if np.linalg.norm(T.matvec(v) - v) < 1e-8:
            S._grow(1); S.B[:, S.r] = v; S.r += 1
            if progress and S.r % 20 == 0:
                progress("  seeds verified so far: %d (candidates looked at: %d)" % (S.r, ncand))
    return S


def ck_approx(H, kmax, log=None, progress=None, b=40, deg=50, a=0.95, exact_max=300000, seed=0):
    """c_1..c_kmax; levels with N^k <= 300000 exact (as ck_seeded_lm), larger levels:
    verified seeds + Chebyshev-filtered random block on the complement (probabilistic).
    log(k, (c, next_ritz, seeds)) ; next_ritz = largest Ritz value below 1 (gap indicator)."""
    N = H.shape[0]
    bases = {}
    for k in range(1, kmax + 1):
        if N ** k <= exact_max:
            U0 = generate_level_lowmem(N, k, bases, dtype=float if np.allclose(H.imag, 0) else complex) if k >= 2 else None
            U, below = fixed_basis(H, k, U0=U0)
            bases[k] = U
            if log: log(k, (U.shape[1], below, 0 if U0 is None else U0.shape[1]))
            continue
        S = seed_lm_fast(H, k, candidates(N, k, bases), progress=progress)
        seeds = S.r
        if progress: progress("  seeding done: %d fixed vectors; random block of %d on the complement" % (seeds, b))
        bb = b
        while True:
            cnt, w, res, V = count_filtered(H, k, S=S.U, b=bb, deg=deg, a=a, progress=progress, seed=seed)
            if progress: progress("  block %d: %d extra fixed vectors, next Ritz value %.4f, max residual %.1e"
                                  % (bb, cnt, w[cnt] if cnt < len(w) else float('nan'), res.max() if cnt else 0))
            if cnt <= bb - 10:
                break
            bb *= 2
        S.add(V)
        bases[k] = S.U
        if log: log(k, (S.r, float(w[cnt]), seeds))
    return bases


def certify_complement(H, k, U, nev=4, progress=None, tol=1e-7, seed=11):
    """Largest eigenvalues of T_k on the orthogonal complement of the orthonormal columns of U
    (Lanczos, scipy eigsh). If the top value is clearly below 1, c_k = number of columns of U."""
    N = H.shape[0]; n = N ** k
    T = T_operator_blas(H, k)
    cnt = [0]
    def mv(x):
        x = np.asarray(x).reshape(-1).astype(T.dtype, copy=True)
        x -= U @ (U.conj().T @ x)
        y = T.matvec(x)
        y -= U @ (U.conj().T @ y)
        cnt[0] += 1
        if progress and cnt[0] % 50 == 0:
            progress("  certificate: %d Lanczos steps" % cnt[0])
        return y
    op = sla.LinearOperator((n, n), matvec=mv, dtype=T.dtype)
    v0 = np.random.default_rng(seed).standard_normal(n).astype(T.dtype)
    vals = sla.eigsh(op, k=nev, which='LA', v0=v0, tol=tol, return_eigenvectors=False)
    return np.sort(vals)[::-1], cnt[0]


def seed_lm_fast(H, k, cand_iter, progress=None, chunk=64, nverify=8):
    """Block version of seed_lm: candidates are projected in chunks (BLAS-3), independent directions
    extracted from the Gram matrix; seeds are fixed by construction, a sample is verified."""
    N = H.shape[0]; n = N ** k
    T = T_operator_blas(H, k) if k >= 3 else T_operator_fast(H, k)
    S = SpanLM(n, T.dtype, cap=400 if n < 500000 else 240)
    buf = []; ncand = [0]; last = [0]
    def flush():
        if not buf: return
        C = np.column_stack(buf).astype(T.dtype); buf.clear()
        C /= np.linalg.norm(C, axis=0)
        for _ in range(2):
            if S.r:
                C -= S.U @ (S.U.conj().T @ C)
        G = C.conj().T @ C; G = (G + G.conj().T) / 2
        w, V = np.linalg.eigh(G)
        keep = w > 1e-10 * max(1.0, w.max())
        if not keep.any(): return
        Y = C @ (V[:, keep] / np.sqrt(w[keep]))
        if S.r:
            Y -= S.U @ (S.U.conj().T @ Y)
        Q, R = np.linalg.qr(Y)
        good = np.abs(np.diag(R)) > 1e-7
        Q = Q[:, good]
        ok = [j for j in range(Q.shape[1]) if np.linalg.norm(T.matvec(Q[:, j]) - Q[:, j]) < 1e-8]
        Q = Q[:, ok]
        S._grow(Q.shape[1]); S.B[:, S.r:S.r + Q.shape[1]] = Q; S.r += Q.shape[1]
        if progress and S.r - last[0] >= 20:
            progress("  seeds so far: %d (candidates looked at: %d)" % (S.r, ncand[0])); last[0] = S.r
    for v in cand_iter:
        ncand[0] += 1; buf.append(v)
        if len(buf) == chunk: flush()
    flush()
    # verify a sample of seeds
    if S.r:
        idx = np.linspace(0, S.r - 1, min(nverify, S.r)).astype(int)
        err = max(np.linalg.norm(T.matvec(S.U[:, j]) - S.U[:, j]) for j in idx)
        if progress: progress("  seed check on %d samples: max residual %.1e" % (len(idx), err))
        if err > 1e-6: raise RuntimeError("seed verification failed: %.1e" % err)
    return S


def noncrossing_partitions(k):
    """All noncrossing partitions of {0..k-1}, as lists of blocks."""
    if k == 0:
        return [[]]
    out = []
    # block containing 0: {0} union S, S subset of 1..k-1, splitting the rest into noncrossing pieces
    def rec(rest):
        if not rest:
            yield []
            return
        first = rest[0]; others = rest[1:]
        from itertools import combinations
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                blk = [first] + list(S)
                # gaps between consecutive elements of blk must be partitioned independently
                pts = blk + [None]
                gaps = []
                for a, b in zip(blk, blk[1:]):
                    gaps.append([x for x in rest if a < x < b])
                gaps.append([x for x in rest if x > blk[-1]])
                def combine(gs):
                    if not gs:
                        yield []
                        return
                    for p in rec(gs[0]):
                        for q in combine(gs[1:]):
                            yield p + q
                for tail in combine(gaps):
                    yield [blk] + tail
    return list(rec(list(range(k))))


def nc_vectors(N, k):
    """Matrix whose columns are the vectors xi_pi, pi in NC(k): xi_pi(i)=1 iff i constant on the blocks of pi."""
    P = noncrossing_partitions(k)
    n = N ** k
    M = np.zeros((n, len(P)))
    strides = [N ** (k - 1 - s) for s in range(k)]
    for c, blocks in enumerate(P):
        idx = np.zeros(1, dtype=np.int64)
        for blk in blocks:
            w = sum(strides[s] for s in blk)
            idx = (idx[:, None] + w * np.arange(N)[None, :]).reshape(-1)
        M[idx, c] = 1.0
    return M


def ck_tl(H, k, b=20, deg=50, a=0.95, progress=None, seed=0):
    """c_k for a matrix expected to be Temperley-Lieb: seeds = the Catalan(k) vectors xi_pi, pi in NC(k)
    (fixed for any G in S_N^+), then the random block check on their complement.
    Returns (c_k, number of NC seeds, next Ritz value)."""
    N = H.shape[0]
    T = T_operator_blas(H, k)
    M = nc_vectors(N, k).astype(T.dtype)
    G = M.T @ M                              # Gram matrix of the 0/1 vectors, small
    w, E = np.linalg.eigh(G)
    keep = w > 1e-10 * w.max()
    M = M @ (E[:, keep] / np.sqrt(w[keep]))   # orthonormalize in place of QR, to save memory
    Q = M; del M
    idx = np.linspace(0, Q.shape[1] - 1, min(6, Q.shape[1])).astype(int)
    err = max(np.linalg.norm(T.matvec(Q[:, j]) - Q[:, j]) for j in idx)
    if progress: progress("  %d noncrossing seeds, check residual %.1e" % (Q.shape[1], err))
    if err > 1e-6: raise RuntimeError("NC seeds not fixed: %.1e" % err)
    bb = b
    while True:
        cnt, w, res, V = count_filtered(H, k, S=Q, b=bb, deg=deg, a=a, progress=progress, seed=seed)
        if progress: progress("  block %d: %d extra fixed vectors beyond Temperley-Lieb" % (bb, cnt))
        if cnt <= bb - 10:
            break
        bb *= 2
    return Q.shape[1] + cnt, Q.shape[1], float(w[cnt]) if cnt < len(w) else float('nan')
