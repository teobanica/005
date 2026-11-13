"""Lowest weight fixed vectors Fix(G^{otimes k}) cap Lambda_k, per rotation sector, by a filtered random block.
Lambda_k = leg sums 0, zero on words with cyclically adjacent equal letters. Everything commutes with rotation."""
import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw")
import ck


def lam_tools(N, k):
    sh = (N,) * k
    mask = np.ones(sh, bool); ar = np.arange(N)
    for s in range(k):
        t = (s + 1) % k
        u = ar.reshape([N if a == s else 1 for a in range(k)]); v = ar.reshape([N if a == t else 1 for a in range(k)])
        mask &= (u != v)
    mask = mask[..., None]

    def P(X, sweeps):
        Y = X.reshape(sh + (X.shape[1],))
        for _ in range(sweeps):
            for a in range(k):
                Y -= Y.mean(axis=a, keepdims=True)
            Y *= mask
        return X

    def rot(X, j=1):   # cyclic rotation of the legs, by j
        Y = X.reshape(sh + (X.shape[1],))
        return np.ascontiguousarray(np.moveaxis(Y, list(range(k)), [(a + j) % k for a in range(k)])).reshape(X.shape)

    def sector(X, w):  # projection onto rho = exp(2 pi i w / k)
        om = np.exp(2j * np.pi * w / k); out = np.zeros_like(X, dtype=complex); Y = X.astype(complex)
        for j in range(k):
            out += om ** (-j) * Y; Y = rot(Y)
        return out / k
    return P, rot, sector


def lw_fixed(H, k, w, b, deg=50, a=0.95, chunk=16, seed=0, say=None):
    N = H.shape[0]; n = N ** k
    P, rot, sector = lam_tools(N, k)
    Tb = ck.T_block_blas(H, k)

    def T(X):
        out = np.empty_like(X)
        for j in range(0, X.shape[1], chunk):
            out[:, j:j + chunk] = Tb(np.ascontiguousarray(X[:, j:j + chunk]))
        return out

    def A(X):
        return P(T(X), 5)
    rng = np.random.default_rng(seed)
    X = sector(rng.standard_normal((n, b)) + 1j * rng.standard_normal((n, b)), w)
    P(X, 60); X /= np.linalg.norm(X, axis=0)
    L = lambda Y: (lambda Z: (Z.__imul__(2.0 / a), Z.__isub__(Y), Z)[-1])(A(Y))
    t0, t1 = X, L(X)
    for j in range(2, deg + 1):
        t2 = L(t1); t2 *= 2; t2 -= t0; t0 = t1; t1 = t2
        sc = np.linalg.norm(t1, axis=0).max(); t0 /= sc; t1 /= sc
    P(t1, 80); t1 = sector(t1, w); P(t1, 20)
    for _ in range(2):
        G = t1.conj().T @ t1; G = (G + G.conj().T) / 2; ev, EV = np.linalg.eigh(G); keep = ev > ev.max() * 1e-20
        t1 = t1 @ (EV[:, keep] / np.sqrt(ev[keep]))
    Q = t1; TQ = T(Q); M = Q.conj().T @ TQ; M = (M + M.conj().T) / 2
    e, Z = np.linalg.eigh(M); o = np.argsort(e)[::-1]; e = e[o]; Z = Z[:, o]
    near = e > 1 - 1e-6
    V = Q @ Z[:, near]
    res = float(np.abs(T(V) - V).max()) if near.any() else 0.0
    nxt = float(e[~near].max()) if (~near).any() else None
    return V, res, nxt
