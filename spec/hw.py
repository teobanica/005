"""Highest weight part of T_k: Phi_k = T_k restricted to X_k = range of the Jones-Wenzl projection p_k,
X_k = vectors killed by the 2k-1 lowering maps (merge legs s,s+1; sum out leg s)."""
import sys, time, numpy as np
import scipy.sparse.linalg as sla
sys.path.insert(0, '/home/claude/work/night')
import ck

def lowering(N, k):
    sh = (N,) * k; n = N ** k; m = N ** (k - 1); idx = np.arange(N)
    def L(x):
        X = x.reshape(sh); out = []
        for s in range(k - 1):
            sl = (slice(None),) * s + (idx, idx)
            out.append(X[sl].reshape(m))
        for s in range(k):
            out.append(X.sum(axis=s).reshape(m))
        return np.concatenate(out)
    def Lt(y):
        X = np.zeros(sh, dtype=y.dtype); ys = y.reshape(2 * k - 1, *((N,) * (k - 1)))
        for s in range(k - 1):
            sl = (slice(None),) * s + (idx, idx)
            X[sl] += ys[s]
        for s in range(k):
            X += np.expand_dims(ys[k - 1 + s], s)
        return X.reshape(n)
    return L, Lt, (2 * k - 1) * m

def projector(N, k, m=40):
    """Projection onto X_k = range C cap range D, by alternating projections (DC)^m, on blocks.
    C = (1-J/N)^{otimes k} (all leg sums zero), D = zero out the words with 2 equal adjacent letters.
    Both commute with T_k, and the convergence is geometric, about 0.63 per sweep."""
    L, Lt, M = lowering(N, k)
    sh = (N,) * k; idx = np.indices(sh); mask = np.ones(sh, bool)
    for s in range(k - 1): mask &= idx[s] != idx[s + 1]
    mask = mask[..., None]; del idx
    def Pblock(X, mm=m):
        b = X.shape[1]; Y = X.reshape(sh + (b,)).copy()
        for _ in range(mm):
            for a in range(k): Y -= Y.mean(axis=a, keepdims=True)
            Y *= mask
        return Y.reshape(-1, b)
    return Pblock, L

def top_phi(H, k, nev=4, steps=8, tol=1e-10, maxrestart=40, seed=0, say=None):
    """Top eigenvalues of Phi_k by restarted block Krylov on X_k (block size N)."""
    N = H.shape[0]; n = N ** k; b = N
    Tb = ck.T_block_blas(H, k); P, L = projector(N, k)
    A = lambda X: P(Tb(np.array(X, dtype=complex, order="C", copy=True)))
    rng = np.random.default_rng(seed)
    Q = P(rng.standard_normal((n, b)) + 1j * rng.standard_normal((n, b)))
    Q, _ = np.linalg.qr(Q)
    nmv = 0
    for it in range(maxrestart):
        K = [Q]; AK = []
        for j in range(steps):
            W = A(K[-1]); nmv += 1; AK.append(W)
            if j == steps - 1: break
            Bsp = np.concatenate(K, axis=1); nrm0 = np.linalg.norm(W, axis=0).max()
            for _ in range(2): W = W - Bsp @ (Bsp.conj().T @ W)
            U, sv, _ = np.linalg.svd(W, full_matrices=False)
            keep = sv > 1e-8 * max(nrm0, 1e-300)
            if not keep.any(): break
            W = U[:, keep]
            for _ in range(2): W = W - Bsp @ (Bsp.conj().T @ W)
            W, _ = np.linalg.qr(W); K.append(W)
        Bsp = np.concatenate(K[:len(AK)], axis=1); ABsp = np.concatenate(AK, axis=1)
        M = Bsp.conj().T @ ABsp; M = (M + M.conj().T) / 2
        w, Z = np.linalg.eigh(M); o = np.argsort(w)[::-1]; w = w[o]; Z = Z[:, o]
        bb = min(b, Bsp.shape[1]); X = Bsp @ Z[:, :bb]; AX = ABsp @ Z[:, :bb]
        res = np.linalg.norm(AX - X * w[:bb], axis=0)
        if say: say("    restart %d: top %s, residuals %s" % (it, np.round(w[:nev], 10), np.array2string(res[:nev], precision=1)))
        if res[:nev].max() < tol: break
        Q, _ = np.linalg.qr(X)
    Xc = P(X[:, :nev], 120); Xc, _ = np.linalg.qr(Xc)
    AXc = Tb(np.array(Xc, order="C")); Mc = Xc.conj().T @ AXc; Mc = (Mc + Mc.conj().T) / 2
    wc, Zc = np.linalg.eigh(Mc); o = np.argsort(wc)[::-1]; wc = wc[o]; Zc = Zc[:, o]
    resc = np.linalg.norm(AXc @ Zc - (Xc @ Zc) * wc, axis=0)
    leak = max(np.linalg.norm(L(Xc[:, j])) for j in range(Xc.shape[1]))
    return wc, resc, nmv, leak
