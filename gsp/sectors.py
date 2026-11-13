"""Fixed vectors of T_k, sector by sector.

T_k commutes with the simultaneous cyclic rotation of the k legs, (T)_{s(i),s(j)} = T_{i,j}, because
tr(P_{i1j1}...P_{ikjk}) is invariant under cyclic permutations. So T_k is block diagonal with respect to
the k eigenspaces of the rotation, of dimension about N^k/k each, and c_k is the sum of the
dimensions of the fixed spaces in the k sectors. We count in each sector with the Chebyshev-filtered
random block of ck.count_filtered, the vectors being stored in reduced coordinates (one per orbit),
which divides the memory by about k."""
import numpy as np
import scipy.sparse as sp
import ck


def sector_maps(N, k):
    """For each sector s = 0..k-1, the isometry E_s : C^{d_s} -> C^{N^k} onto the omega^s eigenspace
    of the rotation, omega = e^{2 pi i/k}, as a sparse matrix."""
    n = N ** k
    w = np.arange(n, dtype=np.int64)
    top = N ** (k - 1)
    R = [w]
    for j in range(1, k):
        R.append((R[-1] % top) * N + R[-1] // top)          # rotate the digits left
    R = np.array(R)                                          # R[j] = sigma^j(w)
    rep = R.min(axis=0)
    size = np.full(n, k)
    for j in range(k - 1, 0, -1):
        size[R[j] == w] = j                                  # orbit size = least j > 0 with sigma^j w = w
    jrep = np.argmax(R == rep[None, :], axis=0)              # sigma^jrep(w) = rep
    m = (k - jrep) % k                                       # sigma^m(rep) = w
    maps = []
    for s in range(k):
        ok = (s * size) % k == 0
        reps = np.unique(rep[ok])
        col = np.searchsorted(reps, rep[ok])
        val = np.exp(2j * np.pi * s * m[ok] / k) / np.sqrt(size[ok])
        E = sp.csr_matrix((val, (w[ok], col)), shape=(n, len(reps)))
        maps.append(E)
    return maps


def count_sector(H, k, E, b, deg=50, a=0.95, seed=0, progress=None, chunk=None):
    """Number of fixed vectors of T_k inside the range of E, by the filtered random block method.
    Returns (count, ritz values desc, max residual of the counted vectors)."""
    N = H.shape[0]
    Tb = ck.T_block_blas(H, k)
    EH = E.conj().T.tocsr()
    chunk = chunk or 4 * N
    def A(C):
        out = np.empty_like(C)
        for j0 in range(0, C.shape[1], chunk):
            Y = np.asarray(E @ C[:, j0:j0 + chunk])
            out[:, j0:j0 + chunk] = EH @ Tb(Y)
        return out
    d = E.shape[1]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((d, b)) + 1j * rng.standard_normal((d, b))
    X /= np.linalg.norm(X, axis=0)
    def L(Y):
        Z = A(Y); Z *= 2.0 / a; Z -= Y
        return Z
    t0, t1 = X, L(X); del X
    for j in range(2, deg + 1):
        t2 = L(t1); t2 *= 2; t2 -= t0; t0 = t1; t1 = t2; del t2
        sc = np.linalg.norm(t1, axis=0).max(); t0 /= sc; t1 /= sc
        if progress and j % 10 == 0:
            progress("    filter step %d/%d" % (j, deg))
    del t0
    for _ in range(2):                                       # Cholesky-QR twice, via eigh
        G = t1.conj().T @ t1; G = (G + G.conj().T) / 2
        ev, EV = np.linalg.eigh(G); keep = ev > ev.max() * 1e-20
        t1 = t1 @ (EV[:, keep] / np.sqrt(ev[keep]))
    Q = t1; del t1
    AQ = A(Q)
    M = Q.conj().T @ AQ; M = (M + M.conj().T) / 2
    w, Z = np.linalg.eigh(M); idx = np.argsort(w)[::-1]; w = w[idx]; Z = Z[:, idx]
    near = w > 1 - 1e-4
    V = Q @ Z[:, near]; AV = AQ @ Z[:, near]; del Q, AQ
    res = float(np.linalg.norm(AV - V, axis=0).max()) if near.any() else 0.0
    return int(near.sum()), w, res


def ck_sectors(H, k, b, progress=None, seed=0, a=0.95):
    """c_k as a sum over the k sectors. b = initial block size per sector (doubled if needed).
    Returns (c_k, list of per-sector counts, next Ritz value below 1, max residual)."""
    N = H.shape[0]
    maps = sector_maps(N, k)
    counts, nxt, resmax = [], 0.0, 0.0
    for s, E in enumerate(maps):
        bb = b
        while True:
            if progress:
                progress("  sector %d/%d: dimension %d, block %d" % (s, k, E.shape[1], bb))
            c, w, res = count_sector(H, k, E, min(bb, E.shape[1]), progress=progress, seed=seed + s, a=a)
            if c <= bb - 10 or bb >= E.shape[1]:
                break
            bb *= 2
        below = w[w <= 1 - 1e-4]
        nxt = max(nxt, float(below[0]) if len(below) else 0.0)
        resmax = max(resmax, res)
        counts.append(c)
        if progress:
            progress("  sector %d: %d fixed vectors (next Ritz value %.4f, residual %.1e)"
                     % (s, c, float(below[0]) if len(below) else 0.0, res))
    return sum(counts), counts, nxt, resmax
