"""Full spectrum of T_k, sector by sector, and the S_N^+ multiplicity test."""
import sys, time, numpy as np
sys.path.insert(0, '/home/claude/work/night')
import ck, sectors

def spectrum(H, k, chunk=None):
    N = H.shape[0]
    if k == 1:
        return np.array([1.0] + [0.0] * (N - 1))
    if k == 2:
        V = np.array([[H[i] / H[j] for j in range(N)] for i in range(N)])   # V[i,j] = H_i/H_j
        G = np.einsum('abx,cdx->abcd', V, V.conj()).reshape(N * N, N * N)   # <v_ij, v_kl>
        T = np.abs(G) ** 2 / N ** 3
        return np.linalg.eigvalsh(T)
    Tb = ck.T_block_blas(H, k)
    out = []
    for E in sectors.sector_maps(N, k):
        d = E.shape[1]; EH = E.conj().T.tocsr(); B = np.empty((d, d), complex)
        ch = chunk or 4 * N
        for j0 in range(0, d, ch):
            cols = np.zeros((d, min(ch, d - j0)), complex)
            cols[np.arange(j0, j0 + cols.shape[1]), np.arange(cols.shape[1])] = 1
            B[:, j0:j0 + cols.shape[1]] = EH @ Tb(np.asarray(E @ cols))
        B = (B + B.conj().T) / 2
        out.append(np.linalg.eigvalsh(B))
    return np.concatenate(out)

def mult_table(K):
    """m[k][s] = multiplicity of r_s in u^{otimes k} for S_N^+ (N >= 4)."""
    m = [[1]]
    for k in range(1, K + 1):
        prev = m[-1] + [0, 0]; new = [0] * (k + 1)
        for s, c in enumerate(m[-1]):
            if s == 0: new[0] += c; new[1] += c
            else: new[s - 1] += c; new[s] += 2 * c; new[s + 1] += c
        m.append(new)
    return m

def dims(N, K):
    d = [1, N - 1]
    while len(d) <= K: d.append((N - 2) * d[-1] - d[-2])
    return d

def subtract(S, parts, tol):
    """Remove from the sorted array S the multiset sum of (array, multiplicity) parts.
    Returns (remainder, worst matching error) or (None, err) if impossible."""
    S = list(np.sort(S)); worst = 0.0
    need = np.sort(np.concatenate([np.repeat(a, m) for a, m in parts])) if parts else np.array([])
    rem = []; i = 0
    for x in need:
        while i < len(S) and S[i] < x - tol:
            rem.append(S[i]); i += 1
        if i < len(S) and abs(S[i] - x) <= tol:
            worst = max(worst, abs(S[i] - x)); i += 1
        else:
            return None, x
    rem += S[i:]
    return np.array(rem), worst

def test(H, K, tol=1e-8, say=print):
    N = H.shape[0]; m = mult_table(K); d = dims(N, K)
    Phi = {0: np.array([1.0]), 1: np.zeros(N - 1)}
    for k in range(1, K + 1):
        t = time.time(); S = spectrum(H, k)
        parts = [(Phi[s], m[k][s]) for s in range(k + 1) if s in Phi and (s < k or k == 1)]
        rem, err = subtract(S, parts, tol)
        if rem is None:
            say("  level %d: FAILS, the eigenvalue %.10f is missing" % (k, err)); return False
        ok = len(rem) == d[k] if k >= 2 else len(rem) == 0
        say("  level %d: dim %d, matched %d eigenvalues (max error %.1e), remainder %d, expected d_%d = %d  %s  (%.0f s)"
            % (k, len(S), len(S) - len(rem), err, len(rem), k, d[k], "OK" if ok else "MISMATCH", time.time() - t))
        if k >= 2: Phi[k] = rem
        if not ok: return False
        top = np.sort(rem)[-3:] if len(rem) else []
        say("           top eigenvalues of Phi_%d: %s" % (k, np.round(top, 6)))
    return True
