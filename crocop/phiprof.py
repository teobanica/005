"""Norms of the highest weight parts Phi_k = T_k restricted to range(p_k), p_k the Jones-Wenzl projection
of S_N^+ on (C^N)^{otimes k}. range(p_k) = {all leg sums zero} cap {zero on words with 2 equal adjacent
letters}; both are explicit projections commuting with T_k, and we project by alternating between them.
||Phi_k|| < 1 for k <= K means c_k = Catalan_k for k <= K; the profile k -> ||Phi_k|| measures the gap."""
import json, os, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ck


def projector(N, k):
    sh = (N,) * k; idx = np.indices(sh); mask = np.ones(sh, bool)
    for s in range(k - 1): mask &= idx[s] != idx[s + 1]
    mask = mask[..., None]; del idx
    def P(X, sweeps):
        b = X.shape[1]; Y = X.reshape(sh + (b,)).copy()
        for _ in range(sweeps):
            for a in range(k): Y -= Y.mean(axis=a, keepdims=True)
            Y *= mask
        return Y.reshape(-1, b)
    def leak(x):
        X = x.reshape(sh); r = [np.linalg.norm(X.sum(axis=a)) for a in range(k)]
        r += [np.linalg.norm(X * (~mask[..., 0])) ]
        return max(r) / np.linalg.norm(x)
    return P, leak


def top_phi_dense(H, k, nev=4):
    """Small cases (N^k <= 2000): T_k dense on an orthonormal basis of range(p_k)."""
    N = H.shape[0]; n = N ** k
    if k == 2:
        V = np.array([[H[i] / H[j] for j in range(N)] for i in range(N)])
        G = np.einsum('abx,cdx->abcd', V, V.conj()).reshape(n, n); T = np.abs(G) ** 2 / N ** 3
    else:
        T = ck.T_block_blas(H, k)(np.eye(n, dtype=complex))
    P, _ = projector(N, k); B = P(np.eye(n, dtype=complex), 200)
    U, s, _ = np.linalg.svd(B, full_matrices=False); U = U[:, s > 0.5]
    w = np.sort(np.linalg.eigvalsh(U.conj().T @ T @ U))[::-1]
    return w[:nev], np.zeros(min(nev, len(w))), U.shape[1]


def top_phi(H, k, nev=4, steps=None, sweeps=5, tol=1e-7, maxrestart=30, seed=0, say=None, big=3_000_000):
    """Top eigenvalues of Phi_k, restarted block Krylov on range(p_k), block size N,
    then an accurate final projection and exact Rayleigh-Ritz. Returns (values, residuals, dim used)."""
    N = H.shape[0]; n = N ** k; b = N
    if n <= 2000: return top_phi_dense(H, k, nev)
    steps = steps or max(3, min(5, 30 // N))
    P, leak = projector(N, k)
    if n * N > big:     # one column at a time: much less memory than the block version
        To = ck.T_operator_blas(H, k)
        Tb = lambda X: np.stack([To.matvec(np.ascontiguousarray(X[:, j])) for j in range(X.shape[1])], axis=1)
    else:
        Tb = ck.T_block_blas(H, k)
    A = lambda X: P(Tb(np.array(X, dtype=complex, order="C")), sweeps)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(P(rng.standard_normal((n, b)) + 1j * rng.standard_normal((n, b)), 60))
    Bsp = np.empty((n, steps * b), complex); ABsp = np.empty((n, steps * b), complex)
    for it in range(maxrestart):
        c = Q.shape[1]; Bsp[:, :c] = Q; nb = 0; cols = [(0, c)]
        for j in range(steps):
            c0, c1 = cols[-1]
            ABsp[:, c0:c1] = A(Bsp[:, c0:c1]); nb = c1
            if j == steps - 1: break
            W = ABsp[:, c0:c1].copy(); nrm0 = np.linalg.norm(W, axis=0).max()
            for _ in range(2): W -= Bsp[:, :c1] @ (Bsp[:, :c1].conj().T @ W)
            U, sv, _ = np.linalg.svd(W, full_matrices=False); keep = sv > 1e-8 * max(nrm0, 1e-300)
            if not keep.any(): break
            W = U[:, keep]; del U
            for _ in range(2): W -= Bsp[:, :c1] @ (Bsp[:, :c1].conj().T @ W)
            W, _ = np.linalg.qr(W); Bsp[:, c1:c1 + W.shape[1]] = W; cols.append((c1, c1 + W.shape[1])); del W
        M = Bsp[:, :nb].conj().T @ ABsp[:, :nb]; M = (M + M.conj().T) / 2
        w, Z = np.linalg.eigh(M); o = np.argsort(w)[::-1]; w = w[o]; Z = Z[:, o]
        bb = min(b, nb); X = Bsp[:, :nb] @ Z[:, :bb]; AX = ABsp[:, :nb] @ Z[:, :bb]
        res = np.linalg.norm(AX - X * w[:bb], axis=0); del AX
        if say and it % 5 == 4: say("      restart %d: top %.8f, residual %.1e" % (it + 1, w[0], res[:nev].max()))
        if res[:nev].max() < tol: break
        Q, _ = np.linalg.qr(X)
    del Bsp, ABsp
    Xc, _ = np.linalg.qr(P(X[:, :nev], 150))
    AXc = Tb(np.array(Xc, order="C")); Mc = Xc.conj().T @ AXc; Mc = (Mc + Mc.conj().T) / 2
    wc, Zc = np.linalg.eigh(Mc); o = np.argsort(wc)[::-1]; wc = wc[o]; Zc = Zc[:, o]
    resc = np.linalg.norm(AXc @ Zc - (Xc @ Zc) * wc, axis=0)
    lk = max(leak(Xc[:, j]) for j in range(Xc.shape[1]))
    return wc, resc, lk


def run(tag, jobs, hours, doc):
    """jobs: (label, builder, list of k). Resumable; log and json in tag-output/."""
    out = os.path.join(HERE, tag + "-output"); os.makedirs(out, exist_ok=True)
    LOG = open(os.path.join(out, tag + ".log"), "a"); JS = os.path.join(out, tag + ".json")
    try: res = json.load(open(JS))
    except Exception: res = {}
    T0 = time.time()
    def say(s):
        s = "[%s | %5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - T0) / 60, s)
        print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    def watchdog():
        time.sleep(hours * 3600); say("%.0f hours reached, stopping. Results so far are in %s" % (hours, JS)); os._exit(0)
    threading.Thread(target=watchdog, daemon=True).start()
    say("start; numpy %s, python %s; %d matrices. %s" % (np.__version__, sys.version.split()[0], len(jobs), doc))
    for label, build, ks in jobs:
        H = build(); N = H.shape[0]
        assert np.abs(np.abs(H) - 1).max() < 1e-9 and np.abs(H @ H.conj().T - N * np.eye(N)).max() < 1e-8, label
        r = res.setdefault(label, {})
        for k in ks:
            if str(k) in r: continue
            t0 = time.time(); say("=== %s, level %d (dimension %d) ===" % (label, k, N ** k))
            try:
                w, rs, lk = top_phi(H, k, say=say)
            except MemoryError:
                say("  out of memory, skipping"); break
            r[str(k)] = {"top": [float(x) for x in w], "residual": float(np.max(rs)) if len(rs) else 0.0,
                         "seconds": round(time.time() - t0)}
            json.dump(res, open(JS, "w"), indent=1)
            say("  ||Phi_%d|| = %.8f   (next %s, residual %.1e, %.0f s)%s" % (k, w[0], np.round(w[1:4], 6),
                np.max(rs) if len(rs) else 0, time.time() - t0, "   <<< EIGENVALUE 1, NOT CATALAN" if w[0] > 1 - 1e-6 else ""))
        prof = [round(r[str(k)]["top"][0], 6) for k in ks if str(k) in r]
        say("%s: profile ||Phi_k||, k = %s: %s" % (label, ks[:len(prof)], prof))
    say("all done.")
