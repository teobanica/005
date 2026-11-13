"""Jones-Wenzl count: e_k = multiplicity of the eigenvalue 1 of Phi_k, where Phi_k is T_k restricted to
range(p_k), p_k the Jones-Wenzl projection of S_N^+ on (C^N)^{otimes k}. By Theorem 6.2 of invariants.tex,
c_k = sum_s m_k(s) e_s with m_k(k) = 1, so if c_{k-1} = Cat_{k-1}, that is e_2 = ... = e_{k-1} = 0, then
c_k = Cat_k + e_k. No Temperley-Lieb seeds need to be stored: we work directly on range(p_k), which is
{all leg sums zero} cap {zero on words with 2 equal adjacent letters}, by alternating projections.

Count: random block on range(p_k), Chebyshev filter T_deg(2x/a - 1), which is bounded by 1 on [0,a] and
equals about 5e5 at x = 1 for deg = 50, a = 0.95, then Rayleigh-Ritz, counting the Ritz values > 1 - 1e-4,
whose vectors are checked directly (residual ||T v - v||, and leak out of range(p_k))."""
import numpy as np
import ck


def projector(N, k):
    """P(X, sweeps): in place, alternating projections onto range(p_k); leak(X): distance to range(p_k)."""
    sh = (N,) * k
    mask = np.ones(sh, bool)
    ar = np.arange(N)
    for s in range(k - 1):
        u = ar.reshape([N if t == s else 1 for t in range(k)])
        v = ar.reshape([N if t == s + 1 else 1 for t in range(k)])
        mask &= (u != v)
    mask = mask[..., None]

    def P(X, sweeps):
        Y = X.reshape(sh + (X.shape[1],))
        for _ in range(sweeps):
            for a in range(k):
                Y -= Y.mean(axis=a, keepdims=True)
            Y *= mask
        return X

    def leak(X):
        Y = X.reshape(sh + (X.shape[1],))
        nr = np.linalg.norm(X, axis=0)
        r = np.zeros(X.shape[1])
        for a in range(k):
            r = np.maximum(r, np.linalg.norm(Y.sum(axis=a).reshape(-1, X.shape[1]), axis=0) / np.sqrt(N))
        r = np.maximum(r, np.linalg.norm((Y * ~mask).reshape(-1, X.shape[1]), axis=0))
        return float((r / nr).max())
    return P, leak


def count_ek(H, k, b, deg=50, a=0.95, sweeps=5, final_sweeps=60, seed=0, say=None):
    """Returns dict: e (count), ritz (top Ritz values), residual (max over counted vectors), leak."""
    N = H.shape[0]; n = N ** k
    real = bool(np.allclose(H.imag, 0))
    dt = float if real else complex
    P, leak = projector(N, k)
    if n <= 400000:
        Tb = ck.T_block_blas(H, k)
        def T(X):
            return Tb(np.ascontiguousarray(X, dtype=dt))
    else:                                    # one column at a time: much less memory
        mv = ck.T_operator_blas(H, k).matvec
        def T(X):
            out = np.empty_like(X)
            for j in range(X.shape[1]):
                out[:, j] = mv(np.ascontiguousarray(X[:, j]))
            return out
    def A(X):
        return P(T(X), sweeps)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, b)) if real else rng.standard_normal((n, b)) + 1j * rng.standard_normal((n, b))
    P(X, 60); X /= np.linalg.norm(X, axis=0)
    def L(Y):
        Z = A(Y); Z *= 2.0 / a; Z -= Y
        return Z
    t0, t1 = X, L(X); del X
    for j in range(2, deg + 1):
        t2 = L(t1); t2 *= 2; t2 -= t0; t0 = t1; t1 = t2; del t2
        sc = np.linalg.norm(t1, axis=0).max(); t0 /= sc; t1 /= sc
        if say and j % 10 == 0:
            say("    filter step %d/%d" % (j, deg))
    del t0
    P(t1, final_sweeps)
    for _ in range(2):                       # Cholesky-QR twice, via eigh
        G = t1.conj().T @ t1; G = (G + G.conj().T) / 2
        ev, EV = np.linalg.eigh(G); keep = ev > ev.max() * 1e-20
        t1 = t1 @ (EV[:, keep] / np.sqrt(ev[keep]))
    Q = t1; del t1
    lk = leak(Q)
    TQ = T(Q)
    M = Q.conj().T @ TQ; M = (M + M.conj().T) / 2
    w, Z = np.linalg.eigh(M); o = np.argsort(w)[::-1]; w = w[o]; Z = Z[:, o]
    near = w > 1 - 1e-4
    if near.any():
        V = Q @ Z[:, near]; TV = TQ @ Z[:, near]
        res = float(np.linalg.norm(TV - V, axis=0).max())
    else:
        res = 0.0
    return {"e": int(near.sum()), "ritz": [float(x) for x in w[:6]], "residual": res, "leak": lk}


def mult(K):
    """m_k(s), multiplicity of r_s in u^{otimes k} for S_N^+, k = 1..K, as a list of dicts."""
    v = {0: 1}; out = []
    for _ in range(K):
        w = {}
        for s, c in v.items():
            w[s] = w.get(s, 0) + c
            for t in ([1] if s == 0 else [s - 1, s, s + 1]):
                w[t] = w.get(t, 0) + c
        v = w; out.append(dict(v))
    return out


def catalan(k):
    from math import comb
    return comb(2 * k, k) // (k + 1)


def run(tag, here, jobs, hours, doc, max_block_gb=1.0):
    """jobs: (label, builder, k, b). Resumable; log and json in tag-output/, next to the script.
    If e_k comes out larger than b/2, the job is redone with block 2b (up to max_block_gb per block)."""
    import json, os, sys, threading, time
    out = os.path.join(here, tag + "-output"); os.makedirs(out, exist_ok=True)
    LOG = open(os.path.join(out, tag + ".log"), "a"); JS = os.path.join(out, tag + ".json")
    try:
        res = json.load(open(JS))
    except Exception:
        res = {}
    T0 = time.time()
    def say(s):
        s = "[%s | %5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - T0) / 60, s)
        print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    def watchdog():
        time.sleep(hours * 3600)
        say("%.0f hours reached, stopping. Results so far are in %s" % (hours, JS)); os._exit(0)
    threading.Thread(target=watchdog, daemon=True).start()
    say("start; numpy %s, python %s; %d jobs. %s" % (np.__version__, sys.version.split()[0], len(jobs), doc))
    for label, build, k, b in jobs:
        key = "%s, e_%d" % (label, k)
        if key in res:
            say("%s already done (e_%d = %d), skipping" % (label, k, res[key]["e"])); continue
        H = build(); N = H.shape[0]
        assert np.abs(np.abs(H) - 1).max() < 1e-9 and np.abs(H @ H.conj().T - N * np.eye(N)).max() < 1e-8, label
        cplx = 1 if np.allclose(H.imag, 0) else 2
        while True:
            t0 = time.time()
            say("=== %s, level %d (dimension %d), block %d ===" % (label, k, N ** k, b))
            try:
                r = count_ek(H, k, b, say=say)
            except MemoryError:
                say("  out of memory, skipping this job"); r = None; break
            if r["e"] <= b // 2:
                break
            if 2 * b * N ** k * 8 * cplx > max_block_gb * 1e9:
                say("  e_%d = %d with block %d, block cap reached: e_%d >= %d" % (k, r["e"], b, k, r["e"])); break
            say("  e_%d = %d, too close to the block size, redoing with block %d" % (k, r["e"], 2 * b)); b *= 2
        if r is None:
            continue
        r.update({"k": k, "N": N, "block": b, "seconds": round(time.time() - t0)})
        res[key] = r; json.dump(res, open(JS, "w"), indent=1)
        if r["e"] == 0:
            msg = "c_%d = Cat_%d = %d, if c_%d = Cat_%d" % (k, k, catalan(k), k - 1, k - 1)
        else:
            msg = "<<< EIGENVALUE 1 IN PHI_%d: c_%d = Cat_%d + %d = %d, if c_%d = Cat_%d" % (
                k, k, k, r["e"], catalan(k) + r["e"], k - 1, k - 1)
        say("  %s: e_%d = %d   (Ritz values %s, residual %.1e, leak %.1e, %.0f s)   %s" % (
            label, k, r["e"], np.round(r["ritz"][:3], 4), r["residual"], r["leak"], time.time() - t0, msg))
    say("all jobs done.")
