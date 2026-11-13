"""Converged norms ||Phi_k|| of the Jones-Wenzl pieces, and multiplicities e_k of their eigenvalue 1.
Phi_k is T_k restricted to range(p_k) = {all leg sums zero} cap {zero on words with 2 equal adjacent letters}.
Since T_k commutes with p_k, the Lanczos iteration started in range(p_k) stays there; we re-project at each step
(alternating projections) to remove the rounding drift.
norm_phi: plain Lanczos for the top eigenvalue, with the Paige bound |lambda - theta| <= beta_m |s_m| as residual.
mult_phi: ARPACK (scipy eigsh) for the top eigenvalues, giving e_k and the next eigenvalue below 1."""
import numpy as np, time, warnings
warnings.filterwarnings("ignore")
import ck
from jwcount import projector


def operator(H, k):
    N = H.shape[0]
    real = bool(np.allclose(H.imag, 0))
    Hc = H.real if real else H
    if k >= 3:
        mv = ck.T_operator_blas(Hc, k).matvec
    else:
        Tb = ck.T_block_blas(Hc, k)
        mv = lambda x: Tb(np.ascontiguousarray(x.reshape(-1, 1)))[:, 0]
    return mv, (float if real else complex)


def norm_phi(H, k, tol=1e-9, maxit=400, sweeps=4, seed=0, say=None):
    N = H.shape[0]; n = N ** k
    mv, dt = operator(H, k)
    P, leak = projector(N, k)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) if dt == float else rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = P(v.reshape(-1, 1), 80)[:, 0]; v /= np.linalg.norm(v)
    alphas, betas = [], []
    v_old = np.zeros_like(v); beta = 0.0
    theta, res = 0.0, 1.0
    t0 = time.time()
    lk = 0.0
    for m in range(1, maxit + 1):
        w = mv(v)
        w = P(w.reshape(-1, 1), sweeps)[:, 0]
        alpha = float(np.real(np.vdot(v, w)))
        w = w - alpha * v - beta * v_old
        beta_new = float(np.linalg.norm(w))
        alphas.append(alpha); betas.append(beta_new)
        done = beta_new < 1e-12
        if m % 5 == 0 or done or m == maxit:
            Tm = np.diag(alphas) + np.diag(betas[:-1], 1) + np.diag(betas[:-1], -1)
            ev, S = np.linalg.eigh(Tm)
            theta = float(ev[-1]); res = abs(beta_new * S[-1, -1])
            if say and m % 20 == 0:
                say("      Lanczos step %d: theta = %.10f, residual %.1e (%.0f s)" % (m, theta, res, time.time() - t0))
            if res < tol or done:
                break
        v_old, v, beta = v, w / beta_new, beta_new
        if m % 10 == 0:
            lk = max(lk, leak(v.reshape(-1, 1)))
    return {"norm": theta, "residual": res, "steps": m, "leak": lk, "seconds": round(time.time() - t0)}


def mult_phi(H, k, nev, sweeps=30, tol=1e-10, say=None):
    """top nev eigenvalues of Phi_k by ARPACK; e_k = number above 1 - 1e-8, next = the largest below."""
    import scipy.sparse.linalg as sla
    N = H.shape[0]; n = N ** k
    mv, dt = operator(H, k)
    P, leak = projector(N, k)
    def A(x):
        y = P(np.array(x, dtype=dt).reshape(-1, 1), sweeps)[:, 0]
        y = mv(y)
        return P(y.reshape(-1, 1), sweeps)[:, 0]
    op = sla.LinearOperator((n, n), matvec=A, dtype=dt)
    rng = np.random.default_rng(1)
    v0 = P(rng.standard_normal((n, 1)), 80)[:, 0]
    t0 = time.time()
    w, V = sla.eigsh(op, k=nev, which='LA', v0=v0, tol=tol, ncv=min(n - 1, 2 * nev + 40))
    w = np.sort(w)[::-1]
    e = int((w > 1 - 1e-8).sum())
    nxt = float(w[e]) if e < len(w) else None
    R = np.array([np.linalg.norm(A(V[:, j]) - np.sort(w)[::-1][0] * 0) for j in range(0)])
    return {"e": e, "next": nxt, "top": [float(x) for x in w[:min(len(w), e + 5)]], "seconds": round(time.time() - t0)}


def mult_block(H, k, b, sweeps=30, tol=1e-9, maxiter=300, seed=3, say=None):
    """Block method (LOBPCG with locking) for the multiplicity e_k of the eigenvalue 1 of Phi_k.
    Rounds of block size b: the converged eigenvectors with eigenvalue 1 are locked (passed as constraints),
    until a round finds fewer than b - 4 of them; then e_k is the total, and the next eigenvalue is known."""
    import scipy.sparse.linalg as sla
    N = H.shape[0]; n = N ** k
    mv, dt = operator(H, k)
    P, leak = projector(N, k)
    def A(X):
        X = np.array(X, dtype=dt).reshape(n, -1)
        Y = P(X.copy(), sweeps)
        Z = np.column_stack([mv(np.ascontiguousarray(Y[:, j])) for j in range(Y.shape[1])])
        return P(Z, sweeps)
    op = sla.LinearOperator((n, n), matvec=lambda x: A(x)[:, 0], matmat=A, dtype=dt)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    Lock = None; res = 0.0; lk = 0.0; rounds = 0
    while True:
        rounds += 1
        X = P(rng.standard_normal((n, b)).astype(dt), 80)
        if Lock is not None:
            X -= Lock @ (Lock.conj().T @ X)
        w, V = sla.lobpcg(op, X, Y=Lock, largest=True, tol=tol, maxiter=maxiter)
        o = np.argsort(w)[::-1]; w = w[o]; V = V[:, o]
        e = int((w > 1 - 1e-7).sum())
        R = A(V[:, :e + 1]) - V[:, :e + 1] * w[:e + 1]
        res = max(res, float(np.linalg.norm(R, axis=0).max())); lk = max(lk, leak(V[:, :e + 1]))
        if e:
            Q, _ = np.linalg.qr(V[:, :e])
            Lock = Q if Lock is None else np.linalg.qr(np.hstack([Lock, Q]))[0]
        if say:
            say("      round %d: %d new eigenvectors with eigenvalue 1, next Ritz value %.8f (%.0f s)" % (
                rounds, e, w[e] if e < b else -1, time.time() - t0))
        if e < b - 4:
            nxt = float(w[e]); break
    total = 0 if Lock is None else Lock.shape[1]
    return {"e": total, "next": nxt, "rounds": rounds, "residual": res, "leak": lk, "seconds": round(time.time() - t0)}
