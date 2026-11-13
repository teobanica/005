"""Linear certificate for the Jordan test: find a real combination of the real quadratic forms (Re/Im parts of all
contraction components, on x = (Re c, Im c) in R^{2f}) which is positive definite. If found, no nonzero lowest
weight vector satisfies all the constraints, exactly (up to the final eigenvalue check)."""
import numpy as np, scipy.optimize as so


def real_forms_span(F, f):
    U = np.hstack([np.eye(f), 1j * np.eye(f)])          # c = U x
    d = 2 * f; G = np.zeros((d * d, d * d))
    for kind, B in F:
        # C_n = U^T B_n conj(U) (sesq) or U^T B_n U (bil), for all n at once: shape (d, d, M)
        if kind == "sesq":
            C = np.einsum('ai,abn,bj->ijn', U, B, U.conj())
        else:
            C = np.einsum('ai,abn,bj->ijn', U, B, U)
        for part in (C.real, C.imag):
            S = (part + part.transpose(1, 0, 2)) / 2
            Sv = S.reshape(d * d, -1)
            G += Sv @ Sv.T
    w, Vv = np.linalg.eigh(G)
    keep = w > w.max() * 1e-12
    basis = [Vv[:, j].reshape(d, d) for j in np.where(keep)[0]]
    return [(S + S.T) / 2 for S in basis]


def certify(basis, iters=3, seed=0, beta=200.0):
    m = len(basis); d = basis[0].shape[0]
    Sstack = np.array(basis)                              # (m, d, d)
    def neg_softmin(y):
        ny = np.linalg.norm(y); A = np.einsum('i,ijk->jk', y / ny, Sstack)
        lam, V = np.linalg.eigh(A); t = -beta * lam; tm = t.max(); wts = np.exp(t - tm); Z = wts.sum()
        soft = -(tm + np.log(Z)) / beta                   # smooth lower approx of lambda_min
        p = wts / Z
        grad_A = np.einsum('j,aj,bj->ab', p, V, V)        # d soft / dA
        g = np.einsum('ijk,jk->i', Sstack, grad_A)
        g = (g - (g @ (y / ny)) * (y / ny)) / ny
        return -soft, -g
    rng = np.random.default_rng(seed); best = -np.inf; besty = None
    for _ in range(iters):
        r = so.minimize(neg_softmin, rng.standard_normal(m), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        y = r.x / np.linalg.norm(r.x); lam = np.linalg.eigvalsh(np.einsum('i,ijk->jk', y, Sstack)).min()
        if lam > best: best, besty = lam, y
    return best, besty
