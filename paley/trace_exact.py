"""Exact Paley I certificate, in integer arithmetic only (no floating point anywhere).

For q = 3 mod 4 a prime power and N = q+1, the even nontrivial characters psi of F_q^* give eigenvectors
X_psi(x,y) = psi(y-x) of A = N^3 T_2, with eigenvalues m_psi = 4 mu(psi) mu'(psi), where mu, mu' are the
character sums of the autocorrelations C, C' of g(a) = h(-a)h(1-a) and g'(a) = h(a)h(a-1), h(0)=1, h = chi
elsewhere.  The Galois orbit traces P_d = sum_{ord psi = d} m_psi are integers, given by
    P_d = 4 sum_{r in Z_n} c_d(r) (c * c')(r),      n = (q-1)/2,
with c_d the Ramanujan sums, c(k) = C(s^k), c'(k) = C'(s^k), s a generator of the squares, and * the cyclic
convolution on Z_n.  If N^2 does not divide P_d, then m_psi/N^2, an eigenvalue of N T_2, is not an algebraic
integer for ord(psi) = d, and the quantum group of P^1_{q+1} is infinite.

All convolutions are computed exactly, by Kronecker substitution, the big integer products being done by
GMP (gmpconv.py).  The autocorrelation of chi itself is computed too, and checked to be -1 off 0, which is
the Paley (Jacobi sum) identity making P^1_{q+1} Hadamard.  Usage:
    python3 trace_exact.py primes LO HI      all primes q = 3 mod 4 in [LO, HI)
    python3 trace_exact.py powers            the 11 proper prime powers q = 3 mod 4 below 10^5
    python3 trace_exact.py brute Q ...       brute force check of the P_d formula on the full matrix A
Output goes to the screen and to trace_exact-<mode>.log and trace_exact-<mode>.json."""
import sys, json, itertools, numpy as np
from gmpconv import conv

def isprime(p): return p > 1 and all(p % r for r in range(2, int(p ** .5) + 1))
def factor(n):
    f, p = [], 2
    while p * p <= n:
        if n % p == 0:
            f.append(p)
            while n % p == 0: n //= p
        p += 1
    return f + ([n] if n > 1 else [])
def mob(n):
    r, p = 1, 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            if n % p == 0: return 0
            r = -r
        p += 1
    return -r if n > 1 else r
def ramanujan(d, n):
    """c_d(r) for r in Z_n, d | n, as an int64 array: sum over e | gcd(r,d) of mu(d/e) e."""
    g = np.gcd(np.arange(n), d); out = np.zeros(n, dtype=np.int64)
    for e in range(1, d + 1):
        if d % e == 0 and mob(d // e): out += np.where(g % e == 0, mob(d // e) * e, 0)
    return out

class Field:
    """F_q, q = p^r, elements 0..q-1 = base p digit vectors (digit i = coefficient of x^i)."""
    def __init__(self, p, r):
        self.p, self.r, self.q = p, r, p ** r; q = self.q
        self.w = p ** np.arange(r)
        self.D = (np.arange(q)[:, None] // self.w[None, :]) % p          # digits
        if r == 1:
            fs = factor(q - 1); g = next(x for x in range(2, q) if all(pow(x, (q - 1) // f, q) != 1 for f in fs))
            B = int(np.ceil(np.sqrt(q))); lo = np.array([pow(g, i, q) for i in range(B)], dtype=np.int64)
            hi = np.array([pow(g, B * j, q) for j in range(B)], dtype=np.int64)
            self.exp = ((hi[:, None] * lo[None, :]) % q).reshape(-1)[:q - 1]
        else:
            def mulmod(u, v, co):                                          # polynomials mod x^r + sum co_i x^i
                w = [0] * (2 * r - 1)
                for i, a in enumerate(u):
                    for j, b in enumerate(v): w[i + j] = (w[i + j] + a * b) % p
                for i in range(2 * r - 2, r - 1, -1):
                    t = w[i]; w[i] = 0
                    for j in range(r): w[i - r + j] = (w[i - r + j] - t * co[j]) % p
                return w[:r]
            def xpow(e, co):
                res, b = [1] + [0] * (r - 1), [0, 1] + [0] * (r - 2)
                while e:
                    if e & 1: res = mulmod(res, b, co)
                    b = mulmod(b, b, co); e >>= 1
                return res
            one, fs = [1] + [0] * (r - 1), factor(q - 1)
            co = next(co for co in itertools.product(range(p), repeat=r) if co[0] and xpow(q - 1, co) == one
                      and all(xpow((q - 1) // f, co) != one for f in fs))    # x has order q-1: primitive
            cur = [1] + [0] * (r - 1); exp = []; wl = [p ** i for i in range(r)]
            for k in range(q - 1):
                exp.append(sum(a * b for a, b in zip(cur, wl))); top = cur[-1]; cur = [0] + cur[:-1]
                cur = [(cur[i] - top * co[i]) % p for i in range(r)]
            self.exp = np.array(exp, dtype=np.int64)
        assert np.unique(self.exp).size == q - 1 and 0 not in self.exp      # primitive element
        self.log = np.full(q, -1, dtype=np.int64); self.log[self.exp] = np.arange(q - 1)
    def idx(self, dig): return (dig % self.p) @ self.w
    def neg(self, a): return self.idx(-self.D[a])
    def add(self, a, b): return self.idx(self.D[a] + self.D[b])
    def corr(self, f, g):
        """C(t) = sum_a f(a) g(a+t), exactly, on the additive group (Z_p)^r, by one Kronecker product."""
        p, r = self.p, self.r; L = 2 * p - 1
        F = np.zeros((L,) * r, dtype=np.int64); G = np.zeros((L,) * r, dtype=np.int64)
        F[(slice(0, p),) * r] = f.reshape((p,) * r, order='F')
        G[(slice(0, p),) * r] = g.reshape((p,) * r, order='F')[(slice(None, None, -1),) * r]
        lin = conv(F.reshape(-1), G.reshape(-1))[:L ** r].reshape((L,) * r)   # lin[j] = sum_i f(i) g(i+p-1-j)
        assert max(abs(int(x)) for x in lin.reshape(-1)) < 2 ** 62 // L ** r     # so int64 folding is exact
        lin = lin.astype(np.int64); fold = (p - 1 - np.arange(L)) % p
        for ax in range(r):                                                       # fold axis by axis mod p
            out = np.zeros(lin.shape[:ax] + (p,) + lin.shape[ax + 1:], dtype=np.int64)
            np.add.at(out, (slice(None),) * ax + (fold,), lin); lin = out
        return lin.reshape(-1, order='F')

def run(p, r):
    K = Field(p, r); q = K.q; N = q + 1; n = (q - 1) // 2; a = np.arange(q)
    chi = np.where(K.log % 2 == 0, 1, -1); chi[0] = 0
    assert chi[K.neg(1)] == -1
    J = K.corr(chi, chi); assert J[0] == q - 1 and all(x == -1 for x in J[1:])      # Paley identity
    h = chi.copy(); h[0] = 1
    one = np.ones(q, dtype=np.int64)
    g = h[K.neg(a)] * h[K.add(one * 1, K.neg(a))]           # h(-a) h(1-a)
    gp = h[a] * h[K.add(a, K.neg(one))]                      # h(a) h(a-1)
    C = K.corr(g, g); Cp = K.corr(gp, gp)
    s = K.exp[0::2]                                          # s^k, s = gamma^2 generates the squares
    c = np.array([int(x) for x in C[s]], dtype=np.int64); cp = np.array([int(x) for x in Cp[s]], dtype=np.int64)
    lin = conv(c, cp); cc = np.array(lin[:n], dtype=object); cc[:lin.size - n] += lin[n:]   # cyclic on Z_n
    P = {}
    for d in [d for d in range(2, n + 1) if n % d == 0]:
        cd = ramanujan(d, n); P[d] = 4 * sum(int(x) * int(y) for x, y in zip(cd, cc) if x)
    return q, N, n, P, C, Cp

def brute(p, r):
    """recompute P_d from the full matrix A = N^3 T_2 = S o S, as an exact integer trace with Ramanujan sums."""
    K = Field(p, r); q = K.q; N = q + 1; n = (q - 1) // 2
    chi = np.where(K.log % 2 == 0, 1, -1); chi[0] = 0
    H = np.zeros((N, N), dtype=np.int64); H[0, :] = 1; H[1:, 0] = -1
    for x in range(q): H[1 + x, 1:] = chi[K.add(np.arange(q), K.neg(np.full(q, x)))] + (np.arange(q) == x)
    assert (H @ H.T == N * np.eye(N, dtype=np.int64)).all()
    R = (H[:, None, :] * H[None, :, :]).reshape(N * N, N); S = R @ R.T; A = S * S
    ind = np.full((N, N), -1)                                # ind(x,y) = k with y-x = +-s^k
    for x in range(q):
        for y in range(q):
            if x != y:
                u = K.add(np.array([y]), K.neg(np.array([x])))[0]; l = K.log[u]
                ind[1 + x, 1 + y] = (l if l % 2 == 0 else K.log[K.neg(np.array([u]))[0]]) // 2
    sup = np.flatnonzero(ind.reshape(-1) >= 0); k = ind.reshape(-1)[sup]; As = A[np.ix_(sup, sup)]
    # exact eigenvector check: with Y_t the indicator of {ind = t}, A Y_t = sum_u B(u-t) Y_u + Z, Z independent
    # of t and supported outside sup; so A X_psi = m_psi X_psi for psi nontrivial, with m_psi = sum_u B(u) psi(u)
    ful = ind.reshape(-1); out_sup = np.flatnonzero(ful < 0)
    AY0 = A[:, ful == 0].sum(axis=1); B0 = np.zeros(n, dtype=np.int64)
    for u in range(n):
        vals = np.unique(AY0[sup][k == u]); assert vals.size == 1; B0[u] = vals[0]   # constant on each class
    for t in range(n):
        AY = A[:, ful == t].sum(axis=1)
        assert (AY[sup] == B0[(k - t) % n]).all() and (AY[out_sup] == AY0[out_sup]).all()
    out = {}
    for d in [d for d in range(2, n + 1) if n % d == 0]:
        cd = ramanujan(d, n); M = cd[(k[:, None] - k[None, :]) % n]
        tot = int((M * As).sum()); assert tot % (q * (q - 1)) == 0; out[d] = tot // (q * (q - 1))
        assert out[d] == int((cd * B0).sum())                 # the same trace, from the circulant symbol
    return out

if __name__ == "__main__":
    mode = sys.argv[1]; logf = open("trace_exact-%s.log" % mode, "w"); res = {}
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    if mode == "brute":
        for Q in map(int, sys.argv[2:]):
            p = next(x for x in range(2, Q + 1) if Q % x == 0); r = round(np.log(Q) / np.log(p))
            B = brute(p, r); E = run(p, r)[3]
            log("q=%d  brute force P_d = %s  formula P_d = %s  %s" % (Q, B, E, "agree" if B == E else "DISAGREE"))
            res[Q] = {"brute": B, "formula": E}
    else:
        if mode == "primes":
            lo, hi = int(sys.argv[2]), int(sys.argv[3]); qs = [(q, 1) for q in range(lo, hi) if q % 4 == 3 and isprime(q)]
        else:
            qs = [(p, r) for p in range(3, 100) if isprime(p) for r in range(2, 20) if p ** r < 10 ** 5 and p ** r % 4 == 3]
        bad = []
        for p, r in qs:
            q, N, n, P, C, Cp = run(p, r); good = [d for d, x in P.items() if x % (N * N)]
            if not good: bad.append(q)
            T = sum(P.values())
            log("%d N=%d T=%d T mod N^2=%d orders certifying: %s" % (q, N, T, T % (N * N), good))
            res[q] = {"p": p, "r": r, "T": T, "certifying": good, "P": {d: P[d] for d in good[:1]}}
        log("checked %d values of q; uncertified: %s" % (len(qs), bad))
    json.dump(res, open("trace_exact-%s.json" % mode, "w"), indent=0, default=int)
