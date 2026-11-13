"""The point (a,b,c,d,e) = (z,z^2,z^4,z^8,z^16), z = exp(2 pi i/101), of F_{8,5b} (remark after Conjecture 8.1).
All entries are 202-th roots of unity, stored exactly as exponents of zeta = exp(2 pi i/202), and a sum of such
roots is tested for vanishing exactly, by reduction modulo the cyclotomic polynomial Phi_202(x) = Phi_101(-x).
Computes: (1) exact Hadamard check; (2) c_2, as the number of components of the profile graph;
(3) all the Dita block structures, of sizes 2x4 and 4x2, and for each 2x4 one, the Haagerup sets of the inner
blocks. With the argument t, the same for the transpose. Output: f85bpoint.log, f85bpoint-transpose.log"""
import itertools
Z = 202
def e(k, sign=1): return (2 * k + (0 if sign > 0 else 101)) % Z        # +-z^k as an exponent of zeta
# F_{8,5b}(a,b,c,d,e) with a,b,c,d,e = z, z^2, z^4, z^8, z^16: bd = z^10, bd/a = z^9, cde = z^28, 1/c = z^-4
s, t, f, ic, E = 10, 9, 28, -4, 16
H = [[e(0)] * 8,
     [e(0), e(0, -1), e(s, -1), e(s), e(s, -1), e(s), e(0), e(0, -1)],
     [e(0), e(ic, -1), e(t, -1), e(8), e(t), e(8, -1), e(0, -1), e(ic)],
     [e(0), e(ic), e(t, -1), e(8, -1), e(t), e(8), e(0, -1), e(ic, -1)],
     [e(0), e(0), e(0, -1), e(0, -1), e(0, -1), e(0, -1), e(0), e(0)],
     [e(0), e(0, -1), e(s), e(s, -1), e(s), e(s, -1), e(0), e(0, -1)],
     [e(0), e(E), e(t), e(f), e(t, -1), e(f, -1), e(0, -1), e(E, -1)],
     [e(0), e(E, -1), e(t), e(f, -1), e(t, -1), e(f), e(0, -1), e(E)]]
PHI = [(-1) ** j for j in range(101)]               # Phi_202(x) = sum_j (-x)^j, degree 100, monic up to sign
def is_zero(exps):
    p = [0] * Z
    for x in exps: p[x % Z] += 1
    for deg in range(Z - 1, 99, -1):                 # reduce modulo Phi_202, leading coefficient (-1)^100 = 1
        c = p[deg]
        if c:
            for j in range(101): p[deg - 100 + j] -= c * PHI[j]
    return not any(p[:100])
import sys
if len(sys.argv) > 1 and sys.argv[1] == "t": H = [list(r) for r in zip(*H)]     # the transpose
n = 8
assert all(is_zero([H[i][x] - H[k][x] for x in range(n)]) == (i != k) for i in range(n) for k in range(n))
log = ["exact Hadamard check: ok"]
def sp(a, b, c, d): return [H[a][x] - H[b][x] - H[c][x] + H[d][x] for x in range(n)]   # <H_a/H_b, H_c/H_d>
pairs = list(itertools.product(range(n), repeat=2)); par = {p: p for p in pairs}
def find(p):
    while par[p] != p: p = par[p]
    return p
for (a, c), (b, d) in itertools.product(pairs, repeat=2):
    if not is_zero(sp(a, b, c, d)): par[find((a, c))] = find((b, d))
comp = {}
for p in pairs: comp.setdefault(find(p), []).append(p)
log.append("profile graph: %d components, of sizes %s, so c_2 = %d" % (len(comp), sorted(len(v) for v in comp.values()), len(comp)))
def row_partitions(M):
    size = n // M
    def rec(free):
        if not free: yield []; return
        r = free[0]
        for rest in itertools.combinations(free[1:], size - 1):
            g = (r,) + rest
            for tail in rec([x for x in free if x not in g]): yield [g] + tail
    return rec(list(range(n)))
def haagerup(rows, cols):
    return frozenset((H[i][j] - H[k][j] + H[k][l] - H[i][l]) % Z for i in rows for k in rows for j in cols for l in cols)
for M in (2, 4):
    N = n // M; found = 0
    for groups in row_partitions(M):
        agree = {c: c for c in range(n)}             # columns on which all the quotients within groups agree
        cls = [[c for c in range(n) if all((H[g[0]][c] - H[r][c]) % Z == (H[g[0]][c0] - H[r][c0]) % Z
                                            for g in groups for r in g)] for c0 in range(n)]
        classes = {tuple(x) for x in cls}
        if not all(len(x) % M == 0 for x in classes): continue
        found += 1
        msg = "size %dx%d: row groups %s, column agreement classes %s" % (M, N, groups, sorted(classes))
        if M == 2:
            reps = [sorted(classes)[i][0] for i in range(len(classes))] if len(classes) == N else None
            if reps:
                hs = [haagerup(g, reps) for g in groups]
                msg += "; inner Haagerup sets equal: %s, exponents (mod 202) %s / %s" % (hs[0] == hs[1], sorted(hs[0]), sorted(hs[1]))
        if M == 4 and len(classes) == N:               # outer blocks: one row per group, one column class
            reps = [g[0] for g in groups]
            hs = [haagerup(reps, list(cl)) for cl in sorted(classes)]
            msg += "; outer Haagerup sets equal: %s, exponents (mod 202) %s / %s" % (hs[0] == hs[1], sorted(hs[0]), sorted(hs[1]))
        log.append(msg)
    log.append("size %dx%d: %d block structures" % (M, N, found))
print("\n".join(log)); open("f85bpoint%s.log" % ("-transpose" if len(sys.argv) > 1 else ""), "w").write("\n".join(log) + "\n")
