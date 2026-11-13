"""Exact value of c_2 for the Haagerup matrix H_6^1, which belongs to the Karlsson family K_6.
c_2 is the number of connected components of the graph on pairs (a,c), with (a,c) ~ (b,d) when
<H_a/H_b, H_c/H_d> != 0 (section 2.4). The entries of H_6^1 are in {1,-1,i,-i}, so all the scalar
products are Gaussian integers, computed here exactly, as pairs of Python integers.
Output: c2exact.log"""
import itertools
I = (0, 1); ONE = (1, 0)
def mul(x, y): return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0])
def conj(x): return (x[0], -x[1])
def neg(x): return (-x[0], -x[1])
q = ONE                                       # H_6^q at q = 1
i, m = I, neg(I)
H = [[ONE] * 6,
     [ONE, neg(ONE), i, i, m, m],
     [ONE, i, neg(ONE), m, q, neg(q)],
     [ONE, i, m, neg(ONE), neg(q), q],
     [ONE, m, conj(q), neg(conj(q)), i, neg(ONE)],
     [ONE, m, neg(conj(q)), conj(q), neg(ONE), i]]
N = 6
for a in range(N):                            # Hadamard check, exactly
    for b in range(N):
        s = (0, 0)
        for x in range(N): t = mul(H[a][x], conj(H[b][x])); s = (s[0] + t[0], s[1] + t[1])
        assert s == ((N, 0) if a == b else (0, 0))
def quot(a, b): return [mul(H[a][x], conj(H[b][x])) for x in range(N)]      # H_a/H_b, entries unimodular
def sp(u, v):
    s = (0, 0)
    for x in range(N): t = mul(u[x], conj(v[x])); s = (s[0] + t[0], s[1] + t[1])
    return s
pairs = list(itertools.product(range(N), repeat=2)); parent = {p: p for p in pairs}
def find(p):
    while parent[p] != p: parent[p] = parent[parent[p]]; p = parent[p]
    return p
for (a, c), (b, d) in itertools.product(pairs, repeat=2):
    if sp(quot(a, b), quot(c, d)) != (0, 0): parent[find((a, c))] = find((b, d))
comp = {}
for p in pairs: comp.setdefault(find(p), []).append(p)
sizes = sorted(len(v) for v in comp.values())
out = "H_6^1: exact Hadamard check ok; profile graph components: %d, of sizes %s; so c_2 = %d" % (len(sizes), sizes, len(sizes))
print(out); open("c2exact.log", "w").write(out + "\n")
