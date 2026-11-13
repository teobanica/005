"""Equivalence tests for Hadamard matrices.

Real case: H ~ K (row/column permutations and sign changes) iff the colored graphs below are
isomorphic (McKay): vertices r_i^{+-}, c_j^{+-}, edge r_i^e -- c_j^d iff e*d*H_ij = +1,
rows and columns colored differently. Invariant first: the 4-profile.
"""
import itertools
from collections import Counter
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism as iso


def profile4(H):
    """Multiset of |sum_x h_a h_b h_c h_d| over 4-subsets of rows."""
    H = np.real(H).round().astype(int)
    N = H.shape[0]
    c = Counter()
    for a, b, cc, d in itertools.combinations(range(N), 4):
        c[abs(int((H[a] * H[b] * H[cc] * H[d]).sum()))] += 1
    return dict(sorted(c.items()))


def row_signatures(H):
    """Sorted multiset, over rows i, of the multiset of |sum| over 4-subsets of rows containing i,
    and likewise for triples/pairs of rows refined by 4th row. Invariant under equivalence."""
    H = np.real(H).round().astype(int)
    N = H.shape[0]
    sig = [Counter() for _ in range(N)]
    for quad in itertools.combinations(range(N), 4):
        v = abs(int((H[quad[0]] * H[quad[1]] * H[quad[2]] * H[quad[3]]).sum()))
        for i in quad:
            sig[i][v] += 1
    return sorted(tuple(sorted(c.items())) for c in sig)


def graph(H):
    H = np.real(H).round().astype(int)
    N = H.shape[0]
    G = nx.Graph()
    for i in range(N):
        for e in (1, -1):
            G.add_node(("r", i, e), color="r")
            G.add_node(("c", i, e), color="c")
    for i in range(N):
        for j in range(N):
            for e in (1, -1):
                for d in (1, -1):
                    if e * d * H[i, j] == 1:
                        G.add_edge(("r", i, e), ("c", j, d))
    return G


def real_equivalent(H, K):
    """Return (True, mapping) / (False, reason)."""
    pH, pK = profile4(H), profile4(K)
    if pH != pK:
        return False, "row 4-profiles differ"
    pHt, pKt = profile4(H.T), profile4(K.T)
    if pHt != pKt:
        return False, "column 4-profiles differ"
    if row_signatures(H) != row_signatures(K):
        return False, "row signatures differ"
    if row_signatures(H.T) != row_signatures(K.T):
        return False, "column signatures differ"
    GM = iso.GraphMatcher(graph(H), graph(K), node_match=lambda a, b: a["color"] == b["color"])
    if GM.is_isomorphic():
        return True, GM.mapping
    return False, "graphs not isomorphic"


def apply_mapping(H, mapping):
    """Check a found mapping really gives an equivalence: build K' = D1 P H Q D2."""
    N = H.shape[0]
    rows = {}
    cols = {}
    for (t, i, e), (t2, j, d) in mapping.items():
        if e == 1:
            (rows if t == "r" else cols)[i] = (j, d)
    return rows, cols


# ---------------------------------------------------------------- backtracking equivalence (real case)
def _colkey(cols):
    """columns (as rows of a 2D array, one per column) normalized by their first entry, as a sorted multiset"""
    c = cols * cols[:, :1]
    return sorted(map(bytes, (c > 0).astype(np.uint8)))


def real_equivalent_bt(H, K, max_nodes=5_000_000):
    """Exact test: find sigma, d, tau, e with K[r,c] = d_r e_c H[sigma r, tau c].
    Returns (True, (sigma, d, tau, e)) or (False, None). Complete search with pruning."""
    H = np.real(H).round().astype(np.int8)
    K = np.real(K).round().astype(np.int8)
    N = H.shape[0]
    nodes = [0]

    def finish(sigma, d):
        m_ = len(sigma)
        A = (H[sigma] * np.array(d)[:, None]).T     # columns of H restricted, as rows
        B = K[:m_].T
        An = A * A[:, :1]
        Bn = B * B[:, :1]
        ka = {bytes((An[i] > 0).astype(np.uint8)): i for i in range(N)}
        if len(ka) < N:
            return None
        tau, e = [], []
        for c in range(N):
            key = bytes((Bn[c] > 0).astype(np.uint8))
            a = ka.get(key)
            if a is None:
                return None
            tau.append(a)
            e.append(int(B[c, 0] * A[a, 0]))
        Ht = H[:, tau] * np.array(e)[None, :]
        used = set(sigma)
        sig, dd = list(sigma), list(d)
        rows = {bytes((Ht[i] > 0).astype(np.uint8)): (i, 1) for i in range(N)}
        rows.update({bytes((Ht[i] < 0).astype(np.uint8)): (i, -1) for i in range(N)})
        for r in range(m_, N):
            hit = rows.get(bytes((K[r] > 0).astype(np.uint8)))
            if hit is None or hit[0] in used:
                return None
            used.add(hit[0]); sig.append(hit[0]); dd.append(hit[1])
        return sig, dd, tau, e

    def rec(sigma, d):
        nodes[0] += 1
        if nodes[0] > max_nodes:
            raise RuntimeError("node limit")
        m_ = len(sigma)
        A = (H[sigma] * np.array(d)[:, None]).T
        if _colkey(A) != _colkey(K[:m_].T):
            return None
        if len(set(map(bytes, ((A * A[:, :1]) > 0).astype(np.uint8)))) == N:
            return finish(sigma, d)
        for i in range(N):
            if i in sigma:
                continue
            for s in ((1,) if m_ == 0 else (1, -1)):
                r = rec(sigma + [i], d + [s])
                if r:
                    return r
        return None

    r = rec([], [])
    return (True, r) if r else (False, None)


def check_bt(H, K, sol):
    sigma, d, tau, e = sol
    H = np.real(H).round(); K = np.real(K).round()
    return np.array_equal(K, np.array(d)[:, None] * H[np.ix_(sigma, tau)] * np.array(e)[None, :])
