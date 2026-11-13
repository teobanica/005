"""Is H (up to equivalence) a Dita deformation K x_Q L, with K of size M and L of size N, MN = size of H?

Note: a successful search only gives a CANDIDATE structure. For H to be of Dita type, in the form
H_{ia,jb} = Q_{ib} K_{ij} (L_i)_{ab} of section 1, the outer blocks (one row of each group, restricted to one
column class) must moreover all be equivalent to K; see checks/f85bpoint.py and checks/f85cpoint.py for this test.

Right Dita, (K x_Q L)_{ia,jb} = Q_{ib} K_{ij} L_{ab}: rows split into M groups R_i = {(i,a)}, columns into
N classes C_b = {(j,b)}, and for r, r' in the same group the quotient H_r/H_r' = L_a./L_a'. is constant on
each class C_b. Conversely, given such row groups and column classes, with H Hadamard, H is of the form
(D K x_Q L) after reordering. Left Dita = right Dita for H^t.

We search for the row groups: r ~ r' (candidate same group) iff the column-equality partition of H_r/H_r'
can be coarsened... we simply require: there is a column partition into N classes of size M, common to all
pairs in a group, on which all their quotients are constant. Search: grow the group of row 0 (then of the
next free row), keeping the common 'agreement' partition, which must stay splittable into classes of size M.
"""
import itertools
import numpy as np


def agree_classes(Q, tol=1e-7):
    """Columns c ~ c' iff all quotient vectors in Q (rows) agree at c and c'. Returns list of classes."""
    n = Q.shape[1]
    lab = [-1] * n
    classes = []
    for c in range(n):
        if lab[c] >= 0:
            continue
        cl = [c]
        lab[c] = len(classes)
        for d in range(c + 1, n):
            if lab[d] < 0 and np.all(np.abs(Q[:, c] - Q[:, d]) < tol):
                lab[d] = len(classes)
                cl.append(d)
        classes.append(cl)
    return classes


def splittable(classes, M):
    return all(len(c) % M == 0 for c in classes)


def right_dita(H, M, tol=1e-7, max_nodes=200000):
    """Search row groups of size N = n/M with a common column partition into classes of size M.
    Returns (row_groups, column_classes) or None."""
    n = H.shape[0]
    if n % M:
        return None
    N = n // M
    nodes = [0]

    def quotients(group):
        r0 = group[0]
        return np.array([H[r] / H[r0] for r in group[1:]]) if len(group) > 1 else np.zeros((0, n))

    def grow(groups, free, common):
        nodes[0] += 1
        if nodes[0] > max_nodes:
            raise RuntimeError("node limit")
        if not free:
            return groups, common
        r0 = free[0]
        # build the group of r0
        def rec(group, rest):
            if len(group) == N:
                Qg = quotients(group)
                allq = Qg if common is None else np.vstack([common, Qg]) if Qg.size else common
                # columns must agree for quotients within each group, with the SAME column partition for all groups:
                # we keep a stack of all within-group quotients; their common agreement classes must split into size M
                cl = agree_classes(allq if allq is not None and allq.size else np.zeros((1, n)), tol)
                if not splittable(cl, M):
                    return None
                r = grow(groups + [group], [x for x in free if x not in group], allq)
                return r
            for k, r in enumerate(rest):
                g = group + [r]
                Qg = quotients(g)
                allq = Qg if common is None else (np.vstack([common, Qg]) if Qg.size else common)
                if not splittable(agree_classes(allq, tol), M):
                    continue
                out = rec(g, rest[k + 1:])
                if out:
                    return out
            return None
        return rec([r0], free[1:])

    try:
        return grow([], list(range(n)), None)
    except RuntimeError:
        return "undecided"


def dita_report(H, name=""):
    n = H.shape[0]
    res = []
    for M in range(2, n):
        if n % M or n // M < 2:
            continue
        for side, X in (("right", H), ("left", H.T)):
            r = right_dita(X, M)
            if r == "undecided":
                res.append(f"{side} M={M}: undecided")
            elif r:
                groups, q = r
                cl = agree_classes(q, 1e-7)
                res.append(f"{side} Dita with M={M}, N={n//M}: row groups {groups}, column agreement classes {cl}")
    return res if res else ["no Dita structure"]


if __name__ == "__main__":
    import matrices as m
    tests = [("F_2 x_Q F_4", m.dita(m.fourier(2), m.fourier(4), m.dephased_Q(2, 4, [m.E(x) for x in (.1, .37, .71)]))),
             ("F_4 x_Q F_2", m.dita(m.fourier(4), m.fourier(2), m.dephased_Q(4, 2, [m.E(x) for x in (.1, .37, .71)]))),
             ("F_{8,4}", m.F84(*[m.E(x) for x in (.13, .29, .41, .77)])),
             ("P_7^q", m.P7(m.E(.3)))]
    for nm, H in tests:
        print(nm, dita_report(H))
