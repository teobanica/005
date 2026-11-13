"""Stage 2, layer 1: Dita type families. Writes results/stage2.json and results/stage2.log."""
import json, time
from math import gcd
import numpy as np
import matrices as m
import ck

log = open("results/stage2.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


def cvec(H, kmax, tol=1e-7):
    """c_1..c_kmax by dense Hermitian eigenvalues; also smallest distance of a non-1 eigenvalue to 1."""
    out, gaps = [], []
    for k in range(1, kmax + 1):
        T = ck.T_matrix(H, k)
        w = np.linalg.eigvalsh((T + T.conj().T) / 2)
        c = int(np.sum(w > 1 - tol))
        out.append(c)
        rest = w[w <= 1 - tol]
        gaps.append(float(1 - rest.max()) if rest.size else 1.0)
    return out, min(gaps)


roots = [(f"e({a}/{n})", m.E(a / n)) for n in range(1, 13) for a in range(n) if gcd(a, n) == 1]

families = [
    ("F_6^(rs) = F_2 x_Q F_3", m.F6rs, 2, 4),
    ("F_6^(r/s) = F_3 x_Q F_2", m.F6rs_col, 2, 4),
    ("F_2 x_Q F_4", lambda *q: m.dita(m.fourier(2), m.fourier(4), m.dephased_Q(2, 4, q)), 3, 3),
    ("F_4 x_Q F_2", lambda *q: m.dita(m.fourier(4), m.fourier(2), m.dephased_Q(4, 2, q)), 3, 3),
    ("F_2 x_Q W_4", lambda *q: m.dita(m.fourier(2), m.kron(m.fourier(2), m.fourier(2)), m.dephased_Q(2, 4, q)), 3, 3),
    ("W_4 x_Q F_2", lambda *q: m.dita(m.kron(m.fourier(2), m.fourier(2)), m.fourier(2), m.dephased_Q(4, 2, q)), 3, 3),
    ("F_{8,5a}", m.F85a, 5, 3), ("F_{8,5b}", m.F85b, 5, 3), ("F_{8,5c}", m.F85c, 5, 3),
    ("F_{8,4}", m.F84, 4, 3), ("F_{8,4}^t", m.F84t, 4, 3),
    ("F_{8,2}", m.F82, 2, 3), ("F_{8,2}^t", m.F82t, 2, 3),
    ("F_3 x_Q F_3", lambda *q: m.dita(m.fourier(3), m.fourier(3), m.dephased_Q(3, 3, q)), 4, 3),
]

rng = np.random.default_rng(2026)
out = []
for name, f, d, kscan in families:
    t0 = time.time()
    fam = {"name": name, "params": d, "generic": [], "jumps": []}
    # generic points, up to k = 4
    for g in range(3):
        q = [m.E(x) for x in rng.random(d)]
        H = f(*q)
        cH, gH = cvec(H, 4); cT, gT = cvec(H.T, 4)
        fam["generic"].append({"q": [float(np.angle(x) / (2 * np.pi)) % 1 for x in q], "H": cH, "Ht": cT,
                               "gap": min(gH, gT)})
        say(f"{name:26s} generic {g}: H {cH}  H^t {cT}  gap {min(gH, gT):.2e}")
    gen = fam["generic"][0]
    genH, genT = gen["H"][:kscan], gen["Ht"][:kscan]
    # scans at k <= kscan
    pts = []
    for p in range(d):
        for lab, z in roots:
            pts.append((f"q{p+1}={lab}", p, z))
    for lab, z in roots:
        pts.append((f"all={lab}", None, z))
    njump = 0
    for lab, p, z in pts:
        q = [m.E(x) for x in rng.random(d)]
        if p is None:
            q = [z] * d
        else:
            q[p] = z
        H = f(*q)
        cH, _ = cvec(H, kscan); cT, _ = cvec(H.T, kscan)
        if cH != genH or cT != genT:
            njump += 1
            fam["jumps"].append({"at": lab, "H": cH, "Ht": cT})
    say(f"{name:26s} scan: {len(pts)} points, {njump} with jumps  ({time.time()-t0:.0f}s)")
    out.append(fam)
    json.dump(out, open("results/stage2.json", "w"), indent=1)
say("done")
