"""Stages 4, 5, 6, layer 1. Writes results/stage46.json and results/stage46.log."""
import json, time
from math import gcd
import numpy as np
import matrices as m
import ck

log = open("results/stage46.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


def cvec(H, kmax, tol=1e-7):
    N = H.shape[0]
    if N >= 9 and kmax >= 4:
        r = ck.ck_seeded(H, kmax)
        return [r[k][0] for k in range(1, kmax + 1)], min(1 - r[k][1] for k in range(2, kmax + 1))
    out, gaps = [], []
    for k in range(1, kmax + 1):
        T = ck.T_matrix(H, k)
        w = np.linalg.eigvalsh((T + T.conj().T) / 2)
        c = int(np.sum(w > 1 - tol)); out.append(c)
        rest = w[w <= 1 - tol]
        gaps.append(float(1 - rest.max()) if rest.size else 1.0)
    return out, min(gaps)


def both(H, kmax):
    assert m.hadamard_defect(H) < 1e-8
    cH, g1 = cvec(H, kmax); cT, g2 = cvec(H.T, kmax)
    return cH, cT, min(g1, g2)


rng = np.random.default_rng(2026)
roots = [(f"e({a}/{n})", m.E(a / n)) for n in range(1, 13) for a in range(n) if gcd(a, n) == 1]
out = {}

def family(name, f, d, kscan, kgen=4):
    t0 = time.time()
    fam = {"generic": [], "jumps": []}
    for _ in range(3):
        q = [m.E(x) for x in rng.random(d)]
        cH, cT, g = both(f(*q), kgen)
        fam["generic"].append({"H": cH, "Ht": cT, "gap": g})
        say(f"{name:12s} generic: H {cH}  H^t {cT}  gap {g:.2e}")
    gen = (fam["generic"][0]["H"][:kscan], fam["generic"][0]["Ht"][:kscan])
    pts = [(f"q{p+1}={lab}", p, z) for p in range(d) for lab, z in roots]
    if d > 1:
        pts += [(f"all={lab}", None, z) for lab, z in roots]
    for lab, p, z in pts:
        q = [m.E(x) for x in rng.random(d)]
        if p is None:
            q = [z] * d
        else:
            q[p] = z
        cH, cT, _ = both(f(*q), kscan)
        if (cH, cT) != gen:
            fam["jumps"].append({"at": lab, "H": cH, "Ht": cT})
    say(f"{name:12s} scan: {len(pts)} points, {len(fam['jumps'])} with jumps  ({time.time()-t0:.0f}s)")
    for j in fam["jumps"]:
        say("    ", j)
    out[name] = fam

say("== Stage 4, N = 7")
family("P_7^q", m.P7, 1, 4)
family("R_7^q", m.R7, 1, 4)
json.dump(out, open("results/stage46.json", "w"), indent=1)

say("\n== Stage 5, N = 8")
family("K_{8,2}", m.K82, 2, 3)
family("M_{8,1}", m.M81, 1, 3)
for name, L, r in [("I_{8,12}", m.I812_L, 12), ("K_{8,30}", m.K830_L, 30), ("L_{8,30}", m.L830_L, 30), ("L_{8,30}^t", m.L830t_L, 30)]:
    cH, cT, g = both(m.from_log(L, r), 4)
    out[name] = {"H": cH, "Ht": cT, "gap": g}
    say(f"{name:12s} H {cH}  H^t {cT}  gap {g:.2e}")
json.dump(out, open("results/stage46.json", "w"), indent=1)

say("\n== Stage 6, N = 9, 10, 11")
for name, H in [("H_9", m.H9()), ("BN_9", m.from_log(m.BN9_L, 10)), ("BN_10", m.BN10()),
                ("N_11", m.N11(0)), ("N'_11", m.N11(1))]:
    t0 = time.time()
    cH, cT, g = both(H, 4)
    out[name] = {"H": cH, "Ht": cT, "gap": g}
    say(f"{name:12s} H {cH}  H^t {cT}  gap {g:.2e}  ({time.time()-t0:.0f}s)")
    json.dump(out, open("results/stage46.json", "w"), indent=1)
say("done")
