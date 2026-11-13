"""Stage 3, layer 1: wild candidates at order 6. Writes results/stage3.json and results/stage3.log."""
import json, time
from math import gcd
import numpy as np
import matrices as m
import ck

log = open("results/stage3.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


def cvec(H, kmax, tol=1e-7):
    out, gaps = [], []
    for k in range(1, kmax + 1):
        T = ck.T_matrix(H, k)
        w = np.linalg.eigvalsh((T + T.conj().T) / 2)
        c = int(np.sum(w > 1 - tol)); out.append(c)
        rest = w[w <= 1 - tol]
        gaps.append(float(1 - rest.max()) if rest.size else 1.0)
    return out, min(gaps)


def both(H, kmax):
    cH, g1 = cvec(H, kmax); cT, g2 = cvec(H.T, kmax)
    return cH, cT, min(g1, g2)


rng = np.random.default_rng(2026)
roots = [(f"e({a}/{n})", m.E(a / n)) for n in range(1, 13) for a in range(n) if gcd(a, n) == 1]
out = {}

say("== isolated")
for name, H in [("T_6", m.T6()), ("BF_6 (root a)", m.BF6(0)), ("BF_6 (root abar)", m.BF6(1))]:
    assert m.hadamard_defect(H) < 1e-9
    cH, cT, g = both(H, 4)
    out[name] = {"H": cH, "Ht": cT, "gap": g}
    say(f"{name:20s} H {cH}  H^t {cT}  gap {g:.2e}")

say("\n== H_6^q")
fam = []
pts = roots + [(f"random e({t:.6f})", m.E(t)) for t in rng.random(3)]
for lab, q in pts:
    H = m.H6(q); cH, cT, g = both(H, 4)
    fam.append({"q": lab, "H": cH, "Ht": cT, "gap": g})
    say(f"H_6^q  q={lab:22s} H {cH}  H^t {cT}  gap {g:.2e}")
out["H_6^q"] = fam

say("\n== BN_6^q on its arc Re q <= (sqrt3-1)/2")
fam = []
bound = (np.sqrt(3) - 1) / 2
arc_roots = [(lab, q) for lab, q in roots if q.real <= bound + 1e-12]
ts = []
while len(ts) < 3:
    t = rng.random()
    if np.cos(2 * np.pi * t) <= bound:
        ts.append(t)
pts = arc_roots + [(f"random e({t:.6f})", m.E(t)) for t in ts]
for lab, q in pts:
    H = m.BN6(q)
    if m.hadamard_defect(H) > 1e-8:
        say(f"BN_6^q q={lab}: not Hadamard (endpoint/branch), skipped"); continue
    cH, cT, g = both(H, 4)
    fam.append({"q": lab, "H": cH, "Ht": cT, "gap": g})
    say(f"BN_6^q q={lab:22s} H {cH}  H^t {cT}  gap {g:.2e}")
out["BN_6^q"] = fam

say("\n== Karlsson K_6(theta, phi, psi)")
fam = {"generic": [], "grid_jumps": [], "grid_points": 0, "skipped": 0}
for _ in range(3):
    th, ph, ps = np.pi * rng.random(3)
    H = m.K6(th, ph, ps)
    if m.hadamard_defect(H) > 1e-8:
        continue
    cH, cT, g = both(H, 4)
    fam["generic"].append({"p": [th, ph, ps], "H": cH, "Ht": cT, "gap": g})
    say(f"K_6 generic ({th:.4f},{ph:.4f},{ps:.4f})  H {cH}  H^t {cT}  gap {g:.2e}")
gen = fam["generic"][0]["H"][:3], fam["generic"][0]["Ht"][:3]
for i in range(6):
    for j in range(6):
        for l in range(6):
            th, ph, ps = i * np.pi / 6, j * np.pi / 6, l * np.pi / 6
            try:
                H = m.K6(th, ph, ps)
            except Exception:
                fam["skipped"] += 1; continue
            if not np.all(np.isfinite(H)) or m.hadamard_defect(H) > 1e-8:
                fam["skipped"] += 1; continue
            fam["grid_points"] += 1
            cH, cT, g = both(H, 3)
            if (cH, cT) != gen:
                fam["grid_jumps"].append({"p": f"({i},{j},{l})*pi/6", "H": cH, "Ht": cT})
say(f"K_6 grid: {fam['grid_points']} valid points, {fam['skipped']} skipped (degenerate), {len(fam['grid_jumps'])} with jumps")
for jmp in fam["grid_jumps"]:
    say("   ", jmp)
out["K_6"] = fam
json.dump(out, open("results/stage3.json", "w"), indent=1, default=str)
say("done")
