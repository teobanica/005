"""Stage 0, layer 1: calibration. Writes results/stage0.json and results/stage0.log."""
import json, os, time
from math import gcd
import numpy as np
import matrices as m
import ck

os.makedirs("results", exist_ok=True)
log = open("results/stage0.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); log.write(s + "\n"); log.flush()

out = {"fourier": [], "F4s": []}
K = [1, 2, 3, 4]

say("== Fourier matrices, expected c_k = N^(k-1)")
cases = [(f"F_{n}", m.fourier(n)) for n in range(2, 9)] + [
    ("F_2^2", m.kron(m.fourier(2), m.fourier(2))),
    ("F_2^3", m.kron(m.fourier(2), m.fourier(2), m.fourier(2))),
    ("F_2 x F_4", m.kron(m.fourier(2), m.fourier(4)))]
for name, H in cases:
    N = H.shape[0]
    row = {"name": name, "N": N, "c": {}, "gap": {}}
    for k in K:
        if N ** k > 5000:
            continue
        t = time.time()
        d, gap = ck.ck(H, k)
        row["c"][k] = d; row["gap"][k] = [float(gap[0]), float(gap[1])]
        ok = "ok" if d == N ** (k - 1) else "MISMATCH"
        say(f"{name:10s} k={k} c_k={d:5d} expected {N**(k-1):5d} {ok}  gap=({gap[0]:.1e},{gap[1]:.1e}) {time.time()-t:.1f}s")
    out["fourier"].append(row)

say("\n== F_4^s, s = e(a/n), n <= 12, gcd(a,n)=1, plus 3 random points; H and H^t")
pts = [(f"e({a}/{n})", m.E(a / n)) for n in range(1, 13) for a in range(n) if gcd(a, n) == 1]
rng = np.random.default_rng(2026)
pts += [(f"random e({t:.6f})", m.E(t)) for t in rng.random(3)]
for label, s in pts:
    for which in ["H", "Ht"]:
        H = m.F4s(s)
        if which == "Ht":
            H = H.T
        row = {"s": label, "which": which, "c": {}, "mingap": None}
        gaps = []
        for k in K:
            d, gap = ck.ck(H, k)
            row["c"][k] = d; gaps.append(gap[1] / max(gap[0], 1e-16))
        row["mingap_ratio"] = float(min(gaps))
        out["F4s"].append(row)
        say(f"s={label:22s} {which:2s} c_1..c_4 = {[row['c'][k] for k in K]}  min gap ratio {row['mingap_ratio']:.1e}")

json.dump(out, open("results/stage0.json", "w"), indent=1)
say("done")
