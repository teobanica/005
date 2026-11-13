"""Layer 2: c_5 for the wild candidates with N <= 12. Writes results/level5.json and results/level5.log."""
import json, time
import numpy as np
import matrices as m
import ck

log = open("results/level5.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

rng = np.random.default_rng(55)
R = lambda d: [m.E(x) for x in rng.random(d)]
bound = (np.sqrt(3) - 1) / 2
tq = next(t for t in rng.random(100) if np.cos(2 * np.pi * t) <= bound)

cases = [
    ("T_6", m.T6()),
    ("BF_6", m.BF6(0)),
    ("H_6^q, q random", m.H6(*R(1))),
    ("H_6^q, q = 1", m.H6(1 + 0j)),
    ("BN_6^q, q random on arc", m.BN6(m.E(tq))),
    ("BN_6^q, q = i (D_6)", m.BN6(1j)),
    ("K_6, random", m.K6(*(np.pi * rng.random(3)))),
    ("P_7^q, q random", m.P7(*R(1))),
    ("P_7^q, q = 1", m.P7(1 + 0j)),
    ("R_7^q, q random", m.R7(*R(1))),
    ("F_{8,4}, random", m.F84(*R(4))),
    ("F_{8,2}, random", m.F82(*R(2))),
    ("K_{8,2}, random", m.K82(*R(2))),
    ("M_{8,1}, random", m.M81(*R(1))),
    ("I_{8,12}", m.from_log(m.I812_L, 12)),
    ("K_{8,30}", m.from_log(m.K830_L, 30)),
    ("L_{8,30}", m.from_log(m.L830_L, 30)),
    ("L_{8,30}^t", m.from_log(m.L830t_L, 30)),
    ("H_9", m.H9()),
    ("BN_9", m.from_log(m.BN9_L, 10)),
    ("BN_10", m.BN10()),
    ("N_11", m.N11(0)),
    ("N'_11", m.N11(1)),
    ("P_12", m.paley1(11)),
]
out = []
for name, H in cases:
    assert m.hadamard_defect(H) < 1e-8, name
    row = {"name": name}
    for which, M in [("H", H), ("Ht", H.T.copy())]:
        t0 = time.time()
        r = ck.ck_seeded(M, 5)
        c = [r[k][0] for k in range(1, 6)]
        nxt = r[5][1]
        row[which] = c; row[which + "_next5"] = nxt; row[which + "_seeded5"] = r[5][2]
        say(f"{name:26s} {which:2s} c_1..c_5 = {c}  next eigenvalue at k=5: {nxt:.4f}  ({time.time()-t0:.0f}s)")
    out.append(row)
    json.dump(out, open("results/level5.json", "w"), indent=1)
say("done")
