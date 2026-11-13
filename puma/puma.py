"""puma.py -- overnight job for laptop 2 (wild side): c_6 for wild matrices of order 6-10, then c_5 at order 20.
Run:  python3 puma.py
Log: puma-output/puma.log ; numbers: puma-output/puma.json ; stops by itself after 7 hours.
If interrupted, running it again skips the jobs already done."""
import numpy as np
import nightlib, matrices as m
rng = np.random.default_rng(55); R = lambda d: [m.E(x) for x in rng.random(d)]
b6 = (np.sqrt(3) - 1) / 2
tq = next(t for t in rng.random(100) if np.cos(2 * np.pi * t) <= b6)
W = [
    ("T_6", lambda: m.T6()),
    ("H_6^q, q random", lambda q=R(1): m.H6(*q)),
    ("BN_6^q, q random on arc", lambda: m.BN6(m.E(tq))),
    ("K_6, random", lambda a=np.pi * rng.random(3): m.K6(*a)),
    ("P_7^q, q random", lambda q=R(1): m.P7(*q)),
    ("R_7^q, q random", lambda q=R(1): m.R7(*q)),
    ("F_{8,4}, random", lambda q=R(4): m.F84(*q)),
    ("F_{8,2}, random", lambda q=R(2): m.F82(*q)),
    ("K_{8,2}, random", lambda q=R(2): m.K82(*q)),
    ("M_{8,1}, random", lambda q=R(1): m.M81(*q)),
    ("H_9", lambda: m.H9()),
    ("BN_10", lambda: m.BN10()),
]
jobs = [(nm + " c_6", f, 6, 5000, 30) for nm, f in W]
jobs += [("P_20^1 c_5", lambda: m.paley1(19), 5, 5000, 30)]
nightlib.run("puma", jobs)
