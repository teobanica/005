"""caracal.py -- c_7 for the order 6 non-Dita matrices, against Catalan's 429, via the noncrossing
partition vectors (always fixed) plus a random block check for anything beyond Temperley-Lieb.
Run:  python3 caracal.py
Log: caracal-output/caracal.log ; numbers: caracal-output/caracal.json ; stops by itself after 7 hours.
If interrupted, running it again skips the jobs already done. Memory: about 2 GB."""
import numpy as np
import nightlib, matrices as m
rng = np.random.default_rng(55); R = lambda d: [m.E(x) for x in rng.random(d)]
b6 = (np.sqrt(3) - 1) / 2
tq = next(t for t in rng.random(100) if np.cos(2 * np.pi * t) <= b6)
qH = R(1); aK = np.pi * rng.random(3)
jobs = [
    ("T_6 c_7", lambda: m.T6(), 7, 20),
    ("H_6^q, q random c_7", lambda: m.H6(*qH), 7, 20),
    ("BN_6^q, q random on arc c_7", lambda: m.BN6(m.E(tq)), 7, 20),
    ("K_6, random c_7", lambda: m.K6(*aK), 7, 20),
    ("BF_6 c_7", lambda: m.BF6(), 7, 20),
    ("H_6^q, q = 1 c_7", lambda: m.H6(1.0 + 0j), 7, 20),
    ("BN_6^q, q = i (D_6) c_7", lambda: m.BN6(1j), 7, 20),
]
nightlib.run_tl("caracal", jobs)
