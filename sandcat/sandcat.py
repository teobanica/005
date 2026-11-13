"""sandcat.py -- first, the multiplicities e_5, e_6 of the eigenvalue 1 of Phi_5, Phi_6 for the families F_{8,5}
(block method, about 1 hour, and 2-3 hours for e_6 of F_{8,5a}, done after the norms), giving c_5, c_6 exactly; then converged norms ||Phi_k|| for the wild matrices of orders 7 to 12, and for the Paley matrices
of orders 20 and 24 (8GB laptop, overnight). ||Phi_k|| < 1 for k <= K gives c_k = Cat_k for k <= K,
replacing the starred values of invariants.tex. Levels: 8 at order 7, 7 at orders 8 to 10, 6 at orders 11, 12,
and 5 at orders 20, 24, then the Paley matrices of orders 32 to 108, at levels 3 or 4.
Each top level takes about 5-45 minutes, memory up to 5 GB.
Run:  python3 sandcat.py      Log: sandcat-output/sandcat.log ; numbers: sandcat-output/sandcat.json
Stops by itself after 10 hours. If interrupted, running it again skips the jobs already done."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gaprun, cases, matrices as m
C = dict(cases.C)
C["N_11'"] = lambda: m.N11(1)
C["P_20^1"] = lambda: m.paley1(19); C["P_20^2"] = lambda: m.paley2(3, 2); C["P_24^1"] = lambda: m.paley1(23)
plan = [("P_12", 6), ("N_11", 6), ("N_11'", 6),
        ("F_{8,4}, random", 7), ("F_{8,2}, random", 7), ("K_{8,2}, random", 7), ("M_{8,1}, random", 7),
        ("I_{8,12}", 7), ("K_{8,30}", 7), ("L_{8,30}", 7),
        ("P_7^q, q random", 8), ("P_7^q, q = 1", 8), ("R_7^q, q random", 8),
        ("P_20^1", 5), ("P_20^2", 5), ("H_9", 7), ("BN_9", 7), ("P_24^1", 5), ("BN_10", 7)]
# then the Paley matrices of orders 32 to 108 (Jones-Wenzl counts e_k = 0 so far), as converged norms
P1 = lambda p, r=1: (lambda: m.paley1(p, r)); P2 = lambda p, r=1: (lambda: m.paley2(p, r))
paley = [("Paley I, q = 31", P1(31), 4), ("Paley II, q = 17", P2(17), 4), ("Paley I, q = 43", P1(43), 4),
         ("Paley I, q = 47", P1(47), 4), ("Paley II, q = 25", P2(5, 2), 3), ("Paley I, q = 59", P1(59), 3),
         ("Paley II, q = 29", P2(29), 3), ("Paley I, q = 67", P1(67), 3), ("Paley I, q = 71", P1(71), 3),
         ("Paley II, q = 37", P2(37), 3), ("Paley I, q = 79", P1(79), 3), ("Paley I, q = 83", P1(83), 3),
         ("Paley II, q = 41", P2(41), 3), ("Paley II, q = 49", P2(7, 2), 3), ("Paley I, q = 103", P1(103), 3),
         ("Paley I, q = 107", P1(107), 3), ("Paley II, q = 53", P2(53), 3)]
import numpy as np
_rng = np.random.default_rng(7); _p = [np.exp(2j * np.pi * x) for x in _rng.random(5)]     # generic point
F85 = {"F_{8,5a}": lambda: m.F85a(*_p), "F_{8,5b}": lambda: m.F85b(*_p), "F_{8,5c}": lambda: m.F85c(*_p)}
jobs = [("F_{8,5b}", F85["F_{8,5b}"], 5, 16), ("F_{8,5c}", F85["F_{8,5c}"], 5, 16), ("F_{8,5a}", F85["F_{8,5a}"], 5, 64),
        ("F_{8,5b}", F85["F_{8,5b}"], 6, 40), ("F_{8,5c}", F85["F_{8,5c}"], 6, 40)]
for nm, K in plan:
    jobs += [(nm, C[nm], k) for k in range(2, K + 1)]
jobs += [("F_{8,5a}", F85["F_{8,5a}"], 6, 64)]
for nm, b, K in paley:
    jobs += [(nm, b, k) for k in range(2, K + 1)]
gaprun.run("sandcat", HERE, jobs, 10.0, "Converged norms ||Phi_k||, orders 7 to 24.")
