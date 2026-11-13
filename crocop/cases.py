"""The wild matrices of the paper, with the same random parameters (seed 55, as in the c_5 computations)."""
import numpy as np
import matrices as m
rng = np.random.default_rng(55)
R = lambda d: [m.E(x) for x in rng.random(d)]
bound = (np.sqrt(3) - 1) / 2
tq = next(t for t in rng.random(100) if np.cos(2 * np.pi * t) <= bound)
C = {}
C["T_6"] = lambda: m.T6()
C["BF_6"] = lambda: m.BF6(0)
_q = R(1); C["H_6^q, q random"] = lambda: m.H6(*_q)
C["H_6^q, q = 1"] = lambda: m.H6(1 + 0j)
C["BN_6^q, q random on arc"] = lambda: m.BN6(m.E(tq))
C["BN_6^q, q = i (D_6)"] = lambda: m.BN6(1j)
_a = np.pi * rng.random(3); C["K_6, random"] = lambda: m.K6(*_a)
_p = R(1); C["P_7^q, q random"] = lambda: m.P7(*_p)
C["P_7^q, q = 1"] = lambda: m.P7(1 + 0j)
_r = R(1); C["R_7^q, q random"] = lambda: m.R7(*_r)
_f4 = R(4); C["F_{8,4}, random"] = lambda: m.F84(*_f4)
_f2 = R(2); C["F_{8,2}, random"] = lambda: m.F82(*_f2)
_k2 = R(2); C["K_{8,2}, random"] = lambda: m.K82(*_k2)
_m1 = R(1); C["M_{8,1}, random"] = lambda: m.M81(*_m1)
C["I_{8,12}"] = lambda: m.from_log(m.I812_L, 12)
C["K_{8,30}"] = lambda: m.from_log(m.K830_L, 30)
C["L_{8,30}"] = lambda: m.from_log(m.L830_L, 30)
C["H_9"] = lambda: m.H9()
C["BN_9"] = lambda: m.from_log(m.BN9_L, 10)
C["BN_10"] = lambda: m.BN10()
C["N_11"] = lambda: m.N11(0)
C["P_12"] = lambda: m.paley1(11)
