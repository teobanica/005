"""crocop.py -- the gap profile ||Phi_k|| of the wild matrices of orders 6 to 12, at the levels that fit in
little memory: k <= 7 for order 6, k <= 6 for orders 7 to 9, k <= 5 for orders 10 to 12.
Phi_k = T_k on the range of the Jones-Wenzl projection p_k; ||Phi_k|| < 1 for all k <= K means
c_k = Catalan_k up to level K. Fine to run with Firefox on top: memory at most about 1.5 GB.
Run:  python3 crocop.py     Log: crocop-output/crocop.log ; numbers: crocop-output/crocop.json
Stops by itself after 8 hours. If interrupted, running it again skips the levels already done."""
import phiprof, cases
C = cases.C
jobs = []
for lab in ["T_6", "H_6^q, q random", "BN_6^q, q random on arc", "BF_6", "K_6, random", "H_6^q, q = 1", "BN_6^q, q = i (D_6)"]:
    jobs.append((lab, C[lab], [2, 3, 4, 5, 6, 7]))
for lab in ["P_7^q, q random", "P_7^q, q = 1", "R_7^q, q random", "F_{8,4}, random", "F_{8,2}, random",
            "K_{8,2}, random", "M_{8,1}, random", "I_{8,12}", "K_{8,30}", "L_{8,30}", "H_9", "BN_9"]:
    jobs.append((lab, C[lab], [2, 3, 4, 5, 6]))
for lab in ["BN_10", "N_11", "P_12"]:
    jobs.append((lab, C[lab], [2, 3, 4, 5]))
phiprof.run("crocop", jobs, 8.0, "Gap profiles, orders 6 to 12.")
