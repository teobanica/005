"""jaguar.py -- Jones-Wenzl counts e_k (eigenvalue 1 of Phi_k), up to level 8, for the wild matrices of
invariants.tex (same random parameters, seed 55). If c_{k-1} = Cat_{k-1}, then c_k = Cat_k + e_k.
In order of priority:
  1. e_6 for P_7^1, I_{8,12}, K_{8,30}, L_{8,30}, BN_9, N_11   (c_6 open in the paper)
  2. e_8 for the 7 matrices of order 6                          (c_8, beyond the c_7 = 429 counts)
  3. e_7 for P_7^q, R_7^q, P_7^1                                (c_7 at order 7)
  4. e_7 for the 7 matrices of order 8                          (c_7 at order 8)
  5. e_6 for P_12                                               (c_6 at order 12)
Run:  python3 jaguar.py      Log: jaguar-output/jaguar.log ; numbers: jaguar-output/jaguar.json
Stops by itself after 7 hours. If interrupted, running it again skips the jobs already done.
Memory: at most about 3 GB."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jwcount, cases
C = cases.C
jobs = []
for lab in ["P_7^q, q = 1", "I_{8,12}", "K_{8,30}", "L_{8,30}", "BN_9", "N_11"]:
    jobs.append((lab, C[lab], 6, 8))
for lab in ["T_6", "BF_6", "K_6, random", "H_6^q, q = 1", "BN_6^q, q = i (D_6)", "H_6^q, q random", "BN_6^q, q random on arc"]:
    jobs.append((lab, C[lab], 8, 8))
for lab in ["P_7^q, q random", "R_7^q, q random", "P_7^q, q = 1"]:
    jobs.append((lab, C[lab], 7, 8))
for lab in ["F_{8,4}, random", "F_{8,2}, random", "K_{8,2}, random", "M_{8,1}, random", "I_{8,12}", "K_{8,30}", "L_{8,30}"]:
    jobs.append((lab, C[lab], 7, 8))
jobs.append(("P_12", C["P_12"], 6, 8))
jwcount.run("jaguar", HERE, jobs, 7.0, "Jones-Wenzl counts e_k, levels 6 to 8.")
