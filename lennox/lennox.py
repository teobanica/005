"""lennox.py -- Jones-Wenzl counts e_k (eigenvalue 1 of Phi_k), pushing c_k = Catalan one level further for the
wild matrices of invariants.tex (same random parameters, seed 55). If c_{k-1} = Cat_{k-1}, then c_k = Cat_k + e_k.
In order of priority (cheap first):
  1. e_8 for P_7^q, P_7^1, R_7^q        (dimension 7^8 = 5.8 million, about 10 minutes each)
  2. e_6 for P_12, e_7 for BN_10        (dimensions 3.0 and 10 million)
  3. e_8 for the 7 matrices of order 8  (dimension 8^8 = 16.8 million, about 25 minutes each)
  4. e_7 for N_11                        (dimension 11^7 = 19.5 million)
Run:  python3 lennox.py      Log: lennox-output/lennox.log ; numbers: lennox-output/lennox.json
Stops by itself after 7 hours. If interrupted, running it again skips the jobs already done.
Memory: about 2-3 GB for the first jobs, up to about 7 GB for the last ones (orders 8 and 11), so better not to
run anything else on this Mac meanwhile. If a job runs out of memory, it is skipped and the next one starts."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jwcount, cases
C = cases.C
jobs = []
for lab in ["P_7^q, q random", "P_7^q, q = 1", "R_7^q, q random"]:
    jobs.append((lab, C[lab], 8, 2))
jobs.append(("P_12", C["P_12"], 6, 2))
jobs.append(("BN_10", C["BN_10"], 7, 2))
for lab in ["F_{8,2}, random", "K_{8,30}", "I_{8,12}", "K_{8,2}, random", "M_{8,1}, random", "F_{8,4}, random", "L_{8,30}"]:
    jobs.append((lab, C[lab], 8, 2))
jobs.append(("N_11", C["N_11"], 7, 2))
jwcount.run("lennox", HERE, jobs, 7.0, "Jones-Wenzl counts: e_8 at orders 7 and 8, e_6 at 12, e_7 at 10 and 11.")
