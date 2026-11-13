"""aerts.py -- Jones-Wenzl counts e_9 for the 7 matrices of order 6 (dimension 6^9 = 10,077,696),
then e_7 for H_9 and BN_9 (dimension 9^7 = 4,782,969). Same matrices and random parameters as in
invariants.tex (seed 55). If c_8 = Cat_8, then c_9 = Cat_9 + e_9 = 4862 + e_9.
Run:  python3 aerts.py      Log: aerts-output/aerts.log ; numbers: aerts-output/aerts.json
Stops by itself after 7 hours. If interrupted, running it again skips the jobs already done.
Memory: about 4 GB at level 9, so better not to run anything else heavy on this Mac meanwhile."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jwcount, cases
C = cases.C
jobs = []
for lab in ["H_6^q, q random", "BN_6^q, q random on arc", "T_6", "K_6, random", "BF_6", "H_6^q, q = 1", "BN_6^q, q = i (D_6)"]:
    jobs.append((lab, C[lab], 9, 2))
for lab in ["H_9", "BN_9"]:
    jobs.append((lab, C[lab], 7, 8))
jwcount.run("aerts", HERE, jobs, 7.0, "Jones-Wenzl counts e_9 at order 6, then e_7 at order 9.")
