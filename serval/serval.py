"""serval.py -- the gap profile ||Phi_k|| at the next levels: k = 8 for the order 6 matrices,
k = 7 for P_7 and R_7, k = 6 for BN_10 and N_11. Needs the night: about 3 GB of memory and several hours.
Run:  python3 serval.py     Log: serval-output/serval.log ; numbers: serval-output/serval.json
Stops by itself after 9 hours. If interrupted, running it again skips the levels already done."""
import phiprof, cases
C = cases.C
jobs = []
for lab in ["T_6", "H_6^q, q random", "BN_6^q, q random on arc", "BF_6", "K_6, random"]:
    jobs.append((lab, C[lab], [8]))
for lab in ["P_7^q, q random", "R_7^q, q random"]:
    jobs.append((lab, C[lab], [7]))
for lab in ["BN_10", "N_11"]:
    jobs.append((lab, C[lab], [6]))
phiprof.run("serval", jobs, 9.0, "Gap profiles, next levels.")
