"""sefo.py -- converged norms ||Phi_k|| for the 7 wild matrices of order 6, levels 2 to 9 (8GB laptop, overnight).
||Phi_k|| < 1 for k <= 9 gives c_k = Cat_k for k <= 9, replacing the starred values of invariants.tex.
Levels up to 8 take a few minutes each; level 9 (dimension 10 million) about 30-60 minutes each, memory up to 4 GB.
Run:  python3 sefo.py      Log: sefo-output/sefo.log ; numbers: sefo-output/sefo.json
Stops by itself after 10 hours. If interrupted, running it again skips the jobs already done."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gaprun, cases
names = ["T_6", "BF_6", "H_6^q, q random", "H_6^q, q = 1", "BN_6^q, q random on arc", "BN_6^q, q = i (D_6)", "K_6, random"]
jobs = [(nm, cases.C[nm], k) for k in range(2, 9) for nm in names]       # all of levels 2..8 first
jobs += [(nm, cases.C[nm], 9) for nm in names]                            # then level 9
gaprun.run("sefo", HERE, jobs, 10.0, "Converged norms ||Phi_k||, order 6, levels 2 to 9.")
