"""orders.py -- converged norms ||Phi_k|| for McKay's real Hadamard matrices of orders 24 and 28:
k = 2, 3, 4 for the 45 classes of order 24 with c_2 = 2, k = 2, 3 for all the 487 classes of order 28,
and k = 4 for the first 9 classes of order 28. Since T_k(H) and T_k(H^t) have the same nonzero spectrum
(Theorem 2.3 of invariants.tex), H alone suffices."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gaprun, matrices as m
L24 = [l for l in open(HERE + "/had24.txt") if l.strip()]
L28 = [l for l in open(HERE + "/had28.txt") if l.strip()]
W = json.load(open(HERE + "/wild24.json"))
jobs = [("had24.%d" % i, (lambda i=i: m.from_hex(L24[i])), k) for i in W for k in (2, 3)]
jobs += [("had28.%d" % i, (lambda i=i: m.from_hex(L28[i])), k) for i in range(len(L28)) for k in (2, 3)]
jobs += [("had24.%d" % i, (lambda i=i: m.from_hex(L24[i])), 4) for i in W]
jobs += [("had28.%d" % i, (lambda i=i: m.from_hex(L28[i])), 4) for i in range(9)]
gaprun.run("orders", HERE, jobs, float(sys.argv[1]) if len(sys.argv) > 1 else 10.0, "Orders 24 and 28.")
