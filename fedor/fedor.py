"""fedor.py -- overnight job: c_1..c_3 for all 487 real Hadamard matrices of order 28 (McKay's classes,
numbered from 0 in had28.txt), first for H, then for H^t, then c_4 for H, as long as time allows.
Run:  python3 fedor.py
Log: fedor-output/fedor.log ; numbers: fedor-output/fedor.json ; stops by itself after 7 hours.
If interrupted, running it again skips the jobs already done."""
import numpy as np
import nightlib, matrices as m
L = [l for l in open(nightlib.HERE + "/had28.txt") if l.strip()]
jobs = [("had28.%d H c_3" % i, (lambda i=i: m.from_hex(L[i])), 3, 1000, 30) for i in range(len(L))]
jobs += [("had28.%d Ht c_3" % i, (lambda i=i: m.from_hex(L[i]).T.copy()), 3, 1000, 30) for i in range(len(L))]
jobs += [("had28.%d H c_4" % i, (lambda i=i: m.from_hex(L[i])), 4, 1000, 30) for i in range(len(L))]
nightlib.run("fedor", jobs)
