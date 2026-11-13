"""lynx.py -- overnight job: c_1..c_4 for the 45 non-Dita real Hadamard matrices of order 24
(McKay's classes, numbered from 0 in had24.txt), first for H, then for H^t.
Run:  python3 lynx.py
Log: lynx-output/lynx.log ; numbers: lynx-output/lynx.json ; stops by itself after 7 hours.
If interrupted, running it again skips the jobs already done."""
import json, numpy as np
import nightlib, matrices as m
L = [l for l in open(nightlib.HERE + "/had24.txt") if l.strip()]
W = json.load(open(nightlib.HERE + "/wild24.json"))
jobs = [("had24.%d H c_4" % i, (lambda i=i: m.from_hex(L[i])), 4, 1000, 30) for i in W]
jobs += [("had24.%d Ht c_4" % i, (lambda i=i: m.from_hex(L[i]).T.copy()), 4, 1000, 30) for i in W]
nightlib.run("lynx", jobs)
