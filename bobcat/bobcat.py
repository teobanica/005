"""bobcat.py -- Paley matrices: Jones-Wenzl counts e_k, pushing c_k = Catalan to higher levels and larger orders.
If c_{k-1} = Cat_{k-1}, then c_k = Cat_k + e_k (Theorem 6.2 of invariants.tex), so a full chain e_2 = ... = e_k = 0
gives c_1, ..., c_k = 1, 2, 5, 14, 42, ... Real matrices, so everything is done in real arithmetic (half the memory).
In order of priority:
  1. e_5 at N = 20 (Paley I q = 19, Paley II q = 9) and N = 24 (Paley I q = 23)          [lower levels known]
  2. e_2, e_3, e_4 at N = 32 (Paley I q = 31) and N = 36 (Paley II q = 17)                 [new matrices]
  3. e_5 at N = 28 (Paley I q = 27, Paley II q = 13)                                        [lower levels known]
  4. e_2, e_3, e_4 at N = 44, 48 (Paley I q = 43, 47) and N = 52 (Paley II q = 25)
  5. e_2, e_3, e_4 at N = 60 (Paley I q = 59, Paley II q = 29)
  6. bonus: e_7 at N = 12 (dimension 35.8 million, may run out of memory: then skipped)
Run:  python3 bobcat.py      Log: bobcat-output/bobcat.log ; numbers: bobcat-output/bobcat.json
Stops by itself after 7 hours. If interrupted, running it again skips the jobs already done. Memory: up to 4-5 GB."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jwcount
import pmatrices as pm
P1 = lambda p, r=1: (lambda: pm.paley1(p, r))
P2 = lambda p, r=1: (lambda: pm.paley2(p, r))
jobs = []
jobs += [("Paley I, q = 19", P1(19), 5, 2), ("Paley II, q = 9", P2(3, 2), 5, 2), ("Paley I, q = 23", P1(23), 5, 2)]
for lab, b in [("Paley I, q = 31", P1(31)), ("Paley II, q = 17", P2(17))]:
    jobs += [(lab, b, k, 2) for k in (2, 3, 4)]
jobs += [("Paley I, q = 27", P1(3, 3), 5, 2), ("Paley II, q = 13", P2(13), 5, 2)]
for lab, b in [("Paley I, q = 43", P1(43)), ("Paley I, q = 47", P1(47)), ("Paley II, q = 25", P2(5, 2)),
               ("Paley I, q = 59", P1(59)), ("Paley II, q = 29", P2(29))]:
    jobs += [(lab, b, k, 2) for k in (2, 3, 4)]
jobs += [("Paley I, q = 11", P1(11), 7, 2)]
jwcount.run("bobcat", HERE, jobs, 7.0, "Jones-Wenzl counts for Paley matrices, orders 12 to 60.")
