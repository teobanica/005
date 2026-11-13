"""cormier.py -- Paley matrices, Jones-Wenzl counts e_k at larger orders (8GB laptop, 5 hours).
If c_{k-1} = Cat_{k-1}, then c_k = Cat_k + e_k, so a chain e_2 = ... = e_k = 0 gives c_1..c_k = 1, 2, 5, 14, ...
Jobs, in this order:
  1. e_2, e_3 at N = 44, 48, 52   (Paley I q = 43, 47; Paley II q = 25)
  2. e_4 at N = 44                (Paley I q = 43; peak memory about 4 GB, about 20-30 min)
  3. e_2, e_3 at N = 100, 104, 108, 108   (Paley II q = 49; Paley I q = 103, 107; Paley II q = 53; up to 4 GB)
  4. e_4 at N = 48                (Paley I q = 47; peak memory about 6 GB, last on purpose)
Run:  python3 cormier.py      Log: cormier-output/cormier.log ; numbers: cormier-output/cormier.json
Everything should be done in about 2 hours. Stops by itself after 5 hours. If interrupted, running it again skips the jobs already done."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jwcount
import pmatrices as pm
P1 = lambda p, r=1: (lambda: pm.paley1(p, r))
P2 = lambda p, r=1: (lambda: pm.paley2(p, r))
jobs = []
for lab, b in [("Paley I, q = 43", P1(43)), ("Paley I, q = 47", P1(47)), ("Paley II, q = 25", P2(5, 2))]:
    jobs += [(lab, b, k, 2) for k in (2, 3)]
jobs += [("Paley I, q = 43", P1(43), 4, 2)]
for lab, b in [("Paley II, q = 49", P2(7, 2)), ("Paley I, q = 103", P1(103)), ("Paley I, q = 107", P1(107)),
               ("Paley II, q = 53", P2(53))]:
    jobs += [(lab, b, k, 2) for k in (2, 3)]
jobs += [("Paley I, q = 47", P1(47), 4, 2)]
jwcount.run("cormier", HERE, jobs, 5.0, "Jones-Wenzl counts for Paley matrices, orders 44 to 108.")
