"""ocelot.py -- overnight job for laptop 1 (Dita side). Run:  python3 ocelot.py
Log: ocelot-output/ocelot.log ; numbers: ocelot-output/ocelot.json ; stops by itself after 7 hours.
If interrupted, running it again skips the jobs already done."""
import numpy as np
import nightlib, matrices as m
rng = np.random.default_rng(31); q = [m.E(x) for x in rng.random(5)]   # same parameters as in invariants.tex
jobs = [
    # label, builder, kmax, exact_max, block size
    ("F_{8,5a} c_5", lambda: m.F85a(*q), 5, 5000, 60),
    ("had_16.1 c_4", lambda: m.from_hex(m.HAD16[1]), 4, 5000, 60),
    ("had_16.4 c_5", lambda: m.from_hex(m.HAD16[4]), 5, 70000, 40),
    ("F_{8,5b} c_6", lambda: m.F85b(*q), 6, 5000, 40),
]
nightlib.run("ocelot", jobs)
