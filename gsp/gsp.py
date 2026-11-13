"""gsp.py -- wild checks beyond Temperley-Lieb, via the noncrossing seeds + random block check:
c_6 for the isolated matrices of order 8, for BN_9 and for P_7, then c_5 for the 3 real classes of order 20
(H and H^t). Run:  python3 gsp.py
Log: gsp-output/gsp.log ; numbers: gsp-output/gsp.json ; stops by itself after 7 hours.
If interrupted, running it again skips the jobs already done. Memory: at most about 3 GB (during the order 20 jobs)."""
import nightlib, matrices as m
L20 = m.HAD20
jobs = [
    ("I_{8,12} c_6", lambda: m.from_log(m.I812_L, 12), 6, 20),
    ("K_{8,30} c_6", lambda: m.from_log(m.K830_L, 30), 6, 20),
    ("L_{8,30} c_6", lambda: m.from_log(m.L830_L, 30), 6, 20),
    ("L_{8,30}^t c_6", lambda: m.from_log(m.L830t_L, 30), 6, 20),
    ("P_7 c_6", lambda: m.P7(1.0 + 0j), 6, 20),
    ("BN_9 c_6", lambda: m.from_log(m.BN9_L, 10), 6, 16),
    ("had_20.0 H c_5", lambda: m.from_hex(L20[0]), 5, 12),
    ("had_20.1 H c_5", lambda: m.from_hex(L20[1]), 5, 12),
    ("had_20.0 Ht c_5", lambda: m.from_hex(L20[0]).T.copy(), 5, 12),
    ("had_20.1 Ht c_5", lambda: m.from_hex(L20[1]).T.copy(), 5, 12),
    ("had_20.2 Ht c_5", lambda: m.from_hex(L20[2]).T.copy(), 5, 12),
]
nightlib.run_tl("gsp", jobs)
