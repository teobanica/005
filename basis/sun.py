"""Class 3 ("here comes the sun"): non-wild, non-Fourier matrices. Higher c_k.
F_{8,5a}, F_{8,5b}, F_{8,5c} at a random point: c_1..c_6 (H and H^t).
had_16.1 .. had_16.4: c_1..c_4.
Writes results/sun.json and results/sun.log."""
import json, sys, time
import numpy as np
import matrices as m
import ck

log = open("results/sun.log", "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

rng = np.random.default_rng(31)
which = sys.argv[1] if len(sys.argv) > 1 else "all"
cases = []
if which in ("all", "f8"):
    q = [m.E(x) for x in rng.random(5)]
    cases += [("F_{8,5a}", m.F85a(*q), 5), ("F_{8,5b}", m.F85b(*q), 5), ("F_{8,5c}", m.F85c(*q), 5)]
if which in ("all", "h16"):
    cases += [("had_16.%d" % k, m.from_hex(m.HAD16[k]), 4) for k in (3, 4, 2, 1)]

try:
    out = json.load(open("results/sun.json"))
except Exception:
    out = {}
for name, H, kmax in cases:
    for side, M in [("H", H), ("Ht", H.T.copy())]:
        if name.startswith("had") and side == "Ht" and name in ("had_16.3", "had_16.4"):
            pass
        t0 = [time.time()]
        def logk(k, r, name=name, side=side):
            out.setdefault(name, {}).setdefault(side, {})[str(k)] = {"c": r[0], "next": r[1], "seeded": r[2]}
            json.dump(out, open("results/sun.json", "w"), indent=1)
            say(f"{name:10s} {side:2s} k={k} c_k={r[0]:6d}  next eigenvalue {r[1]:.4f}  seeded {r[2]}  {time.time()-t0[0]:.0f}s")
            t0[0] = time.time()
        ck.ck_seeded(M, kmax, log=logk)
        if name.startswith("had") and side == "H" and name in ("had_16.0",):
            break
say("done " + which)
