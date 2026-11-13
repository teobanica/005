"""Run one case: python3 one.py NAME SIDE KMAX ; appends to results/sun.json and results/sun.log."""
import json, sys, time
import numpy as np
import matrices as m
import ck

name, side, kmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = np.random.default_rng(31)
q = [m.E(x) for x in rng.random(5)]
MATS = {"F_{8,5a}": lambda: m.F85a(*q), "F_{8,5b}": lambda: m.F85b(*q), "F_{8,5c}": lambda: m.F85c(*q)}
for k in range(5):
    MATS["had_16.%d" % k] = (lambda k=k: m.from_hex(m.HAD16[k]))
H = MATS[name]()
if side == "Ht":
    H = H.T.copy()
log = open("results/sun.log", "a")
try:
    out = json.load(open("results/sun.json"))
except Exception:
    out = {}
t0 = [time.time()]
def logk(k, r):
    out.setdefault(name, {}).setdefault(side, {})[str(k)] = {"c": r[0], "next": r[1], "seeded": r[2]}
    json.dump(out, open("results/sun.json", "w"), indent=1)
    s = f"{name:10s} {side:2s} k={k} c_k={r[0]:6d}  next eigenvalue {r[1]:.4f}  seeded {r[2]}  {time.time()-t0[0]:.0f}s"
    print(s, flush=True); log.write(s + "\n"); log.flush()
    t0[0] = time.time()
ck.ck_seeded_lm(H, kmax, log=logk)
