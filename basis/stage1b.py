"""Stage 1, layer 1: real matrices, c_1..c_4, one representative per (class, transpose class)."""
import json, time
import numpy as np
import matrices as m
import ck

log = open("results/stage1b.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

cases = [
    ("had16.1 (H~H^t)", m.from_hex(m.HAD16[1])),
    ("had16.2 (H~H^t)", m.from_hex(m.HAD16[2])),
    ("had16.3 (= had16.4^t)", m.from_hex(m.HAD16[3])),
    ("had16.4 (= had16.3^t)", m.from_hex(m.HAD16[4])),
]
out = []
for name, H in cases:
    N = H.shape[0]
    row = {"name": name, "N": N, "c": {}, "next": {}, "time": {}}
    kmax = 3
    t0 = [time.time()]
    def logk(k, r, row=row, name=name, N=N):
        c, below, seeds = r
        row["c"][k] = c; row["next"][k] = below; row["seeds"] = row.get("seeds", {}); row["seeds"][k] = seeds
        row["time"][k] = round(time.time() - t0[0], 1)
        say(f"{name:32s} N={N:2d} k={k} c_k={c:4d} (Cat={ck.catalan(k)})  next eigenvalue {below:.4f}  seeded {seeds}  {time.time()-t0[0]:.1f}s")
        t0[0] = time.time()
    ck.ck_seeded(H, kmax, log=logk)

    out.append(row)
    json.dump(out, open("results/stage1b.json", "w"), indent=1)
say("done")
