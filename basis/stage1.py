"""Stage 1, layer 1: real matrices, c_1..c_4, one representative per (class, transpose class)."""
import json, time
import numpy as np
import matrices as m
import ck

log = open("results/stage1.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

cases = [
    ("P12 (unique class, H~H^t)", m.paley1(11)),
    ("had20.0 = P20^2 = W20 (H~H^t)", m.from_hex(m.HAD20[0])),
    ("had20.1 (H~H^t)", m.from_hex(m.HAD20[1])),
    ("had20.2 = P20^1 (H~H^t)", m.from_hex(m.HAD20[2])),
    ("P24^1 (H~H^t)", m.paley1(23)),
    ("P28^1 (H~H^t)", m.paley1(3, 3)),
    ("P28^2 = W28 (H~H^t)", m.paley2(13)),
    ("had16.0 = W16 (H~H^t)", m.from_hex(m.HAD16[0])),
    ("had16.1 (H~H^t)", m.from_hex(m.HAD16[1])),
    ("had16.2 (H~H^t)", m.from_hex(m.HAD16[2])),
    ("had16.3 (= had16.4^t)", m.from_hex(m.HAD16[3])),
    ("had16.4 (= had16.3^t)", m.from_hex(m.HAD16[4])),
]
out = []
for name, H in cases:
    N = H.shape[0]
    row = {"name": name, "N": N, "c": {}, "next": {}, "time": {}}
    kmax = 3 if name.startswith("had16.0") else 4
    t0 = [time.time()]
    def logk(k, r, row=row, name=name, N=N):
        c, below, seeds = r
        row["c"][k] = c; row["next"][k] = below; row["seeds"] = row.get("seeds", {}); row["seeds"][k] = seeds
        row["time"][k] = round(time.time() - t0[0], 1)
        say(f"{name:32s} N={N:2d} k={k} c_k={c:4d} (Cat={ck.catalan(k)})  next eigenvalue {below:.4f}  seeded {seeds}  {time.time()-t0[0]:.1f}s")
        t0[0] = time.time()
    ck.ck_seeded(H, kmax, log=logk)
    if kmax == 3:
        row["c"][4] = 4096
        say(f"{name:32s} N=16 k=4 c_k=4096 (Fourier, known, not recomputed)")
    out.append(row)
    json.dump(out, open("results/stage1.json", "w"), indent=1)
say("done")
