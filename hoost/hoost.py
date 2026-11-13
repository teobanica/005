"""hoost.py -- c_6 for the 3 families F_{8,5a}, F_{8,5b}, F_{8,5c}, testing the conjectured values
903 (little Schroder, for 5b and 5c) and 2974 (ternary Boolean cumulants, for 5a).
T_6 commutes with the cyclic rotation of the 6 legs, so the count is done separately in the 6 rotation
sectors, of dimension about 43,700 each, which keeps the memory low. Run:  python3 hoost.py
Log: hoost-output/hoost.log ; numbers: hoost-output/hoost.json ; stops by itself after 9 hours.
If interrupted, running it again skips the sectors already done. Memory: at most about 2.5 GB."""
import json, os, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import matrices as m, sectors

rng = np.random.default_rng(31); q = [m.E(x) for x in rng.random(5)]   # same parameters as in invariants.tex
jobs = [  # label, matrix, k, blocks for the 6 sectors (doubled if too small), conjectured value
    ("F_{8,5b} c_6", m.F85b(*q), 6, [240, 200, 210, 220, 210, 200], 903),
    ("F_{8,5a} c_6", m.F85a(*q), 6, [760, 600, 640, 660, 640, 600], 2974),
    ("F_{8,5c} c_6", m.F85c(*q), 6, [240, 200, 210, 220, 210, 200], 903),
]
TAG = "hoost"; HOURS = 9.0
out = os.path.join(HERE, TAG + "-output"); os.makedirs(out, exist_ok=True)
LOG = open(os.path.join(out, TAG + ".log"), "a"); JS = os.path.join(out, TAG + ".json")
try:
    res = json.load(open(JS))
except Exception:
    res = {}
T0 = time.time()
def say(s):
    s = "[%s | %5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - T0) / 60, s)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def watchdog():
    time.sleep(HOURS * 3600); say("%.0f hours reached, stopping. Results so far are in %s" % (HOURS, JS)); os._exit(0)
threading.Thread(target=watchdog, daemon=True).start()
say("start; numpy %s, python %s; %d jobs" % (np.__version__, sys.version.split()[0], len(jobs)))

for label, H, k, b, conj in jobs:
    r = res.setdefault(label, {"sectors": {}})
    if "c" in r:
        say("%s already done (c_%d = %d), skipping" % (label, k, r["c"])); continue
    say("=== %s (conjectured value %d) ===" % (label, conj))
    maps = sectors.sector_maps(H.shape[0], k)
    for s, E in enumerate(maps):
        if str(s) in r["sectors"]:
            say("  sector %d already done: %d" % (s, r["sectors"][str(s)]["c"])); continue
        bb = b[s]
        while True:
            t0 = time.time()
            say("  sector %d/%d: dimension %d, block %d" % (s, k, E.shape[1], bb))
            c, w, resid = sectors.count_sector(H, k, E, min(bb, E.shape[1]), progress=say, seed=s)
            if c <= bb - 10 or bb >= E.shape[1]:
                break
            say("  sector %d: %d found, too close to the block size, redoing with block %d" % (s, c, 2 * bb)); bb *= 2
        below = w[w <= 1 - 1e-4]; nxt = float(below[0]) if len(below) else 0.0
        r["sectors"][str(s)] = {"c": c, "next": nxt, "residual": resid, "block": bb, "seconds": round(time.time() - t0)}
        json.dump(res, open(JS, "w"), indent=1)
        say("  sector %d: %d fixed vectors   (next Ritz value %.4f, residual %.1e, %.0f s)" % (s, c, nxt, resid, time.time() - t0))
    r["c"] = sum(v["c"] for v in r["sectors"].values())
    json.dump(res, open(JS, "w"), indent=1)
    say("%s: c_%d = %d   (by sector %s; conjectured %d)%s" % (label, k, r["c"],
        [r["sectors"][str(s)]["c"] for s in range(k)], conj, "   <<< MATCH" if r["c"] == conj else "   <<< DIFFERENT"))
say("all jobs done.")
