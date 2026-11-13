"""margay.py -- independent check of c_5 = 212 for had_16.3. Run:  python3 margay.py
1) recomputes c_1..c_5 with a new random seed; 2) Lanczos for the top eigenvalues of T_5
on the complement of the fixed vectors found: if they are clearly below 1, then c_5 is exactly that number.
Log: margay-output/margay.log ; numbers: margay-output/margay.json ; stops by itself after 7 hours."""
import json, os, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ck, matrices as m
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = os.path.join(HERE, "margay-output"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "margay.log"), "a"); JS = os.path.join(OUT, "margay.json")
T0 = time.time(); res = {}
def say(s):
    s = "[%s | %5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - T0) / 60, s)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def watchdog():
    time.sleep(7 * 3600); say("7 hours reached, stopping."); os._exit(0)
threading.Thread(target=watchdog, daemon=True).start()
H = m.from_hex(m.HAD16[3])
say("start: had_16.3, levels 1..%d, new random seed" % K)
def logk(k, r):
    res[str(k)] = {"c": int(r[0]), "seeded": int(r[2])}
    json.dump(res, open(JS, "w"), indent=1)
    say("c_%d = %d   (%d from seeds)" % (k, r[0], r[2]))
bases = ck.ck_approx(H, K, log=logk, progress=say, b=40, seed=7, exact_max=70000 if K == 5 else 300)
U = bases[K]
say("certificate: Lanczos on the complement of the %d fixed vectors found" % U.shape[1])
vals, steps = ck.certify_complement(H, K, U, progress=say)
res["certificate"] = {"c": int(U.shape[1]), "top_eigenvalues_on_complement": [float(v) for v in vals], "lanczos_steps": steps}
json.dump(res, open(JS, "w"), indent=1)
say("top eigenvalues of T_%d on the complement: %s  (%d steps)" % (K, ", ".join("%.6f" % v for v in vals), steps))
say("CONCLUSION: c_%d = %d" % (K, U.shape[1]) if vals[0] < 0.999 else "WARNING: eigenvalue near 1 on the complement, c_%d > %d" % (K, U.shape[1]))
