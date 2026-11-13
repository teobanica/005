"""Common runner for the overnight jobs. Each job: (label, builder, kmax, exact_max, b)."""
import json, os, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ck


def run(tag, jobs, hours=7.0):
    out = os.path.join(HERE, tag + "-output"); os.makedirs(out, exist_ok=True)
    LOG = open(os.path.join(out, tag + ".log"), "a")
    JS = os.path.join(out, tag + ".json")
    try:
        res = json.load(open(JS))
    except Exception:
        res = {}
    T0 = time.time()

    def say(s):
        s = "[%s | %5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - T0) / 60, s)
        print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

    def watchdog():
        time.sleep(hours * 3600)
        say("%.0f hours reached, stopping. Results so far are in %s-output/%s.json" % (hours, tag, tag))
        os._exit(0)
    threading.Thread(target=watchdog, daemon=True).start()
    say("start; numpy %s, python %s; %d jobs" % (np.__version__, sys.version.split()[0], len(jobs)))
    for label, build, kmax, exact_max, b in jobs:
        if label in res and res[label].get("done"):
            say("%s already done, skipping" % label); continue
        H = build()
        say("=== %s : c_1..c_%d ===" % (label, kmax))
        t0 = [time.time()]

        def logk(k, r):
            res.setdefault(label, {})[str(k)] = {"c": int(r[0]), "next": float(r[1]), "seeded": int(r[2])}
            json.dump(res, open(JS, "w"), indent=1)
            say("%s  c_%d = %d   (next %.4f, %d from seeds, %.0f s)" % (label, k, r[0], r[1], r[2], time.time() - t0[0]))
            t0[0] = time.time()
        try:
            ck.ck_approx(H, kmax, log=logk, progress=say, b=b, exact_max=exact_max)
            res[label]["done"] = True
            json.dump(res, open(JS, "w"), indent=1)
            say("%s done: c = %s" % (label, [res[label][str(k)]["c"] for k in range(1, kmax + 1)]))
        except MemoryError:
            say("%s: out of memory, skipping to the next job" % label)
    say("all jobs done.")


def run_tl(tag, jobs, hours=7.0):
    """jobs: (label, builder, k, block). Computes c_k with the noncrossing seeds + random block check."""
    out = os.path.join(HERE, tag + "-output"); os.makedirs(out, exist_ok=True)
    LOG = open(os.path.join(out, tag + ".log"), "a"); JS = os.path.join(out, tag + ".json")
    try:
        res = json.load(open(JS))
    except Exception:
        res = {}
    T0 = time.time()
    def say(s):
        s = "[%s | %5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - T0) / 60, s)
        print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    def watchdog():
        time.sleep(hours * 3600); say("%.0f hours reached, stopping." % hours); os._exit(0)
    threading.Thread(target=watchdog, daemon=True).start()
    say("start; numpy %s, python %s; %d jobs" % (np.__version__, sys.version.split()[0], len(jobs)))
    for label, build, k, b in jobs:
        if label in res:
            say("%s already done, skipping" % label); continue
        t0 = time.time(); say("=== %s ===" % label)
        c, cat, nxt = ck.ck_tl(build(), k, b=b, progress=say)
        res[label] = {"k": k, "c": int(c), "catalan": int(cat), "next": nxt}
        json.dump(res, open(JS, "w"), indent=1)
        say("%s: c_%d = %d   (Catalan %d, next Ritz value %.4f, %.0f s)%s" % (label, k, c, cat, nxt, time.time() - t0,
            "" if c == cat else "   <<< BEYOND TEMPERLEY-LIEB"))
    say("all jobs done.")
