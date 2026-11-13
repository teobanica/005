"""Resumable runner for converged norms ||Phi_k||: log in <tag>-output/<tag>.log, numbers in <tag>-output/<tag>.json.
A job is (label, builder, k). If ||Phi_k|| < 1 with a small residual for all 2 <= k <= K, then e_2 = ... = e_K = 0,
and c_k = Cat_k for all k <= K, by Theorem jw of invariants.tex."""
import json, os, sys, threading, time
import numpy as np
import gapcheck


def run(tag, here, jobs, hours, doc):
    out = os.path.join(here, tag + "-output"); os.makedirs(out, exist_ok=True)
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
        time.sleep(hours * 3600)
        say("%.0f hours reached, stopping. Results so far are in %s" % (hours, JS)); os._exit(0)
    threading.Thread(target=watchdog, daemon=True).start()
    say("start; numpy %s, python %s; %d jobs. %s" % (np.__version__, sys.version.split()[0], len(jobs), doc))
    for job in jobs:
        if len(job) == 4:
            run_mult(job, res, JS, say); continue
        label, build, k = job
        key = "%s, k = %d" % (label, k)
        if key in res:
            say("%s already done (||Phi_%d|| = %.6f), skipping" % (label, k, res[key]["norm"])); continue
        H = np.array(build(), dtype=complex); N = H.shape[0]
        assert np.abs(np.abs(H) - 1).max() < 1e-9 and np.abs(H @ H.conj().T - N * np.eye(N)).max() < 1e-8, label
        say("=== %s, level %d (dimension %d) ===" % (label, k, N ** k))
        try:
            r = gapcheck.norm_phi(H, k, sweeps=6, say=say)
        except MemoryError:
            say("  out of memory, skipping this job"); continue
        r = {a: (float(b) if isinstance(b, (np.floating, float)) else b) for a, b in r.items()}
        r.update({"k": k, "N": N})
        res[key] = r; json.dump(res, open(JS, "w"), indent=1)
        say("  %s: ||Phi_%d|| = %.8f   (residual %.1e, %d Lanczos steps, leak %.1e, %d s)%s" % (
            label, k, r["norm"], r["residual"], r["steps"], r["leak"], r["seconds"],
            "" if r["norm"] < 1 - 1e-6 else "   <<< EIGENVALUE 1"))
    say("all jobs done.")


def run_mult(job, res, JS, say):
    """job = (label, builder, k, b): multiplicity e_k of the eigenvalue 1 of Phi_k, by a block method of size b."""
    label, build, k, b = job
    key = "%s, k = %d, multiplicity" % (label, k)
    if key in res:
        say("%s already done (e_%d = %d), skipping" % (label, k, res[key]["e"])); return
    H = np.array(build(), dtype=complex); N = H.shape[0]
    say("=== %s, level %d (dimension %d), multiplicity of the eigenvalue 1, block %d ===" % (label, k, N ** k, b))
    try:
        r = gapcheck.mult_block(H, k, b, say=say)
    except MemoryError:
        say("  out of memory, skipping this job"); return
    r.update({"k": k, "N": N, "block": b})
    res[key] = r; json.dump(res, open(JS, "w"), indent=1)
    say("  %s: e_%d = %d, next eigenvalue %.8f   (max residual %.1e, leak %.1e, %d s)%s" % (
        label, k, r["e"], r["next"] if r["next"] is not None else -1, r["residual"], r["leak"], r["seconds"],
        ""))
