"""bones.py -- glued complex Hadamard matrices of order 10 (section 6, "Glued matrices of order 10", of invariants.tex).

H has 2x2 blocks H_ij = A_i diag(K_ij, L_ij) B_j, with K = F_5 and L_ij = z^{sigma(i) tau(j)}, z = exp(2 pi i/5).
It is Hadamard iff the Bloch vectors of the A_i, B_j satisfy 16 equations (Proposition 6.1), solved here by local
minimization from random points, in the normal form X_0 = 1, X_i = F_2 D F_2 / 2, Y_j = D F_2, with the axis at
angle theta.

Stage 1: for 16 choices of (sigma, tau), find a solution with c_2 = 2, and compute c_3, c_4, the defect, the local
         dimension of the solution set, and e_5 (Jones-Wenzl count, c_5 = 42 + e_5 when c_4 = 14).
Stage 2: e_6 for the example of the paper (example.npy, sigma = id, tau = (34)) and for 3 more matrices (about 10 min each).
Stage 3: e_7 for the example of the paper (dimension 10^7, block of 4 vectors, about 1 to 2 hours, about 3 GB).
Stage 4: e_6 for the remaining matrices, as time allows.
Memory: at most about 3.5 GB (stage 3). Stops by itself after 7 hours.

Run:  python3 bones.py      Log: bones-output/bones.log ; numbers: bones-output/bones.json ; matrices: pair-*.npy
Each item is saved as soon as it is done, so running again resumes."""
import os, sys, json, time, threading
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ck, jwcount

OUT = os.path.join(HERE, "bones-output"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "bones.log"), "a"); JS = os.path.join(OUT, "bones.json")
try:
    res = json.load(open(JS))
except Exception:
    res = {}
T0 = time.time()
def say(s):
    s = "[%s | %5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - T0) / 60, s)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def save():
    json.dump(res, open(JS, "w"), indent=1)
def watchdog():
    time.sleep(7 * 3600); say("7 hours reached, stopping. Results so far are in " + JS); os._exit(0)
threading.Thread(target=watchdog, daemon=True).start()

q = 5; z = np.exp(2j * np.pi / q)
F = np.array([[z ** (i * j) for j in range(q)] for i in range(q)])
F2 = np.array([[1, 1], [1, -1]], dtype=complex)
sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1., -1])
D = lambda w: np.diag([1, w])

def rot(n, ang):
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
def nvec(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
def resid(v, A, th):
    ph = v[0]; al = v[1:q]; be = v[q:]; n = nvec(th, ph)
    a = [np.array([0, 0, 1.])] + [np.array([0, np.sin(t), np.cos(t)]) for t in al]
    b = [np.array([1., 0, 0])] + [np.array([np.cos(t), np.sin(t), 0]) for t in be]
    return np.array([a[i] @ rot(n, A[i, j]) @ b[j] for i in range(1, q) for j in range(1, q)])
def build(L, th, ph, al, be, sgn):
    n = nvec(th, ph); Lam = n[0] * sx + n[1] * sy + n[2] * sz
    w, U = np.linalg.eigh(Lam); U = U[:, ::-1]
    if sgn < 0: U = U[:, ::-1]
    W = lambda i, j: U @ np.diag([F[i, j], L[i, j]]) @ U.conj().T
    X = [np.eye(2)] + [0.5 * F2 @ D(np.exp(1j * t)) @ F2 for t in al]
    Y = [F2] + [D(np.exp(1j * t)) @ F2 for t in be]
    return np.block([[X[i] @ W(i, j) @ Y[j] for j in range(q)] for i in range(q)])
def defect(H):
    N = H.shape[0]; rows = []
    for i in range(N):
        for j in range(i + 1, N):
            c = np.zeros((N, N), dtype=complex)
            c[i, :] += H[i] * np.conj(H[j]); c[j, :] -= H[i] * np.conj(H[j])
            rows.append(c.real.ravel()); rows.append(c.imag.ravel())
    return int(N * N - np.linalg.matrix_rank(np.array(rows), 1e-9) - (2 * N - 1))
def c_k(H, k):
    if k <= 3:
        return int(ck.ck(H, k)[0])
    return int(ck.ck_deflated(H, k)[0])

def find(s, t, seed):
    """a solution with c_2 = 2 for the pair (sigma, tau), with its theta, parameters and local dimension"""
    L = F[np.ix_((0,) + s, (0,) + t)]; A = np.angle(L * F.conj())
    rng = np.random.default_rng(seed)
    for trial in range(60):
        th = rng.uniform(0.3, 1.3)
        r = least_squares(lambda v: resid(v, A, th), rng.uniform(0, 2 * np.pi, 2 * q - 1),
                          xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.abs(r.fun).max() > 1e-11: continue
        for sgn in (1, -1):
            for ca in (1, -1):
                H = build(L, th, r.x[0], ca * r.x[1:q], r.x[q:], sgn)
                if np.abs(np.abs(H) - 1).max() < 1e-9 and np.abs(H @ H.conj().T - 2 * q * np.eye(2 * q)).max() < 1e-9:
                    if c_k(H, 2) != 2: break
                    f = lambda w: resid(w[1:], A, w[0]); w0 = np.r_[th, r.x]; hh = 1e-6
                    J = np.array([(f(w0 + hh * e) - f(w0 - hh * e)) / (2 * hh) for e in np.eye(len(w0))]).T
                    dim = len(w0) - np.linalg.matrix_rank(J, 1e-6)
                    return H, th, r.x, int(dim)
            else:
                continue
            break
    return None

PAIRS = [((1, 2, 3, 4), (1, 2, 4, 3)),
         ((1, 3, 4, 2), (3, 1, 4, 2)), ((2, 4, 1, 3), (4, 1, 3, 2)), ((2, 4, 1, 3), (4, 1, 2, 3)), ((3, 1, 4, 2), (2, 1, 4, 3)),
         ((2, 4, 1, 3), (1, 3, 4, 2)), ((4, 3, 1, 2), (4, 1, 3, 2)), ((4, 3, 2, 1), (3, 1, 2, 4)), ((4, 3, 1, 2), (3, 2, 1, 4)),
         ((1, 2, 4, 3), (2, 3, 4, 1)), ((1, 4, 3, 2), (1, 2, 4, 3)), ((4, 1, 3, 2), (1, 4, 3, 2)), ((2, 1, 3, 4), (2, 4, 1, 3)),
         ((3, 1, 2, 4), (3, 4, 2, 1)), ((4, 3, 2, 1), (3, 2, 4, 1)), ((3, 1, 4, 2), (3, 2, 4, 1))]

say("start; numpy %s, python %s. Glued Hadamard matrices of order 10." % (np.__version__, sys.version.split()[0]))
H10 = np.load(os.path.join(HERE, "example.npy"))
say("example H10: |H| - 1 = %.1e, H H^* - 10 = %.1e" % (np.abs(np.abs(H10) - 1).max(), np.abs(H10 @ H10.conj().T - 10 * np.eye(10)).max()))
mats = {"example": H10}

# stage 1
for idx, (s, t) in enumerate(PAIRS):
    key = "pair %d: sigma=%s tau=%s" % (idx, s, t); npy = os.path.join(OUT, "pair-%s.npy" % chr(97 + idx))
    if key in res and os.path.exists(npy):
        mats[idx] = np.load(npy); say("%s already done" % key); continue
    tt = time.time(); out = find(s, t, idx)
    if out is None:
        res[key] = {"found": False}; save(); say("%s: no solution with c_2 = 2 found" % key); continue
    H, th, x, dim = out; np.save(npy, H); mats[idx] = H
    c = [1, 2, c_k(H, 3), c_k(H, 4)]
    e5 = jwcount.count_ek(H, 5, 8)
    c5 = 42 + e5["e"] if c[3] == 14 else None
    res[key] = {"found": True, "theta": th, "params": list(x), "local_dim": dim, "defect": defect(H),
                "c1..c4": c, "e5": e5["e"], "c5": c5, "e5_ritz": e5["ritz"][:3], "seconds": round(time.time() - tt)}
    save()
    say("%s: theta %.3f, local dim %d, defect %d, c = %s, e5 = %d (top Ritz %.3f), %.0f s"
        % (key, th, dim, res[key]["defect"], c + [c5], e5["e"], e5["ritz"][0], time.time() - tt))

def do_e(name, H, k, b):
    key = "%s: e_%d" % (name, k)
    if key in res:
        say("%s already done: %d" % (key, res[key]["e"])); return
    tt = time.time(); say("%s: start (dimension %d, block %d)" % (key, 10 ** k, b))
    r = jwcount.count_ek(H, k, b, say=say)
    r["seconds"] = round(time.time() - tt); res[key] = r; save()
    say("%s = %d, top Ritz %s, residual %.1e, leak %.1e, %.0f s" % (key, r["e"], np.round(r["ritz"][:3], 4).tolist(), r["residual"], r["leak"], time.time() - tt))

# stage 2
do_e("example", H10, 6, 8)
for idx in [1, 2, 3]:
    if idx in mats: do_e("pair %d" % idx, mats[idx], 6, 8)
# stage 3
do_e("example", H10, 7, 4)
# stage 4
for idx in range(4, len(PAIRS)):
    if idx in mats: do_e("pair %d" % idx, mats[idx], 6, 8)
say("all done.")
