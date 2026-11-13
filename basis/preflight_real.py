"""Pre-flight for the real matrices: H vs H^t, and identification of the named constructions."""
import json, time
import numpy as np
import matrices as m
import equiv

log = open("results/preflight_real.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


def verified(H, K, mapping):
    rows, cols = equiv.apply_mapping(H, mapping)
    H = np.real(H).round().astype(int); K = np.real(K).round().astype(int)
    for i, (j, d) in rows.items():
        for a, (b, e) in cols.items():
            if K[j, b] != d * e * H[i, a]:
                return False
    return len(rows) == H.shape[0] and len(cols) == H.shape[0]


def test(name1, H, name2, K):
    t = time.time()
    pr = (equiv.profile4(H) == equiv.profile4(K)) and (equiv.profile4(H.T) == equiv.profile4(K.T))
    if not pr:
        say(f"{name1} vs {name2}: NOT equivalent (4-profiles differ)  {time.time()-t:.1f}s")
        return False
    try:
        ok, info = equiv.real_equivalent_bt(H, K, max_nodes=20_000_000)
    except RuntimeError:
        say(f"{name1} vs {name2}: UNDECIDED (node limit)  {time.time()-t:.1f}s")
        return None
    if ok:
        v = equiv.check_bt(H, K, info)
        say(f"{name1} ~ {name2}: EQUIVALENT (mapping verified: {v})  {time.time()-t:.1f}s")
        return True
    say(f"{name1} vs {name2}: NOT equivalent (exhaustive search)  {time.time()-t:.1f}s")
    return False


real = {
    "P12^1": m.paley1(11), "P12^2": m.paley2(5), "W12": m.williamson(3),
    **{f"had16.{k}": m.from_hex(l) for k, l in enumerate(m.HAD16)},
    **{f"had20.{k}": m.from_hex(l) for k, l in enumerate(m.HAD20)},
    "P20^1": m.paley1(19), "P20^2": m.paley2(3, 2), "W20": m.williamson(5),
    "P24^1": m.paley1(23),
    "P28^1": m.paley1(3, 3), "P28^2": m.paley2(13), "W28": m.williamson(7),
}
res = {}
say("== H versus H^t")
for name, H in real.items():
    res[name] = test(name, H, name + "^t", H.T)

say("\n== named constructions versus each other")
test("P12^1", real["P12^1"], "P12^2", real["P12^2"])
test("P12^1", real["P12^1"], "W12", real["W12"])
for nm in ["P20^1", "P20^2", "W20"]:
    for k in range(3):
        if test(nm, real[nm], f"had20.{k}", real[f"had20.{k}"]):
            break
test("P28^1", real["P28^1"], "P28^2", real["P28^2"])
test("P28^1", real["P28^1"], "W28", real["W28"])
test("P28^2", real["P28^2"], "W28", real["W28"])
test("had16.3^t", real["had16.3"].T, "had16.4", real["had16.4"])
json.dump(res, open("results/preflight_real.json", "w"), indent=1)
say("done")
