import sys,time; sys.path.insert(0,'/home/claude/work/night'); sys.path.insert(0,'/home/claude/work/spec')
import numpy as np, matrices as m, hw
rng=np.random.default_rng(5)
mats=[("T6",m.T6()),("H6q",m.H6(np.exp(2j*np.pi*rng.random()))),("BN6",m.BN6(np.exp(2j*np.pi*rng.random())))]
for lab,H in mats:
    for k in (3,4,5,6):
        t=time.time(); w,res,nmv,leak=hw.top_phi(H,k)
        print(lab,"k=%d"%k, np.round(w,8), "res %.1e"%res.max(), nmv, "leak %.1e"%leak, round(time.time()-t,1),flush=True)
