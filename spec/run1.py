import sys; sys.path.insert(0,'/home/claude/work/night'); sys.path.insert(0,'/home/claude/work/spec')
import numpy as np, matrices as m, spectro
rng=np.random.default_rng(5)
for lab,H in [("T6",m.T6()),("H6q",m.H6(np.exp(2j*np.pi*rng.random()))),("P7",m.P7(np.exp(2j*np.pi*rng.random())))]:
    print(lab, flush=True); spectro.test(H, 4 if H.shape[0]>6 else 5)
