import sys; sys.path.insert(0,'/home/claude/work/night'); sys.path.insert(0,'/home/claude/work/spec')
import numpy as np, matrices as m, spectro
from run2 import phis, incl
rng=np.random.default_rng(31); q=[m.E(x) for x in rng.random(5)]
for lab,H,K in [("F4s",m.F4s(np.exp(0.7j)),6),("F85b",m.F85b(*q),4),("P7",m.P7(np.exp(2j*np.pi*0.123)),4),("F85a",m.F85a(*q),4)]:
    Phi=phis(H,K)
    print(lab, [ (s, incl(Phi[s],Phi[s+1])) for s in range(2,K)], "tops",[round(float(Phi[s].max()),6) for s in range(2,K+1)], "ones",[int(np.sum(np.abs(Phi[s]-1)<1e-9)) for s in range(2,K+1)],flush=True)
