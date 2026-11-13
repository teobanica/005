import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import cases, matrices as m
from phis import jw_basis, phi_spec
def contain(a,b):
    b=list(b); miss=[]
    for x in a:
        j=int(np.argmin(np.abs(np.array(b)-x)))
        if abs(b[j]-x)<1e-8: b.pop(j)
        else: miss.append(x)
    return miss
tests={"F_4^s, s generic": m.F4s(np.exp(0.7j)), "F_4 (Fourier)": m.fourier(4),
       "F_6 (Fourier)": m.fourier(6), "F_6(r,s) Dita random": m.F6rs(np.exp(0.4j),np.exp(1.9j)),
       "F_2 x F_3": np.kron(m.fourier(2),m.fourier(3))}
for lab,H in tests.items():
    N=H.shape[0]; ss=[2,3,4] if N<=6 else [2,3]
    sp={s:phi_spec(H,s,jw_basis(N,s)) for s in ss}
    print(lab, " e_s:", [int((sp[s]>1-1e-9).sum()) for s in ss])
    for s in ss[:-1]:
        mi=contain(sp[s],sp[s+1]); print("   Phi_%d -> Phi_%d: %d unmatched %s"%(s,s+1,len(mi),np.round(mi[:6],4)))
