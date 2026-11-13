import numpy as np, sys
from eigcheck import paley1
from sectors import Aop
for q in map(int,sys.argv[1:]):
    H=paley1(q); N=q+1; A=Aop(H)
    ev=np.linalg.eigvalsh(A)
    vals=[];mult=[]
    for e in np.sort(ev):
        if vals and abs(e-vals[-1])<1e-6*max(1,abs(e)): mult[-1]+=1
        else: vals.append(e); mult.append(1)
    print("q=%d: "%q+", ".join("%.4f[%d]"%(v,m) for v,m in zip(vals,mult)), flush=True)
