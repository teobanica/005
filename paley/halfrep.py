import numpy as np, sys
from eigcheck import paley1
from sectors import Aop
for q in map(int,sys.argv[1:]):
    H=paley1(q); N=q+1; A=Aop(H)
    ev=np.sort(np.linalg.eigvalsh(A))
    vals=[];mult=[]
    for e in ev:
        if vals and abs(e-vals[-1])<1e-6*max(1,abs(e)): mult[-1]+=1
        else: vals.append(e); mult.append(1)
    rat=[(round(v),m) for v,m in zip(vals,mult) if abs(v-round(v))<1e-5 and round(v)!=0]
    print("q=%d: rational eigenvalues"%q, rat, " /64:", [r/64 for r,m in rat], flush=True)
