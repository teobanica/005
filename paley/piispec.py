import numpy as np, sys
sys.path.insert(0,'/home/claude/kar/day6/bobcat')
from pmatrices import paley2
from sectors import Aop
for args in [(5,),(3,2),(13,),(17,)]:
    H=np.array(paley2(*args)).real.astype(int); N=H.shape[0]; q=N//2-1
    A=Aop(H); ev=np.linalg.eigvalsh(A)
    vals=[];mult=[]
    for e in np.sort(ev):
        if vals and abs(e-vals[-1])<1e-6*max(1,abs(e)): mult[-1]+=1
        else: vals.append(e); mult.append(1)
    tgt=16*q/(q+1)**2*N**2
    print("Paley II q=%d N=%d target %.4f:"%(q,N,tgt), ", ".join("%.4f[%d]"%(v,m) for v,m in zip(vals,mult)), flush=True)
