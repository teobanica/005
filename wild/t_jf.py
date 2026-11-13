import sys, time, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m
from jordan_full import nc_partitions, forms, min_defect, tl_basis
from lwsector import lw_fixed
print("NC counts:", [len(list(nc_partitions(n))) for n in range(1,7)], " TL dims N=6:", [tl_basis(6,j).shape[1] for j in range(1,5)])
# gradient check
rng=np.random.default_rng(1)
H=m.F6rs(np.exp(0.4j),np.exp(1.9j))
V,_,_=lw_fixed(H,4,0,b=12); F=forms(V,6,4); f=V.shape[1]
best,fg=min_defect(F,f,starts=1)
x=rng.standard_normal(2*f); v,g=fg(x); e=1e-6; num=np.array([(fg(x+e*np.eye(2*f)[i])[0]-fg(x-e*np.eye(2*f)[i])[0])/(2*e) for i in range(2*f)])
print("gradient check:", np.abs(num-g).max())
for lab,HH,ks in [("Z_6",m.fourier(6),[3,4]),("2x3 Dita",H,[3,4,5])]:
    for k in ks:
        t=time.time(); res=[]
        for w in range(k):
            V,_,_=lw_fixed(HH,k,w,b=40 if lab=="Z_6" else 16)
            if V.shape[1]==0: res.append((w,0,None)); continue
            F=forms(V,6,k); b,_=min_defect(F,V.shape[1],starts=20); res.append((w,V.shape[1],b))
        print("%s level %d: (sector, dim, min defect) %s  [%.0f s]"%(lab,k,[(w,d,None if b is None else float('%.3g'%b)) for w,d,b in res],time.time()-t),flush=True)
