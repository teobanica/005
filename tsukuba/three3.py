import numpy as np, itertools
from nongen import counts
from index8 import subgroup_size
E=lambda t: np.exp(2j*np.pi*t)
def index33(ints,L):   # ints[(j,d)] = a_{jd}, Q_{jd}=e(a/L)
    M=N=3
    th=np.zeros((3,3),int)
    for (j,d),a in ints.items(): th[(-j)%3,d]=a       # theta(j,d)=arg Q_{-j,d}
    def mul(u,v):
        w=np.zeros((3,3),int)
        for a in range(3):
            for b in range(3):
                if u[a,b]: w+=u[a,b]*np.roll(np.roll(v,a,0),b,1)
        return w
    xy=np.zeros((3,3),int); xy[1,1]=1; xy[1,0]=-1; xy[0,1]=-1; xy[0,0]=1   # (x-1)(y-1)
    base=mul(xy,th)
    gens=[]
    for i in (1,2):
        for c in (1,2):
            r=np.zeros((3,3),int); r[i,c]+=1; r[i,0]-=1; r[0,c]-=1; r[0,0]+=1
            gens.append([int(v)%L for v in mul(r,base).ravel()])
    return subgroup_size(gens,L,9)
ex={"Fourier, Q=1":({},1),
    "Q11=w":({(1,1):1},3),
    "Q11=Q22=w":({(1,1):1,(2,2):1},3),
    "Q12=Q21=w":({(1,2):1,(2,1):1},3),
    "Q_jd=w^(jd)":({(1,1):1,(1,2):2,(2,1):2,(2,2):1},3),
    "Q11=i":({(1,1):1},4),
    "Q11=e(1/6)":({(1,1):1},6)}
for nm,(ints,L) in ex.items():
    idx=index33(ints,L) if L>1 else 1
    Q=[[1,1,1]]+[[1]+[E(ints.get((j,d),0)/L) for d in (1,2)] for j in (1,2)]
    print(nm,": index",idx,"dim",9*idx," c_k",counts(3,3,Q,4),flush=True)
