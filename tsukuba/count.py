import numpy as np, itertools, sys
def cp(M,N,p,inN):
    cnt=0
    for i in itertools.product(range(M),repeat=p):
        for d in itertools.product(range(N),repeat=p):
            w=np.zeros((M,N),int)
            for r in range(p):
                w[i[r],d[r]]+=1; w[i[r],d[r-1]]-=1
            if inN(w[1:]): cnt+=1
    return cnt/(M*N)
gen=lambda w: not w.any()
alt=np.array([1,-1,1,-1])
def rel(w):
    m=w[0]@alt/4
    return np.array_equal(w[0],round(m)*alt) if abs(m-round(m))<1e-9 else False
for p in range(1,int(sys.argv[1])+1):
    print(p,"generic",cp(2,4,p,gen),"q2=q1q3",cp(2,4,p,rel),flush=True)
