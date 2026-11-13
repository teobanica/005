# Paley I certificate over GF(p^r), q = p^r = 3 mod 4. Elements = base-p integers (digit i = coeff of x^i).
import sys, itertools, numpy as np
from trace_fast import mob
def primitive_poly(p,r):
    q=p**r
    for coeffs in itertools.product(range(p),repeat=r):   # monic x^r + sum c_i x^i
        if coeffs[0]==0: continue
        cur=[0]*r; cur[0]=1; seen=0
        for k in range(1,q):
            # multiply cur by x
            top=cur[-1]; cur=[0]+cur[:-1]
            cur=[(cur[i]-top*coeffs[i])%p for i in range(r)]
            if cur==[1]+[0]*(r-1): break
        if k==q-1: return coeffs
def field(p,r):
    q=p**r; co=primitive_poly(p,r)
    w=p**np.arange(r)
    exp=np.zeros(q-1,np.int64); cur=[1]+[0]*(r-1)
    for k in range(q-1):
        exp[k]=int(np.dot(cur,w))
        top=cur[-1]; cur=[0]+cur[:-1]; cur=[(cur[i]-top*co[i])%p for i in range(r)]
    log=np.full(q,-1,np.int64); log[exp]=np.arange(q-1)
    digits=np.array([[ (e//p**i)%p for i in range(r)] for e in range(q)])
    return q,exp,log,digits,w
def run(p,r):
    q,exp,log,D,w=field(p,r); n=(q-1)//2; N=q+1
    def idx(dig): return (dig%p)@w
    neg=idx(-D); one=np.zeros(r,int); one[0]=1
    oneminus=idx(one[None,:]-D); minusone=idx(D-one[None,:])
    h=np.where(log%2==0,1,-1); h[0]=1          # log[0]=-1 -> fix
    h[0]=1
    g=h[neg]*h[oneminus]; gp=h[np.arange(q)]*h[minusone]
    shp=(p,)*r
    def autoc(f):
        F=np.fft.fftn(f.reshape(shp,order='F'))
        return np.rint(np.fft.ifftn(np.abs(F)**2).real).astype(np.int64).reshape(-1,order='F')
    C=autoc(g); Cp=autoc(gp)
    # autocorrelation index: C[t] = sum_a g(a) g(a+t) ; fftn convention gives sum_a f(a+t)f(a) -> same by symmetry check
    sqs=exp[0::2]
    c=C[sqs]; cp=Cp[sqs]
    conv=np.rint(np.fft.ifft(np.fft.fft(c)*np.fft.fft(cp)).real).astype(np.int64)
    out={}
    for d in [d for d in range(2,n+1) if n%d==0]:
        gg=np.gcd(np.arange(n),d); cd=np.zeros(n,np.int64)
        for e in range(1,d+1):
            if d%e==0 and mob(d//e): cd+=np.where(gg%e==0,mob(d//e)*e,0)
        out[d]=4*sum(int(x)*int(y) for x,y in zip(cd,conv) if x)
    return q,N,out
for arg in sys.argv[1:]:
    p,r=map(int,arg.split('^'))
    q,N,out=run(p,r)
    good=[d for d,P in out.items() if P%(N*N)]
    print("q=%d^%d=%d N=%d"%(p,r,q,N), {d:P for d,P in out.items()} if q<100 else "", "certifying orders:",good,flush=True)
