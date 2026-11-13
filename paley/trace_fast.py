import sys, numpy as np
from math import gcd
from trace_cert import mob
def isprime(p): return p>1 and all(p%r for r in range(2,int(p**.5)+1))
def primroot(q):
    fs=[p for p in range(2,q) if (q-1)%p==0 and isprime(p)]
    return next(x for x in range(2,q) if all(pow(x,(q-1)//p,q)!=1 for p in fs))
def run(q):
    N=q+1; n=(q-1)//2
    sq=np.zeros(q,bool); sq[(np.arange(1,q)**2)%q]=True
    h=np.where(sq,1,-1); h[0]=1
    a=np.arange(q)
    g=h[(-a)%q]*h[(1-a)%q]; gp=h[a]*h[(a-1)%q]
    # autocorrelations via FFT (exact after rounding)
    C=np.rint(np.fft.ifft(np.abs(np.fft.fft(g))**2).real).astype(np.int64)
    Cp=np.rint(np.fft.ifft(np.abs(np.fft.fft(gp))**2).real).astype(np.int64)
    hh=primroot(q)**2%q
    sqs=np.array([pow(hh,k,q) for k in range(n)])
    c=C[sqs]; cp=Cp[sqs]
    conv=np.rint(np.fft.ifft(np.fft.fft(c)*np.fft.fft(cp)).real).astype(np.int64)  # cyclic conv on Z_n
    out={}
    for d in [d for d in range(2,n+1) if n%d==0]:
        r=np.arange(n); gg=np.gcd(r,d)
        cd=np.zeros(n,np.int64)
        for e in range(1,d+1):
            if d%e==0 and mob(d//e): cd+=np.where(gg%e==0,mob(d//e)*e,0)
        out[d]=4*sum(int(x)*int(y) for x,y in zip(cd,conv) if x)
    T=sum(out.values())
    return N,n,out,T
if __name__=="__main__":
    lo,hi=int(sys.argv[1]),int(sys.argv[2]); bad=[]; cnt=0
    for q in range(lo,hi):
        if q%4!=3 or not isprime(q): continue
        N,n,out,T=run(q); cnt+=1
        good=[d for d,P in out.items() if P%(N*N)]
        if not good: bad.append(q)
        print(q,N,"T=%d"%T,"T mod N^2=%d"%(T%(N*N)),"orders certifying:",good,flush=True)
    print("checked",cnt,"primes; uncertified:",bad)
