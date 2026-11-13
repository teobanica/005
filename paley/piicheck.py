import numpy as np, sys
from piiinv import R_matrix
from trace_fast import isprime
def two_squares(q):
    for a in range(1,int(q**.5)+1,2):
        b2=q-a*a; b=int(round(b2**.5))
        if b*b==b2: return a,b
def _main():
 bad=[]
 for q in range(5,int(sys.argv[1])):
     if q%4!=1 or not isprime(q): continue
     R=R_matrix(q); N=2*(q+1); a,b=two_squares(q)
     ev=np.sort(np.linalg.eigvals(R).real)
     alpha=32*((a*a-b*b)**2+1); beta=8*(q**3-5*q**2+7*q+5)
     pred=sorted([0]*14+[64*q]*4+[alpha]*2+[beta]*2+[N**3]*2)
     ok=np.allclose(ev,pred,atol=1e-6*N**3)
     if not ok: bad.append(q)
     print(q,(a,b),"alpha",alpha,"beta",beta,"match:",ok,flush=True)
 print("mismatches:",bad)

if __name__=="__main__": _main()
