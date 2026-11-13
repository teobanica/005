import sys, numpy as np
sys.path.insert(0, "/home/claude/jw/jw"); sys.path.insert(0, "/home/claude/day")
import matrices as m, ck
from lwsector import lam_tools
def top_ritz(H,k,w,b,thr=0.9,deg=40,a=0.85,seed=0):
    N=H.shape[0]; n=N**k; P,rot,sector=lam_tools(N,k); Tb=ck.T_block_blas(H,k)
    T=lambda X: np.hstack([Tb(np.ascontiguousarray(X[:,j:j+16])) for j in range(0,X.shape[1],16)])
    A=lambda X: P(T(X),5)
    rng=np.random.default_rng(seed); X=sector(rng.standard_normal((n,b))+1j*rng.standard_normal((n,b)),w); P(X,60)
    X/=np.linalg.norm(X,axis=0)
    L=lambda Y: (lambda Z:(Z.__imul__(2/a),Z.__isub__(Y),Z)[-1])(A(Y))
    t0,t1=X,L(X)
    for j in range(2,deg+1):
        t2=L(t1); t2*=2; t2-=t0; t0=t1; t1=t2; sc=np.linalg.norm(t1,axis=0).max(); t0/=sc; t1/=sc
    P(t1,80); t1=sector(t1,w); P(t1,20)
    for _ in range(2):
        G=t1.conj().T@t1; ev,EV=np.linalg.eigh((G+G.conj().T)/2); keep=ev>ev.max()*1e-20; t1=t1@(EV[:,keep]/np.sqrt(ev[keep]))
    M=t1.conj().T@T(t1); e=np.sort(np.linalg.eigvalsh((M+M.conj().T)/2))[::-1]
    return e[e>thr]
if __name__=="__main__":
    q0=np.pi/6*np.array([0,2,1]); d=np.array([1.0,0,0])
    if __name__!="__main__": pass
    for k in [3,4,5]:
        for w in range(k):
            out=[]
            for eps in [0.0,0.02,0.01]:
                H=m.K6(*(q0+eps*d)); out.append(top_ritz(H,k,w,b=40 if k<5 else 48))
            f=len(out[0][out[0]>1-1e-8])
            if f==0: continue
            a1=(1-out[1][:f])/0.02**2; a2=(1-out[2][:f])/0.01**2
            print("level %d sector %d: f=%d  a_i (eps=.02): %s\n%28s(eps=.01): %s"%(k,w,f,np.round(np.sort(a1),3),"",np.round(np.sort(a2),3)),flush=True)
