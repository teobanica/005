# Real Dita deformations of the Walsh matrix W_16: F_X (x)_Q F_Y with X, Y elementary abelian 2-groups, Q = +-1.
# Find which of had_16.0..16.4 they are, and compute their quantum groups via Proposition nq (exact c_k by counting).
import sys, itertools, numpy as np
sys.path.insert(0,'/home/claude/dita/tsukuba'); sys.path.insert(0,'/home/claude/dita'); sys.path.insert(0,'/home/claude/jw/jw'); sys.path.insert(0,'/home/claude/v21/code-29/paper')
from abelian import Ab, shifts
from magic import magic, ck_fast
def walsh(a):
    F=np.array([[1.]]); 
    for _ in range(a): F=np.kron(F,np.array([[1,1],[1,-1]]))
    return F
def dita(a,b,Qm):
    FX,FY=walsh(a),walsh(b); M,N=2**a,2**b
    H=np.zeros((M*N,M*N))
    for i,x,j,y in itertools.product(range(M),range(N),range(M),range(N)):
        H[i*N+x,j*N+y]=Qm[i,y]*FX[i,j]*FY[x,y]
    return H
def counts(a,b,Qm,K):
    X,Y=Ab((2,)*a),Ab((2,)*b); M,N=X.n,Y.n
    # element order of Ab((2,)*a) is lexicographic in bits = same as Walsh index ordering (binary digits, big endian)
    Q={(X.els[i],Y.els[y]):complex(Qm[i,y]) for i in range(M) for y in range(N)}
    P=shifts(X,Y,Q); n=M*N
    gens=[]
    for (i,c),A in P.items():
        p=np.argmax(np.abs(A),axis=0); ph=A[p,np.arange(n)]
        gens.append((p,ph))
    def key(p,ph):
        ph=ph.copy()
        for k in range(M):
            ph[k*N:(k+1)*N]/=ph[k*N]
        return (tuple(p),tuple(np.round(ph.real).astype(int)),tuple(np.round(ph.imag).astype(int)))
    idp=np.arange(n); idph=np.ones(n,complex)
    states={key(idp,idph):(idp,idph,1)}
    ck=[]
    for k in range(1,K+1):
        new={}
        for (p,ph,cnt) in states.values():
            for (q,qh) in gens:
                np_=p[q]; nph=ph[q]*qh
                kk=key(np_,nph)
                if kk in new: new[kk]=(new[kk][0],new[kk][1],new[kk][2]+cnt)
                else: new[kk]=(np_,nph,cnt)
        states=new
        bs=[v[2] for kk,v in states.items() if np.array_equal(np.array(kk[0]),idp) and all(x==1 for x in kk[1]) and all(x==0 for x in kk[2])]
        ck.append(sum(bs)//M if sum(bs)%M==0 else sum(bs)/M)
    return ck, len(states)
if __name__=="__main__":
    sys.path.insert(0,"/home/claude/v21/code-29/paper"); import importlib; Mt=importlib.import_module("matrices")
    from equiv import real_equivalent_bt
    had=[Mt.from_hex(l) for l in Mt.HAD16]
    sig={}
    sig={0:[16,256],1:[7,58],2:[4,21],3:[3,11],4:[3,11]}   # from the paper's table
    print("had16 signatures c2,c3:",sig,flush=True)
    found={}
    for (a,b) in [(1,3),(2,2),(3,1)]:
        M,N=2**a,2**b
        free=[(i,y) for i in range(1,M) for y in range(1,N)]
        for bits in itertools.product([1,-1],repeat=len(free)):
            Qm=np.ones((M,N))
            for (i,y),s in zip(free,bits): Qm[i,y]=s
            s=counts(a,b,Qm,3)[0][1:]
            for k in range(5):
                if s==sig[k] and (k not in found):
                    H=dita(a,b,Qm)
                    ok=real_equivalent_bt(H,had[k])
                    if ok:
                        found[k]=(a,b,Qm.copy()); print("had16.%d = F_{Z2^%d} x_Q F_{Z2^%d}, Q =\n"%(k,a,b),Qm.astype(int),flush=True)
        if len(found)==5: break
    for k,(a,b,Qm) in sorted(found.items()):
        c,ns=counts(a,b,Qm,7)
        print("had16.%d: formula c1..c7 = %s, |Gamma/N_Q| = %d, dim C(G) = %d"%(k,c,ns,ns*2**a),flush=True)
