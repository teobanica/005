import numpy as np, itertools, sys
sys.path.insert(0,'/home/claude/jw/jw')
import matrices as M
def magic(H):
    N=H.shape[0]
    return np.array([[np.outer(H[i]*H[j].conj(), (H[i]*H[j].conj()).conj())/N for j in range(N)] for i in range(N)])
def find_outer(W, M_out):
    """partitions of rows into M_out blocks of equal size such that sum_a W[(i,a),(j,b)] is independent of b"""
    N=W.shape[0]; n=N//M_out; out=[]
    for blocks in set_partitions(list(range(N)), n):
        ok=True
        for I in blocks:
            for J in blocks:
                S=[sum(W[a,b] for a in I) for b in J]
                if max(np.abs(S[k]-S[0]).max() for k in range(len(S)))>1e-9: ok=False; break
            if not ok: break
        if ok: out.append(blocks)
    return out
def set_partitions(lst, n):
    if not lst: yield []; return
    f=lst[0]
    for rest in itertools.combinations(lst[1:], n-1):
        blk=[f]+list(rest); rem=[x for x in lst if x not in blk]
        for p in set_partitions(rem, n): yield [blk]+p
def Tk(U, k):
    """T_k[(i),(j)] = tr(U_{i1j1}...U_{ikjk}) / dim, for a magic matrix U[i][j] of d x d projections"""
    n=U.shape[0]; d=U.shape[2]; idx=list(itertools.product(range(n),repeat=k))
    T=np.zeros((len(idx),len(idx)),complex)
    for x,I in enumerate(idx):
        for y,J in enumerate(idx):
            P=np.eye(d)
            for s in range(k): P=P@U[I[s],J[s]]
            T[x,y]=np.trace(P)/d
    return T
def ck_model(U,k,tol=1e-7):
    ev=np.linalg.eigvals(Tk(U,k)); return int(np.sum(np.abs(ev-1)<tol))

def Tk_fast(U, k, batch=256):
    n=U.shape[0]; d=U.shape[2]
    Uop=U.transpose(0,2,1,3); m=n**k; T=np.zeros((m,m),complex)
    for c0 in range(0,m,batch):
        cols=np.arange(c0,min(m,c0+batch)); bsz=len(cols)
        B=np.zeros((m,bsz)); B[cols,np.arange(bsz)]=1; B=B.reshape((n,)*k+(bsz,))
        for e in range(d):
            Y=np.zeros((n,)*k+(d,bsz),complex); Y[...,e,:]=B
            for s in reversed(range(k)):
                Y=np.moveaxis(Y,s,0); Y=contract(Uop,Y); Y=np.moveaxis(Y,0,s)
            T[:,cols]+=Y[...,e,:].reshape(m,bsz)
    return T/d
def contract(Uop,Y):
    # Y shape (n, ..., d, m): apply operator on (leg0, d-axis)
    n=Uop.shape[0]; d=Uop.shape[1]
    sh=Y.shape; Yr=np.moveaxis(Y,-2,1)            # (n,d,...,m)
    Z=np.tensordot(Uop.reshape(n*d,n*d), Yr.reshape(n*d,-1), axes=(1,0)).reshape(Yr.shape)
    return np.moveaxis(Z,1,-2)
def ck_fast(U,k,tol=1e-7):
    ev=np.linalg.eigvals(Tk_fast(U,k)); return int(np.sum(np.abs(ev-1)<tol))
