# B-invariant sector of A=N^3T_2 for Paley II, q prime = 1 mod 4. Orbit-sum matrix R (24x24).
import numpy as np, sys
def paley2p(q):
    sq=np.zeros(q,bool); sq[(np.arange(1,q)**2)%q]=True
    chi=np.where(sq,1,-1); chi[0]=0
    m=q+1; C=np.zeros((m,m),int); C[0,1:]=1; C[1:,0]=1
    a=np.arange(q); C[1:,1:]=chi[(a[None,:]-a[:,None])%q]
    F=np.array([[1,1],[1,-1]]); G=np.array([[1,-1],[-1,-1]])
    return np.kron(C,F)+np.kron(np.eye(m,dtype=int),G), sq
def orbit_label(q,sq,i,j):
    x,a=divmod(i,2); y,b=divmod(j,2)
    if x==0 and y==0: t=0
    elif x==0: t=1
    elif y==0: t=2
    elif x==y: t=3
    else: t=4 if sq[(y-x)%q] else 5
    return t*4+a*2+b
def R_matrix(q):
    H,sq=paley2p(q); N=H.shape[0]
    lab=np.array([[orbit_label(q,sq,i,j) for j in range(N)] for i in range(N)]).ravel()
    reps={}
    for k,l in enumerate(lab):
        if l not in reps: reps[l]=k
    Rw=(H[:,None,:]*H[None,:,:]).reshape(N*N,N)
    R=np.zeros((24,24),np.int64)
    for l,k in reps.items():
        Srow=Rw@Rw[k]; Arow=Srow.astype(np.int64)**2
        R[l]=np.bincount(lab,weights=Arow,minlength=24).astype(np.int64)
    return R
if __name__=="__main__":
    for q in map(int,sys.argv[1:]):
        R=R_matrix(q); N=2*(q+1)
        ev=np.sort(np.linalg.eigvals(R).real)
        print("q=%d N=%d eigenvalues of B-invariant sector:"%(q,N), np.round(ev,3), flush=True)
