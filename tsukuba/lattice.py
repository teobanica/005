import numpy as np, itertools
def setup(M,N):
    var={(j,d):n for n,(j,d) in enumerate((j,d) for j in range(1,M) for d in range(1,N))}
    nv=len(var)
    def vec(j,d):
        v=np.zeros(nv,int); j%=M; d%=N
        if j and d: v[var[(j,d)]]=1
        return v
    gens=[(i,c) for i in range(1,M) for c in range(1,N)]
    def D(k,e,i,c):
        return vec(i,e-c)+vec(i-k,e)-vec(i,e)-vec(i-k,e-c)+vec(-k,e-c)-vec(-k,e)
    # A: R -> (D^k_e(R)-D^k_0(R)) stacked
    blocks=[(k,e) for k in range(M) for e in range(1,N)]
    A=np.array([np.concatenate([D(k,e,i,c)-D(k,0,i,c) for (k,e) in blocks]) for (i,c) in gens]).T
    Dfull={(k,e):np.array([D(k,e,i,c) for (i,c) in gens]).T for k in range(M) for e in range(N)}
    return var,gens,blocks,A,Dfull
def rank(X): return np.linalg.matrix_rank(X) if X.size else 0
def nq_dim(M,N,rels):
    var,gens,blocks,A,_=setup(M,N); nv=len(var)
    L=np.array(rels).T if rels else np.zeros((nv,0))
    Lbig=np.kron(np.eye(len(blocks)),L)
    # dim{R: A R in span(Lbig)} = dim ker of [A | -Lbig] projected... = rank(A)+rank(Lbig)-rank([A Lbig]) (A injective)
    return rank(A)+rank(Lbig)-rank(np.hstack([A,Lbig])) if rels else 0, rank(A), len(gens)
if __name__=="__main__":
    for M,N in [(2,2),(2,3),(3,2),(2,4),(4,2),(2,5),(3,3)]:
        var,gens,blocks,A,_=setup(M,N)
        print((M,N),"vars",len(var),"rank A",rank(A))
