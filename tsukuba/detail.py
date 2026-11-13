from lattice import *
M,N=2,4; var,gens,blocks,A,Dfull=setup(M,N)
R=np.array([1,-1,1])
for k in range(2):
    print("block",k,[list(Dfull[(k,e)]@R) for e in range(N)])
# multi-relation lattices: rank 2 and 3 tests
tests={"q1=q2=q3":[[1,-1,0],[0,1,-1]],"q2=q1^2,q3=q1^3":[[2,-1,0],[3,0,-1]],"q1=q3":[[1,0,-1]],
       "q2=q1q3 & q1=q3":[[1,-1,1],[1,0,-1]],"q1q3/q2 root (2v)":[[2,-2,2]]}
for nm,rels in tests.items(): print(nm,"dim N_Q (rational):",nq_dim(M,N,rels)[0])
