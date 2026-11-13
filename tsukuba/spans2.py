from lattice import *
M,N=2,4; var,gens,blocks,A,Dfull=setup(M,N)
for R in [(1,0,-1),(0,1,0),(1,1,-1),(1,-1,-1),(2,1,-2)]:
    R=np.array(R)
    S=np.array([Dfull[(k,e)]@R-Dfull[(k,0)]@R for k in range(M) for e in range(N)])
    print(R, "diag exps block1:",[list(Dfull[(1,e)]@R) for e in range(N)],"block0:",[list(Dfull[(0,e)]@R) for e in range(N)])
