from lattice import *
for M,N in [(2,4),(4,2)]:
    var,gens,blocks,A,Dfull=setup(M,N)
    print((M,N),"vars",var,"gens",gens)
    for k in range(M):
        for e in range(N):
            print("  k",k,"e",e,[tuple(int(x) for x in Dfull[(k,e)][:,j]) for j in range(len(gens))])
