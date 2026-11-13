from lattice import *
for M,N in [(2,3),(3,2)]:
    var,gens,blocks,A,Dfull=setup(M,N)
    print((M,N),"vars",var,"gens",gens)
    for k in range(M):
        for e in range(N):
            print("  k",k,"e",e," D(t_g) per gen:",[list(Dfull[(k,e)][:,j]) for j in range(len(gens))])
