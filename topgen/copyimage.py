# rho restricted to each copy of C(H2+): should have the Hopf image H2+ (c_k = 1,3,10,35)
exec(open('smashtest.py').read().split("print(\"c_k(theta_1)")[0].replace("K=int(sys.argv[1]) if len(sys.argv)>1 else 4","K=4"))
w2,Vec2=np.linalg.eigh(np.eye(8)-E0); T=Vec2[:,w2>0.5]
Th=np.zeros((2,4,4,4,4),complex)
for x in range(2):
    for ai in range(4):
        for ci in range(4):
            Ux=sum(P[blocks[x][ai],blocks[y][ci]] for y in range(2))
            Th[x,ai,ci]=T.conj().T@Ux@T
for x in range(2):
    R=np.zeros((4,4,20,20),complex)
    for ai in range(4):
        for bi in range(4):
            R[ai,bi,:4,:4]=U[4*x+ai,4*x+bi]
            R[ai,bi,4:,4:]=sum(np.kron(Th[x,ai,ci],Th[1-x,ci,bi]) for ci in range(4))
    print("copy",x,"c_k:",[ck_fast(R,k) for k in range(1,6)])
