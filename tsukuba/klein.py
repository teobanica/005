from abelian import *
X,Y=Ab((2,)),Ab((2,2)); a,b,c=(1,0),(0,1),(1,1); g=0.1234567; h=0.2718281
def test(ph):   # ph: dict d->phase for row j=1
    phase={(j,d):0.0 for j in X.els for d in Y.els}
    for d,v in ph.items(): phase[((1,),d)]=v
    Q={k:E(v) for k,v in phase.items()}
    th={(j,d):phase[(X.neg(j),d)] for j in X.els for d in Y.els}
    gens=[(i,cc) for i in X.els[1:] for cc in Y.els[1:]]
    pts=[R for R in itertools.product(range(-3,4),repeat=3) if any(R) and formula(X,Y,th,dict(zip(gens,R)))]
    return np.linalg.matrix_rank(np.array(pts)) if pts else 0, pts[:2]
print("generic        :",test({a:g,b:h,c:0.377}))
print("q_a=q_b q_c    :",test({a:g+h,b:g,c:h}))
print("q_b=q_a q_c    :",test({a:g,b:g+h,c:h}))
print("q_a=q_b q_c, q_b=q_a q_c (=> q_c^2=1):",test({a:g+0.5,b:g,c:0.5}))
