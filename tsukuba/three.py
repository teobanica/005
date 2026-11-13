from gen_fit import formula_t
from nongen import counts
import numpy as np, itertools
E=lambda t: np.exp(2j*np.pi*t)
g=0.1234567  # irrational-ish
def th_from(Qexp):  # Qexp: dict (j,d)->(power of gamma, torsion turns)
    th=np.zeros((3,3))
    for (j,d),(a,t) in Qexp.items(): th[j,d]=(a*g+t)%1
    return th
cases={"generic":{(1,1):(1,0),(1,2):(2,0.3),(2,1):(-1,0.1),(2,2):(3,0)},
       "(a) Q12=Q21=γ, Q11Q22=γ":{(1,1):(2,0),(1,2):(1,0),(2,1):(1,0),(2,2):(-1,0)},
       "(a) with torsion":{(1,1):(2,0.25),(1,2):(1,0),(2,1):(1,1/3),(2,2):(-1,0)},
       "(b) Q11=Q22=γ, Q12Q21=γ":{(1,1):(1,0),(1,2):(3,0),(2,1):(-2,0),(2,2):(1,0)}}
gens=[(i,c) for i in (1,2) for c in (1,2)]
for nm,qe in cases.items():
    th=th_from(qe)
    pts=[R for R in itertools.product(range(-3,4),repeat=4) if any(R) and formula_t(3,3,th,dict(zip(gens,R)),-1,1,1,1)]
    rank=np.linalg.matrix_rank(np.array(pts)) if pts else 0
    Q=[[E(th[j,d]) for d in range(3)] for j in range(3)]
    print(nm,": N_Q points in box",len(pts),"rank",rank," c_k:",counts(3,3,Q,4),flush=True)
