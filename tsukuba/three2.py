from three import *
import sys; sys.path.insert(0,'/home/claude/dita')
from magic import magic, ck_fast
from nongen import dita_Q
tha=th_from(cases["(a) Q12=Q21=γ, Q11Q22=γ"]); tht=th_from(cases["(a) with torsion"])
pts=[R for R in itertools.product(range(-3,4),repeat=4) if any(R) and formula_t(3,3,tha,dict(zip(gens,R)),-1,1,1,1)]
print("component sample:",pts[:3])
print("12*R in N_Q (torsion case):",[formula_t(3,3,tht,dict(zip(gens,[12*x for x in R])),-1,1,1,1) for R in pts[:5]])
for nm in ["generic","(a) Q12=Q21=γ, Q11Q22=γ","(b) Q11=Q22=γ, Q12Q21=γ"]:
    th=th_from(cases[nm]); Q=[[E(th[j,d]) for d in range(3)] for j in range(3)]
    print(nm,"model c1..c3:",[ck_fast(magic(dita_Q(3,3,Q)),k) for k in (1,2,3)],flush=True)
