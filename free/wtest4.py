import sys; sys.argv=['x']
exec(open('wtest.py').read().split("for nm,W in tests.items():")[0])
for nm in ["W=1","W diagonal in Pauli basis","W random"]:
    print(nm,"c4",ck_fast(blk(Px,adj(tests[nm],Py)),4),flush=True)
