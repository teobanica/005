from blocks import *
np.set_printoptions(precision=4, suppress=True, linewidth=150)
rng=np.random.default_rng(3); r=lambda: np.exp(2j*np.pi*rng.random())
def ev(w): return np.angle(np.linalg.eigvals(w))/(2*np.pi)
for name,H in [('F84',M.F84(r(),r(),r(),r())),('F82',M.F82(r(),r())),('K82',M.K82(r(),r())),('M81',M.M81(r()))]:
    print('=====',name)
    for rm,cm in h2_structures(H)[:3]:
        X,Y,W=normalize(to_blocks(H,rm,cm))
        print(rm,cm)
        for i in range(1,4):
            print('  ', [np.round(np.sort(ev(W[i][j])),4) for j in range(1,4)], ' det', [np.round(np.angle(np.linalg.det(W[i][j]))/(2*np.pi),4) for j in range(1,4)])
        flat=[W[i][j] for i in range(1,4) for j in range(1,4)]
        print('  commutators:', np.round([[np.abs(a@b-b@a).max() for b in flat] for a in flat],2).max())
