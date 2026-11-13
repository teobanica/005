"""Banica-Bichon moment formula for generic Q (Random walk questions..., Prop. 5.1):
c_p = 1/(|X||Y|) #{ i in X^p, d in Y^p : multiset{(i_r,d_r)} = multiset{(i_r,d_{r-1})}, d_0 = d_p }."""
import itertools
from collections import Counter


def bb(nx, ny, p):
    cnt = 0
    for i in itertools.product(range(nx), repeat=p):
        for d in itertools.product(range(ny), repeat=p):
            a = Counter(zip(i, d))
            b = Counter(zip(i, d[-1:] + d[:-1]))
            if a == b:
                cnt += 1
    assert cnt % (nx * ny) == 0
    return cnt // (nx * ny)


if __name__ == "__main__":
    for nx, ny in [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)]:
        pmax = 6 if nx * ny <= 6 else 5
        print(f"|X|={nx} |Y|={ny}:", [bb(nx, ny, p) for p in range(1, pmax + 1)], flush=True)
