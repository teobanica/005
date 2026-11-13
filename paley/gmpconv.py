"""Exact integer convolutions by Kronecker substitution, the big integer products being done by GMP (through
ctypes; GMP is the arithmetic library used by most computer algebra systems). No floating point is involved."""
import ctypes, ctypes.util, numpy as np

_gmp = ctypes.CDLL(ctypes.util.find_library("gmp"))
class _mpz(ctypes.Structure):
    _fields_ = [("alloc", ctypes.c_int), ("size", ctypes.c_int), ("d", ctypes.c_void_p)]
_P = ctypes.POINTER(_mpz)
for name, args in [("__gmpz_init", [_P]), ("__gmpz_clear", [_P]), ("__gmpz_mul", [_P, _P, _P]),
                   ("__gmpz_import", [_P, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]),
                   ("__gmpz_export", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_int, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t, _P])]:
    getattr(_gmp, name).argtypes = args
_gmp.__gmpz_export.restype = ctypes.c_void_p

def _mul_words(a, b):
    """a, b: 1-d uint64 arrays (little-endian words); returns their product as a uint64 array."""
    x, y, z = _mpz(), _mpz(), _mpz()
    for t in (x, y, z): _gmp.__gmpz_init(ctypes.byref(t))
    a = np.ascontiguousarray(a, dtype=np.uint64); b = np.ascontiguousarray(b, dtype=np.uint64)
    _gmp.__gmpz_import(ctypes.byref(x), a.size, -1, 8, 0, 0, a.ctypes.data)
    _gmp.__gmpz_import(ctypes.byref(y), b.size, -1, 8, 0, 0, b.ctypes.data)
    _gmp.__gmpz_mul(ctypes.byref(z), ctypes.byref(x), ctypes.byref(y))
    out = np.zeros(a.size + b.size + 1, dtype=np.uint64); cnt = ctypes.c_size_t(0)
    _gmp.__gmpz_export(out.ctypes.data, ctypes.byref(cnt), -1, 8, 0, 0, ctypes.byref(z))
    for t in (x, y, z): _gmp.__gmpz_clear(ctypes.byref(t))
    return out

def conv_nonneg(a, b, bound):
    """exact linear convolution of nonnegative integer arrays, all output coefficients being < 2^64 and < bound."""
    assert bound < 2 ** 64 and min(a.min(), b.min()) >= 0
    return _mul_words(a.astype(np.uint64), b.astype(np.uint64))[:a.size + b.size - 1].astype(object)

def conv(a, b):
    """exact linear convolution of integer arrays (any signs), returned as a Python int array."""
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    ap, an, bp, bn = np.maximum(a, 0), np.maximum(-a, 0), np.maximum(b, 0), np.maximum(-b, 0)
    bound = int(max(ap.max(), an.max())) * int(max(bp.max(), bn.max())) * min(a.size, b.size) + 1
    r = np.zeros(a.size + b.size - 1, dtype=object)
    for s, u, v in ((1, ap, bp), (-1, ap, bn), (-1, an, bp), (1, an, bn)):
        if u.any() and v.any(): r = r + s * conv_nonneg(u, v, bound)
    return r

def cyclic_corr(f, g, n):
    """C(t) = sum_a f(a) g(a+t) on Z_n, exactly."""
    lin = conv(f[::-1], g)                        # lin[n-1+t] = sum_a f(a) g(a+t), t in (-n, n)
    out = np.zeros(n, dtype=object)
    for idx in range(lin.size): out[(idx - (n - 1)) % n] += lin[idx]
    return out

def cyclic_conv(f, g, n):
    lin = conv(f, g); out = np.zeros(n, dtype=object)
    for idx in range(lin.size): out[idx % n] += lin[idx]
    return out
