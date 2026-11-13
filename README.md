# Higher invariants of complex Hadamard matrices: programs

This archive has the Python programs used in the paper *Higher invariants of complex Hadamard matrices*, by T. Banica and C. Opus. The output files are included next to the programs that wrote them.

Everything uses `numpy` and `scipy`, and `dita/paley/gmpconv.py` uses the GMP library `libgmp`, through `ctypes`. Some programs import the modules of their own folder (`matrices.py`, `ck.py`, ...), so run each program from its folder, with `python3 name.py`. A few exploratory scripts still contain the absolute paths of the machine where they were written. In those, edit the `sys.path` lines before running.

## Basic modules

| module | role |
|---|---|
| `matrices.py` | all the matrices of the paper, with the conventions used there |
| `ck.py` | the numbers c_k: exact computation, seeded computation, probabilistic count with Chebyshev filter |
| `equiv.py` | equivalence tests, including McKay's graph reduction for real matrices |
| `dita_test.py` | search for candidate Dita block structures (see the note in its docstring) |
| `sectors.py`, `lwsector.py` | computations by rotation sectors, and lowest weight vectors |
| `jwcount.py` | Jones-Wenzl counts e_k, i.e. the multiplicity of the eigenvalue 1 of Phi_k |
| `phiprof.py` | Krylov norms of Phi_k |
| `gapcheck.py`, `gaprun.py` | converged Lanczos norms of Phi_k with the Paige bound, and LOBPCG multiplicities with locking |

Several folders keep their own copy of these modules, as they were at the time of the computation.

## Folders

| folder | content | paper |
|---|---|---|
| `basis/` | first sweeps through the basis: `stage0.py` to `stage46.py`, `level5.py`, `sun.py`, `scan.py`, `scan4.py`, `sweep_real.py`, `sweep_c2.py`, `bbformula.py`, `bb_moments.py`, results in `results/` | sections 3, 6, 7, 8 |
| `dita/tsukuba/` | quantum groups of the deformed Fourier matrices, for all parameters: brute force checks of Theorem 4.4 and its corollaries | section 4 |
| `dita/` (top), `dita/free/`, `dita/topgen/`, `dita/f85a/` | the families F_{8,5}: Pauli models, inner copies, freeness tests, placement models, F_{8,5a} | section 5 |
| `dita/real/` | real Dita deformations of the Walsh matrix | section 6.3 |
| `dita/paley/` | Paley I and II: eigenvectors, integrality certificates. Theorem 6.6 is proved by `trace_exact.py` (integer arithmetic only, with `gmpconv.py`), output `trace_exact-*.log` and `trace_exact-*.json`; the first sweep `trace_fast.py`, `trace_gf.py`, with rounded FFTs, is in `trace_small.log` and `trace_full.log` | sections 6.5 to 6.7 |
| `norms/` | converged norms for the real matrices of orders 24 and 28 (`orders.py`) | section 6.4 |
| `karlsson/` | H_2-reducible and glued matrices, Bloch sphere, order 8 block structures, order 10 | sections 7, 8 |
| `spec/`, `phi/`, `jw/` | spectra of T_k, Jones-Wenzl pieces | section 9 |
| `wild/` | annular pieces, and some exploratory tests | section 9 |
| `checks/` | exact checks: `c2exact.py` (c_2 of H_6^1), `f85bpoint.py`, `f85cpoint.py` (points of F_{8,5b}, F_{8,5c} and their transposes: c_2 and Dita block structures, exact; run with argument `t` for the transpose) | sections 7, 8 |
| `laptops/` | the long computations, run on 2 laptops (see below) | all |

## Laptop programs

Each folder in `laptops/` has the program, the modules it uses, and its output folder `name-output/` with `name.log` and `name.json` when available. These programs stop by themselves after a fixed time, and resume where they stopped when launched again.

| program | computation | section |
|---|---|---|
| `ocelot` | c_5 for F_{8,5a} and had_{16.4}, c_4 for had_{16.1}, c_6 for F_{8,5b} | 3.4, 6.3 |
| `hoost` | c_6 for F_{8,5a}, F_{8,5b}, F_{8,5c}, by rotation sectors | 3.4 |
| `sandcat` | e_5, e_6 for the families F_{8,5}, then converged norms at orders 7 to 24, and for Paley up to order 108, output in `sandcat-output` | 3.4, 6.2, 9.1 |
| `margay` | independent check of c_5 = 212 for had_{16.3} | 6.3 |
| `lynx` | c_1..c_4 for the 45 classes of order 24 with c_2 = 2 | 6.4 |
| `fedor` | c_1..c_3 for the 487 classes of order 28, then c_4 | 6.4 |
| `bobcat` | Jones-Wenzl counts for the Paley matrices of orders 20 to 36 | 6.2 |
| `cormier` | Jones-Wenzl counts for the Paley matrices of orders 44 to 108 | 6.2 |
| `bones` | glued matrices of order 10 | 8.2 |
| `caracal` | c_7 at order 6, with noncrossing seeds | 8.3 |
| `puma` | c_6 for the matrices of orders 6 to 10, c_5 for P_20^1 | 8.3 |
| `gsp` | c_6 for the isolated matrices of order 8, P_7^1, BN_9, and c_5 at order 20 | 8.3 |
| `crocop` | norms of Phi_k at the low levels | 9.1 |
| `serval` | norms of Phi_k at the next levels | 9.1 |
| `jaguar` | Jones-Wenzl counts e_6, e_7, e_8 | 9.1 |
| `aerts` | e_9 at order 6, e_7 for H_9 and BN_9 | 9.1 |
| `lennox` | e_8 at order 7, e_6 for P_12, e_7 for BN_10 | 9.1 |
| `sefo` | converged norms at order 6, up to level 9, output in `sefo-output/` | 9.1 |

