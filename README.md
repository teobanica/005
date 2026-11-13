 # Higher invariants of complex Hadamard matrices

**T. Banica and C. Opus**, November 2026

Associated to any complex Hadamard matrix $H\in M_N(\mathbb T)$ is a quantum permutation group $G\subset S_N^+$, and its invariants $c_k=\dim(Fix(u^{\otimes k}))$ are the dimensions of the relative commutants of the subfactor associated to $H$. This paper computes these invariants well beyond the levels $k=2,3$ previously known, up to $k=9$ in the best cases, by using Jones-Wenzl and annular decompositions. Among the results:

- for the Diţă deformations $F_X\otimes_QF_Y$ of the Fourier matrices, the quantum group is computed for all values of the parameters;
- the order 8 families $F_{8,5}$ are studied in connection with the free wreath products $S_4^+\wr_*\mathbb Z_2$ and $H_2^+\wr_*\mathbb Z_2$;
- for the matrices of small order with $c_2=2$, such as the Karlsson ones at generic parameters, $c_k=Cat_k$ at all the levels computed, up to $k=9$ at order 6;
- if $G$ is finite, the eigenvalues of $NT_k$ are algebraic integers, and this gives infinite depth for the matrices coming from symmetric conference matrices, including all the Paley II matrices;
- the Paley I matrices have infinite depth for all the prime powers $q\equiv3$ mod 4 with $11\le q<10^5$, via character sums and exact certificates.

## Contents

- `005.pdf` — the paper (58 pages)
- `005.tex` — its LaTeX source (self-contained; compile with `pdflatex`, twice)
- the folders below — the Python programs used in the paper, with their output files next to them

| folder | content | paper |
|---|---|---|
| `basis/` | first sweeps through the basis: `stage0.py` to `stage46.py`, `level5.py`, `sun.py`, `scan.py`, `scan4.py`, `sweep_real.py`, `sweep_c2.py`, `bbformula.py`, `bb_moments.py`, results in `results/` | Sections 3, 6, 7, 8 |
| `tsukuba/` | quantum groups of the deformed Fourier matrices, for all parameters, Theorem 4.4 and its corollaries | Section 4 |
| `pauli.py`, `fibers.py`, `finite.py`, `magic.py`, `rklcheck.py`, and `free/`, `topgen/`, `f85a/` | the families $F_{8,5}$: Pauli models, inner copies, freeness tests, placement models | Section 5 |
| `real/` | real Diţă deformations of the Walsh matrix | Section 6.3 |
| `paley/` | Paley I and II: eigenvectors, and the integrality certificates of `trace_exact.py`, in integer arithmetic | Sections 6.5 to 6.7 |
| `norms/` | converged norms for the real matrices of orders 24 and 28 | Section 6.4 |
| `karlsson/` | $H_2$-reducible and glued matrices, Bloch sphere, order 8 block structures, order 10 | Sections 7, 8 |
| `checks/` | exact checks: $c_2$ of $H_6^1$, points of $F_{8,5b}$, $F_{8,5c}$ and their transposes | Sections 7, 8 |
| `spec/`, `phi/`, `jw/`, `wild/` | spectra of $T_k$, Jones-Wenzl and annular pieces | Section 9 |
| the folders named after felines and fighters | the long computations, run on 2 laptops, listed below | all |

| laptop program | computation | paper |
|---|---|---|
| `ocelot` | $c_5$ for $F_{8,5a}$ and $had_{16.4}$, $c_4$ for $had_{16.1}$, $c_6$ for $F_{8,5b}$ | Sections 3.4, 6.3 |
| `hoost` | $c_6$ for $F_{8,5a}$, $F_{8,5b}$, $F_{8,5c}$, by rotation sectors | Section 3.4 |
| `sandcat` | $e_5,e_6$ for $F_{8,5}$, converged norms at orders 7 to 12, and at level 5 for $P_{20}^1,P_{20}^2,P_{24}^1$ | Sections 3.4, 6.2, 9.1 |
| `margay` | independent check of $c_5=212$ for $had_{16.3}$ | Section 6.3 |
| `lynx` | $c_1,\ldots,c_4$ for the 45 classes of order 24 with $c_2=2$ | Section 6.4 |
| `fedor` | $c_1,\ldots,c_4$ for the 487 classes of order 28 | Section 6.4 |
| `bobcat`, `cormier` | Jones-Wenzl counts for the Paley matrices of orders 20 to 108 | Section 6.2 |
| `bones` | glued matrices of order 10 | Section 8.2 |
| `caracal`, `puma`, `gsp` | $c_6,c_7$ with noncrossing seeds | Section 8.3 |
| `crocop`, `serval` | norms of $\Phi_k$ | Section 9.1 |
| `jaguar`, `aerts`, `lennox` | Jones-Wenzl counts $e_6,\ldots,e_9$ | Section 9.1 |
| `sefo` | converged norms at order 6, up to level 9 | Section 9.1 |

## Running the code

Python 3 with `numpy` and `scipy`, and the GMP library `libgmp` for `paley/gmpconv.py`. Some programs import the modules of their own folder (`matrices.py`, `ck.py`, ...), so run each program from its folder, with `python3 name.py`, and a few exploratory scripts still contain, in their `sys.path` lines, the absolute paths of the machine where they were written. Most programs run in seconds or minutes, while the laptop programs run from several minutes to a full night each, as recorded in their log files. Each laptop folder has the program, the modules it uses, and its output folder `name-output/`.
