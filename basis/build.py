"""Generate basis.tex: the horizontal basis, every matrix displayed with its convention."""
import itertools
import numpy as np
import matrices as m

out = []
W = out.append

# ------------------------------------------------------------ helpers
def signs(M, size=r"\scriptsize", sep=1.6, name=""):
    M = np.real(M).round().astype(int)
    n = M.shape[0]
    rows = [" & ".join("+" if x > 0 else "-" for x in r) for r in M]
    return (r"{" + size + r"\setlength{\arraycolsep}{" + str(sep) + r"pt}$$" + (name + "=" if name else "") + r"\left(\begin{array}{" + "c" * n + "}"
            + r"\\ ".join(rows) + r"\end{array}\right)$$}")


def logmat(L, name, root):
    rows = [" & ".join(str(x) for x in r) for r in L]
    return (r"{\setlength{\arraycolsep}{4pt}$$" + name + r"=\begin{pmatrix}" + r"\\ ".join(rows)
            + r"\end{pmatrix}$$}")


# Dita, symbolic: entries Q_{ib} H_{ij} K_{ab} with H, K among F_2, F_3, F_4, W_4
def unit_exp(kind, i, j):
    """exponent mod 12 of the (i,j) entry"""
    if kind == "F2": return 6 * i * j % 12
    if kind == "F3": return 4 * i * j % 12
    if kind == "F4": return 3 * i * j % 12
    if kind == "W4":
        i1, i2, j1, j2 = i // 2, i % 2, j // 2, j % 2
        return 6 * (i1 * j1 + i2 * j2) % 12
SIZE = {"F2": 2, "F3": 3, "F4": 4, "W4": 4}
UNIT = {0: "", 6: "-", 3: "i", 9: "-i", 4: "w", 8: "w^2"}


def dita_tex(hk, kk, name):
    M, N = SIZE[hk], SIZE[kk]
    qn = {}
    c = 1
    for i in range(1, M):
        for b in range(1, N):
            qn[(i, b)] = f"q_{{{c}}}" if (M - 1) * (N - 1) > 1 else "q"
            c += 1
    rows = []
    for i, a in itertools.product(range(M), range(N)):
        ents = []
        for j, b in itertools.product(range(M), range(N)):
            e = (unit_exp(hk, i, j) + unit_exp(kk, a, b)) % 12
            u = UNIT[e]
            q = qn.get((i, b), "")
            if q:
                ents.append(u + q)
            else:
                ents.append({"": "1", "-": "-1"}.get(u, u))
        rows.append(" & ".join(ents))
    return (r"{\small\setlength{\arraycolsep}{3pt}$$" + name + r"=\begin{pmatrix}" + r"\\ ".join(rows)
            + r"\end{pmatrix}$$}")


def item(title, text):
    W(r"\medskip\noindent\textbf{" + title + r"} " + text + "\n")


# ------------------------------------------------------------ preamble
W(r"""\documentclass[11pt]{amsart}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amssymb,amsmath}
\usepackage{booktabs,adjustbox}
\usepackage{cite}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=magenta,urlcolor=magenta,citecolor=magenta]{hyperref}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{conjecture}[theorem]{Conjecture}
\newcommand{\T}{\mathbb T}
\setcounter{MaxMatrixCols}{30}
\newcommand{\arx}[1]{\href{https://arxiv.org/pdf/#1}{arxiv:#1}}
\title{Higher invariants of complex Hadamard matrices}
\author{Claude and Teo}
\date{November 2026}
\begin{document}
\subjclass[2020]{46L37 (05B20, 46L65)}
\keywords{Complex Hadamard matrix, quantum permutation group, subfactor}
\begin{abstract}
To any complex Hadamard matrix $H\in M_N(\mathbb T)$ corresponds a quantum permutation group $G\subset S_N^+$, whose invariants $c_k=\dim Fix(P^{\otimes k})$ are those of the associated subfactor. We build here a basis of complex and real Hadamard matrices, with explicit conventions, and we compute the invariants $c_k$ well beyond the levels $k=2,3$ previously known, with the aim of formulating a general conjecture relating the algebraic, geometric and analytic invariants of $H$.
\end{abstract}
\maketitle
\tableofcontents
""")
W(open('front.tex').read())
W(r"""
""")

# ------------------------------------------------------------ rules
W(r"""
\section{Conventions and methods}\label{sec:conv}

\noindent\textbf{Invariants.} For a complex Hadamard matrix $H\in M_N(\T)$ with rows $H_1,\ldots,H_N$, we use the magic matrix $P_{ij}=Proj(H_i/H_j)$, and its Hopf image $G\subset S_N^+$. We compute
$$c_k=\dim Fix(P^{\otimes k})=\int_G\chi^k$$
normalized so that $F_N$ gives $c_k=N^{k-1}$, and the Temperley--Lieb value is $Cat_k=1,2,5,14,42,132$ at $k=1,\ldots,6$. Scalar products are linear at left, $\langle x,y\rangle=\sum_ix_i\bar{y}_i$. By Proposition \ref{prop:conj}, only $H$ and $H^t$ need to be computed.

\medskip

\noindent\textbf{Equivalences.} For each matrix, we first determine the equivalences between $H,H^t,\bar{H},H^*$, by placing them in the known parametrizations and by an exact equivalence test, so that each equivalence class is computed once.

\medskip

\noindent\textbf{Notations.} We write $e(t)=e^{2\pi it}$, $w=e(1/3)$, $i=e(1/4)$. Indices run over $0,\ldots,N-1$. The Fourier matrix is $(F_N)_{ij}=e(ij/N)$. For Di\c t\u a deformations we use the convention of \cite[Theorem 5.12]{banicaHM}, \cite{dit}
$$(H\otimes_QK)_{ia,jb}=Q_{ib}H_{ij}K_{ab}$$
with $Q\in M_{M\times N}(\T)$ dephased, and the double index $ia$ standing for $iN+a$. A matrix in logarithmic form is given by $H'$, with $H_{ij}=\omega^{H'_{ij}}$ for a stated root of unity $\omega$. Real matrices are displayed as sign patterns, with $\pm$ standing for $\pm1$. A starred value $X^*$ is obtained by the probabilistic count of section \ref{sec:approx}, as opposed to an exact computation. In all such cases the inequality $X\geq X^*$ is rigorous, and equality holds with high probability. The values which are not matched by OEIS, or which break an observed pattern, were certified by a Lanczos computation on the complement of the fixed vectors found, and are written without star.

\medskip

\noindent\textbf{Method.} With $\xi_{ij}=(H_i/H_j)/\sqrt N$ the unit vector spanning $P_{ij}$, consider the matrix $T_k\in M_{N^k}(\mathbb C)$ given by
$$(T_k)_{i_1\ldots i_k,j_1\ldots j_k}=tr(P_{i_1j_1}\ldots P_{i_kj_k})=\frac{1}{N}\prod_{s=1}^k\langle\xi_{a_{s+1}},\xi_{a_s}\rangle$$
where $a_s=(i_s,j_s)$, with $a_{k+1}=a_1$, and with $tr$ the normalized trace. We have $T_k=N^{-k-1}VV^*$, with $V_{i,x}=\prod_sH_{i_sx_{s-1}}\bar{H}_{i_sx_s}$ for $i,x\in\{1,\ldots,N\}^k$, with $x_0=x_k$, so $T_k$ is positive, and it is a contraction, being the image of the unitary $P^{\otimes k}\in M_{N^k}(M_N(\mathbb C))$ under the conditional expectation $id\otimes tr$. Thus its spectrum is contained in $[0,1]$, and $c_k=\dim\ker(T_k-1)$ is the multiplicity of the eigenvalue 1. We compute this dimension from the singular values of $T_k-1$, and we record the ratio between the smallest nonzero singular value and the largest zero one. This direct method is used for $N^k\leq5000$, and the larger cases are treated by the methods of section \ref{sec:approx}.

\medskip

\noindent\textbf{Checks.} Every matrix below was checked numerically to be Hadamard, $|H_{ij}|=1$ and $HH^*=N1_N$ to precision $10^{-9}$, at 5 random points for the families. The formulas used in the computations are the ones displayed here.
""")

# ------------------------------------------------------------ stage 0
W(r"""
\section{Fourier matrices and $F_4^s$}

We first consider the matrices for which the answer is known, or partly known.

""")
item("Fourier matrices.", r"$F_N$ for $N=2,\ldots,8$, with $(F_N)_{ij}=e(ij/N)$. Also $F_G=F_{N_1}\otimes\ldots\otimes F_{N_r}$ for $G=\mathbb Z_2^2,\mathbb Z_2^3,\mathbb Z_2\times\mathbb Z_4$, the first 2 being the Walsh matrices $W_4,W_8$. Here $G$ is the group itself, and $c_k=N^{k-1}$.")
item("The matrices $F_4^s$.", r"Source: \cite[Theorem 5.13]{banicaHM}. This is $F_2\otimes_QF_2$ with $Q=(^1_1{\ }^1_s)$, and every $4\times4$ complex Hadamard matrix is of this form. We have $F_4^s\sim F_4^{-s}\sim F_4^{\bar{s}}\sim F_4^{-\bar{s}}$. Known: $c_2,c_3,c_4=3,10,35$ for generic $s$, with $c_4=36$ when $s^8=1$ is primitive \cite{cn}.")
W(r"""$$F_4^s=\begin{pmatrix}1&1&1&1\\1&-1&1&-1\\1&s&-1&-s\\1&-s&-1&s\end{pmatrix}$$

\noindent\textbf{Results.} For the Fourier matrices $F_2,\ldots,F_8$ and $F_G$ with $G=\mathbb Z_2^2,\mathbb Z_2^3,\mathbb Z_2\times\mathbb Z_4$, we find $c_k=N^{k-1}$ for all $k\leq4$, as it should be. For $F_4^s$, with $s=e(a/n)$ for all $n\leq12$ and $(a,n)=1$, plus 3 random points, we find, for $H$ and $H^t$ alike:

\medskip

\begin{center}
\begin{tabular}{lcccc}
\toprule
$s$ & $c_1$ & $c_2$ & $c_3$ & $c_4$\\
\midrule
$s^4=1$ (Fourier, $\mathbb Z_2^2$ or $\mathbb Z_4$) & 1 & 4 & 16 & 64\\
$s^8=1$, primitive & 1 & 3 & 10 & 36\\
all other tested $s$, including 3 random ones & 1 & 3 & 10 & 35\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent This agrees with \cite{cn}. The singular value gaps are all above $10^{13}$, so these values are reliable. Since $F_4^s\sim F_4^{-s}\sim F_4^{\bar{s}}$, the relevant parameter is $s^2$, up to conjugation: up to level 4, only $s^2$ of order 4 produces a jump. For $s^2$ of order 3 or 6 (that is, $s$ of order 3, 6 or 12), any jump must happen at $k\geq5$. Testing the $D_{2k}^{(1)}$ conjecture of \cite{cn} in these cases thus requires the levels $k\geq5$.
""")

# ------------------------------------------------------------ stage 1, real
W(r"""
\section{Real Hadamard matrices}\label{sec:real}

Here $\chi$ is the quadratic character of $\mathbb F_q$, and $Q_{ab}=\chi(b-a)$. The Paley I matrix, for $q\equiv3$ mod 4, is $P^1_{q+1}=1+\begin{pmatrix}0&\mathbb I\\-\mathbb I&Q\end{pmatrix}$, and the Paley II matrix, for $q\equiv1$ mod 4, is obtained from $\begin{pmatrix}0&\mathbb I\\ \mathbb I&Q\end{pmatrix}$ by replacing $0\to\begin{pmatrix}1&-1\\-1&-1\end{pmatrix}$ and $\pm1\to\pm\begin{pmatrix}1&1\\1&-1\end{pmatrix}$, as in \cite[Theorem 1.19]{banicaHM}, see also \cite{pal}. The elements of $\mathbb F_q$ are ordered as $0,1,\ldots,q-1$ for $q$ prime, and as polynomials $c_0+c_1x+c_2x^2$ in lexicographic order of $(c_0,c_1,c_2)$, for $q=9,27$. The Williamson matrices are built from symmetric circulant $A,B,C,D\in M_K(\pm1)$ with first entry $+1$ and $A^2+B^2+C^2+D^2=4K$, the first such quadruple found by our search, placed as in \cite[Theorem 1.24]{banicaHM}. The matrices named $\mathrm{had}_{N.k}$ are those of McKay's library \cite{mck}, in his order, with $\mathrm{had}_{16.0}$ being the Walsh matrix $W_{16}$ (we checked this).

\medskip

\noindent\textbf{Order 12.} There is 1 class. It contains the Paley I matrix $P_{12}^1$, displayed below, as well as the Paley II matrix $P_{12}^2$ and the Williamson matrix $W_{12}$ (with $K=3$), which are both equivalent to it, by the equivalence test below.
""")
for nm, M in [("P_{12}^1", m.paley1(11))]:
    W(signs(M, r"\normalsize", 3.5, name=nm))
W(r"""
\noindent\textbf{Order 16.} There are 5 classes (Hall), $\mathrm{had}_{16.0}=W_{16}$, the Walsh matrix, and $\mathrm{had}_{16.1},\ldots,\mathrm{had}_{16.4}$, taken from McKay's library \cite{mck}, in his order. The last 2 are transpose to each other, up to equivalence: $\mathrm{had}_{16.4}\sim\mathrm{had}_{16.3}^t$.
""")
W(r"""
\noindent\textbf{Order 20.} There are 3 classes (Hall): the Paley I matrix $P_{20}^1$ ($q=19$), which is equivalent to $\mathrm{had}_{20.2}$; the Paley II matrix $P_{20}^2$ ($q=9$), which is equivalent to $\mathrm{had}_{20.0}$ and to the Williamson matrix $W_{20}$ ($K=5$); and the remaining class $\mathrm{had}_{20.1}$, from McKay's library \cite{mck}. For space reasons, the matrices of order $\geq20$ are not displayed: they are given by the constructions above, and by \cite{mck}, and all of them are in \texttt{matrices.py}.
""")
W(r"""
\noindent\textbf{Orders 24 and 28.} There are 60 and 487 classes. We take the Paley I matrices $P_{24}^1$ ($q=23$) and $P_{28}^1$ ($q=27$), the Paley II matrix $P_{28}^2$ ($q=13$), and the Williamson matrix $W_{28}$ ($K=7$), all built as above. The Williamson matrix $W_{28}$ is equivalent to $P_{28}^2$, while $P_{28}^1$ and $P_{28}^2$ are not equivalent.
""")

# ------------------------------------------------------------ stage 2, Dita
W(r"""
\noindent\textbf{Equivalences.} With an exact equivalence test (row by row backtracking, complete, with explicit equivalences checked), we find: at order 12, $P_{12}^1\sim P_{12}^2\sim W_{12}$; at order 16, $\mathrm{had}_{16.k}\sim\mathrm{had}_{16.k}^t$ for $k=0,1,2$, while $\mathrm{had}_{16.3}\not\sim\mathrm{had}_{16.3}^t\sim\mathrm{had}_{16.4}$; at order 20, all 3 classes are self-transpose, $P_{20}^1\sim\mathrm{had}_{20.2}$ and $P_{20}^2\sim W_{20}\sim\mathrm{had}_{20.0}$; at orders 24 and 28, $P_{24}^1$, $P_{28}^1$, $P_{28}^2$, $W_{28}$ are all self-transpose, with $P_{28}^2\sim W_{28}\not\sim P_{28}^1$. Thus 12 computations cover all the matrices, and their transposes. All the matrices are self-transpose up to equivalence, except for $\mathrm{had}_{16.3}$ and $\mathrm{had}_{16.4}$.

\medskip

\noindent\textbf{Results.}

\medskip

\begin{center}
\begin{tabular}{lcccc}
\toprule
matrix & $c_1$ & $c_2$ & $c_3$ & $c_4$\\
\midrule
$P_{12}$ & 1 & 2 & 5 & 14\\
$\mathrm{had}_{16.0}=W_{16}$ & 1 & 16 & 256 & 4096\\
$\mathrm{had}_{16.1}$ & 1 & 7 & 58 & \\
$\mathrm{had}_{16.2}$ & 1 & 4 & 21 & \\
$\mathrm{had}_{16.3}$, $\mathrm{had}_{16.4}=\mathrm{had}_{16.3}^t$ & 1 & 3 & 11 & \\
$P_{20}^1$, $P_{20}^2$, $\mathrm{had}_{20.1}$ & 1 & 2 & 5 & 14\\
$P_{24}^1$ & 1 & 2 & 5 & 14\\
$P_{28}^1$, $P_{28}^2$ & 1 & 2 & 5 & 14\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent Orders 12, 20, 24, 28: Temperley--Lieb up to level 4, so wild candidates. Order 16: none is wild, and the 5 classes form a decreasing chain, with only $W_{16}$ having a commuting magic matrix. The non-self-transpose pair $\mathrm{had}_{16.3}$, $\mathrm{had}_{16.4}$ has equal $c_2,c_3$. At level 4 we get $c_4=142$ for $\mathrm{had}_{16.2}$, and $c_4=46$ for $\mathrm{had}_{16.3}$ and $\mathrm{had}_{16.4}$, which thus leave the little Schr\"oder sequence $1,3,11,45$ at level 4 (see section \ref{sec:dita}). At higher levels we get $c_4=568^*$ for $\mathrm{had}_{16.1}$, and $c_5=212^*$ for both $\mathrm{had}_{16.3}$ and $\mathrm{had}_{16.4}$, see section \ref{sec:approx}.

\medskip

\noindent\textbf{Di\c t\u a structure.} A direct search for the Di\c t\u a block structure (row groups whose quotients are constant on a common column partition) shows that all 4 non-Fourier classes $\mathrm{had}_{16.1},\ldots,\mathrm{had}_{16.4}$ are real Di\c t\u a deformations of Walsh type products, with $\pm1$ parameters. The same holds for $F_{8,5a},F_{8,5b},F_{8,5c}$ in section \ref{sec:dita}, while $F_{8,2},F_{8,4},K_{8,2},M_{8,1}$ and $P_7^q$ have no such structure. Thus all the matrices considered here fall into 2 classes: Di\c t\u a type, with $c_2\geq3$, and the others, with $1,2,5,14,\ldots$

\section{Matrices of Di\c t\u a type}\label{sec:dita}

\noindent\textbf{Deformations.} The Di\c t\u a deformations $H\otimes_QK$ \cite{dit} are the basic source of families of complex Hadamard matrices. A conceptual approach to the deformations of the Fourier matrices, and more generally of the group commuting squares, is that of Nicoara and White \cite{nwh}: for a finite abelian group, explicit partial isometries $a$ produce one-parameter families $U_t=\exp(it(a+a^*))$, with $U_tF_N$ Hadamard for any $t\in\mathbb R$, which attain the defect bound, and which were combined into multi-parameter families by Worley \cite{wor}. See also \cite{nw4} for the second order deformations. We will use here the Di\c t\u a parametrization, and the comparison with the families of \cite{nwh,wor} remains to be done.

\medskip

\noindent\textbf{Order 6.} Source: \cite[Proposition 6.19]{banicaHM}, \cite{bbs}. With our convention, $F_6^{(rs)}=F_2\otimes_QF_3$ with $Q=(^1_1{\ }^1_r{\ }^1_s)$, and $F_6^{(^r_s)}=F_3\otimes_QF_2$ with $Q^t=(^1_1{\ }^1_r{\ }^1_s)$ (checked numerically).
$$F_6^{(rs)}=\begin{pmatrix}
1&1&1&1&1&1\\
1&w&w^2&1&w&w^2\\
1&w^2&w&1&w^2&w\\
1&r&s&-1&-r&-s\\
1&wr&w^2s&-1&-wr&-w^2s\\
1&w^2r&ws&-1&-w^2r&-ws
\end{pmatrix}\qquad
F_6^{(^r_s)}=\begin{pmatrix}
1&1&1&1&1&1\\
1&-1&1&-1&1&-1\\
1&r&w&wr&w^2&w^2r\\
1&-r&w&-wr&w^2&-w^2r\\
1&s&w^2&w^2s&w&ws\\
1&-s&w^2&-w^2s&w&-ws
\end{pmatrix}$$

\noindent\textbf{Order 8, general.} The 4 right Di\c t\u a deformations of the order 8 tensor products, with $Q$ dephased, and the parameters $q_1,q_2,q_3$ filled into $Q$ row by row:
""")
W(dita_tex("F2", "F4", r"F_2\otimes_QF_4"))
W(dita_tex("F4", "F2", r"F_4\otimes_QF_2"))
W(dita_tex("F2", "W4", r"F_2\otimes_QW_4"))
W(dita_tex("W4", "F2", r"W_4\otimes_QF_2"))
W(r"""
\noindent\textbf{Order 9.} The same, for $F_3\otimes_QF_3$:
""")
W(dita_tex("F3", "F3", r"F_3\otimes_QF_3"))
W(r"""
\noindent\textbf{Order 8, regular families.} The 7 regular families of \cite{gt} in which only cycle structures $A,B$ appear, and which were expected to be of Di\c t\u a type; the results below show that 4 of them are not. The parameters $a,b,c,d,e\in\T$ are independent. These are the displays of \cite{gt}, verbatim.
""")
blocks = open("f8families.tex").read().split(r"\begin{center}")[1:]
blocks = [r"\begin{center}" + b for b in blocks]
for b in blocks[:7]:
    W(b)

# ------------------------------------------------------------ stage 3, N = 6
W(r"""
\noindent\textbf{Results.} At 3 random points of each family, for $H$ and $H^t$ alike, and stable on all of them:

\medskip

\begin{center}
\begin{tabular}{lccccc}
\toprule
family & params & $c_1$ & $c_2$ & $c_3$ & $c_4$\\
\midrule
$F_6^{(rs)}=F_2\otimes_QF_3$, $F_6^{(^r_s)}=F_3\otimes_QF_2$ & 2 & 1 & 4 & 18 & 86\\
$F_2\otimes_QF_4$, $F_4\otimes_QF_2$, $F_2\otimes_QW_4$, $W_4\otimes_QF_2$ & 3 & 1 & 5 & 28 & 167\\
$F_3\otimes_QF_3$ & 4 & 1 & 5 & 29 & 181\\
$F_{8,5a}$ & 5 & 1 & 4 & 19 & 98\\
$F_{8,5b}$, $F_{8,5c}$ & 5 & 1 & 3 & 11 & 45\\
$F_{8,4}$, $F_{8,4}^t$ & 4 & 1 & 2 & 5 & 14\\
$F_{8,2}$, $F_{8,2}^t$ & 2 & 1 & 2 & 5 & 14\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent Jumps, found by setting each parameter in turn to a root of unity of order $\leq12$, the others random, and all parameters equal to such a root (at $k\leq4$ for $N=6$, $k\leq3$ for $N=8,9$):
\begin{enumerate}
\item $N=6$: all parameters $=\pm1$ gives $F_6$, with $6,36,216$. All parameters of order 3 or 6 gives $c_4=90$, all parameters $=\pm i$ gives $c_4=88$.
\item The four 3-parameter families at $N=8$, and $F_3\otimes_QF_3$: only the Fourier point.
\item $F_{8,5a},F_{8,5b},F_{8,4},F_{8,4}^t$: all parameters $=\pm1$ gives $8,64$, all $=\pm i$ gives $5,28$. For $F_{8,5c}$, also $c_2,c_3=4,19$ at $c=\pm1$, and along the diagonal at all other tested roots.
\item $F_{8,2},F_{8,2}^t$: $c_2,c_3=5,28$ at $a=\pm1$, and the Fourier point.
\end{enumerate}

\noindent Observations. The generic values of $F_M\otimes_QF_N$ depend only on $\{M,N\}$, not on the order of the factors, nor on $F_4$ versus $W_4$, as predicted by \cite{bb3}. The 2- and 4-parameter families $F_{8,2}$, $F_{8,4}$ of \cite{gt} have $c_2=2$ (checked independently with the profile graph), so they are \emph{not} of Di\c t\u a type, and they are wild candidates up to level 4. They become Di\c t\u a-like only on special loci.

\medskip

\noindent\textbf{Comparison with \cite{bb3}.} For generic $Q$, the moment formula of \cite[Proposition 5.1]{bb3} gives, for $F_X\otimes_QF_Y$:

\medskip

\begin{center}
\begin{tabular}{lccccc}
\toprule
$|X|\times|Y|$ & $c_1$ & $c_2$ & $c_3$ & $c_4$ & $c_5$\\
\midrule
$2\times2$ & 1 & 3 & 10 & 35 & 126\\
$2\times3$, $3\times2$ & 1 & 4 & 18 & 86 & 426\\
$2\times4$, $4\times2$ & 1 & 5 & 28 & 167 & 1036\\
$3\times3$ & 1 & 5 & 29 & 181 & 1181\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent This agrees with all our generic values above. The families $F_{8,5a},F_{8,5b},F_{8,5c}$ are Di\c t\u a deformations of non-Fourier or iterated type, and are not covered by this formula.

\medskip

\noindent\textbf{Remark (conjectural).} The sequence $1,3,11,45$ of $F_{8,5b},F_{8,5c}$ is the beginning of the little Schr\"oder numbers (OEIS A001003), which are, up to a factor 2, the moments of the free Poisson law of parameter 2, namely $c_k=\frac{1}{2}\sum_{\pi\in NC(k)}2^{|\pi|}$. If this holds, then $\lim_kc_k^{1/k}=(1+\sqrt2)^2=3+2\sqrt2$, which lies strictly between the free value 4 and the value $N=8$ of the coamenable case, so the quantum group would not be coamenable, unlike for the generic deformations \cite{bb3}. We have now checked $c_5=197$, for both $F_{8,5b}$ and $F_{8,5c}$, and $c_6\geq880$ for $F_{8,5b}$, against the predicted 903. In terms of free cumulants, the conjecture reads $\kappa_n(\chi)=2^{n-1}$, see section \ref{sec:approx}.

\section{Wild candidates, order 6}

\noindent\textbf{Tao matrix.} Source: \cite[Proposition 6.16]{banicaHM}, \cite{tao}. Isolated, regular.
$$T_6=\begin{pmatrix}
1&1&1&1&1&1\\
1&1&w&w&w^2&w^2\\
1&w&1&w^2&w^2&w\\
1&w&w^2&1&w&w^2\\
1&w^2&w^2&w&1&w\\
1&w^2&w&w^2&w&1
\end{pmatrix}$$

\noindent\textbf{Haagerup matrix.} Source: \cite[Proposition 6.17]{banicaHM}, \cite{ha1}. Regular, $q\in\T$.
$$H_6^q=\begin{pmatrix}
1&1&1&1&1&1\\
1&-1&i&i&-i&-i\\
1&i&-1&-i&q&-q\\
1&i&-i&-1&-q&q\\
1&-i&\bar{q}&-\bar{q}&i&-1\\
1&-i&-\bar{q}&\bar{q}&-1&i
\end{pmatrix}$$

\noindent\textbf{Bj\"orck--Fr\"oberg matrix.} Source: \cite[Proposition 5.17]{banicaHM}, from \cite{bjf}. Circulant, non-regular, with $a$ a root of $a^2+(\sqrt3-1)a+1=0$. The 2 roots are conjugate, $a$ and $\bar{a}$, giving the matrix and a second circulant matrix.
$$BF_6=\begin{pmatrix}
1&ia&-a&-i&-\bar{a}&i\bar{a}\\
i\bar{a}&1&ia&-a&-i&-\bar{a}\\
-\bar{a}&i\bar{a}&1&ia&-a&-i\\
-i&-\bar{a}&i\bar{a}&1&ia&-a\\
-a&-i&-\bar{a}&i\bar{a}&1&ia\\
ia&-a&-i&-\bar{a}&i\bar{a}&1
\end{pmatrix}$$

\noindent\textbf{Beauchamp--Nicoara matrices.} Source: \cite[Theorem 5.18]{banicaHM}, from \cite{ben}. Self-adjoint, non-regular, and every self-adjoint $6\times6$ complex Hadamard matrix is of this form.
$$BN_6^q=\begin{pmatrix}
1&1&1&1&1&1\\
1&-1&\bar{x}&-y&-\bar{x}&y\\
1&x&-1&t&-t&-x\\
1&-\bar{y}&\bar{t}&-1&\bar{y}&-\bar{t}\\
1&-x&-\bar{t}&y&1&\bar{z}\\
1&\bar{y}&-\bar{x}&-t&z&1
\end{pmatrix}$$
$$x=\frac{1+2q+q^2-\sqrt{2}\sqrt{1+2q+2q^3+q^4}}{1+2q-q^2},\qquad t=\frac{1+2q+q^2-\sqrt{2}\sqrt{1+2q+2q^3+q^4}}{-1+2q+q^2}$$
$$y=q,\qquad z=\frac{1+2q-q^2}{q(-1+2q+q^2)}$$
with the principal branch of the square root. \textbf{Warning:} this is Hadamard only on the arc $Re(q)\leq(\sqrt3-1)/2$, i.e.\ $|\arg q|\geq68.53^\circ$, and not on the whole circle as stated in \cite{banicaHM}. At the endpoints $q^4+2q^3+2q+1=0$, and the constant $\sqrt3-1$ is that of the Bj\"orck--Fr\"oberg equation. We also have $D_6$ at $q=i$, i.e.\ $\theta=\pi/2$.

\medskip

\noindent\textbf{Karlsson matrices.} Source: \cite{kar}. Non-affine, 3 real parameters $\theta,\phi,\psi\in[0,\pi)$, containing all the previously known 1- and 2-parameter families. With
$$F_2=\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad Z_{1,2}=\begin{pmatrix}1&1\\z_{1,2}&-z_{1,2}\end{pmatrix},\quad Z_{3,4}=\begin{pmatrix}1&z_{3,4}\\1&-z_{3,4}\end{pmatrix}$$
the matrix is
$$K_6(\theta,\phi,\psi)=\begin{pmatrix}F_2&Z_1&Z_2\\Z_3&\frac{1}{2}Z_3AZ_1&\frac{1}{2}Z_3BZ_2\\Z_4&\frac{1}{2}Z_4BZ_1&\frac{1}{2}Z_4AZ_2\end{pmatrix}$$
where $A=F_2(-\frac{1}{2}+\frac{i\sqrt3}{2}\Lambda)$, $B=-F_2-A$, with $\Lambda=\begin{pmatrix}\cos\theta&e^{i\phi}\sin\theta\\e^{-i\phi}\sin\theta&-\cos\theta\end{pmatrix}$, and with $z_1=e^{i\psi}$ and
$$z_3^2=\mathcal M_A(z_1^2),\quad z_4^2=\mathcal M_B(z_1^2),\quad z_2^2=\mathcal M_A^{-1}(\mathcal M_B(z_1^2)),\qquad \mathcal M_X(z)=\frac{X_{12}^2z-X_{11}^2}{\bar{X}_{11}^2z-\bar{X}_{12}^2}$$
taking principal square roots (the signs give equivalent matrices, according to Karlsson). The degenerate points where $\mathcal M_A$ or $\mathcal M_B$ is undefined are excluded.

\medskip

\noindent\textbf{Sz\"oll\H{o}si family.} The conjectured generic $6\times6$ matrices, a 4-parameter family with no closed form. If needed, points will be produced numerically, and recorded as such.
""")

# ------------------------------------------------------------ stage 4, N = 7
W(r"""
\noindent\textbf{Results.} Here $c_k(\bar{H})=c_k(H)$ and $c_k(H^*)=c_k(H^t)$, so we compute $H$ and $H^t$, and these always agree below. With $k\leq4$ everywhere:
\begin{enumerate}
\item $T_6$, and $BF_6$ for both roots $a,\bar{a}$: $c_1,\ldots,c_4=1,2,5,14$.

\item $H_6^q$, at all roots of unity $q$ of order $\leq12$ and 3 random points: $1,2,5,14$ everywhere. No jumps.

\item $BN_6^q$, at all roots of unity of order $\leq12$ on its arc (including $q=i$, which gives $D_6$, and $q=-1$) and 3 random points of the arc: $1,2,5,14$ everywhere. No jumps.

\item $K_6(\theta,\phi,\psi)$ at 3 random points: $1,2,5,14$. On the grid $\theta,\phi,\psi\in\frac{\pi}{6}\{0,\ldots,5\}$, at $k\leq3$: $1,2,5$ at 164 of the 185 valid points, and $1,4,18$ at the other 21 (31 degenerate points skipped). These are the generic values of the order 6 Di\c t\u a families, which are contained in $K_6$.
\end{enumerate}

\noindent So at order 6, every known non-Di\c t\u a matrix is Temperley--Lieb up to level 4, which extends the $c_2=2$ observations of \cite{cn}. The Sz\"oll\H{o}si family has no closed form, and is not treated here.

\section{Regular matrices, order 7}

Source: \cite{gt}, see also \cite{pet}. Every regular complex Hadamard matrix of order 7 is equivalent to $F_7$, $P_7^q$ or $R_7^q$, with $q\in\T$. Here $R_7^q$ is a monomially normalized conjugate of $P_7^q$, kept because conjugation is not part of equivalence. For $c_k$ purposes, $c_k(R_7^q)$ is thus a $c_k(P_7^{q'})$.
$$P_7^q=\begin{pmatrix}
-q&q&w&1&w&1&w\\
q&-q&w&1&1&w&w\\
w&w&-w&1&w&w&1\\
1&1&1&-1&w&w&w\\
w&1&w&w&-\bar{q}w&\bar{q}w&1\\
1&w&w&w&\bar{q}w&-\bar{q}w&1\\
w&w&1&w&1&1&-1
\end{pmatrix}$$
$$R_7^q=\begin{pmatrix}
1&1&1&1&1&1&1\\
1&1&1&w&-1&-1&w^2\\
1&-w^2&w^2&w^2&q&-q&w\\
1&w&-1&1&qw^2&-qw^2&w^2\\
1&-1&-w&-1&-q&q&-w^2\\
1&w^2&w&qw&q&-qw&-q\\
1&w^2&w&-qw&qw&-q&q
\end{pmatrix}$$
""")

# ------------------------------------------------------------ stage 5, N = 8
W(r"""
\noindent\textbf{Results.} For $P_7^q$ and $R_7^q$, at 3 random points and at all roots of unity $q$ of order $\leq12$, for $H$ and $H^t$ alike: $c_1,\ldots,c_4=1,2,5,14$ everywhere. No jumps, at $k\leq4$. In particular $c_3=5$ at $q=1$, as found in \cite{cn}.

\section{Other matrices, order 8}

Source: \cite{gt}. The 2 remaining families, and the 4 isolated matrices, which are Butson. The isolated ones are in logarithmic form, with $\omega=e(1/12)$ for $I_{8,12}$ and $\omega=e(1/30)$ for the other 3.
""")
for b in blocks[7:]:
    W(b)
W(logmat(m.I812_L, "I'_{8,12}", 12))
W(logmat(m.K830_L, "K'_{8,30}", 30))
W(logmat(m.L830_L, "L'_{8,30}", 30))
W(logmat(m.L830t_L, "L^{t\\prime}_{8,30}", 30))

# ------------------------------------------------------------ stage 6, N = 9, 10, 11
W(r"""
\noindent\textbf{Results.} For $H$ and $H^t$ alike:
\begin{enumerate}
\item $K_{8,2}$: generically $1,2,5,14$. Jumps, at $k\leq3$: $c_2,c_3=4,19$ at $b=\pm1$, and along the whole diagonal $a=b$, except at $a=b=\pm1$, where we get $8,64$, and at $a=b=\pm i$, where we get $6,40$.

\item $M_{8,1}$: generically $1,2,5,14$. Jumps, at $k\leq3$: $c_2,c_3=4,19$ at $a^6=1$.

\item $I_{8,12}$, $K_{8,30}$, $L_{8,30}$, $L_{8,30}^t$: $1,2,5,14$.
\end{enumerate}

\noindent Note that $4,19$ are the generic values of $F_{8,5a}$, so both families meet a Di\c t\u a-like locus at special parameters.

\section{Matrices of orders 9--11}

Source: \cite[Section 3]{ben}, isolated matrices found by computer. Here $y=\frac{1-i\sqrt{15}}{4}$ for $H_9$, $\epsilon=e(1/10)$ for $BN_9$, given in logarithmic form, and $a=\frac{-1+i\sqrt{15}}{4}$ for $BN_{10}$. The regular matrices at $N=9$--$11$ from the GPT classification will be added as they come.
{\small$$H_9=\begin{pmatrix}
1&1&1&1&1&1&1&1&1\\
1&-1&-1&y&y^{-1}&-y^{-3}&y^{-3}&y&y^{-1}\\
1&y&y^3&-y^2&y^3&y&-1&-y&y\\
1&y&y&-1&y^{-1}&y^{-3}&-y^{-3}&-1&y^{-1}\\
1&-y&-y^2&y^3&y^3&-1&y&y&y\\
1&y&y^3&y^3&y&y^{-1}&y^{-1}&y&y^2\\
1&y^3&y&y&y^2&y^{-1}&y^{-1}&y^3&y\\
1&-y^2&-y&y&y&-1&y&y^3&y^3\\
1&y^3&y&-y&y&y&-1&-y^2&y^3
\end{pmatrix}$$}
""")
W(logmat(m.BN9_L, "BN'_9", 10))
W(r"""
{\small$$BN_{10}=\begin{pmatrix}
1&1&1&1&1&1&1&1&1&1\\
1&1&a^{-2}&a^{-1}&a^{-2}&a^2&1&a^2&1&a\\
1&a^2&1&a^{-2}&a^{-2}&1&a&a^2&1&a^{-1}\\
1&1&1&1&a^{-2}&a^2&a^2&a&a^{-2}&a^{-1}\\
1&a^2&a^{-2}&1&1&a&1&a^2&a^{-2}&a^{-1}\\
1&1&a^{-1}&a^{-2}&1&1&a^2&a^2&a^{-2}&a\\
1&a^2&1&a^{-2}&a^{-1}&a^2&1&1&a^{-2}&a\\
1&a^2&a^{-2}&1&a^{-2}&1&a^2&1&a^{-1}&a\\
1&a&a^{-2}&a^{-2}&1&a^2&a^2&1&1&a^{-1}\\
1&a&a^{-1}&a^{-1}&a^{-1}&a&a&a&a^{-1}&1
\end{pmatrix}$$}

\noindent The 2 matrices $N_{11},N_{11}'$ have $x=\frac{3}{4}-\frac{i\sqrt7}{4}$, and $(a,b,y)=(-x,x,-x^2)$, respectively $(a,b,y)=(1,-1,\bar{x})$:
{\small\setlength{\arraycolsep}{3pt}$$N_{11},N_{11}'=\begin{pmatrix}
1&1&1&1&1&1&1&1&1&1&1\\
1&y&y&a&b&b&-1&x&-1&x&-1\\
1&y&b&b&a&y&x&x&-1&-1&-1\\
1&a&b&y&y&b&-1&-1&x&-1&x\\
1&b&a&y&b&y&x&-1&-1&-1&x\\
1&b&y&b&y&a&-1&-1&x&x&-1\\
1&-1&x&-1&x&-1&-\frac{x}{a}&-\frac{x}{b}&-\frac{x}{y}&-\frac{x}{y}&-\frac{x}{b}\\
1&x&x&-1&-1&-1&-\frac{x}{b}&-\frac{x}{y}&-\frac{x}{b}&-\frac{x}{y}&-\frac{x}{a}\\
1&-1&-1&x&-1&x&-\frac{x}{y}&-\frac{x}{b}&-\frac{x}{b}&-\frac{x}{a}&-\frac{x}{y}\\
1&x&-1&-1&-1&x&-\frac{x}{y}&-\frac{x}{y}&-\frac{x}{a}&-\frac{x}{b}&-\frac{x}{b}\\
1&-1&-1&x&x&-1&-\frac{x}{b}&-\frac{x}{a}&-\frac{x}{y}&-\frac{x}{b}&-\frac{x}{y}
\end{pmatrix}$$}
""")

W(r"""
\noindent\textbf{Results.} For $H_9$, $BN_9$, $BN_{10}$, $N_{11}$, $N_{11}'$, and their transposes: $c_1,\ldots,c_4=1,2,5,14$.
""")
import json as _json
_L5 = _json.load(open("results/level5.json"))
_rows = []
for _r in _L5:
    _nm = _r["name"]
    for a, b in [("_6", "$_6$"), ("_7", "$_7$"), ("_9", "$_9$"), ("_10", "$_{10}$"), ("_11", "$_{11}$"), ("_12", "$_{12}$")]:
        pass
    _rows.append((_r["name"], _r["H"], _r["Ht"], _r["H_next5"]))
def _tex(nm):
    rep = {"T_6": "$T_6$", "BF_6": "$BF_6$", "H_6^q, q random": "$H_6^q$, $q$ random", "H_6^q, q = 1": "$H_6^q$, $q=1$",
           "BN_6^q, q random on arc": "$BN_6^q$, $q$ random", "BN_6^q, q = i (D_6)": "$BN_6^q$, $q=i$ ($D_6$)",
           "K_6, random": "$K_6$, random", "P_7^q, q random": "$P_7^q$, $q$ random", "P_7^q, q = 1": "$P_7^q$, $q=1$",
           "R_7^q, q random": "$R_7^q$, $q$ random", "F_{8,4}, random": "$F_{8,4}$, random", "F_{8,2}, random": "$F_{8,2}$, random",
           "K_{8,2}, random": "$K_{8,2}$, random", "M_{8,1}, random": "$M_{8,1}$, random", "I_{8,12}": "$I_{8,12}$",
           "K_{8,30}": "$K_{8,30}$", "L_{8,30}": "$L_{8,30}$", "L_{8,30}^t": "$L_{8,30}^t$", "H_9": "$H_9$", "BN_9": "$BN_9$",
           "BN_10": "$BN_{10}$", "N_11": "$N_{11}$", "N'_11": "$N_{11}'$", "P_12": "$P_{12}$"}
    return rep.get(nm, nm)
_body = "\\\\\n".join("%s & %s & %.3f" % (_tex(n), ",".join(map(str, h)), x) for n, h, ht, x in _rows)
W(r"""
\section{Level 5, wild candidates}\label{sec:level5}

We computed $c_5$ for all the wild candidates of order $N\leq12$, at a random point for the families, plus a few special points, for $H$ and $H^t$ alike. The computation is in $(\mathbb C^N)^{\otimes5}$, of dimension up to $12^5=248832$, by the seeded iterative method: the fixed vectors coming from level 4 (leg doublings, tensor products, rotations) are verified, and a deflated Lanczos run on their orthogonal complement checks that nothing else is fixed. The last column is the largest eigenvalue of $T_5$ below 1, measuring how clean the separation is.

\medskip

\begin{center}
\begin{tabular}{lcc}
\toprule
matrix & $c_1,\ldots,c_5$ & next eigenvalue\\
\midrule
""" + _body + r"""\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent In all cases, and for $H^t$ as well, we get $c_5=42=Cat_5$. Thus all wild candidates of order $\leq12$ remain Temperley--Lieb at level 5. At level 6, see section \ref{sec:approx}, we get $c_6=132^*$ for 12 of them, and $c_5=42^*$ for $P_{20}$. The other real matrices of orders 20--28 are not computed at this level, the cost being of order $N^7$ per matrix-vector product.
""")
W(r"""
\section{Probabilistic counts, levels 5--6}\label{sec:approx}

\noindent\textbf{Method.} At level $k$, with $n=N^k$, we first generate the seeds, namely the leg doublings, tensor products and rotations of the fixed vectors of lower levels, and we verify each of them, $\|T_kv-v\|<10^{-8}$, after orthogonalization. This gives an orthonormal family $S$ of $s$ fixed vectors, so $c_k\geq s$, rigorously. On $S^\perp$ we take a random Gaussian block $X\in M_{n\times b}$, and apply the Chebyshev filter $p(T_k)$, where $p(x)=T_{50}(2x/a-1)$ with $a=0.95$, which is bounded by 1 on $[0,a]$, and equals $\cosh(50\,\mathrm{arccosh}(2/a-1))\simeq5\cdot10^{5}$ at $x=1$. After Rayleigh--Ritz on the span of $p(T_k)X$, we count the Ritz values $>1-10^{-4}$, and check the corresponding vectors directly, with residuals $\|T_kv-v\|$ of order $10^{-8}$. This gives $X^*=s+\#$, with $c_k\geq X^*$ rigorous, the counted vectors being verified, and equality holding with high probability. The only failure modes are a random block nearly orthogonal to a fixed vector, which is excluded by taking $b$ well above the number found (if this number exceeds $b-10$, we restart with $2b$), and an eigenvalue in $(a,1)$, which would show up as an intermediate Ritz value; for $F_{8,5a}$ such a value $0.966$ indeed appears, and is well separated from 1. Cost: $50b$ matrix-vector products, about $1.5$s each at $N=16$, $k=5$, and memory about $4$GB. As a test, against known exact values, we recover $c_5=197$ for $F_{8,5b}$, $c_4=142$ for $\mathrm{had}_{16.2}$, and $c_4=46$ for $\mathrm{had}_{16.3}$. The seeds are orthogonalized in blocks of 64 candidates, the independent directions being extracted from the Gram matrix of each block, and each new seed is verified. For the matrices expected to be wild, we use instead as seeds the $Cat_k$ vectors $\xi_\pi$ with $\pi\in NC(k)$, which are fixed for any $G\subset S_N^+$, and the random block then only checks whether anything beyond Temperley--Lieb is fixed. With these seeds, $c_7$ at $N=6$, in dimension $6^7=279936$, takes a few minutes.

\medskip

\noindent\textbf{Results.} We obtained the following values:

\medskip

\begin{center}
\begin{tabular}{lccc}
\toprule
matrix & $c_1,c_2,\ldots$ & seeds $+$ new & next Ritz value\\
\midrule
$\mathrm{had}_{16.3}$ & $1,3,11,46,212$ & $207+5$ & 0.042\\
$\mathrm{had}_{16.4}$ & $1,3,11,46,212^*$ & $207+5$ & 0.042\\
$\mathrm{had}_{16.1}$ & $1,7,58,568^*$ & $385+183$ & 0.049\\
$F_{8,5a}$ & $1,4,19,98,531^*$ & $526+5$ & 0.966\\
$F_{8,5b}$ & $1,3,11,45,197,c_6\geq880$ & seeding unfinished & --\\
12 wild matrices, see below & $1,2,5,14,42,132^*$ & $132+0$ & 0.10--0.16\\
$P_{20}$ & $1,2,5,14,42^*$ & $42+0$ & 0.049\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent The 12 wild matrices are $T_6$, $H_6^q$, $BN_6^q$, $K_6$, $P_7^q$, $R_7^q$, $F_{8,4}$, $F_{8,2}$, $K_{8,2}$, $M_{8,1}$, with the random parameters of section \ref{sec:level5}, and $H_9$, $BN_{10}$. For them, $c_6=132^*$ is entirely produced by seeds, so the fixed spaces are generated by Temperley--Lieb up to level 6. Since $\mathrm{had}_{16.4}\simeq\mathrm{had}_{16.3}^t$, the agreement $c_k(H)=c_k(H^t)$ continues at level 5 for this pair. For $F_{8,5a}$, note that the 3-fold analogue of the little Schr\"oder numbers, namely the sequence shifting left under 3 iterations of the INVERT transform, is $1,4,19,100,562$, so this is not it. For $F_{8,5b}$ at level 6 the computation was stopped during seeding, at 880 verified vectors.

\medskip

\noindent\textbf{Positivity.} Since $\chi=\sum_iu_{ii}$ is a sum of projections, we have $0\leq\chi\leq N$, so $(c_k)$ is the moment sequence of a probability measure on $[0,N]$, with $c_0=1$. Thus the Hankel matrices $(c_{i+j})$, $(c_{i+j+1})$ and $(Nc_{i+j}-c_{i+j+1})$ must be positive. For $\mathrm{had}_{16.3}$ with $c_1,\ldots,c_4=1,3,11,46$, the second condition gives
$$\det\begin{pmatrix}1&3&11\\3&11&46\\11&46&c_5\end{pmatrix}=2c_5-411\geq0$$
so $c_5\geq206$, a free rigorous bound, which is in fact sharp up to 1 for the seeds, since these alone give $c_5\geq207$. The value $c_5=212$ gives $2c_5-411=13$.

\medskip

\noindent\textbf{OEIS.} The search for $1,3,11,46$ returns 34 sequences, with the following next terms:

\medskip

\begin{center}
\begin{tabular}{lcp{7.5cm}}
\toprule
next term & number & sequences\\
\midrule
180--205 & 10 & excluded by $c_5\geq206$\\
206 & 2 & A271959, A151140\\
207 & 7 & A006605, A155134, A275757, A291090, A151141, A357233, A306822\\
208 & 2 & A193074, A287891\\
209 & 2 & A371428, A233389\\
210 & 3 & A281548, A086521, A225293\\
211 & 2 & A046996, A248426\\
213 & 2 & A337913, A129579\\
219, 230, 233, 234 & 4 & A393791, A030814, A030976, A264416\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent Thus $1,3,11,46,212$ is not in OEIS. Note that the seeds alone give 207, the value of A006605, the number of modes of connections of $2n$ points, related to Baxter's generalization of the Temperley--Lieb operators. Since the candidates are consecutive integers, approximate values $c_5\pm1$ would have been useless here, and the probabilistic count, which produces an integer, is the right tool.

\medskip

\noindent\textbf{Free cumulants.} The free cumulants $\kappa_n=\kappa_n(\chi)$, computed from $c_1,c_2,\ldots$ via the moment-cumulant formula over $NC(n)$, are as follows:

\medskip

\begin{center}
\begin{tabular}{lll}
\toprule
matrix & $\kappa_1,\kappa_2,\ldots$ & comment\\
\midrule
wild & $1,1,1,1,1,1^*$ & free Poisson, as for $S_N^+$\\
$F_{8,5b}$, $F_{8,5c}$ & $1,2,4,8,16$ & $\kappa_n=2^{n-1}$, so $\chi\sim2\pi_{1/2}$\\
$\mathrm{had}_{16.3}$, $\mathrm{had}_{16.4}$ & $1,2,4,9,26^*$ & as above, plus $1,10^*$ at $n=4,5$\\
$F_{8,5a}$ & $1,3,9,25,60^*$ & close to, but not, $3^{n-1}$\\
$\mathrm{had}_{16.2}$ & $1,3,11,61$ & \\
$\mathrm{had}_{16.1}$ & $1,6,39,303^*$ & \\
$W_{16}$ & $1,15,210,2715$ & \\
$F_2\otimes_QF_3$, generic & $1,3,8,17,20$ & from \cite{bb3}\\
$F_2\otimes_QF_2$, generic & $1,2,3,2,-5$ & not freely infinitely divisible\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent Here $\pi_t$ is the free Poisson law of parameter $t$, and $\chi\sim2\pi_{1/2}$ is the reformulation of the little Schr\"oder conjecture of section \ref{sec:dita}, the moments $\sum_{\pi\in NC(k)}2^{k-|\pi|}$ being the little Schr\"oder numbers. For $\mathrm{had}_{16.3}$ the deviation from this law starts at $\kappa_4$, matching the single fixed vector at level 4 not produced by seeds, and possibly coming from a new generator of the corresponding planar algebra at level 4. However, the Boolean cumulants below seem to be better coordinates than the free ones.

\medskip

\noindent\textbf{Computational cost.} On a laptop with 8GB of memory, typical times are: $\mathrm{had}_{16.4}$ at $k=5$, 40 minutes; $\mathrm{had}_{16.1}$ at $k=4$, 18 minutes; $F_{8,5a}$ at $k=5$, 3 minutes; the wild matrices at $k=6$, from 1 minute ($N=6$) to 2.5 hours ($BN_{10}$); $P_{20}$ at $k=5$, 75 minutes. At high level the bottleneck is the seeding, whose orthogonalizations cost $n\cdot s$ per candidate. The cases $\mathrm{had}_{16.2}$ at $k=5$, and $F_{8,5a},F_{8,5b}$ at $k=6$, remain to be computed.

\medskip

\noindent\textbf{Certification of $c_5=212$.} For $\mathrm{had}_{16.3}$, an independent run with a new random seed gave again $207+5$ fixed vectors, and a Lanczos computation of the top eigenvalues of $T_5$ on the orthogonal complement of these 212 vectors gave $0.875,0.8536,0.8248,0.8042$, in 64 steps. Thus $c_5=212$, at the same level of certainty as our exact values, and the OEIS candidate A129579, the only one with next term $\geq212$ not of exponential generating function type, with next term 213, is ruled out. The value $0.875$ is the same as the second eigenvalue at level 4.

\medskip

\noindent\textbf{Boolean cumulants.} The Boolean cumulants $b_n$ are given by $M(z)=1/(1-B(z))$, with $M(z)=1+\sum_kc_kz^k$ and $B(z)=\sum_nb_nz^n$, that is, $c_k=\sum_{\pi}\prod_{\beta\in\pi}b_{|\beta|}$, sum over interval partitions. Equivalently, $1/G(z)=z-\sum_kb_{k+1}z^{-k}$, with $G$ the Cauchy transform of the law of $\chi$. Combinatorially, $c_k$ counts the closed paths of length $k$ at $1$ on the fusion graph of $u$, and $b_k$ counts those returning to $1$ for the first time at step $k$, that is, the fixed vectors of $u^{\otimes k}$ not factorizing as $\xi\otimes\eta$ over a prefix of the legs. We have:

\medskip

\begin{center}
\begin{tabular}{lll}
\toprule
matrix & $b_1,b_2,\ldots$ & identification\\
\midrule
wild & $1,1,2,5,14,42^*$ & Catalan, permutations avoiding a pattern of length 3\\
$F_{8,5b}$, $F_{8,5c}$ & $1,2,6,22,90$ & large Schr\"oder, separable permutations, $\mathrm{Av}(2413,3142)$\\
$\mathrm{had}_{16.3}$, $\mathrm{had}_{16.4}$ & $1,2,6,23,103$ & permutations avoiding 1 pattern of length 4\\
$F_{8,5a}$ & $1,3,12,55,273^*$ & ternary numbers $\frac{1}{2n+1}\binom{3n}{n}$\\
$\mathrm{had}_{16.2}$ & $1,3,14,95$ & \\
$\mathrm{had}_{16.1}$ & $1,6,45,423^*$ & \\
$F_2\otimes_QF_2$, generic & $1,2,5,14,42,132$ & Catalan, shifted\\
$F_2\otimes_QF_4$, generic & $1,4,19,100,562$ & \\
$W_{16}$ & $1,15,225,3375$ & $15^{n-1}$\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent For $\mathrm{had}_{16.3}$, the passage from $F_{8,5b}$ amounts to dropping 1 of the 2 forbidden patterns, $22\to23$ and $90\to103$, matching the single new fixed vector at level 4. The OEIS searches for the free cumulants $1,2,4,9,26$, for the moments $1,0,2,4,17,66$ of $\chi-1$, and for the binomial transform $1,2,6,24,113,588$ give nothing relevant. This suggests exploring a Boolean trail, in the context of honest compact quantum groups, as opposed to the Boolean quantum semigroups of Liu \cite{liu}. Predictions: $c_6=2974$ for $F_{8,5a}$, from the ternary numbers, and $c_6\in\{1054,1055\}$ for $\mathrm{had}_{16.3}$, according to the class of the avoided pattern, since $|\mathrm{Av}_6(\tau)|=512$ or $513$.
""")
W(r"""
\section{Non-Di\c t\u a matrices, systematic search}\label{sec:nondita}

\noindent\textbf{Conjecture W.} If $H$ is not of Di\c t\u a type, then $G(H)=S_N^+$, that is, $c_k=Cat_k$ for any $k$.

\medskip

\noindent\textbf{Evidence.} All the matrices with $c_2=2$ tested are Temperley--Lieb as far as computed: the 24 matrices of section \ref{sec:level5} up to $c_5$ exact, 12 of them up to $c_6=132^*$, the 7 matrices $T_6$, $H_6^q$, $BN_6^q$, $K_6$, $BF_6$, $H_6^1$, $D_6$ up to $c_7=429^*$, with next eigenvalue below $0.16$ in all cases, the real matrices of orders 12--28 of section \ref{sec:real} up to $c_4$, and $P_{20}$ up to $c_5=42^*$. The Di\c t\u a test was run directly on $F_{8,2},F_{8,4},K_{8,2},M_{8,1},P_7$, and for the others non-Di\c t\u a follows from $c_2=2$, by the easy statement that a Di\c t\u a structure produces a block partition invariant under $G$, hence $c_2\geq3$.

\medskip

\noindent\textbf{Real matrices.} We computed $c_2,c_3$ for all 60 classes of real Hadamard matrices of order 24, and $c_2$ for all 487 classes of order 28, from the data of McKay \cite{mck}, for both $H$ and $H^t$, which agree in all cases. Then, for the 45 non-Di\c t\u a classes of order 24, we found $c_4=14^*$ for both $H$ and $H^t$, and for the 487 classes of order 28, we found $c_3=5^*$ for $H$, and for $H^t$ in the 146 cases computed:

\medskip

\begin{center}
\begin{tabular}{llll}
\toprule
order & classes & $c_1,c_2,c_3$ & Di\c t\u a test\\
\midrule
24 & 45 & $1,2,5$ & no Di\c t\u a structure\\
24 & 14 & $1,3,11$ & Di\c t\u a, $2\times12$\\
24 & 1 & $1,4,20$ & Di\c t\u a, $2\times12$\\
28 & 487 & $c_2=2$ & no Di\c t\u a structure\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent Thus the dichotomy $\textrm{Di\c t\u a}\iff c_2\geq3$ holds for all these 547 matrices, and Conjecture W holds at level 4 for order 24, and at level 3 for order 28.

\medskip

\noindent\textbf{Special parameters.} The non-Di\c t\u a families were scanned on fine grids, together with all the points having roots of unity of order $\leq24$ (1 parameter), $\leq12$ (2 parameters), $\leq8$ (4 parameters) as parameters, plus random points, for a total of about 13,600 points with $c_2=2$. At all of them, $c_3=5$ and $c_4=14$, the 15 points where the probabilistic count gave $c_4>14$ being, by exact recomputation, artifacts of points close to a special locus, with eigenvalues $1-10^{-5}$ or $1-10^{-6}$. The special loci found are:

\medskip

\begin{center}
\begin{tabular}{lll}
\toprule
family & special locus & $c_2,c_3,c_4$\\
\midrule
$F_{8,2}$ & line $a=1$ & $5,28,167$, generic $2\times4$ Di\c t\u a\\
$K_{8,2}$ & lines $a=b$ and $b=1$ & $4,19,98$, as for $F_{8,5a}$\\
$K_{8,2}$ & isolated points & $6,40$\\
$F_{8,4}$ & 69 points & $3,11$\\
$F_{8,4}$ & other points & $5,28$\\
$M_{8,1}$ & 12 points & $4,19$\\
$K_6$ & 45 points & $4,18$, generic $2\times3$ Di\c t\u a\\
$F_{8,2},K_{8,2},F_{8,4}$ & a few points & $8,64$, Fourier type\\
$H_6^q,BN_6^q,P_7^q,R_7^q$ & none & --\\
\bottomrule
\end{tabular}
\end{center}

\medskip

\noindent In all cases the jump starts at level 2, so we have here evidence for a stronger statement: each matrix either has $c_2=2$ and is Temperley--Lieb, or jumps at level 2, and is of Di\c t\u a type. The first step towards Conjecture W would be the level 2 dichotomy, $H$ non-Di\c t\u a $\iff$ $G(H)$ quantum 2-transitive.
""")
W(open("bibliography.tex").read().replace(r"\begin{thebibliography}{999}", r"\begin{thebibliography}{999}" + "\n" + r"\raggedright\hypersetup{urlcolor=black,linkcolor=black,citecolor=black}"))
W(r"\end{document}" + "\n")
open("invariants.tex", "w").write("\n".join(out))
print("written")
