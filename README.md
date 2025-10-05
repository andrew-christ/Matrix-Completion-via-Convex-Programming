# Matrix Completion via Convex Programming

This repository provides implementations of various matrix completion algorithms based on convex optimization. Convex optimization is particularly useful in this context because it offers theoretical guarantees for the optimality of a solution. The algorithms here are designed to recover missing entries in partially observed and possibly corrupted matrices.

While many heuristic method for matrix completion exist in the literature, they often lack rigorous guarantees for accurately recovering unseen entries. Moreover, these methods are frequently non-convex, meaning converegnce to a global solution is not assured.

To address these challenges, prior work has focused on recovering matrices by seeking low-rank or low-norm factorizations that minimize the least-squares error over the observed entries. Directly minimizing the rank of a matrix is non-convex, but can be efficiently approximated using its convex surrogate, the nuclear norm (i.e., the sum of the singular values) [[1](#ref1)].

Matrix completion has numerous applications, including recommendation systems [[2](#ref2)], collaborative filtering [[3](#ref3)], compressed sensing [[4](#ref4)], and phase retrieval [[5](#ref5)]. Many of these applications involve large-scale data, requiring algorithms that are both accurate and scalable. This repository emphasizes Frank-Wolfe-based algorithms [[6](#ref6)], which offer nearly linear runtime performance comparable to SVD-based methods, making them suitable for large-scale matrix completion tasks.

## Algorithms

### Nuclear Norm Matrix Factorization

This algorithm solves the convex relaxation of rank minimization by replacing the matrix rank with its nuclear norm (sum of singular values):

$$
\begin{aligned}
\underset{\mathbf{X}}{\text{minimize}}  \quad & \| P_\Omega(X - Y) \|_F^2 \\
\text{subject to} \quad & \| X \|_* \leq \tau
\end{aligned}
$$

where $P_\Omega$ is the projection onto the set of observed entries. This formulation promotes a low-rank structure in $\mathbf{X}$ and is robust to additive noise in the observed entries. The low-rank factorization is computed efficiently using a *Frank-Wolfe* algorithm. Additionally, the equation above can be represented as a semidefinite program (SDP):

$$
\begin{aligned}
    \underset{\mathbf{A}, \mathbf{B}, \mathbf{X}}{\text{minimize}}  \quad & \| P_\Omega(X - Y) \|_F^2 \\
    \text{subject to} \quad & \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B}) \leq 2\tau \\
    & \begin{bmatrix}
    \mathbf{A} & \mathbf{X} \\
    \mathbf{X}^T & \mathbf{B}
    \end{bmatrix} \succeq 0
\end{aligned}
$$

➡️ **View [Numpy](matrix_completion/numpy/NNMF.py)  implementation**

### Maximum Margin Matrix Factorization

$$
\begin{aligned}
    \underset{\mathbf{A}, \mathbf{B}, \mathbf{X}}{\text{minimize}}  \quad & \| P_\Omega(X - Y) \|_F^2 \\
    \text{subject to} \quad 
    & \begin{bmatrix}
    \mathbf{A} & \mathbf{X} \\
    \mathbf{X}^T & \mathbf{B}
    \end{bmatrix} \succeq 0 \\
    & \mathbf{A}_{ii}, \mathbf{B}_{jj} \leq \tau \quad \forall i, j \\
\end{aligned}
$$

## Examples

```bash
./run.sh experiments.run_NNMF --max_iter=500
```

## References

<a id="ref1"></a>[1] Fazel, Maryam. *Matrix rank minimization with applications*. Diss. PhD thesis, Stanford University, 2002.

<a id="ref2"></a>[2] Kang, Zhao, Chong Peng, and Qiang Cheng. "Top-n recommender system via matrix completion." *Proceedings of the AAAI conference on artificial intelligence*. Vol. 30. No. 1. 2016.

<a id="ref3"></a>[3] Srebro, Nathan, Jason Rennie, and Tommi Jaakkola. "Maximum-margin matrix factorization." *Advances in neural information processing systems* 17 (2004).

<a id="ref4"></a>[4] Recht, Benjamin, Maryam Fazel, and Pablo A. Parrilo. "Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization." *SIAM review* 52.3 (2010): 471-501.

<a id="ref5"></a>[5] Candes, Emmanuel J., et al. "Phase retrieval via matrix completion." SIAM review 57.2 (2015): 225-251.

<a id="ref6"></a>[6] Jaggi, Martin. "Revisiting Frank-Wolfe: Projection-free sparse convex optimization." *International conference on machine learning*. PMLR, 2013.
