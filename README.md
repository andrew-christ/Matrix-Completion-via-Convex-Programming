# Matrix Completion via Convex Programming

Tis repository provides implementations of various matrix completion algorithms based on convex optimization. Convex optimization is particularly useful in this context because it offers theoretical guarantees for the optimality of a solution. The algorithms here are designed to recover missing entries in partially observed and possibly corrupted matrices.

While many heuristic method for matrix completion exist in the literature, they often lack rigorous guarantees for accurately recovering unseen entries. Moreover, these methods are frequently non-convex, meaning converegnce to a global solution is not assured.

To address these challenges, prior work has focused on recovering matrices by seeking low-rank or low-norm factorizations that minimize the least-squares error over the observed entries. Directly minimizing the rank of a matrix is non-convex, but can be efficiently approximated using its convex surrogate, the nuclear norm (i.e., the sum of the singular values).

Matrix completion has numerous applications, including recommendation systems, collaborative filtering, compressed sesning, and phase retrieval. Many of these applications involve large-scale data, requiring algorithms tha are both accurate and scalable. This repsoitory emphasizes Frank-Wolfe-based algorithms, which offer nearly linear runtime performance comparable to SVD-based methods, making them suitable for large-scale matrix completion tasks.

## Examples

```bash
./run.sh experiments.run_NNMF --max_iter=500
