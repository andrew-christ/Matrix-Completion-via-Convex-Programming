import numpy as np

def sparse_power_method(A, max_iter=100, tol=1e-6):
    '''
    Power method for approximating the leading 
    eigenvalue/eigenvector of a block-sparse symmetric 
    matrix. For a matrix X = [0, A; A^T, 0], we 
    assume the block-diagonal elements are sparse.
    Therefore, to find the leading eigenvector of
    X, we simply need to find the leading
    left/right singular vectors of A.

    Args:
        A (numpy.ndarray): matrix of shape (n, m)
        max_iter (int): max number of iterations
        tol (float): stopping tolerance

    Returns:
        x (numpy.ndarray): concatenated [u; v] vector
    '''

    n, m = A.shape

    # Randomly initialize the leading eigenvector of 'X'
    x = np.random.rand(n + m, 1)
    x = x / np.linalg.norm(x)

    lambda_old = 0.0

    for _ in range(max_iter):

        # Split the eigenvector of 'X' into the 
        # left/right singular vectors of A'
        u, v = np.split(x, [n])

        # Update 'u' and v'
        u = A @ v
        v = A.T @ u

        # Normalize 'u' and 'v' to have unit energy
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)

        # Recombine 'u' and 'v'
        x = np.concatenate((u, v))

        # Normalize 'x' to have unit energy
        x = x / np.linalg.norm(x)

        # Compute the new lambda value
        u, v = np.split(x, [n])
        lambda_new = 2 * u.T @ A @ v

        # Check if convergence
        if np.abs(lambda_new - lambda_old) < tol:
            break

        # Update old lambda value
        lambda_old = lambda_new

    return x