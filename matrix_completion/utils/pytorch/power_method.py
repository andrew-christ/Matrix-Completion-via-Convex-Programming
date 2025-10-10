import torch


def sparse_power_method(A, max_iter=100, tol=1e-6, device="cpu"):
    """
    Power method for approximate leading singular value/vector of A
    using sparse-friendly matrix-vector multiplications.

    Args:
        A (torch.Tensor): matrix of shape (n, m)
        max_iter (int): max number of iterations
        tol (float): stopping tolerance
        device (str): "cpu" or "cuda"

    Returns:
        lambda_new (torch.Tensor): approximate leading singular value
        x (torch.Tensor): concatenated [u; v] vector
    """
    n, m = A.shape
    A = A.to(device)

    # initialize x
    x = torch.rand(n + m, 1, device=device, dtype=A.dtype)
    x = x / torch.norm(x)

    lambda_old = torch.tensor(0.0, device=device, dtype=A.dtype)

    for _ in range(max_iter):
        u, v = torch.split(x, [n, m])

        # matrix-vector products
        u = A @ v
        v = A.t() @ u

        # normalize u, v
        u = u / torch.norm(u)
        v = v / torch.norm(v)

        # concatenate back
        x = torch.cat((u, v))
        x = x / torch.norm(x)

        u, v = torch.split(x, [n, m])

        lambda_new = 2 * (u.t() @ (A @ v))

        if torch.abs(lambda_new - lambda_old) < tol:
            break

        lambda_old = lambda_new

    return lambda_new.item(), x