from ..base import MatrixCompletion
from ..utils.pytorch.power_method import sparse_power_method

import torch

from tqdm import tqdm


class NNMF(MatrixCompletion):

    def __init__(self, tau, max_iter=200) -> None:
        self.tau = tau
        self.max_iter = max_iter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, Y, mask):

        n, m = Y.shape

        Y       = torch.tensor(Y, device=self.device, dtype=torch.float32)
        mask    = torch.tensor(mask, device=self.device, dtype=torch.float32)

        # Initialize the PSD matrix 'X' to rank-1 with trace equal to 𝜏
        x = torch.randn(n + m, 1, device=self.device, dtype=torch.float32)
        x = x / torch.norm(x)

        self.X = self.tau * x @ x.t()

        self.loss = []

        for k in tqdm(range(self.max_iter), desc='Running Nuclear Norm Matrix Completion'):

            # Initialize the Frank-Wolfe step size
            g_k = 2 / (2 + k)

            # Select the upper right block matrix of 'X'
            F = self.X[:n, -m:]

            # Compute the gradient
            G = (F - Y) * mask

            # Compute the rank-1 update using the negative gradient
            _, x = sparse_power_method(-G, device=self.device, max_iter=2*(k+1))

            # Perform the Frank-Wolfe rank-1 update
            self.X = (1 - g_k) * self.X + g_k * (self.tau * x @ x.t())

            # Save loss 
            F = self.X[:n, -m:]
            self.loss.append(torch.norm((F - Y) * mask).item())

        return self
    
    def predict(self):
        return self.X.detach().cpu().numpy()