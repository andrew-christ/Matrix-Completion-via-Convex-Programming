from ..base import MatrixCompletion
from ..utils.numpy.power_method import sparse_power_method

import numpy as np

from tqdm import tqdm


class NNMF(MatrixCompletion):

    def __init__(self, tau, max_iter=200):
        super().__init__()
        self.tau = tau
        self.max_iter = max_iter

    def fit(self, Y, mask):

        n, m = Y.shape

        # Initialize the PSD matrix 'X' to rank-1 with trace equal to 𝜏
        x = np.random.randn(n + m, 1)
        x = x / np.linalg.norm(x)

        self.X = self.tau * x @ x.T

        self.loss = []

        for k in tqdm(range(self.max_iter), desc='Running Nuclear Norm Matrix Completion'):

            # Initialize the Frank-Wolfe step size
            g_k = 2 / (2 + k)

            # Select the upper right block matrix of 'X'
            F = self.X[:n, -m:]

            # Compute the gradient
            G = (F - Y) * mask

            # Compute the rank-1 update using the negative gradient
            x = sparse_power_method(-G, max_iter=2*(k+1))

            # Perform the Frank-Wolfe rank-1 update
            self.X = (1 - g_k) * self.X + g_k * (self.tau * x @ x.T)

            # Save loss 
            F = self.X[:n, -m:]
            self.loss.append(np.linalg.norm((F - Y) * mask))

        return self
    
    def predict(self):
        return self.X