import argparse
import numpy as np
import matplotlib.pyplot as plt

from matrix_completion.utils.create_low_rank_matrix import create_low_rank_matrix
from matrix_completion.numpy.NNMF import NNMF as npNNMF
from matrix_completion.pytorch.NNMF import NNMF as tNNMF

# Suppress specific warnings
np.seterr(divide='ignore', over='ignore', invalid='ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Run NNMF matrix completion")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--matrix_size", type=int, nargs=2, default=[1000, 1000])
    parser.add_argument("--missing_fraction", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    Y, X, S = create_low_rank_matrix(args.rank, args.matrix_size[0])

    # Select which entries we observe uniformly at random
    mask = np.random.randint(0, 2, (args.matrix_size[0], args.matrix_size[1]))

    ####################################

    tau = np.sum(S) * 2

    print('tau:', tau)

    ####################################

    if args.backend == 'numpy':

        nnmf = npNNMF(tau=tau, max_iter=args.max_iter)

    else:

        nnmf = tNNMF(tau=tau, max_iter=args.max_iter)
        print(nnmf.device)

    nnmf.fit(Y, mask)

    ####################################

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    fig.suptitle('Nuclear Norm Matrix Factorization Least-Squares Loss')

    ax[0].plot(nnmf.loss)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Loss')

    ax[1].plot(nnmf.loss)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Loss')

    plt.tight_layout()

    ####################################

    X_pred = nnmf.predict()

    print(f'Error between predicted and ground truth: {np.round(np.linalg.norm(X_pred - X), 4)}')

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    cax = ax[0].imshow(X_pred, cmap='jet')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Predicted PSD Matrix')
    fig.colorbar(cax, ax=ax[0])

    cax = ax[1].imshow(X, cmap='jet')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Ground Truth PSD Matrix')
    fig.colorbar(cax, ax=ax[1])

    plt.tight_layout()

    ####################################

    F = X_pred[:args.matrix_size[0], -args.matrix_size[1]:]

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    cax = ax[0].imshow(np.abs(Y), cmap='jet')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Ground Truth Low-Rank Matrix')
    fig.colorbar(cax, ax=ax[0])

    cax = ax[1].imshow(np.abs(Y) * mask, cmap='jet')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Observed Entries of Ground Truth')
    fig.colorbar(cax, ax=ax[1])

    cax = ax[2].imshow(np.abs(F), cmap='jet')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title('Recovered Low-Rank Matrix')
    fig.colorbar(cax, ax=ax[2])

    plt.tight_layout()

    #########################

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    eigvals, eigvecs = np.linalg.eigh(X_pred)
    eigvals = eigvals[::-1]
    ax.plot(eigvals[:2*args.rank] / 2, label='Predicted Eigenvalues Values')

    U, S, V = np.linalg.svd(F)
    ax.plot(S[:2*args.rank], label='Predicted Singular Values')

    U, S, V = np.linalg.svd(Y)
    ax.plot(S[:2*args.rank], label='True Singular Values')

    ax.set_xlabel('Singular Value')
    ax.set_ylabel('Value')
    ax.legend()

    plt.tight_layout()

    ####################################

    plt.show()


if __name__ == "__main__":
    main()

