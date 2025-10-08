import numpy as np

# Suppress specific warnings
np.seterr(divide='ignore', over='ignore', invalid='ignore')

def create_low_rank_matrix(r, n):

    t = np.linspace(0, 1, n)

    # draw r random frequencies between 1 and 50 Hz
    f1 = np.random.uniform(1, 10, size=r)  
    f2 = np.random.uniform(1, 10, size=r)  

    # r random phase offsets between 0 and 2π
    phi1 = np.random.uniform(0, 2*np.pi, size=10)
    phi2 = np.random.uniform(0, 2*np.pi, size=10)

    U = np.cos(2 * np.pi * np.outer(t, f1) + phi1)
    U = np.linalg.qr(U)[0]
    U = np.nan_to_num(U, nan=1e-8, posinf=1e8, neginf=-1e8)
    V = np.sin(2 * np.pi * np.outer(t, f2) + phi2)
    V = np.linalg.qr(V)[0]
    V = np.nan_to_num(V, nan=1e-8, posinf=1e8, neginf=-1e8)

    s = np.random.uniform(1, 10, size=r)
    S = np.diag(s)

    Y = U @ S @ V.T

    UV = np.concatenate((U, V), axis=0)

    XX = UV @ S @ UV.T

    return Y, XX, S