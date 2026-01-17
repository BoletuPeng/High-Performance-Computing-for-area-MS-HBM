"""
vmf_probability.py

Optimized Numba implementation of von Mises-Fisher distribution probability.

API:
    vmf_probability(X, nu, kappa) -> ndarray[float64, 2d]
    
    Parameters:
        X     : ndarray[float64, 2d] - Input data matrix (N x D)
        nu    : ndarray[float64, 2d] - Mean direction matrix (D x L)
        kappa : ndarray[float64, 1d] - Concentration parameters (L,)
    
    Returns:
        ndarray[float64, 2d] - Log probability matrix (N x L)

Usage (mirrors MATLAB):
    # MATLAB:  log_vmf = CBIG_ArealMSHBM_vmf_probability(X, nu, kappa)
    # Python:  log_vmf = vmf_probability(X, nu, kappa)
    
    from vmf_probability import vmf_probability
    import numpy as np
    
    X = np.random.randn(1000, 500).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    nu = np.random.randn(500, 10).astype(np.float64)
    nu /= np.linalg.norm(nu, axis=0, keepdims=True)
    
    kappa = np.array([100.0] * 10)
    
    log_vmf = vmf_probability(X, nu, kappa)

Algorithm:
    log p(x | mu, kappa) = Cdln(kappa, d) + kappa * (x · mu)
    
    Where Cdln is the log partition function computed by Cdln_Par module.

Note:
    This module uses Cdln_Par which employs Numba parallel processing.
    If running multiple subjects in parallel (e.g., 4 subjects on 24 cores),
    consider using numba.set_num_threads() to control per-process thread count.
    
    Example:
        import numba
        numba.set_num_threads(6)  # 24 cores / 4 subjects = 6 threads each

Inspired by CBIG_ArealMSHBM_vmf_probability.

Written by Boletu from UCL Accelerated Computing Group
"""

import numpy as np
from numba import njit, float64, int64

from Cdln_Par import Cdln


# ==============================================================================
# Public API: vmf_probability
# ==============================================================================

@njit(float64[:, :](float64[:, :], float64[:, :], float64[:]), cache=True, fastmath=True)
def vmf_probability(X, nu, kappa):
    """
    Compute log of von Mises-Fisher distribution probability.
    
    Parameters
    ----------
    X : ndarray of float64, shape (N, D)
        Input data matrix, where N is number of vertices, D is dimension.
        Each row should be a unit vector. Must be C-contiguous.
    nu : ndarray of float64, shape (D, L)
        Mean direction matrix, where L is number of clusters.
        Each column should be a unit vector. Must be C-contiguous.
    kappa : ndarray of float64, shape (L,)
        Concentration parameter vector. Must be C-contiguous.
    
    Returns
    -------
    ndarray of float64, shape (N, L)
        Log probability matrix.
    
    Examples
    --------
    >>> import numpy as np
    >>> from vmf_probability import vmf_probability
    >>> X = np.eye(3, dtype=np.float64)
    >>> nu = np.eye(3, dtype=np.float64).T
    >>> kappa = np.array([10.0, 10.0, 10.0])
    >>> log_vmf = vmf_probability(X, nu, kappa)
    """
    N = X.shape[0]
    D = X.shape[1]
    L = nu.shape[1]
    
    # Dimension for vMF (sphere dimension = ambient dimension - 1)
    dim = np.int64(D - 1)
    
    # Compute log normalization constants using Cdln_Par (parallel)
    Cdln_vals = Cdln(kappa, dim, -1.0)
    
    # Matrix multiply: X_nu[i, j] = sum_k(X[i, k] * nu[k, j])
    X_nu = np.dot(X, nu)
    
    # Allocate output array
    log_vmf = np.empty((N, L), dtype=np.float64)
    
    # Compute log_vmf = Cdln + kappa * X_nu
    for i in range(N):
        for j in range(L):
            log_vmf[i, j] = Cdln_vals[j] + kappa[j] * X_nu[i, j]
    
    return log_vmf
