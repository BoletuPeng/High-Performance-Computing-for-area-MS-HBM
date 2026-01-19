"""
vmf_probability_gpu.py

GPU-accelerated implementation of von Mises-Fisher distribution probability.

API:
    vmf_probability_gpu(X, nu, kappa) -> ndarray[float64, 2d]
    
    Parameters:
        X     : ndarray[float64, 2d] - Input data matrix (N x D)
        nu    : ndarray[float64, 2d] - Mean direction matrix (D x L)
        kappa : ndarray[float64, 1d] - Concentration parameters (L,)
    
    Returns:
        ndarray[float64, 2d] - Log probability matrix (N x L)

Usage (mirrors MATLAB):
    # MATLAB:  log_vmf = CBIG_ArealMSHBM_vmf_probability(X, nu, kappa)
    # Python:  log_vmf = vmf_probability_gpu(X, nu, kappa)
    
    from vmf_probability_gpu import vmf_probability_gpu
    import numpy as np
    
    X = np.random.randn(1000, 500).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    nu = np.random.randn(500, 10).astype(np.float64)
    nu /= np.linalg.norm(nu, axis=0, keepdims=True)
    
    kappa = np.array([100.0] * 10)
    
    log_vmf = vmf_probability_gpu(X, nu, kappa)

Algorithm:
    log p(x | mu, kappa) = Cdln(kappa, d) + kappa * (x · mu)
    
    Where Cdln is the log partition function computed by Cdln_Par module.
    The matrix multiplication (x · mu) is computed on GPU using CuPy.

Note:
    This module uses CuPy for GPU-accelerated matrix multiplication.
    The Cdln computation remains on CPU using Cdln_Par (Numba parallel).
    
    For optimal performance with large matrices, the GPU transfer overhead
    is amortized. For very small matrices, CPU version may be faster.
    
    Requirements:
        - CuPy with CUDA support
        - Cdln_Par module

Inspired by CBIG_ArealMSHBM_vmf_probability.

Written by Boletu from UCL Accelerated Computing Group
"""

import numpy as np
import cupy as cp

from Cdln_Par import Cdln


# ==============================================================================
# Public API: vmf_probability_gpu
# ==============================================================================

def vmf_probability_gpu(X, nu, kappa):
    """
    Compute log of von Mises-Fisher distribution probability (GPU-accelerated).
    
    Parameters
    ----------
    X : ndarray of float64, shape (N, D)
        Input data matrix, where N is number of vertices, D is dimension.
        Each row should be a unit vector.
    nu : ndarray of float64, shape (D, L)
        Mean direction matrix, where L is number of clusters.
        Each column should be a unit vector.
    kappa : ndarray of float64, shape (L,)
        Concentration parameter vector.
    
    Returns
    -------
    ndarray of float64, shape (N, L)
        Log probability matrix.
    
    Examples
    --------
    >>> import numpy as np
    >>> from vmf_probability_gpu import vmf_probability_gpu
    >>> X = np.eye(3, dtype=np.float64)
    >>> nu = np.eye(3, dtype=np.float64).T
    >>> kappa = np.array([10.0, 10.0, 10.0])
    >>> log_vmf = vmf_probability_gpu(X, nu, kappa)
    """
    # Ensure proper dtypes and contiguity
    X = np.ascontiguousarray(X, dtype=np.float64)
    nu = np.ascontiguousarray(nu, dtype=np.float64)
    kappa = np.ascontiguousarray(kappa, dtype=np.float64).flatten()
    
    N = X.shape[0]
    D = X.shape[1]
    L = nu.shape[1]
    
    # Dimension for vMF (sphere dimension = ambient dimension - 1)
    dim = np.int64(D - 1)
    
    # Compute log normalization constants using Cdln_Par (CPU, parallel)
    Cdln_vals = Cdln(kappa, dim, -1.0)
    
    # GPU-accelerated matrix multiply: X_nu[i, j] = sum_k(X[i, k] * nu[k, j])
    X_gpu = cp.asarray(X)
    nu_gpu = cp.asarray(nu)
    X_nu_gpu = cp.dot(X_gpu, nu_gpu)
    X_nu = cp.asnumpy(X_nu_gpu)
    
    # Compute log_vmf = Cdln + kappa * X_nu
    log_vmf = Cdln_vals.reshape(1, -1) + kappa.reshape(1, -1) * X_nu
    
    return log_vmf
