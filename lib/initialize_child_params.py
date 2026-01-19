"""
initialize_child_params.py

GPU-accelerated implementation for child parameter initialization.

API:
    initialize_child_params(data_series, g_mu, dim, num_clusters, num_session)
        -> (ndarray[float64, 2d], ndarray[bool, 2d])
    
    Parameters:
        data_series  : ndarray[float64, 3d] - Input data (num_verts x num_features x num_session)
        g_mu         : ndarray[float64, 2d] - Group mean directions (num_features x num_clusters)
        dim          : int                  - Dimension for concentration initialization
        num_clusters : int                  - Number of clusters
        num_session  : int                  - Number of sessions
    
    Returns:
        ini_s_lambda : ndarray[float64, 2d] - Initial s_lambda values (num_verts x num_clusters)
        mask         : ndarray[bool, 2d]    - Mask for vertices (num_verts x num_clusters)

Usage (mirrors MATLAB):
    # MATLAB:  [ini_s_lambda, mask] = CBIG_ArealMSHBM_initialize_child_params(data_series, setting_params, tmp_dir, s)
    # Python:  ini_s_lambda, mask = initialize_child_params(data_series, g_mu, dim, num_clusters, num_session)
    
    from initialize_child_params import initialize_child_params
    import numpy as np
    
    # Load data
    data_series = np.load('data.npy')  # (num_verts, num_features, num_session)
    g_mu = np.load('group_mu.npy')     # (num_features, num_clusters)
    
    ini_s_lambda, mask = initialize_child_params(
        data_series, g_mu, dim=1482, num_clusters=50, num_session=4
    )

Algorithm:
    1. Initialize concentration parameter:
       - dim == 1482: ini_val = 650 (hardcoded for fsLR_32k)
       - Otherwise:   ini_val = initialize_concentration(dim)
    
    2. Compute log vMF probability for each session:
       ini_log_vmf += Cdln(ini_val, dim) + ini_val * (X @ g_mu)
    
    3. Apply softmax-style normalization:
       ini_log_vmf -= max(ini_log_vmf, axis=1)
    
    4. Create mask for zero-sum rows (medial wall vertices)
    
    5. Compute final s_lambda:
       ini_s_lambda = exp(ini_log_vmf)

Note:
    This module uses CuPy for GPU-accelerated matrix multiplication.
    The Cdln computation uses Cdln_Par (high-precision Numba parallel).
    
    The Python API is simplified compared to MATLAB - it takes g_mu directly
    instead of setting_params struct, and does not handle file I/O.

Inspired by CBIG_ArealMSHBM_initialize_child_params.

Written by Boletu from UCL Accelerated Computing Group
"""

import numpy as np
import cupy as cp

from Cdln_Par import Cdln as Cdln_batch
from initialize_concentration import initialize_concentration


# ==============================================================================
# Internal: Cdln wrapper for single value
# ==============================================================================

def _compute_Cdln_single(kappa, dim):
    """
    Compute Cdln for a single kappa value using Cdln_Par.
    
    Wrapper around Cdln_batch that handles single value input.
    """
    k_array = np.array([kappa], dtype=np.float64)
    result = Cdln_batch(k_array, np.int64(dim))
    return result[0]


# ==============================================================================
# Public API: initialize_child_params
# ==============================================================================

def initialize_child_params(data_series, g_mu, dim, num_clusters, num_session):
    """
    Compute initial child parameters for group priors estimation.
    
    Parameters
    ----------
    data_series : ndarray of float64, shape (num_verts, num_features, num_session)
        Input data series for all sessions.
    g_mu : ndarray of float64, shape (num_features, num_clusters)
        Group mean directions.
    dim : int
        Dimension for concentration initialization.
    num_clusters : int
        Number of clusters.
    num_session : int
        Number of sessions.
    
    Returns
    -------
    ini_s_lambda : ndarray of float64, shape (num_verts, num_clusters)
        Initial s_lambda values.
    mask : ndarray of bool, shape (num_verts, num_clusters)
        Mask for vertices (True for medial wall).
    
    Examples
    --------
    >>> import numpy as np
    >>> from initialize_child_params import initialize_child_params
    >>> data_series = np.random.randn(1000, 500, 4).astype(np.float64)
    >>> g_mu = np.random.randn(500, 50).astype(np.float64)
    >>> ini_s_lambda, mask = initialize_child_params(data_series, g_mu, 500, 50, 4)
    """
    num_verts = data_series.shape[0]
    
    # Initialize concentration parameter
    if dim == 1482:
        ini_val = 650.0
    else:
        ini_val = initialize_concentration(float(dim))
    
    # Compute Cdln using Cdln_Par (high-precision)
    Cdln_val = _compute_Cdln_single(ini_val, dim)
    
    # Convert scalars to GPU float64
    ini_val_f64 = cp.float64(ini_val)
    Cdln_f64 = cp.float64(Cdln_val)
    
    # Transfer g_mu to GPU
    d_g_mu = cp.asarray(g_mu, dtype=cp.float64)
    
    # Allocate accumulator on GPU
    d_ini_log_vmf = cp.zeros((num_verts, num_clusters), dtype=cp.float64)
    
    # Accumulate log vMF probability across sessions
    for t in range(num_session):
        d_X_t = cp.asarray(data_series[:, :, t], dtype=cp.float64)
        d_X_nu = cp.dot(d_X_t, d_g_mu)
        d_ini_log_vmf += Cdln_f64 + ini_val_f64 * d_X_nu
    
    # Softmax-style normalization: subtract row max
    d_row_max = cp.max(d_ini_log_vmf, axis=1, keepdims=True)
    d_ini_log_vmf -= d_row_max
    
    # Create mask for zero-sum rows (medial wall)
    d_row_sum = cp.sum(d_ini_log_vmf, axis=1)
    d_mask = cp.broadcast_to(
        (d_row_sum == cp.float64(0.0))[:, cp.newaxis],
        (num_verts, num_clusters)
    ).copy()
    
    # Compute exp
    d_ini_s_lambda = cp.exp(d_ini_log_vmf)
    
    # Transfer results to CPU
    ini_s_lambda = cp.asnumpy(d_ini_s_lambda)
    mask = cp.asnumpy(d_mask)
    
    return ini_s_lambda, mask
