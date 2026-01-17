"""
Cdln.py

High-precision Numba implementation of Cdln for von Mises-Fisher distribution.

API:
    Cdln(k, d, k0) -> float64
    
    Parameters:
        k  : float64 - kappa value (concentration parameter)
        d  : int64   - dimension
        k0 : float64 - overflow threshold, use k0 < 0 for auto
    
    Returns:
        float64 - log partition function value

Usage (mirrors MATLAB):
    # MATLAB:  out = CBIG_ArealMSHBM_Cdln(k, d)
    # Python:  out = Cdln(k, d, -1.0)
    
    from Cdln import Cdln
    
    result = Cdln(100.0, 500, -1.0)

Algorithm:
    For k <= k0: 
        - ν >= 25: Debye asymptotic expansion (DLMF 10.41) with 6 terms
        - ν < 25:  High-precision series expansion
    
    For k > k0: 
        - Numerical integration (midpoint rule, 1000 grids)

Inspired by CBIG_ArealMSHBM_Cdln.

Written by Boletu from UCL Accelerated Computing Group
"""

import numpy as np
from numba import njit, float64, int64
import math


# ==============================================================================
# Constants
# ==============================================================================

_LOG_2PI = 1.8378770664093454835606594728112


# ==============================================================================
# Internal: Log-Gamma (use math.lgamma, natively supported by Numba)
# ==============================================================================

@njit(float64(float64), cache=True, fastmath=False)
def _lgamma(x):
    """Compute log-gamma using Numba's native math.lgamma support."""
    return math.lgamma(x)


# ==============================================================================
# Internal: Series expansion for log(I_ν(x))
# ==============================================================================

@njit(float64(float64, float64), cache=True, fastmath=False)
def _log_bessel_i_series(v, x):
    """Compute log(besseli(v, x)) using series expansion for small ν."""
    if x <= 0.0:
        if v == 0.0:
            return 0.0
        return -math.inf
    
    log_x_2 = math.log(x * 0.5)
    log_gamma_v1 = _lgamma(v + 1.0)
    
    log_term_0 = v * log_x_2 - log_gamma_v1
    log_max = log_term_0
    log_term = log_term_0
    
    x2_4 = x * x * 0.25
    log_x2_4 = math.log(x2_4) if x2_4 > 0.0 else -math.inf
    
    n_max = min(500, int(x + v + 100))
    terms = np.empty(n_max + 1, dtype=np.float64)
    terms[0] = log_term
    count = 1
    
    for k in range(1, n_max + 1):
        fk = float(k)
        log_term = log_term + log_x2_4 - math.log(fk) - math.log(v + fk)
        terms[count] = log_term
        count += 1
        
        if log_term > log_max:
            log_max = log_term
        
        if log_term < log_max - 50.0:
            break
    
    exp_sum = 0.0
    for i in range(count):
        exp_sum += math.exp(terms[i] - log_max)
    
    return log_max + math.log(exp_sum)


# ==============================================================================
# Internal: Debye asymptotic expansion for log(I_ν(x))
# ==============================================================================

@njit(float64(float64, float64), cache=True, fastmath=False)
def _log_bessel_i_debye(nu, x):
    """Compute log(besseli(nu, x)) using Debye expansion (DLMF 10.41)."""
    if x <= 0.0:
        if nu == 0.0:
            return 0.0
        return -math.inf
    
    if nu < 1e-10:
        nu = 1e-10
    
    z = x / nu
    z2 = z * z
    sqrt_1pz2 = math.sqrt(1.0 + z2)
    
    eta = sqrt_1pz2 + math.log(z / (1.0 + sqrt_1pz2))
    
    p = 1.0 / sqrt_1pz2
    p2 = p * p
    p3 = p2 * p
    p4 = p2 * p2
    p5 = p4 * p
    p6 = p4 * p2
    p7 = p4 * p3
    p9 = p4 * p4 * p
    p10 = p5 * p5
    p11 = p5 * p6
    p12 = p6 * p6
    p13 = p6 * p7
    p15 = p7 * p7 * p
    
    u1 = (3.0 * p - 5.0 * p3) / 24.0
    u2 = (81.0 * p2 - 462.0 * p4 + 385.0 * p6) / 1152.0
    u3 = (30375.0 * p3 - 369603.0 * p5 + 765765.0 * p7 - 425425.0 * p9) / 414720.0
    u4 = (4465125.0 * p4 - 94121676.0 * p6 + 349922430.0 * p4 * p4 
          - 446185740.0 * p10 + 185910725.0 * p12) / 39813120.0
    u5 = (1519035525.0 * p5 - 49286948607.0 * p7 + 284499769554.0 * p9 
          - 614135872350.0 * p11 + 566098157625.0 * p13 
          - 188699385875.0 * p15) / 6688604160.0
    
    inv_nu = 1.0 / nu
    inv_nu2 = inv_nu * inv_nu
    inv_nu3 = inv_nu2 * inv_nu
    inv_nu4 = inv_nu2 * inv_nu2
    inv_nu5 = inv_nu4 * inv_nu
    
    correction = 1.0 + u1 * inv_nu + u2 * inv_nu2 + u3 * inv_nu3 + u4 * inv_nu4 + u5 * inv_nu5
    
    log_iv = nu * eta - 0.5 * (_LOG_2PI + math.log(nu)) - 0.25 * math.log(1.0 + z2)
    
    if correction > 0.0:
        log_iv += math.log(correction)
    
    return log_iv


# ==============================================================================
# Public API: Cdln (single value)
# ==============================================================================

@njit(float64(float64, int64, float64), cache=True, fastmath=False)
def Cdln(k, d, k0):
    """
    Compute log partition function of von Mises-Fisher distribution.
    
    Parameters
    ----------
    k : float64
        Kappa value (concentration parameter). Must be positive.
    d : int64
        Dimension.
    k0 : float64
        Overflow threshold. If k0 < 0, auto-determined:
        - d < 1200:  k0 = 500
        - d < 1800:  k0 = 650  
        - d >= 1800: k0 = 800
    
    Returns
    -------
    float64
        Log partition function value.
    
    Examples
    --------
    >>> from Cdln import Cdln
    >>> result = Cdln(100.0, 500, -1.0)
    """
    # Auto-determine k0
    if k0 < 0.0:
        if d < 1200:
            k0 = 500.0
        elif d < 1800:
            k0 = 650.0
        else:
            k0 = 800.0
    
    v = float(d) * 0.5 - 1.0
    use_debye = (v >= 25.0)
    
    # Direct computation for k <= k0
    if k <= k0:
        if use_debye:
            return v * math.log(k) - _log_bessel_i_debye(v, k)
        else:
            return v * math.log(k) - _log_bessel_i_series(v, k)
    
    # Numerical integration for k > k0
    nGrids = 1000
    
    if use_debye:
        fk0 = v * math.log(k0) - _log_bessel_i_debye(v, k0)
    else:
        fk0 = v * math.log(k0) - _log_bessel_i_series(v, k0)
    
    half_d_minus_1 = 0.5 * float(d - 1)
    ofintv = (k - k0) / float(nGrids)
    adsum = 0.0
    
    for j in range(nGrids):
        ks = k0 + ofintv * (float(j) + 0.5)
        ratio = half_d_minus_1 / ks
        adsum += 1.0 / (ratio + math.sqrt(1.0 + ratio * ratio))
    
    return fk0 - ofintv * adsum
