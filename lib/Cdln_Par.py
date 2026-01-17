"""
Cdln_Par.py

High-precision parallel Numba implementation of Cdln for von Mises-Fisher distribution.

API:
    Cdln(k, d, k0) -> ndarray[float64]
    
    Parameters:
        k  : ndarray[float64] - kappa values (1D array)
        d  : int64            - dimension
        k0 : float64          - overflow threshold, use k0 < 0 for auto
    
    Returns:
        ndarray[float64] - log partition function values (same shape as k)

Usage (mirrors MATLAB):
    # MATLAB:  out = CBIG_ArealMSHBM_Cdln(k, d)
    # Python:  out = Cdln(k, d, -1.0)
    
    from Cdln_Par import Cdln
    import numpy as np
    
    k = np.array([100.0, 200.0, 500.0, 1000.0])
    result = Cdln(k, 500, -1.0)  # Internally parallel

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
from numba import njit, prange, float64, int64
import math


# ==============================================================================
# Constants
# ==============================================================================

_LOG_2PI = 1.8378770664093454835606594728112


# ==============================================================================
# Internal: Log-Gamma (use math.lgamma, natively supported by Numba)
# ==============================================================================

@njit(float64(float64), cache=True, fastmath=False, inline='always')
def _lgamma(x):
    """Compute log-gamma using Numba's native math.lgamma support."""
    return math.lgamma(x)


# ==============================================================================
# Internal: Series expansion for log(I_ν(x))
# ==============================================================================

@njit(float64(float64, float64), cache=True, fastmath=False, inline='always')
def _log_bessel_i_series(v, x):
    """Compute log(besseli(v, x)) using series expansion for small ν."""
    if x <= 0.0:
        return 0.0 if v == 0.0 else -math.inf
    
    log_x_2 = math.log(x * 0.5)
    log_term = v * log_x_2 - _lgamma(v + 1.0)
    log_max = log_term
    
    x2_4 = x * x * 0.25
    log_x2_4 = math.log(x2_4) if x2_4 > 0.0 else -math.inf
    
    n_max = min(500, int(x + v + 100))
    terms = np.empty(n_max + 1, dtype=np.float64)
    terms[0] = log_term
    count = 1
    
    for k in range(1, n_max + 1):
        log_term += log_x2_4 - math.log(float(k)) - math.log(v + float(k))
        terms[count] = log_term
        count += 1
        if log_term > log_max:
            log_max = log_term
        if log_term < log_max - 50.0:
            break
    
    s = 0.0
    for i in range(count):
        s += math.exp(terms[i] - log_max)
    
    return log_max + math.log(s)


# ==============================================================================
# Internal: Debye asymptotic expansion for log(I_ν(x))
# ==============================================================================

@njit(float64(float64, float64), cache=True, fastmath=False, inline='always')
def _log_bessel_i_debye(nu, x):
    """Compute log(besseli(nu, x)) using Debye expansion (DLMF 10.41)."""
    if x <= 0.0:
        return 0.0 if nu == 0.0 else -math.inf
    
    if nu < 1e-10:
        nu = 1e-10
    
    z = x / nu
    z2 = z * z
    sqrt_1pz2 = math.sqrt(1.0 + z2)
    
    eta = sqrt_1pz2 + math.log(z / (1.0 + sqrt_1pz2))
    
    p = 1.0 / sqrt_1pz2
    p2, p3, p4 = p*p, p*p*p, p*p*p*p
    p5, p6, p7 = p4*p, p4*p2, p4*p3
    p9, p10, p11 = p4*p5, p5*p5, p5*p6
    p12, p13, p15 = p6*p6, p6*p7, p7*p7*p
    
    u1 = (3.0*p - 5.0*p3) / 24.0
    u2 = (81.0*p2 - 462.0*p4 + 385.0*p6) / 1152.0
    u3 = (30375.0*p3 - 369603.0*p5 + 765765.0*p7 - 425425.0*p9) / 414720.0
    u4 = (4465125.0*p4 - 94121676.0*p6 + 349922430.0*p4*p4 
          - 446185740.0*p10 + 185910725.0*p12) / 39813120.0
    u5 = (1519035525.0*p5 - 49286948607.0*p7 + 284499769554.0*p9 
          - 614135872350.0*p11 + 566098157625.0*p13 
          - 188699385875.0*p15) / 6688604160.0
    
    inv_nu = 1.0 / nu
    correction = (1.0 + u1*inv_nu + u2*inv_nu*inv_nu + u3*inv_nu*inv_nu*inv_nu 
                  + u4*inv_nu*inv_nu*inv_nu*inv_nu + u5*inv_nu*inv_nu*inv_nu*inv_nu*inv_nu)
    
    log_iv = nu*eta - 0.5*(_LOG_2PI + math.log(nu)) - 0.25*math.log(1.0 + z2)
    if correction > 0.0:
        log_iv += math.log(correction)
    
    return log_iv


# ==============================================================================
# Internal: Combined log Bessel I
# ==============================================================================

@njit(float64(float64, float64), cache=True, fastmath=False, inline='always')
def _log_bessel_i(v, x):
    """Auto-select between series and Debye expansion."""
    return _log_bessel_i_debye(v, x) if v >= 25.0 else _log_bessel_i_series(v, x)


# ==============================================================================
# Internal: Single-value Cdln computation
# ==============================================================================

@njit(float64(float64, float64, float64, float64), cache=True, fastmath=False, inline='always')
def _cdln_single(k, v, k0, half_d_m1):
    """Compute Cdln for a single k value."""
    if k <= k0:
        return v * math.log(k) - _log_bessel_i(v, k)
    
    # Numerical integration for k > k0
    fk0 = v * math.log(k0) - _log_bessel_i(v, k0)
    ofintv = (k - k0) * 0.001  # 1/1000
    
    adsum = 0.0
    for j in range(1000):
        ks = k0 + ofintv * (float(j) + 0.5)
        r = half_d_m1 / ks
        adsum += 1.0 / (r + math.sqrt(1.0 + r*r))
    
    return fk0 - ofintv * adsum


# ==============================================================================
# Public API: Cdln (parallel batch processing)
# ==============================================================================

@njit(float64[:](float64[:], int64, float64), parallel=True, cache=True, fastmath=False)
def Cdln(k, d, k0):
    """
    Compute log partition function of von Mises-Fisher distribution.

    Parameters
    ----------
    k : ndarray of float64
        Kappa values (concentration parameters). Must be positive.
    d : int64
        Dimension.
    k0 : float64
        Overflow threshold. If k0 < 0, auto-determined:
        - d < 1200:  k0 = 500
        - d < 1800:  k0 = 650  
        - d >= 1800: k0 = 800
    
    Returns
    -------
    ndarray of float64
        Log partition function values (same shape as k).
    
    Examples
    --------
    >>> import numpy as np
    >>> from Cdln_Par import Cdln
    >>> k = np.array([100.0, 500.0, 1000.0])
    >>> result = Cdln(k, 500, -1.0)
    """
    # Auto-determine k0
    if k0 < 0.0:
        if d < 1200:
            k0_actual = 500.0
        elif d < 1800:
            k0_actual = 650.0
        else:
            k0_actual = 800.0
    else:
        k0_actual = k0
    
    v = float(d) * 0.5 - 1.0
    half_d_m1 = 0.5 * float(d - 1)
    
    n = len(k)
    out = np.empty(n, dtype=np.float64)
    
    # Parallel loop over all k values
    for i in prange(n):
        out[i] = _cdln_single(k[i], v, k0_actual, half_d_m1)
    
    return out
