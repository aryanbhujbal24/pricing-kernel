"""Breeden-Litzenberger extraction and its diagnostics.

    q(K) = e^{rT} d^2C/dK^2

The whole difficulty of this project lives in that second derivative.
It is an ill-posed operation: a price perturbation of size eps at a
single strike moves the estimated density by ~2 eps e^{rT} / h^2, so
halving the strike spacing quadruples the noise amplification while
only quartering the truncation error. Everything downstream -- SVI,
arbitrage constraints, smoothing in IV space -- is machinery for
managing that single trade-off.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "extract_density",
    "density_diagnostics",
    "l1_error",
    "linf_error",
]


def extract_density(strikes, calls, r, T, spacing_tol=1e-8):
    """Risk-neutral density on an equally spaced strike grid.

    Central second difference, so the density is returned on the interior
    strikes only (the two endpoints are dropped).

    Returns (K_interior, q_hat).
    """
    K = np.asarray(strikes, float)
    C = np.asarray(calls, float)
    if K.ndim != 1 or K.shape != C.shape:
        raise ValueError("strikes and calls must be 1-D arrays of equal length")
    if K.size < 3:
        raise ValueError("need at least three strikes for a second difference")

    dK = np.diff(K)
    h = dK[0]
    if np.max(np.abs(dK - h)) > spacing_tol * max(1.0, abs(h)):
        raise ValueError(
            "strikes must be equally spaced; interpolate onto a uniform grid first"
        )

    d2C = (C[2:] - 2.0 * C[1:-1] + C[:-2]) / h**2
    return K[1:-1], np.exp(r * T) * d2C


def density_diagnostics(K, q_hat, forward=None):
    """Sanity checks a panel will ask about before believing any plot.

    - mass:            should integrate to 1 over a wide enough strike range
    - negative_mass:   any negative density is an arbitrage violation
    - implied_forward: E^Q[S_T] should equal the forward, not the spot
    """
    mass = float(np.trapezoid(q_hat, K))
    neg = np.clip(q_hat, None, 0.0)
    out = {
        "mass": mass,
        "negative_mass": float(np.trapezoid(neg, K)),
        "min_density": float(np.min(q_hat)),
        "frac_negative": float(np.mean(q_hat < 0)),
        "implied_mean": float(np.trapezoid(K * q_hat, K)),
    }
    if forward is not None:
        out["forward"] = float(forward)
        out["mean_error"] = out["implied_mean"] - float(forward)
    return out


def l1_error(K, q_hat, q_true):
    """Integrated absolute error -- the honest scalar for a density."""
    return float(np.trapezoid(np.abs(np.asarray(q_hat) - np.asarray(q_true)), K))


def linf_error(q_hat, q_true):
    return float(np.max(np.abs(np.asarray(q_hat) - np.asarray(q_true))))
