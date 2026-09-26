"""Heston (1993): characteristic function, call prices, and exact density.

Ground truth #2. Black-Scholes is too kind a test -- it has no smile, so a
broken estimator can still look fine. Heston gives a genuine smile *and*
a density we can compute to machine precision by inverting the
characteristic function, so we can check the extractor against a
non-lognormal, skewed, fat-tailed truth.

The CF uses the Albrecher et al. (2007) "little Heston trap" formulation:
algebraically identical to Heston's original, but the complex logarithm
stays on its principal branch for long maturities instead of silently
jumping and producing oscillating garbage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

__all__ = ["HestonParams", "char_fn", "call_price", "rn_density"]


@dataclass(frozen=True)
class HestonParams:
    v0: float       # initial variance
    kappa: float    # mean-reversion speed
    theta: float    # long-run variance
    sigma: float    # vol-of-vol
    rho: float      # spot/vol correlation (negative => left skew)

    @property
    def feller(self) -> float:
        """2 kappa theta - sigma^2. Negative means variance can touch zero."""
        return 2 * self.kappa * self.theta - self.sigma**2


def char_fn(u, S, T, r, q, p: HestonParams):
    """E^Q[exp(i u log S_T)], vectorised over complex u."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u

    a = p.kappa - p.rho * p.sigma * iu
    d = np.sqrt(a**2 + (p.sigma**2) * (iu + u**2))
    g = (a - d) / (a + d)                      # "trap" form: |g| <= 1

    exp_dt = np.exp(-d * T)
    C = (p.kappa * p.theta / p.sigma**2) * (
        (a - d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = (p.v0 / p.sigma**2) * (a - d) * (1.0 - exp_dt) / (1.0 - g * exp_dt)

    return np.exp(iu * (np.log(S) + (r - q) * T) + C + D)


def _P(j: int, S, K, T, r, q, p: HestonParams, limit=400):
    """Gil-Pelaez probabilities. j=2 is Q(S_T > K); j=1 is the share measure."""
    logK = np.log(K)
    drift = np.exp((r - q) * T)                # = phi(-i) / S

    def integrand(u):
        if j == 2:
            phi = char_fn(u, S, T, r, q, p)
        else:
            phi = char_fn(u - 1j, S, T, r, q, p) / (S * drift)
        return float(np.real(np.exp(-1j * u * logK) * phi / (1j * u)))

    # Split the range: the integrand oscillates fastest near zero and decays
    # slowly, so a single (0, inf) call trips scipy's roundoff detector.
    v1, _ = quad(integrand, 1e-10, 50.0, limit=limit, epsabs=1e-12, epsrel=1e-10)
    v2, _ = quad(integrand, 50.0, np.inf, limit=limit, epsabs=1e-12, epsrel=1e-10)
    return 0.5 + (v1 + v2) / np.pi


def call_price(S, K, T, r, q, p: HestonParams):
    """European call. Scalar or array K."""
    K_arr = np.atleast_1d(np.asarray(K, float))
    out = np.empty_like(K_arr)
    for i, k in enumerate(K_arr):
        P1 = _P(1, S, k, T, r, q, p)
        P2 = _P(2, S, k, T, r, q, p)
        out[i] = S * np.exp(-q * T) * P1 - k * np.exp(-r * T) * P2
    return out if np.ndim(K) else out[0]


def rn_density(x, S, T, r, q, p: HestonParams, limit=400):
    """Exact risk-neutral density of S_T at x, by inverting the CF.

    f_logS(y) = (1/pi) Int_0^inf Re[e^{-iuy} phi(u)] du, then divide by x
    for the change of variable to price space.
    """
    x_arr = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x_arr)
    for i, xi in enumerate(x_arr):
        y = np.log(xi)

        def integrand(u, y=y):
            return float(np.real(np.exp(-1j * u * y) * char_fn(u, S, T, r, q, p)))

        v1, _ = quad(integrand, 0.0, 50.0, limit=limit, epsabs=1e-13, epsrel=1e-11)
        v2, _ = quad(integrand, 50.0, np.inf, limit=limit, epsabs=1e-13, epsrel=1e-11)
        out[i] = (v1 + v2) / (np.pi * xi)
    return out if np.ndim(x) else out[0]
