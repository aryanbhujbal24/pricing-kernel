"""Black-Scholes: European prices and the *exact* risk-neutral density.

This module exists to provide ground truth. Under Black-Scholes the
risk-neutral law of S_T is lognormal in closed form, so any
Breeden-Litzenberger estimator we write must reproduce `rn_density`
to within its own discretisation error. If it does not, the estimator
is wrong -- not the market.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["call_price", "put_price", "rn_density", "rn_cdf", "vega"]


def _d1_d2(S, K, T, r, q, sigma):
    S, K = np.asarray(S, float), np.asarray(K, float)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def call_price(S, K, T, r, q, sigma):
    """European call under Black-Scholes with continuous dividend yield q."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, q, sigma):
    """European put; used to check put-call parity holds in our conventions."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def vega(S, K, T, r, q, sigma):
    """dC/dsigma. Needed later to weight IV-space fits by quote reliability."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def rn_density(x, S, T, r, q, sigma):
    """Exact risk-neutral density of S_T evaluated at x.

    log S_T ~ N(log S + (r - q - sigma^2/2) T, sigma^2 T), so the density
    in price space picks up the 1/x Jacobian.
    """
    x = np.asarray(x, float)
    mu = np.log(S) + (r - q - 0.5 * sigma**2) * T
    sd = sigma * np.sqrt(T)
    return norm.pdf((np.log(x) - mu) / sd) / (x * sd)


def rn_cdf(x, S, T, r, q, sigma):
    """Exact Q(S_T <= x). Equals 1 + e^{rT} dC/dK, a useful second check."""
    x = np.asarray(x, float)
    mu = np.log(S) + (r - q - 0.5 * sigma**2) * T
    sd = sigma * np.sqrt(T)
    return norm.cdf((np.log(x) - mu) / sd)
