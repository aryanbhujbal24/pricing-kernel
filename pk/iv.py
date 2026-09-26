"""Black-76: options on the forward, and implied-vol inversion.

Working in forward measure rather than with spot and a dividend yield is
the right move for index options. The forward and discount factor come
straight out of put-call parity (see pk.data), so there is no separate
dividend assumption to get wrong, and the smile is naturally a function
of log-moneyness k = log(K/F) -- the coordinate SVI is parameterised in.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = ["call", "put", "vega", "implied_vol", "implied_vol_vector"]


def _d(F, K, sigma, T):
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v**2) / v
    return d1, d1 - v


def call(F, K, D, sigma, T):
    d1, d2 = _d(F, K, sigma, T)
    return D * (F * norm.cdf(d1) - K * norm.cdf(d2))


def put(F, K, D, sigma, T):
    d1, d2 = _d(F, K, sigma, T)
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def vega(F, K, D, sigma, T):
    d1, _ = _d(F, K, sigma, T)
    return D * F * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, F, K, D, T, is_call, lo=1e-4, hi=5.0):
    """Invert Black-76. Returns nan when the quote is outside no-arbitrage bounds.

    Returning nan rather than raising matters: a real chain always contains
    a handful of stale or crossed quotes, and the pipeline should drop them
    and say how many, not abort.
    """
    intrinsic = D * max(F - K, 0.0) if is_call else D * max(K - F, 0.0)
    upper = D * F if is_call else D * K
    if not np.isfinite(price) or price <= intrinsic or price >= upper:
        return np.nan

    f = (lambda s: call(F, K, D, s, T) - price) if is_call else (
        lambda s: put(F, K, D, s, T) - price)
    try:
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_vector(prices, F, strikes, D, T, is_call):
    prices = np.asarray(prices, float)
    strikes = np.asarray(strikes, float)
    flags = np.broadcast_to(np.asarray(is_call), strikes.shape)
    return np.array([
        implied_vol(p, F, k, D, T, bool(c))
        for p, k, c in zip(prices, strikes, flags)
    ])
