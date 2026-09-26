"""SVI: an arbitrage-free parameterisation of the volatility slice.

Part 2 established that differencing quotes twice is hopeless -- the noise
in the raw December extraction ran 47x the signal. The fix is not a better
difference but a change of object: fit a smooth low-dimensional surface in
total-variance space, impose no-arbitrage as a hard constraint, then
differentiate the *fitted* surface analytically. No finite differences
touch market data at any point.

Gatheral's raw SVI, in total variance w = sigma^2 T as a function of
log-moneyness k = log(K/F):

    w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + s^2) ]

Five parameters: a level, b the wing slope, rho the skew, m the shift, s the
smoothing of the kink at the money.

The butterfly condition is Gatheral & Jacquier's g >= 0, where g is defined
below; g < 0 somewhere is exactly the statement that the implied density goes
negative there. Imposing g >= 0 on a dense grid during the fit is what makes
the extracted density a density.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

__all__ = ["SVIParams", "total_variance", "dw", "d2w", "g_function",
           "density_k", "density_K", "call_price", "fit_svi"]


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def as_array(self):
        return np.array([self.a, self.b, self.rho, self.m, self.s])

    @property
    def wing_slope(self):
        """Asymptotic |dw/dk|. Lee's moment formula caps this at 2."""
        return self.b * (1 + abs(self.rho))


def _core(k, p: SVIParams):
    x = np.asarray(k, float) - p.m
    root = np.sqrt(x**2 + p.s**2)
    return x, root


def total_variance(k, p: SVIParams):
    x, root = _core(k, p)
    return p.a + p.b * (p.rho * x + root)


def dw(k, p: SVIParams):
    """dw/dk, analytic."""
    x, root = _core(k, p)
    return p.b * (p.rho + x / root)


def d2w(k, p: SVIParams):
    """d2w/dk2, analytic. Strictly positive, so w is convex in k."""
    x, root = _core(k, p)
    return p.b * p.s**2 / root**3


def g_function(k, p: SVIParams):
    """Gatheral-Jacquier g. The implied density is non-negative iff g >= 0."""
    w = total_variance(k, p)
    w1 = dw(k, p)
    w2 = d2w(k, p)
    return (1 - k * w1 / (2 * w))**2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2


def density_k(k, p: SVIParams):
    """Risk-neutral density of log-moneyness k = log(K/F).

    Standard result: with d2 = -k/sqrt(w) - sqrt(w)/2,
        p(k) = g(k) / sqrt(2 pi w) * exp(-d2^2 / 2).
    """
    w = total_variance(k, p)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2
    return g_function(k, p) / np.sqrt(2 * np.pi * w) * np.exp(-0.5 * d2**2)


def density_K(K, F, p: SVIParams):
    """Risk-neutral density in price space. Jacobian dk/dK = 1/K."""
    K = np.asarray(K, float)
    return density_k(np.log(K / F), p) / K


def call_price(K, F, D, T, p: SVIParams):
    """Black-76 call priced off the fitted slice."""
    K = np.asarray(K, float)
    k = np.log(K / F)
    v = np.sqrt(total_variance(k, p))
    d1 = -k / v + v / 2
    d2 = d1 - v
    return D * (F * norm.cdf(d1) - K * norm.cdf(d2))


# ------------------------------------------------------------------ fitting
def _unpack(theta):
    a, b, rho_raw, m, s = theta
    return SVIParams(a=a, b=b, rho=np.tanh(rho_raw), m=m, s=s)


def fit_svi(k, w_mkt, weights=None, k_grid=None, n_starts=12, seed=0,
            g_floor=1e-6, verbose=False):
    """Fit raw SVI to observed total variance under no-arbitrage constraints.

    Constraints imposed as hard inequalities (SLSQP):
      * b >= 0, s > 0                    -- shape
      * w(k) > 0 everywhere              -- variance is positive
      * b(1 + |rho|) <= 2                -- Lee's bound on the wing slope
      * g(k) >= g_floor on a dense grid  -- butterfly / density non-negativity

    rho is parameterised as tanh(.) so |rho| < 1 holds automatically rather
    than needing a constraint that the optimiser can sit exactly on.

    Multi-start because the SVI objective is not convex and a single start
    lands in a local minimum often enough to matter.
    """
    k = np.asarray(k, float)
    w_mkt = np.asarray(w_mkt, float)
    wts = np.ones_like(k) if weights is None else np.asarray(weights, float)
    wts = wts / wts.sum()

    if k_grid is None:
        pad = 0.25 * (k.max() - k.min())
        k_grid = np.linspace(k.min() - pad, k.max() + pad, 400)

    def objective(theta):
        p = _unpack(theta)
        resid = total_variance(k, p) - w_mkt
        return float(np.sum(wts * resid**2)) * 1e4

    cons = [
        {"type": "ineq", "fun": lambda th: th[1]},                       # b >= 0
        {"type": "ineq", "fun": lambda th: th[4] - 1e-4},                # s > 0
        {"type": "ineq",
         "fun": lambda th: 2.0 - th[1] * (1 + abs(np.tanh(th[2])))},     # Lee
        {"type": "ineq",
         "fun": lambda th: total_variance(k_grid, _unpack(th)) - 1e-8},  # w > 0
        {"type": "ineq",
         "fun": lambda th: g_function(k_grid, _unpack(th)) - g_floor},   # butterfly
    ]

    rng = np.random.default_rng(seed)
    atm_w = float(np.interp(0.0, k, w_mkt))
    best, best_val = None, np.inf

    for i in range(n_starts):
        if i == 0:
            th0 = np.array([0.5 * atm_w, 0.1, np.arctanh(-0.7), 0.0, 0.1])
        else:
            th0 = np.array([
                atm_w * rng.uniform(0.1, 0.9),
                rng.uniform(0.02, 0.4),
                np.arctanh(rng.uniform(-0.95, -0.1)),
                rng.uniform(-0.2, 0.1),
                rng.uniform(0.02, 0.4),
            ])
        try:
            res = minimize(objective, th0, method="SLSQP", constraints=cons,
                           options={"maxiter": 800, "ftol": 1e-14})
        except Exception:
            continue
        if not res.success:
            continue
        p = _unpack(res.x)
        if np.min(g_function(k_grid, p)) < 0:
            continue
        if res.fun < best_val:
            best_val, best = res.fun, p
        if verbose:
            print(f"  start {i:2d}: obj={res.fun:.6g} {'*' if best is p else ''}")

    if best is None:
        raise RuntimeError("no feasible SVI fit found; loosen g_floor or check inputs")

    resid = total_variance(k, best) - w_mkt
    diag = {
        "objective": best_val,
        "rmse_total_var": float(np.sqrt(np.mean(resid**2))),
        "max_abs_resid_total_var": float(np.max(np.abs(resid))),
        "min_g_on_grid": float(np.min(g_function(k_grid, best))),
        "wing_slope": best.wing_slope,
        "n_points": len(k),
    }
    return best, diag
