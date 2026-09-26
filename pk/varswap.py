"""Model-free implied variance, the jump correction, and HAC inference.

Three pieces:

  cboe_variance / interpolate_vix
      The discretisation of the log contract derived in Section 3.3 of the
      theory note, implemented as CBOE implements it. Running it on our own
      snapshot and comparing with the published VIX is the check that ties
      the cross-sectional half of the project to the time-series half: if
      the derivation and the code are right, the number comes out.

  risk_neutral_moments
      Bakshi-Kapadia-Madan style integrals, used to quantify the jump
      correction of Section 3.4 rather than leave it as a caveat.

  realized_variance / newey_west_mean
      The physical side, and standard errors that survive the overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["cboe_variance", "interpolate_vix", "realized_variance",
           "newey_west_mean", "risk_neutral_moments", "log_return_moments"]


def cboe_variance(strikes, call_mid, put_mid, T, r, call_bid=None, put_bid=None):
    """Model-free implied variance for one expiry, CBOE algorithm.

        sigma^2 = (2/T) sum_i (dK_i / K_i^2) e^{rT} O(K_i)
                  - (1/T) (F/K_0 - 1)^2

    The forward comes from the strike where the call and put mids are
    closest, as CBOE specifies, not from the regression used elsewhere in
    this project -- the point here is to reproduce the published index, so
    the published recipe is followed exactly.
    """
    K = np.asarray(strikes, float)
    C = np.asarray(call_mid, float)
    P = np.asarray(put_mid, float)
    order = np.argsort(K)
    K, C, P = K[order], C[order], P[order]

    # 1. forward from the strike with the smallest |C - P|
    i_star = int(np.argmin(np.abs(C - P)))
    F = K[i_star] + np.exp(r * T) * (C[i_star] - P[i_star])

    # 2. K0 = largest strike at or below F
    below = np.flatnonzero(K <= F)
    if below.size == 0:
        raise ValueError("no strike below the forward")
    i0 = int(below[-1])
    K0 = K[i0]

    # 3. OTM selection, with CBOE's two-consecutive-zero-bid stopping rule
    O = np.where(K < K0, P, np.where(K > K0, C, 0.5 * (C[i0] + P[i0])))

    keep = np.ones(K.size, dtype=bool)
    if put_bid is not None:
        pb = np.asarray(put_bid, float)[order]
        zeros = 0
        for i in range(i0 - 1, -1, -1):          # walk down through puts
            zeros = zeros + 1 if pb[i] <= 0 else 0
            if zeros >= 2:
                keep[:i + 1] = False
                break
    if call_bid is not None:
        cb = np.asarray(call_bid, float)[order]
        zeros = 0
        for i in range(i0 + 1, K.size):          # walk up through calls
            zeros = zeros + 1 if cb[i] <= 0 else 0
            if zeros >= 2:
                keep[i:] = False
                break
    keep &= O > 0

    Kk, Ok = K[keep], O[keep]

    # 4. dK: centred differences, one-sided at the ends
    dK = np.empty_like(Kk)
    dK[1:-1] = (Kk[2:] - Kk[:-2]) / 2
    dK[0] = Kk[1] - Kk[0]
    dK[-1] = Kk[-1] - Kk[-2]

    contrib = (dK / Kk**2) * np.exp(r * T) * Ok
    sigma2 = (2.0 / T) * contrib.sum() - (1.0 / T) * (F / K0 - 1.0) ** 2

    return {
        "sigma2": float(sigma2),
        "sigma": float(np.sqrt(max(sigma2, 0.0))),
        "F": float(F), "K0": float(K0), "T": float(T),
        "n_options": int(keep.sum()),
        "K_lo": float(Kk.min()), "K_hi": float(Kk.max()),
        "sum_term": float((2.0 / T) * contrib.sum()),
        "correction_term": float((1.0 / T) * (F / K0 - 1.0) ** 2),
    }


def interpolate_vix(near, next_, days_near, days_next, target_days=30,
                    days_year=365):
    """Blend two expiries to a constant 30-day horizon, CBOE weighting.

    Interpolation is in *total* variance (sigma^2 T), which is the quantity
    that is additive in time, not in sigma^2.
    """
    w_near = (days_next - target_days) / (days_next - days_near)
    w_next = (target_days - days_near) / (days_next - days_near)
    T1 = days_near / days_year
    T2 = days_next / days_year
    blended = (T1 * near["sigma2"] * w_near + T2 * next_["sigma2"] * w_next)
    var_30 = blended * days_year / target_days
    return {
        "vix": float(100 * np.sqrt(var_30)),
        "variance_30d": float(var_30),
        "w_near": float(w_near), "w_next": float(w_next),
    }


# --------------------------------------------------------------- jump side
def log_return_moments(K, q, F):
    """Risk-neutral moments of the log return x = log(S_T/F) under density q(K).

    Integrating the fitted SVI density directly is cleaner than the BKM
    option-portfolio integrals here: the density is already arbitrage-free
    and smooth, so there is no discretisation error to argue about, and the
    two are the same object by construction.
    """
    x = np.log(K / F)
    m0 = np.trapezoid(q, K)
    m1 = np.trapezoid(x * q, K) / m0
    c2 = np.trapezoid((x - m1) ** 2 * q, K) / m0
    c3 = np.trapezoid((x - m1) ** 3 * q, K) / m0
    c4 = np.trapezoid((x - m1) ** 4 * q, K) / m0
    return {"mass": float(m0), "mean_log_return": float(m1),
            "variance": float(c2), "third_central": float(c3),
            "skewness": float(c3 / c2**1.5),
            "excess_kurtosis": float(c4 / c2**2 - 3)}


def risk_neutral_moments(K, q, F, T):
    """Log-contract variance and the jump correction from Section 3.4.

    From equation (30),
        2 e^{rT} int O/K^2 dK = E[QV] - (2/3) E[sum J^3] + O(J^4),
    so an estimate of E[sum J^3] converts the measured implied variance into
    an estimate of expected quadratic variation.

    We estimate E[sum J^3] by the risk-neutral third central moment of the
    log return. That attributes the *entire* third moment to jumps, which
    makes the resulting correction an upper bound: a stochastic-volatility
    diffusion with leverage also generates negative skew without any jumps.
    Stated rather than smuggled, because the bound is what the data supports
    without a further model.
    """
    m = log_return_moments(K, q, F)
    third = m["third_central"]
    correction = -(2.0 / 3.0) * third          # positive when skew is negative
    implied_var_ann = m["variance"] / T
    return {
        **m,
        "third_moment": float(third),
        "jump_correction_total": float(correction),
        "jump_correction_annualised": float(correction / T),
        "implied_variance_annualised": float(implied_var_ann),
        "correction_pct_of_variance": float(correction / m["variance"] * 100),
    }


# ------------------------------------------------------------ physical side
def realized_variance(log_returns: pd.Series, window: int, annualise=252):
    """Forward-looking realised variance over the next `window` sessions.

    Indexed at the *start* of the window, so RV at date t is comparable with
    an implied variance quoted at t. Annualised.
    """
    r2 = log_returns.astype(float) ** 2
    fwd = r2.shift(-1).rolling(window).sum().shift(-(window - 1))
    return (fwd * annualise / window).dropna()


def newey_west_mean(x, lags):
    """Mean with a Newey-West HAC standard error.

    With overlapping windows of length h the series is autocorrelated to
    order h-1 by construction even under the null, so the naive standard
    error is badly oversized. Bartlett weights, lag truncation `lags`.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    e = x - x.mean()
    gamma0 = float(e @ e / n)
    s = gamma0
    for L in range(1, lags + 1):
        g = float(e[L:] @ e[:-L] / n)
        s += 2 * (1 - L / (lags + 1)) * g
    s = max(s, 1e-18)
    se = float(np.sqrt(s / n))
    naive_se = float(np.sqrt(gamma0 / n))
    return {
        "mean": float(x.mean()), "n": int(n), "lags": int(lags),
        "se_nw": se, "t_nw": float(x.mean() / se),
        "se_naive": naive_se, "t_naive": float(x.mean() / naive_se),
        "se_inflation": float(se / naive_se),
    }


def andrews_bandwidth(x):
    """Automatic lag truncation, Newey-West (1994) plug-in, AR(1) based."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    e = x - x.mean()
    rho = float(e[1:] @ e[:-1] / (e[:-1] @ e[:-1]))
    rho = np.clip(rho, -0.97, 0.97)
    alpha = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    return int(np.ceil(1.1447 * (alpha * x.size) ** (1 / 3)))
