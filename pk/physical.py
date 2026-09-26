"""The physical density p, estimated two ways.

The risk-neutral density q from Part 3 is *conditional on today*: it is
what the option market believes about 18 December 2026, priced on
25 September 2026, when the ATM implied vol was 13.5%.

The obvious estimator of p -- a kernel density on historical 59-day
returns -- is *unconditional*. It answers "what does a 59-day return look
like on a randomly chosen day since 1990", and the answer is dominated by
the fact that the average day since 1990 had an 18% vol. Dividing a
13.5%-vol q by an 18%-vol p does not measure a pricing kernel. It measures
the difference between today and an average day, and the result will be
U-shaped whatever the true kernel is.

So the two estimators here are not "a rough one and a better one". They
are answers to two different questions, and the gap between them is the
main quantitative result of this part.

  naive_kde   -- unconditional, overlapping returns. The standard naive
                 choice; reported because it shows the size of the error.
  garch_fhs   -- conditional. Fit GARCH(1,1)-t to daily returns, bootstrap
                 the standardised residuals forward 59 days from *today's*
                 conditional variance (Barone-Adesi filtered historical
                 simulation). Keeps the fat tails and the leverage effect
                 without assuming a parametric terminal law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

__all__ = ["overlapping_returns", "effective_sample_size", "naive_kde",
           "GarchFit", "fit_garch", "fhs_simulate", "density_from_sample",
           "to_price_density"]


def overlapping_returns(close: pd.Series, horizon: int) -> pd.Series:
    """h-period log returns sampled every day. Overlapping by construction."""
    lc = np.log(close.astype(float))
    return (lc.shift(-horizon) - lc).dropna()


def effective_sample_size(n_obs: int, horizon: int) -> float:
    """Non-overlapping equivalent. Overlap by h shrinks information ~h-fold.

    Reporting n rather than n/h is the single most common way to make an
    overlapping-returns standard error look far better than it is.
    """
    return n_obs / horizon


def naive_kde(returns, bw="silverman"):
    """Unconditional KDE of the horizon return.

    Silverman's rule assumes i.i.d. data. These returns overlap by 58 of
    59 days, so the rule is badly miscalibrated -- it sees n observations
    where there are effectively n/59. It undersmooths. Stated rather than
    silently corrected, since the correction is itself arbitrary.
    """
    r = np.asarray(returns, float)
    return gaussian_kde(r, bw_method=bw)


# --------------------------------------------------------------------------
@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    mu: float
    nu: float
    sigma2_last: float      # conditional variance for the next day
    resid_std: np.ndarray   # standardised residuals, for bootstrapping
    persistence: float
    loglik: float
    uncond_vol_ann: float
    current_vol_ann: float


def fit_garch(returns, dist="t"):
    """GARCH(1,1) with Student-t innovations on daily log returns."""
    from arch import arch_model

    r = pd.Series(returns).dropna() * 100.0   # arch prefers percent units
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    res = am.fit(disp="off", show_warning=False)
    pr = res.params

    omega, alpha, beta = pr["omega"], pr["alpha[1]"], pr["beta[1]"]
    sigma2 = res.conditional_volatility.to_numpy() ** 2
    eps = (r - pr["mu"]).to_numpy()

    # one-step-ahead conditional variance, in percent^2
    sigma2_next = omega + alpha * eps[-1] ** 2 + beta * sigma2[-1]
    persistence = alpha + beta

    # Centre the standardised residuals. They do not have mean zero in sample
    # -- high-variance days carry systematically signed residuals -- so
    # bootstrapping them raw injects an uncontrolled drift on top of mu. With
    # a centred pool the horizon drift is exactly mu * horizon, which makes the
    # drift an explicit assumption rather than a side effect of the filter.
    z = eps / np.sqrt(sigma2)
    z = z - z.mean()

    return GarchFit(
        omega=omega, alpha=alpha, beta=beta, mu=pr["mu"],
        nu=pr.get("nu", np.nan),
        sigma2_last=sigma2_next,
        resid_std=z,
        persistence=persistence,
        loglik=res.loglikelihood,
        uncond_vol_ann=float(np.sqrt(omega / (1 - persistence) * 252) / 100),
        current_vol_ann=float(np.sqrt(sigma2_next * 252) / 100),
    )


def fhs_simulate(fit: GarchFit, horizon: int, n_paths=200_000, seed=0,
                 mu_override=None):
    """Filtered historical simulation of the cumulative horizon return.

    Bootstraps standardised residuals rather than drawing from a fitted
    parametric law, so the simulated path inherits the empirical skew and
    kurtosis of the residuals. Starts from today's conditional variance,
    which is what makes the result conditional.
    """
    rng = np.random.default_rng(seed)
    z_pool = fit.resid_std
    mu = fit.mu if mu_override is None else mu_override * 100.0

    sigma2 = np.full(n_paths, fit.sigma2_last, dtype=float)
    total = np.zeros(n_paths, dtype=float)

    for _ in range(horizon):
        z = z_pool[rng.integers(0, z_pool.size, n_paths)]
        eps = np.sqrt(sigma2) * z
        total += mu + eps
        sigma2 = fit.omega + fit.alpha * eps**2 + fit.beta * sigma2

    return total / 100.0      # back to log-return units


def density_from_sample(sample, grid, bw="silverman"):
    return gaussian_kde(np.asarray(sample, float), bw_method=bw)(grid)


def to_price_density(x, S0, return_density_fn):
    """Map a density over log returns R to a density over S_T = S0 exp(R).

    p_S(x) = p_R(log(x/S0)) / x.
    """
    x = np.asarray(x, float)
    return return_density_fn(np.log(x / S0)) / x
