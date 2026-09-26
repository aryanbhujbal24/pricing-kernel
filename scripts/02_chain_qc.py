"""Part 3, step 1 -- what the real chain actually looks like.

Three questions, in order:

  1. Is the snapshot internally consistent? (put-call parity, implied
     rate and dividend yield, quote widths)
  2. Do the raw mid quotes satisfy static no-arbitrage? (convexity,
     monotonicity, bounds)
  3. What happens if we apply the Part 2 extractor directly to them?

The answer to (3) is the point. Part 2 predicted, from quote noise alone,
roughly what fraction of a naively extracted density would come out
negative. This script checks that prediction against live SPX quotes. If
it holds, the case for arbitrage-constrained smoothing in IV space stops
being a stylistic preference and becomes a measured necessity.

Run:  python scripts/02_chain_qc.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pk import iv as blk
from pk.bl import extract_density
from pk.data import build_slice, load_chain, load_index_history, load_vix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG, RES = os.path.join(ROOT, "doc", "figures"), os.path.join(ROOT, "results")
UP = "/mnt/user-data/uploads"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold"})
BLUE, ORANGE, GREEN, RED, PURPLE = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"

EXPIRIES = ["2026-10-16", "2026-11-20", "2026-12-18"]
MAIN = "2026-12-18"   # longest monthly: widest strikes, best-behaved kernel


# ---------------------------------------------------------------- diagnostics
def parity_table(df, meta):
    rows = []
    for e in EXPIRIES:
        s = build_slice(df, meta, e)
        # F = S exp((r - q) T)  =>  q = r - log(F/S)/T
        div_yield = s.rate - np.log(s.forward / s.spot) / s.T
        rows.append({
            "expiry": e,
            "T_years": s.T,
            "n_strikes": len(s.strikes),
            "K_min": s.strikes.min(),
            "K_max": s.strikes.max(),
            "forward": s.forward,
            "discount": s.discount,
            "implied_r": s.rate,
            "implied_div_yield": div_yield,
            "parity_rmse": s.parity_rmse,
            "n_parity_fit": s.n_used_for_parity,
        })
    return pd.DataFrame(rows)


def quote_width_stats(s):
    """Half-spread is the natural scale for the quote noise eps of Prop 2.5."""
    otm_is_put = s.strikes < s.forward
    spread = np.where(otm_is_put, s.put_spread, s.call_spread)
    price = s.otm()
    near = np.abs(np.log(s.strikes / s.forward)) < 0.10
    return {
        "median_half_spread_all": float(np.median(spread) / 2),
        "median_half_spread_near": float(np.median(spread[near]) / 2),
        "median_rel_spread_near": float(np.median((spread / np.maximum(price, 1e-9))[near])),
        "n_near": int(near.sum()),
    }


def arbitrage_violations(K, C):
    """Static no-arbitrage checks on a call curve, in raw quote order."""
    dK = np.diff(K)
    slope = np.diff(C) / dK
    # butterfly: convexity, i.e. non-decreasing slope
    butterfly_bad = int(np.sum(np.diff(slope) < 0))
    monotone_bad = int(np.sum(slope > 0))          # calls must fall in K
    slope_bad = int(np.sum(slope < -1.0))          # dC/dK >= -D >= -1
    return {
        "n_strikes": len(K),
        "butterfly_violations": butterfly_bad,
        "butterfly_frac": butterfly_bad / max(len(slope) - 1, 1),
        "monotonicity_violations": monotone_bad,
        "slope_bound_violations": slope_bad,
    }


def longest_uniform_run(K, tol=1e-9):
    """Largest contiguous block of strikes with constant spacing."""
    d = np.diff(K)
    best = (0, 0, None)
    i = 0
    while i < len(d):
        j = i
        while j + 1 < len(d) and abs(d[j + 1] - d[i]) < tol:
            j += 1
        if (j - i + 2) > best[1] - best[0] + 1:
            best = (i, j + 1, d[i])
        i = j + 1
    lo, hi, h = best
    return lo, hi, h


def naive_bl(s):
    """Apply the Part 2 extractor straight to raw mid quotes, and compare the
    damage with what Proposition 2.5 predicts from the quote width alone."""
    lo, hi, h = longest_uniform_run(s.strikes)
    K = s.strikes[lo:hi + 1]
    C = s.call_mid[lo:hi + 1]
    r = s.rate
    Ki, q_hat = extract_density(K, C, r, s.T)

    # Prop 2.5: noise sd in q_hat is e^{rT} sqrt(6) eps / h^2. Treat the
    # quote as uniform on [bid, ask], so eps = half_spread / sqrt(3).
    half = np.median(s.call_spread[lo:hi + 1]) / 2
    eps = half / np.sqrt(3)
    sd_pred = np.exp(r * s.T) * np.sqrt(6) * eps / h**2

    # Reference signal scale: peak of the lognormal implied by the ATM vol.
    # Using max(q_hat) instead would be circular -- on this grid q_hat is
    # almost entirely noise, so its peak measures the noise, not the density.
    atm = blk.implied_vol(np.interp(s.forward, s.strikes, s.call_mid),
                          s.forward, s.forward, s.discount, s.T, True)
    q_peak = 1.0 / (s.forward * atm * np.sqrt(2 * np.pi * s.T))

    from scipy.stats import norm as _n
    pred_frac = float(_n.cdf(-q_peak / sd_pred))

    # Invert the measured negative fraction to get the *effective* independent
    # noise. If this is well below eps, quote errors are positively correlated
    # across adjacent strikes -- market makers quote off a common smile -- and
    # the second difference differences some of the error away.
    meas_frac = float(np.mean(q_hat < 0))
    sd_eff = q_peak / max(-_n.ppf(min(meas_frac, 0.4999)), 1e-9)
    eps_eff = sd_eff * h**2 / (np.exp(r * s.T) * np.sqrt(6))

    return {
        "h": h, "n_used": len(K),
        "K_lo": K.min(), "K_hi": K.max(),
        "atm_iv": float(atm),
        "median_half_spread": float(half),
        "eps_from_spread": float(eps),
        "predicted_noise_sd": float(sd_pred),
        "lognormal_peak_density": float(q_peak),
        "noise_to_signal": float(sd_pred / q_peak),
        "frac_negative_measured": meas_frac,
        "frac_negative_predicted": pred_frac,
        "eps_effective_implied": float(eps_eff),
        "correlation_discount": float(eps_eff / eps),
    }, Ki, q_hat


# -------------------------------------------------------------------- figures
def fig_smile(df, meta):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = [BLUE, ORANGE, GREEN]
    for e, c in zip(EXPIRIES, colors):
        s = build_slice(df, meta, e)
        k = np.log(s.strikes / s.forward)
        is_call = s.strikes >= s.forward
        sig = blk.implied_vol_vector(s.otm(), s.forward, s.strikes, s.discount,
                                     s.T, is_call)
        ok = np.isfinite(sig)
        axes[0].plot(k[ok], sig[ok], ".", ms=3.5, color=c,
                     label=f"{e}  (T={s.T:.3f})")
        axes[1].plot(k[ok], sig[ok] ** 2 * s.T, ".", ms=3.5, color=c, label=e)

    axes[0].set_title("SPX implied volatility, OTM quotes")
    axes[0].set_ylabel(r"implied vol $\sigma$")
    axes[0].set_ylim(0, 1.2)
    axes[1].set_title("Total variance $\\sigma^2 T$ (calendar arbitrage check)")
    axes[1].set_ylabel(r"$\sigma^2 T$")
    for a in axes:
        a.set_xlabel(r"log-moneyness $k=\log(K/F)$")
        a.set_xlim(-1.2, 0.5)
        a.axvline(0, color="grey", lw=0.8, ls=":")
        a.legend(markerscale=3)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "04_spx_smile.png"), bbox_inches="tight")
    plt.close(fig)


def fig_parity(df, meta):
    s = build_slice(df, meta, MAIN)
    y = s.call_mid - s.put_mid
    fit = s.discount * (s.forward - s.strikes)
    near = np.abs(s.strikes / s.spot - 1) <= 0.05

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(s.strikes, y, ".", ms=3, color=BLUE, label="quoted $C-P$")
    axes[0].plot(s.strikes, fit, "-", color=RED, lw=1.2, label=r"$D(F-K)$ fit")
    axes[0].set_title(f"Put-call parity, {MAIN}")
    axes[0].set_ylabel("$C - P$")
    axes[0].legend()

    axes[1].plot(s.strikes[near], (y - fit)[near], ".", ms=5, color=PURPLE)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title(f"Residuals near ATM (RMSE {s.parity_rmse:.3f} pts)")
    axes[1].set_ylabel("residual (index points)")
    for a in axes:
        a.set_xlabel("strike $K$")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "05_parity.png"), bbox_inches="tight")
    plt.close(fig)


def fig_naive(Ki, q_hat, info):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(Ki, q_hat, lw=0.7, color=RED)
    ax.axhline(0, color="grey", lw=0.9)
    ax.set_title(f"Raw second difference of SPX mids ($h={info['h']:.0f}$)")
    ax.set_ylabel(r"$\hat q(K)$")

    ax = axes[1]
    ax.plot(Ki, q_hat, lw=0.7, color=RED, alpha=0.5, label="naive extraction")
    band = 3 * info["predicted_noise_sd"]
    ax.axhspan(-band, band, color=BLUE, alpha=0.15,
               label=r"$\pm 3\sigma$ predicted by Prop. 2.5")
    ax.axhline(0, color="grey", lw=0.9)
    ax.set_ylim(-4 * band, 4 * band)
    ax.set_title("Noise band predicted from quote width alone")
    ax.legend()

    for a in axes:
        a.set_xlabel("strike $K$")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.suptitle(
        f"{info['frac_negative_measured']:.0%} of the naively extracted density is negative",
        fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "06_naive_bl_failure.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    df, meta = load_chain(os.path.join(UP, "_SPXchain.json"))
    print(f"chain: {len(df)} contracts, spot {meta['spot']:.2f}, "
          f"quotes as of {meta['asof']}")

    pt = parity_table(df, meta)
    pt.to_csv(os.path.join(RES, "parity.csv"), index=False)
    print("\n--- put-call parity ---")
    print(pt.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    s = build_slice(df, meta, MAIN)
    ws = quote_width_stats(s)
    print(f"\n--- quote widths, {MAIN} ---")
    for k, v in ws.items():
        print(f"  {k:>26s}: {v:.5g}")

    av = arbitrage_violations(s.strikes, s.call_mid)
    pd.DataFrame([av]).to_csv(os.path.join(RES, "arbitrage_violations.csv"), index=False)
    print(f"\n--- static arbitrage in raw mids, {MAIN} ---")
    for k, v in av.items():
        print(f"  {k:>26s}: {v:.5g}")

    info, Ki, q_hat = naive_bl(s)
    pd.DataFrame([info]).to_csv(os.path.join(RES, "naive_bl.csv"), index=False)
    print(f"\n--- naive Breeden-Litzenberger on raw mids, {MAIN} ---")
    for k, v in info.items():
        print(f"  {k:>26s}: {v:.5g}")

    hist = load_index_history(os.path.join(UP, "sp500_history.csv"))
    vix = load_vix(os.path.join(UP, "VIX_History.csv"))
    joined = hist.join(vix[["CLOSE"]].rename(columns={"CLOSE": "VIX"}), how="inner")
    print(f"\n--- time series ---")
    print(f"  SPX {hist.index.min().date()} to {hist.index.max().date()}, "
          f"{len(hist)} rows")
    print(f"  VIX {vix.index.min().date()} to {vix.index.max().date()}, "
          f"{len(vix)} rows")
    print(f"  overlap usable for VRP: {len(joined)} days "
          f"({joined.index.min().date()} to {joined.index.max().date()})")

    fig_smile(df, meta)
    fig_parity(df, meta)
    fig_naive(Ki, q_hat, info)
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
