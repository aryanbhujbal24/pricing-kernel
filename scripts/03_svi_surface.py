"""Part 3 -- the arbitrage-free surface and the risk-neutral density.

The December 2026 slice only, per scope. The chain gives OTM mid quotes;
we convert to implied vol, fit raw SVI in total-variance space under hard
no-arbitrage constraints, and read the density off the fitted slice
analytically.

The comparison that matters is against script 02: the same data, the same
expiry, differenced naively, produced a density that was 39% negative with
a noise-to-signal ratio of 47. Everything here is an argument that the
constrained fit fixes that without hiding anything.

Run:  python scripts/03_svi_surface.py
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
from pk import svi
from pk.bl import extract_density
from pk.data import build_slice, load_chain

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG, RES = os.path.join(ROOT, "doc", "figures"), os.path.join(ROOT, "results")
UP = "/mnt/user-data/uploads"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold"})
BLUE, ORANGE, GREEN, RED, PURPLE = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"

EXPIRY = "2026-12-18"
MIN_PRICE = 1.00      # below a dollar the quote is tick noise, not information
MAX_REL_SPREAD = 0.15  # relative quote width; wider means no usable signal
K_WIDE = (1000.0, 20000.0)   # integration range for moment diagnostics


def prepare(s, apply_filter=True):
    """OTM quotes -> implied vol -> total variance, with precision weights.

    The reliability filter is a stated modelling choice, not cleaning. The
    December chain quotes strikes down to k = -1.72, which at this maturity
    is roughly 26 standard deviations out: those contracts trade near fifty
    cents with relative spreads above 30%, and their implied vols are an
    artefact of the tick size. Including them does not add tail information,
    it adds noise with enough leverage to distort the whole slice.
    """
    K = s.strikes
    is_call = K >= s.forward
    price = s.otm()
    spread = np.where(is_call, s.call_spread, s.put_spread)

    sig = blk.implied_vol_vector(price, s.forward, K, s.discount, s.T, is_call)

    keep = np.isfinite(sig) & (price > 0) & (spread > 0)
    if apply_filter:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = spread / np.where(price > 0, price, np.nan)
        keep &= (price >= MIN_PRICE) & (rel <= MAX_REL_SPREAD)

    K, price, spread, sig, is_call = (
        K[keep], price[keep], spread[keep], sig[keep], is_call[keep])

    k = np.log(K / s.forward)
    w = sig**2 * s.T

    # Propagate the quote width into total-variance units:
    #   d(sigma) = dC / vega,  and  w = sigma^2 T  =>  dw = 2 sigma T d(sigma).
    # Weight by inverse variance, so wide or low-vega quotes count for less.
    vega = blk.vega(s.forward, K, s.discount, sig, s.T)
    dsig = (spread / 2) / np.maximum(vega, 1e-8)
    dw = 2 * sig * s.T * dsig
    weights = 1.0 / np.maximum(dw, 1e-10) ** 2

    return pd.DataFrame({
        "K": K, "k": k, "price": price, "spread": spread,
        "iv": sig, "w": w, "dw": dw, "weight": weights, "is_call": is_call,
    })


def moment_diagnostics(p, F, D, T, lo=K_WIDE[0], hi=K_WIDE[1], n=200_000):
    """Integrate the fitted density over a very wide range.

    Mass should be 1 and the mean should be the forward. Both are checks on
    the fit, not on the market: SVI is a five-parameter family and nothing
    forces it to integrate correctly, so if it does, the fit is sane.
    """
    K = np.linspace(lo, hi, n)
    q = svi.density_K(K, F, p)
    mass = np.trapezoid(q, K)
    mean = np.trapezoid(K * q, K)
    var = np.trapezoid((K - mean) ** 2 * q, K)
    m3 = np.trapezoid((K - mean) ** 3 * q, K)
    m4 = np.trapezoid((K - mean) ** 4 * q, K)
    return {
        "mass": float(mass),
        "mean": float(mean),
        "forward": float(F),
        "mean_minus_forward": float(mean - F),
        "mean_rel_error": float(mean / F - 1),
        "std": float(np.sqrt(var)),
        "implied_vol_from_std": float(np.sqrt(np.log(1 + var / mean**2) / T)),
        "skew": float(m3 / var**1.5),
        "excess_kurtosis": float(m4 / var**2 - 3),
    }


def truncation_sensitivity(p, F, D, T, quoted_lo, quoted_hi):
    """How much probability mass lives outside the quoted strike range?

    This is the wing-extrapolation limitation named in the paper spec. SVI
    extrapolates past the last quote by construction; the honest thing is to
    say how much of the answer rests on that extrapolation.
    """
    rows = []
    for lo, hi, label in [
        (quoted_lo, quoted_hi, "quoted range"),
        (0.5 * quoted_lo, 1.5 * quoted_hi, "1.5x wider"),
        (K_WIDE[0], K_WIDE[1], "effectively full"),
    ]:
        K = np.linspace(max(lo, 1.0), hi, 100_000)
        q = svi.density_K(K, F, p)
        rows.append({"range": label, "K_lo": lo, "K_hi": hi,
                     "mass": float(np.trapezoid(q, K))})
    df = pd.DataFrame(rows)
    full = df.loc[df["range"] == "effectively full", "mass"].iloc[0]
    df["mass_fraction_of_full"] = df["mass"] / full
    return df


# -------------------------------------------------------------------- figures
def fig_fit(d, p, s, diag):
    kk = np.linspace(d.k.min() - 0.15, d.k.max() + 0.15, 600)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    ax.plot(d.k, d.iv, ".", ms=3.5, color=BLUE, label="market (OTM mids)")
    ax.plot(kk, np.sqrt(svi.total_variance(kk, p) / s.T), "-", color=RED, lw=1.8,
            label="SVI fit")
    ax.set_ylabel(r"implied vol $\sigma$")
    ax.set_title(f"SPX {EXPIRY} slice, $T={s.T:.3f}$")
    ax.legend(markerscale=3)

    ax = axes[1]
    resid_iv = (d.iv - np.sqrt(svi.total_variance(d.k, p) / s.T)) * 1e4
    halfspread_iv = (d.dw / (2 * d.iv * s.T)) * 1e4
    ax.plot(d.k, resid_iv, ".", ms=3.5, color=PURPLE, label="fit residual")
    ax.plot(d.k, halfspread_iv, "-", color="grey", lw=1, alpha=0.7,
            label="quote half-width")
    ax.plot(d.k, -halfspread_iv, "-", color="grey", lw=1, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylim(-400, 400)
    ax.set_ylabel("vol points (bp)")
    ax.set_title("Residuals vs quote width")
    ax.legend()

    ax = axes[2]
    g = svi.g_function(kk, p)
    ax.plot(kk, g, "-", color=GREEN, lw=1.8)
    ax.axhline(0, color=RED, lw=1.2, ls="--", label="butterfly boundary")
    ax.set_ylim(bottom=min(0, g.min() * 1.1))
    ax.set_ylabel("$g(k)$")
    ax.set_title(f"Butterfly condition (min $g$ = {diag['min_g_on_grid']:.3f})")
    ax.legend()

    for a in axes:
        a.set_xlabel(r"$k = \log(K/F)$")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "07_svi_fit.png"), bbox_inches="tight")
    plt.close(fig)


def fig_density(p, s, d):
    K = np.linspace(2000, 14000, 4000)
    q = svi.density_K(K, s.forward, p)

    # the naive extraction from script 02, for contrast
    from pk.data import build_slice as _bs  # noqa
    lo, hi, h = _longest_uniform(s.strikes)
    Kn = s.strikes[lo:hi + 1]
    Cn = s.call_mid[lo:hi + 1]
    Kni, qn = extract_density(Kn, Cn, s.rate, s.T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(Kni, qn, lw=0.6, color=RED, alpha=0.55,
            label="naive second difference")
    ax.plot(K, q, "-", color=BLUE, lw=2.2, label="SVI, arbitrage-free")
    ax.axvline(s.forward, color="grey", ls=":", lw=1)
    ax.set_xlim(6800, 8400)
    ax.set_ylim(-0.02, 0.02)
    ax.set_title("Same quotes, two estimators")
    ax.set_ylabel("$q(K)$")
    ax.legend()

    ax = axes[1]
    ax.semilogy(K, np.maximum(q, 1e-12), "-", color=BLUE, lw=2)
    ax.axvline(s.strikes.min(), color=ORANGE, ls="--", lw=1.2,
               label="quoted strike range")
    ax.axvline(s.strikes.max(), color=ORANGE, ls="--", lw=1.2)
    ax.axvline(s.forward, color="grey", ls=":", lw=1, label="forward")
    ax.set_ylim(1e-11, 1e-2)
    ax.set_title("Full density, log scale (wings are extrapolated)")
    ax.set_ylabel("$q(K)$, log scale")
    ax.legend()

    for a in axes:
        a.set_xlabel("strike $K$")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "08_risk_neutral_density.png"), bbox_inches="tight")
    plt.close(fig)


def _longest_uniform(K, tol=1e-9):
    d = np.diff(K)
    best, i = (0, 0, None), 0
    while i < len(d):
        j = i
        while j + 1 < len(d) and abs(d[j + 1] - d[i]) < tol:
            j += 1
        if (j - i + 2) > best[1] - best[0] + 1:
            best = (i, j + 1, d[i])
        i = j + 1
    return best


# ------------------------------------------------------------------------ main
def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    df, meta = load_chain(os.path.join(UP, "_SPXchain.json"))
    s = build_slice(df, meta, EXPIRY)
    d_all = prepare(s, apply_filter=False)
    d = prepare(s, apply_filter=True)
    print(f"{EXPIRY}: T={s.T:.4f}  F={s.forward:.2f}  D={s.discount:.6f}  "
          f"r={s.rate:.4%}")
    print(f"quotes with a valid IV: {len(d_all)}  -> after reliability filter: "
          f"{len(d)}  ({len(d_all) - len(d)} dropped)")
    print(f"  fitted k range [{d.k.min():.3f}, {d.k.max():.3f}]  "
          f"(all quotes span [{d_all.k.min():.3f}, {d_all.k.max():.3f}])")

    p, diag = fit_svi_slice(d)
    print("\n--- SVI fit ---")
    print(f"  a={p.a:.6f}  b={p.b:.6f}  rho={p.rho:.4f}  m={p.m:.6f}  s={p.s:.6f}")
    for k_, v in diag.items():
        print(f"  {k_:>24s}: {v:.6g}")

    # residuals in vol points, and how they compare with the quote width
    iv_fit = np.sqrt(svi.total_variance(d.k.to_numpy(), p) / s.T)
    resid_bp = (d.iv.to_numpy() - iv_fit) * 1e4
    halfwidth_bp = (d.dw.to_numpy() / (2 * d.iv.to_numpy() * s.T)) * 1e4
    inside = float(np.mean(np.abs(resid_bp) <= halfwidth_bp))
    rmse_bp = float(np.sqrt(np.mean(resid_bp**2)))
    print(f"  rmse in vol points: {rmse_bp:.1f} bp")
    print(f"  median quote half-width: {np.median(halfwidth_bp):.1f} bp")
    print(f"  fraction of quotes fitted inside their own bid-ask: {inside:.1%}")
    print(f"  ATM skew dw/dk at k=0: {svi.dw(0.0, p):.4f}")

    kk = d.k.to_numpy()
    regions = [("deep left  k<-0.5", kk < -0.5),
               ("left  -0.5<=k<-0.15", (kk >= -0.5) & (kk < -0.15)),
               ("core  |k|<=0.15", np.abs(kk) <= 0.15)]
    print("\n  error by region:")
    reg_rows = []
    for name, sel in regions:
        if sel.sum() == 0:
            continue
        rr = float(np.sqrt(np.mean(resid_bp[sel] ** 2)))
        ii = float(np.mean(np.abs(resid_bp[sel]) <= halfwidth_bp[sel]))
        print(f"    {name:>22s}: n={sel.sum():3d}  rmse={rr:6.1f} bp  inside={ii:5.1%}")
        reg_rows.append({"region": name, "n": int(sel.sum()),
                         "rmse_bp": rr, "frac_inside": ii})
    pd.DataFrame(reg_rows).to_csv(os.path.join(RES, "svi_fit_by_region.csv"),
                                  index=False)

    mom = moment_diagnostics(p, s.forward, s.discount, s.T)
    print("\n--- fitted density moments ---")
    for k_, v in mom.items():
        print(f"  {k_:>24s}: {v:.6g}")

    trunc = truncation_sensitivity(p, s.forward, s.discount, s.T,
                                   s.strikes.min(), s.strikes.max())
    print("\n--- wing extrapolation ---")
    print(trunc.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    out = {"expiry": EXPIRY, "T": s.T, "forward": s.forward,
           "discount": s.discount, "rate": s.rate,
           "a": p.a, "b": p.b, "rho": p.rho, "m": p.m, "s": p.s,
           "rmse_vol_bp": rmse_bp,
           "median_halfwidth_bp": float(np.median(halfwidth_bp)),
           "n_quotes_fitted": len(d), "n_quotes_dropped": len(d_all) - len(d),
           "k_lo_fit": float(d.k.min()), "k_hi_fit": float(d.k.max()),
           "frac_inside_spread": inside, **diag, **mom}
    pd.DataFrame([out]).to_csv(os.path.join(RES, "svi_fit.csv"), index=False)
    trunc.to_csv(os.path.join(RES, "wing_truncation.csv"), index=False)

    # the density itself, for Part 4
    Kg = np.linspace(2000, 16000, 14001)
    pd.DataFrame({"K": Kg, "q": svi.density_K(Kg, s.forward, p)}).to_csv(
        os.path.join(RES, "risk_neutral_density.csv"), index=False)

    fig_fit(d, p, s, diag)
    fig_density(p, s, d)
    print(f"\nfigures written to {FIG}")


def fit_svi_slice(d):
    return svi.fit_svi(d.k.to_numpy(), d.w.to_numpy(),
                       weights=d.weight.to_numpy(), n_starts=16)


if __name__ == "__main__":
    main()
