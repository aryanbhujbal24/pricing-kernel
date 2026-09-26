"""Part 5 -- model-free implied variance, the jump correction, and the VRP.

Four things:

  1. Reconstruct VIX from our own chain using the formula derived in
     Section 3.3. This validates the derivation *and* licenses the use of
     the published VIX history as a model-free implied variance series --
     which is what makes the 1990-2026 time series possible from a single
     option snapshot.

  2. Quantify the jump correction of Section 3.4 on a fitted ~30-day slice,
     so the measured implied variance can be compared with expected
     quadratic variation rather than silently conflated with it.

  3. Build the VRP series and test whether its mean is non-zero, with
     Newey-West standard errors. The windows overlap by 20 of 21 days, so
     the naive t-statistic is meaningless.

  4. Look at the shape of the distribution, because a t-test on the mean of
     a heavily left-skewed series is a statement about an average whose
     economics are dominated by the tail.

Run:  python scripts/05_variance_risk_premium.py
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pk import iv as blk
from pk import svi
from pk import varswap as vs
from pk.data import build_slice, load_chain, load_index_history, load_vix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG, RES = os.path.join(ROOT, "doc", "figures"), os.path.join(ROOT, "results")
UP = "/mnt/user-data/uploads"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold"})
BLUE, ORANGE, GREEN, RED, PURPLE = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"

NEAR, NEXT = "2026-10-23", "2026-10-30"    # bracket 30 days on the snapshot date
RV_WINDOW = 21                             # sessions ~ 30 calendar days
PUBLISHED_VIX = 14.87


def slice_for_vix(df, asof, expiry, root, r):
    e = pd.Timestamp(expiry)
    sub = df[(df["expiry"] == e) & (df["root"] == root)]
    c = sub[sub.cp == "C"].set_index("strike").sort_index()
    p = sub[sub.cp == "P"].set_index("strike").sort_index()
    K = c.index.intersection(p.index)
    days = (e - asof).days
    out = vs.cboe_variance(K.values, c.loc[K, "mid"].values, p.loc[K, "mid"].values,
                           days / 365, r,
                           call_bid=c.loc[K, "bid"].values,
                           put_bid=p.loc[K, "bid"].values)
    return out, days


def fit_slice_svi(df, meta, expiry, root):
    """Reuse the Part 3 pipeline on an arbitrary expiry."""
    m = importlib.import_module("03_svi_surface")
    s = build_slice(df, meta, expiry, root=root)
    d = m.prepare(s, apply_filter=True)
    p, diag = svi.fit_svi(d.k.to_numpy(), d.w.to_numpy(),
                          weights=d.weight.to_numpy(), n_starts=16)
    return s, p, diag, d


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    df, meta = load_chain(os.path.join(UP, "_SPXchain.json"))
    asof = meta["session"]
    r_short = 0.0429     # 1M implied rate from the Part 3 parity table

    # ---------------------------------------------------- 1. reproduce the VIX
    near, d_near = slice_for_vix(df, asof, NEAR, "SPXW", r_short)
    nxt, d_next = slice_for_vix(df, asof, NEXT, "SPXW", r_short)
    blend = vs.interpolate_vix(near, nxt, d_near, d_next)
    err = blend["vix"] - PUBLISHED_VIX
    print("--- VIX reconstruction from our own chain ---")
    for lab, o, dd in [(NEAR, near, d_near), (NEXT, nxt, d_next)]:
        print(f"  {lab}: {dd}d  F={o['F']:.2f}  K0={o['K0']:.0f}  "
              f"n={o['n_options']}  K=[{o['K_lo']:.0f},{o['K_hi']:.0f}]  "
              f"sigma={o['sigma']:.4f}  correction={o['correction_term']:.3e}")
    print(f"  reconstructed VIX = {blend['vix']:.3f}   published = {PUBLISHED_VIX}"
          f"   error = {err:+.3f} ({err/PUBLISHED_VIX:+.2%})")

    # truncation sensitivity of the sum: how much comes from the deep wings?
    trunc = wing_contribution(df, asof, NEXT, "SPXW", r_short, d_next)
    print("\n  contribution by moneyness band (next-term expiry):")
    print("  " + trunc.to_string(index=False,
                                 float_format=lambda v: f"{v:.5g}").replace("\n", "\n  "))

    # ------------------------------------------------- 2. the jump correction
    s30, p30, diag30, _ = fit_slice_svi(df, meta, NEXT, "SPXW")
    Kg = np.linspace(1500, 20000, 60_000)
    q30 = svi.density_K(Kg, s30.forward, p30)
    jm = vs.risk_neutral_moments(Kg, q30, s30.forward, s30.T)
    print(f"\n--- jump correction, {NEXT} slice (T={s30.T:.4f}) ---")
    for k_, v_ in jm.items():
        print(f"  {k_:>30s}: {v_:.6g}")
    print(f"  implied vol from log-contract variance: "
          f"{np.sqrt(jm['implied_variance_annualised']):.4f}")
    corr_var = jm["implied_variance_annualised"] - jm["jump_correction_annualised"]
    print(f"  jump-corrected expected QV (annualised): {corr_var:.6f} "
          f"=> vol {np.sqrt(max(corr_var,0)):.4f}")

    # ------------------------------------------------------ 3. the VRP series
    hist = load_index_history(os.path.join(UP, "sp500_history.csv"))
    vix = load_vix(os.path.join(UP, "VIX_History.csv"))
    rv = vs.realized_variance(hist["log_ret"].dropna(), RV_WINDOW)

    dat = pd.DataFrame({"vix": vix["CLOSE"]}).join(rv.rename("rv"), how="inner").dropna()
    dat["iv2"] = (dat["vix"] / 100.0) ** 2
    dat["vrp"] = dat["iv2"] - dat["rv"]
    # in volatility points, which is how practitioners quote it
    dat["vrp_vol_pts"] = (np.sqrt(dat["iv2"]) - np.sqrt(dat["rv"])) * 100

    print(f"\n--- variance risk premium, {dat.index.min().date()} to "
          f"{dat.index.max().date()} ---")
    print(f"  n = {len(dat)} overlapping daily observations, "
          f"window = {RV_WINDOW} sessions")
    print(f"  effective independent observations ~ {len(dat)/RV_WINDOW:.0f}")
    print(f"  mean IV^2 = {dat.iv2.mean():.5f}  (vol {np.sqrt(dat.iv2.mean()):.3f})")
    print(f"  mean RV^2 = {dat.rv.mean():.5f}  (vol {np.sqrt(dat.rv.mean()):.3f})")

    auto = vs.andrews_bandwidth(dat["vrp"].to_numpy())
    rows = []
    for lags, label in [(0, "naive (no HAC)"), (RV_WINDOW, "NW, lag = window"),
                        (2 * RV_WINDOW, "NW, lag = 2x window"),
                        (auto, f"NW, Andrews plug-in ({auto})")]:
        st = vs.newey_west_mean(dat["vrp"].to_numpy(), max(lags, 1))
        if lags == 0:
            st = {**st, "se_nw": st["se_naive"], "t_nw": st["t_naive"], "lags": 0}
        rows.append({"spec": label, "lags": st["lags"], "mean": st["mean"],
                     "se": st["se_nw"], "t_stat": st["t_nw"]})
    hac = pd.DataFrame(rows)
    print("\n  mean VRP and standard errors:")
    print("  " + hac.to_string(index=False,
                               float_format=lambda v: f"{v:.5g}").replace("\n", "\n  "))

    # ------------------------------------------------- 4. shape of the series
    nw = vs.newey_west_mean(dat["vrp"].to_numpy(), RV_WINDOW)
    shape = {
        "mean": dat.vrp.mean(), "median": dat.vrp.median(),
        "sd": dat.vrp.std(), "skewness": dat.vrp.skew(),
        "excess_kurtosis": dat.vrp.kurt(),
        "frac_positive": float((dat.vrp > 0).mean()),
        "p01": dat.vrp.quantile(0.01), "p99": dat.vrp.quantile(0.99),
        "min": dat.vrp.min(), "max": dat.vrp.max(),
        "min_date": str(dat.vrp.idxmin().date()),
        "mean_vol_points": dat.vrp_vol_pts.mean(),
    }
    print("\n  distribution of the VRP:")
    for k_, v_ in shape.items():
        print(f"    {k_:>18s}: {v_}")

    # how much of the mean comes from the worst 1% of days?
    srt = dat.vrp.sort_values()
    n1 = max(int(0.01 * len(srt)), 1)
    tail_contrib = srt.iloc[:n1].sum() / dat.vrp.sum()
    print(f"    worst 1% of days contribute {tail_contrib:+.1%} of the total")

    # subsamples
    sub_rows = []
    for a, b in [("1990", "1999"), ("2000", "2009"), ("2010", "2019"), ("2020", "2026")]:
        w = dat.loc[a:b, "vrp"]
        st = vs.newey_west_mean(w.to_numpy(), RV_WINDOW)
        sub_rows.append({"period": f"{a}-{b}", "n": st["n"], "mean": st["mean"],
                         "se_nw": st["se_nw"], "t_nw": st["t_nw"],
                         "frac_positive": float((w > 0).mean())})
    subs = pd.DataFrame(sub_rows)
    print("\n  by decade:")
    print("  " + subs.to_string(index=False,
                                float_format=lambda v: f"{v:.5g}").replace("\n", "\n  "))

    # ------------------------------------------------------------------ save
    pd.DataFrame([{
        "reconstructed_vix": blend["vix"], "published_vix": PUBLISHED_VIX,
        "abs_error": err, "rel_error": err / PUBLISHED_VIX,
        "near_expiry": NEAR, "near_days": d_near, "near_sigma": near["sigma"],
        "next_expiry": NEXT, "next_days": d_next, "next_sigma": nxt["sigma"],
        "w_near": blend["w_near"], "w_next": blend["w_next"],
    }]).to_csv(os.path.join(RES, "vix_reconstruction.csv"), index=False)
    trunc.to_csv(os.path.join(RES, "vix_wing_contribution.csv"), index=False)
    pd.DataFrame([jm]).to_csv(os.path.join(RES, "jump_correction.csv"), index=False)
    hac.to_csv(os.path.join(RES, "vrp_hac.csv"), index=False)
    subs.to_csv(os.path.join(RES, "vrp_by_decade.csv"), index=False)
    pd.DataFrame([{**shape, "n": len(dat), "n_eff": len(dat) / RV_WINDOW,
                   "se_nw": nw["se_nw"], "t_nw": nw["t_nw"],
                   "se_naive": nw["se_naive"], "t_naive": nw["t_naive"],
                   "se_inflation": nw["se_inflation"],
                   "tail_1pct_share": tail_contrib}]).to_csv(
        os.path.join(RES, "vrp_summary.csv"), index=False)
    dat.to_csv(os.path.join(RES, "vrp_series.csv"))

    # ------------------------------------- 5. what the jump correction implies
    corr_frac = jm["correction_pct_of_variance"] / 100
    implied_corr_at_mean = corr_frac * dat.iv2.mean()
    print("\n--- jump correction against the measured premium ---")
    print(f"  correction is {corr_frac:.1%} of implied variance on the snapshot date")
    print(f"  at the sample mean IV^2 ({dat.iv2.mean():.5f}) that is "
          f"{implied_corr_at_mean:.5f} in variance units")
    print(f"  measured mean VRP = {dat.vrp.mean():.5f}")
    print(f"  => the correction is {implied_corr_at_mean/dat.vrp.mean():.0%} "
          f"of the measured premium")
    print("  (extrapolated from one cross-section, and an upper bound: the whole\n"
          "   third moment is attributed to jumps, so a leverage-driven diffusion\n"
          "   would need less of a correction)")

    make_figures(dat, hac, nw)
    print(f"\nfigures written to {FIG}")


def wing_contribution(df, asof, expiry, root, r, days):
    """Share of the variance sum coming from each moneyness band.

    The 1/K^2 weight loads on deep OTM puts, which is exactly where a chain
    runs out of quotes. Showing the split is how the truncation limitation
    gets a number instead of a sentence.
    """
    e = pd.Timestamp(expiry)
    sub = df[(df["expiry"] == e) & (df["root"] == root)]
    c = sub[sub.cp == "C"].set_index("strike").sort_index()
    p = sub[sub.cp == "P"].set_index("strike").sort_index()
    K = np.asarray(c.index.intersection(p.index), float)
    C = c.loc[K, "mid"].to_numpy()
    P = p.loc[K, "mid"].to_numpy()
    T = days / 365
    i_star = int(np.argmin(np.abs(C - P)))
    F = K[i_star] + np.exp(r * T) * (C[i_star] - P[i_star])
    K0 = K[K <= F].max()
    O = np.where(K < K0, P, np.where(K > K0, C, 0.5 * (C[i_star] + P[i_star])))
    ok = O > 0
    K, O = K[ok], O[ok]
    dK = np.empty_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    contrib = (dK / K**2) * np.exp(r * T) * O
    m = K / F
    bands = [(0.0, 0.7, "K/F < 0.70"), (0.7, 0.85, "0.70-0.85"),
             (0.85, 0.95, "0.85-0.95"), (0.95, 1.05, "0.95-1.05"),
             (1.05, 10.0, "K/F > 1.05")]
    tot = contrib.sum()
    rows = []
    for lo, hi, lab in bands:
        sel = (m >= lo) & (m < hi)
        rows.append({"band": lab, "n": int(sel.sum()),
                     "share_of_sum": float(contrib[sel].sum() / tot)})
    return pd.DataFrame(rows)


def make_figures(dat, hac, nw):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    ax.plot(dat.index, np.sqrt(dat.iv2) * 100, color=BLUE, lw=0.7,
            label="implied vol (VIX)")
    ax.plot(dat.index, np.sqrt(dat.rv) * 100, color=ORANGE, lw=0.7, alpha=0.8,
            label="subsequent realised vol")
    ax.set_ylabel("annualised vol (%)")
    ax.set_title("Implied vs subsequently realised volatility, 1990-2026")
    ax.legend()

    ax = axes[1]
    ax.fill_between(dat.index, 0, dat.vrp, where=dat.vrp >= 0, color=GREEN,
                    alpha=0.7, lw=0)
    ax.fill_between(dat.index, 0, dat.vrp, where=dat.vrp < 0, color=RED,
                    alpha=0.8, lw=0)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(dat.vrp.mean(), color="black", lw=1, ls="--",
               label=f"mean {dat.vrp.mean():.4f}")
    ax.set_ylabel(r"$\mathrm{IV}^2 - \mathrm{RV}^2$")
    ax.set_title("Variance risk premium: small and positive, then violently negative")
    ax.legend()

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "11_vrp_timeseries.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(dat.vrp, bins=180, color=BLUE, alpha=0.85, edgecolor="none")
    ax.axvline(0, color="grey", lw=1)
    ax.axvline(dat.vrp.mean(), color=RED, lw=1.5, ls="--",
               label=f"mean {dat.vrp.mean():.4f}")
    ax.axvline(dat.vrp.median(), color=GREEN, lw=1.5, ls="--",
               label=f"median {dat.vrp.median():.4f}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{IV}^2 - \mathrm{RV}^2$")
    ax.set_ylabel("count (log)")
    ax.set_title(f"Left-skewed: skew {dat.vrp.skew():.1f}, "
                 f"{(dat.vrp>0).mean():.0%} positive")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(dat.rv, dat.iv2, s=2, alpha=0.15, color=BLUE)
    lim = [0, min(dat.rv.max(), dat.iv2.max()) * 1.05]
    ax.plot(lim, lim, color=RED, lw=1.3, ls="--", label="45 degrees")
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel(r"realised variance $\mathrm{RV}^2$")
    ax.set_ylabel(r"implied variance $\mathrm{IV}^2$")
    ax.set_title("Above the line most days, far below it occasionally")
    ax.legend()

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "12_vrp_distribution.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
