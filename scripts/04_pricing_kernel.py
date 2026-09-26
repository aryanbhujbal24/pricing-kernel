"""Part 4 -- the physical density and the pricing kernel.

    M(x) = e^{-rT} q(x) / p(x)

q comes from the arbitrage-free SVI slice of Part 3. p is estimated two
ways, and the comparison is the point: an unconditional KDE on overlapping
returns answers a different question from the one q answers, and the
difference shows up directly in the kernel.

The result everyone looks for is monotonicity. Theory (Section 1.3 of the
note) says M should decrease in wealth. The literature routinely finds it
does not -- the pricing kernel puzzle. This script reports what we get and,
more usefully, how much of the answer is pinned down by the data versus
assumed through the drift.

Run:  python scripts/04_pricing_kernel.py
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pk import physical as phys
from pk import svi
from pk.data import build_slice, load_chain, load_index_history

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG, RES = os.path.join(ROOT, "doc", "figures"), os.path.join(ROOT, "results")
UP = "/mnt/user-data/uploads"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold"})
BLUE, ORANGE, GREEN, RED, PURPLE = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"

EXPIRY = "2026-12-18"
HORIZON = 59              # NYSE sessions from 2026-09-25 to 2026-12-18
SAMPLE_START = "1990-01-01"   # matches the VIX era used in Part 5
N_PATHS = 200_000
N_BOOT = 120
BOOT_PATHS = 20_000


def get_q():
    """Refit the Part 3 slice so this script stands alone."""
    import importlib
    m = importlib.import_module("03_svi_surface")
    df, meta = load_chain(os.path.join(UP, "_SPXchain.json"))
    s = build_slice(df, meta, EXPIRY)
    d = m.prepare(s, apply_filter=True)
    p, diag = m.fit_svi_slice(d)
    return s, p, diag


def kernel(x, q_vals, p_vals, r, T, p_floor=1e-12):
    """M = e^{-rT} q/p, with p floored.

    Where the KDE returns exactly zero the ratio is undefined, not infinite:
    those are regions the historical sample never visited, so the kernel is
    simply not identified there. Returning nan keeps them out of the plots
    and out of the monotonicity count instead of letting a division blow up.
    """
    M = np.full_like(q_vals, np.nan, dtype=float)
    ok = p_vals > p_floor
    M[ok] = np.exp(-r * T) * q_vals[ok] / p_vals[ok]
    return M


def monotonicity_report(x, M, lo, hi):
    """Where, inside the identified region, is the kernel increasing?"""
    sel = (x >= lo) & (x <= hi)
    sel = sel & np.isfinite(M)
    xs, Ms = x[sel], M[sel]
    if xs.size < 3:
        return {"frac_increasing": np.nan, "region_lo": float(lo),
                "region_hi": float(hi), "n_points": int(xs.size)}
    dM = np.gradient(Ms, xs)
    up = dM > 0
    frac = float(np.mean(up))
    out = {"frac_increasing": frac, "n_points": int(xs.size),
           "region_lo": float(lo), "region_hi": float(hi)}
    if up.any():
        out["increasing_from"] = float(xs[up].min())
        out["increasing_to"] = float(xs[up].max())
        # largest contiguous increasing run
        idx = np.flatnonzero(up)
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        big = max(splits, key=len)
        out["longest_run_lo"] = float(xs[big].min())
        out["longest_run_hi"] = float(xs[big].max())
        out["longest_run_pts"] = int(len(big))
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    s, sp, _ = get_q()
    S0, F, T, r = s.spot, s.forward, s.T, s.rate
    print(f"{EXPIRY}: S0={S0:.2f} F={F:.2f} T={T:.4f} r={r:.4%} "
          f"horizon={HORIZON} sessions")

    hist = load_index_history(os.path.join(UP, "sp500_history.csv"))
    close = hist["Close"]
    daily = hist["log_ret"].dropna()
    daily90 = daily[daily.index >= pd.Timestamp(SAMPLE_START)]

    # ---------------------------------------------------------- physical, naive
    ov = phys.overlapping_returns(close[close.index >= pd.Timestamp(SAMPLE_START)],
                                  HORIZON)
    n_eff = phys.effective_sample_size(len(ov), HORIZON)
    kde_naive = phys.naive_kde(ov.to_numpy())
    print(f"\n--- naive, unconditional ---")
    print(f"  overlapping {HORIZON}-day returns: n={len(ov)}  n_eff={n_eff:.0f}")
    print(f"  mean={ov.mean():.4f}  sd={ov.std():.4f}  "
          f"annualised vol={ov.std()*np.sqrt(252/HORIZON):.3f}")

    # ---------------------------------------------------------- physical, GARCH
    fit = phys.fit_garch(daily90)
    n_years = len(daily90) / 252
    MU_HAT_ANN = float(daily90.mean() * 252)
    # Base case uses the plain sample mean as the physical drift, passed
    # explicitly. The GARCH constant-mean estimate is reported separately: it
    # comes out far higher because GARCH weights low-variance days most, and
    # low-variance days are bull markets. That is an estimator artefact, not
    # a better reading of the equity premium.
    R = phys.fhs_simulate(fit, HORIZON, n_paths=N_PATHS, seed=1,
                          mu_override=MU_HAT_ANN / 252)
    print(f"\n--- GARCH(1,1)-t filtered historical simulation ---")
    print(f"  omega={fit.omega:.5f} alpha={fit.alpha:.4f} beta={fit.beta:.4f} "
          f"persistence={fit.persistence:.5f} nu={fit.nu:.2f}")
    print(f"  unconditional vol={fit.uncond_vol_ann:.3f}  "
          f"today's conditional vol={fit.current_vol_ann:.3f}")
    print(f"  GARCH constant mean={fit.mu*252/100:.2%}/yr vs sample mean="
          f"{MU_HAT_ANN:.2%}/yr (base case uses the sample mean)")
    print(f"  simulated {HORIZON}-day: mean={R.mean():.4f} sd={R.std():.4f} "
          f"annualised vol={R.std()*np.sqrt(252/HORIZON):.3f}")
    print(f"  skew={pd.Series(R).skew():.3f} "
          f"kurtosis={pd.Series(R).kurt()+3:.2f}")

    # ------------------------------------------------------------------ grids
    x = np.linspace(5000, 11000, 3000)
    q = svi.density_K(x, F, sp)
    logm = np.log(x / S0)
    p_naive = kde_naive(logm) / x
    p_fhs = phys.density_from_sample(R, logm) / x

    # identified region: central 90% of q, inside the quoted strike range
    cdf = np.cumsum(q) * (x[1] - x[0])
    cdf /= cdf[-1]
    lo = float(np.interp(0.05, cdf, x))
    hi = float(np.interp(0.95, cdf, x))
    lo = max(lo, s.strikes.min())
    hi = min(hi, s.strikes.max())
    print(f"\nidentified region (central 90% of q): [{lo:.0f}, {hi:.0f}]")

    M_naive = kernel(x, q, p_naive, r, T)
    M_fhs = kernel(x, q, p_fhs, r, T)

    mono_naive = monotonicity_report(x, M_naive, lo, hi)
    mono_fhs = monotonicity_report(x, M_fhs, lo, hi)
    print("\n--- monotonicity inside the identified region ---")
    for name, mo in [("naive KDE", mono_naive), ("GARCH FHS", mono_fhs)]:
        print(f"  {name:>10s}: increasing over {mo['frac_increasing']:.1%} of the region"
              + (f", longest run [{mo['longest_run_lo']:.0f}, {mo['longest_run_hi']:.0f}]"
                 if "longest_run_lo" in mo else ""))

    # --------------------------------------------------------- drift sensitivity
    # The physical drift is the single least well determined input. Over the
    # 1990-2026 sample the annualised mean is ~ mu_hat with a standard error of
    # sigma/sqrt(n_years) -- large enough to move the kernel's slope materially.
    mu_hat_ann = MU_HAT_ANN
    se_mu_ann = float(daily90.std() * np.sqrt(252) / np.sqrt(n_years))
    print(f"\n--- drift ---")
    print(f"  sample mean return {mu_hat_ann:.2%}/yr over {n_years:.1f} years")
    print(f"  standard error {se_mu_ann:.2%}/yr  "
          f"=> 95% CI [{mu_hat_ann-1.96*se_mu_ann:.2%}, {mu_hat_ann+1.96*se_mu_ann:.2%}]")

    drift_rows = []
    M_drift = {}
    for label, mu_ann in [("mu - 2 s.e.", mu_hat_ann - 2 * se_mu_ann),
                          ("mu hat", mu_hat_ann),
                          ("mu + 2 s.e.", mu_hat_ann + 2 * se_mu_ann),
                          ("risk-neutral drift", float(np.log(F / S0) / T)),
                          ("GARCH mean", float(fit.mu * 252 / 100))]:
        Rd = phys.fhs_simulate(fit, HORIZON, n_paths=N_PATHS, seed=2,
                               mu_override=mu_ann / 252)
        pd_ = phys.density_from_sample(Rd, logm) / x
        Md = kernel(x, q, pd_, r, T)
        mo = monotonicity_report(x, Md, lo, hi)
        M_drift[label] = Md
        drift_rows.append({"scenario": label, "mu_annual": mu_ann,
                           "frac_increasing": mo["frac_increasing"]})
    drift_df = pd.DataFrame(drift_rows)
    print(drift_df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ------------------------------------------------------------ bootstrap band
    rng = np.random.default_rng(7)
    boots = np.empty((N_BOOT, x.size))
    pool = fit.resid_std
    for b in range(N_BOOT):
        fb = phys.GarchFit(**{**fit.__dict__,
                              "resid_std": pool[rng.integers(0, pool.size, pool.size)]})
        Rb = phys.fhs_simulate(fb, HORIZON, n_paths=BOOT_PATHS, seed=1000 + b,
                               mu_override=MU_HAT_ANN / 252)
        pb = phys.density_from_sample(Rb, logm) / x
        boots[b] = kernel(x, q, pb, r, T)
    band_lo = np.percentile(boots, 2.5, axis=0)
    band_hi = np.percentile(boots, 97.5, axis=0)
    print(f"\nbootstrap band from {N_BOOT} residual resamples "
          f"({BOOT_PATHS} paths each)")

    # ---------------------------------------------------------------- outputs
    out = pd.DataFrame({"K": x, "q": q, "p_naive": p_naive, "p_fhs": p_fhs,
                        "M_naive": M_naive, "M_fhs": M_fhs,
                        "M_fhs_lo": band_lo, "M_fhs_hi": band_hi})
    out.to_csv(os.path.join(RES, "pricing_kernel.csv"), index=False)
    pd.DataFrame([{**{"estimator": "naive_kde"}, **mono_naive},
                  {**{"estimator": "garch_fhs"}, **mono_fhs}]).to_csv(
        os.path.join(RES, "monotonicity.csv"), index=False)
    drift_df.to_csv(os.path.join(RES, "drift_sensitivity.csv"), index=False)
    pd.DataFrame([{
        "n_overlapping": len(ov), "n_effective": n_eff,
        "naive_mean": ov.mean(), "naive_sd": ov.std(),
        "naive_ann_vol": ov.std() * np.sqrt(252 / HORIZON),
        "garch_omega": fit.omega, "garch_alpha": fit.alpha,
        "garch_beta": fit.beta, "garch_persistence": fit.persistence,
        "garch_nu": fit.nu, "uncond_vol": fit.uncond_vol_ann,
        "conditional_vol_today": fit.current_vol_ann,
        "fhs_mean": R.mean(), "fhs_sd": R.std(),
        "fhs_ann_vol": R.std() * np.sqrt(252 / HORIZON),
        "fhs_skew": pd.Series(R).skew(), "fhs_kurt": pd.Series(R).kurt() + 3,
        "atm_iv": np.sqrt(svi.total_variance(0.0, sp) / T),
        "mu_hat_ann": mu_hat_ann, "se_mu_ann": se_mu_ann,
        "region_lo": lo, "region_hi": hi,
    }]).to_csv(os.path.join(RES, "physical_density.csv"), index=False)

    make_figures(x, q, p_naive, p_fhs, M_naive, M_fhs, band_lo, band_hi,
                 M_drift, lo, hi, S0, F, s)
    print(f"\nfigures written to {FIG}")


def make_figures(x, q, p_naive, p_fhs, M_naive, M_fhs, band_lo, band_hi,
                 M_drift, lo, hi, S0, F, s):
    # --- densities overlaid
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(x, q, color=BLUE, lw=2.2, label="$q$ risk-neutral (SVI)")
    ax.plot(x, p_fhs, color=GREEN, lw=2, label="$p$ GARCH-filtered (conditional)")
    ax.plot(x, p_naive, color=ORANGE, lw=2, ls="--",
            label="$p$ naive KDE (unconditional)")
    ax.axvline(F, color="grey", ls=":", lw=1)
    ax.set_xlim(5800, 10000)
    ax.set_title("Risk-neutral and physical densities")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogy(x, np.maximum(q, 1e-12), color=BLUE, lw=2, label="$q$")
    ax.semilogy(x, np.maximum(p_fhs, 1e-12), color=GREEN, lw=2, label="$p$ GARCH")
    ax.semilogy(x, np.maximum(p_naive, 1e-12), color=ORANGE, lw=2, ls="--",
                label="$p$ naive")
    ax.axvspan(lo, hi, color=PURPLE, alpha=0.08, label="identified region")
    ax.set_xlim(5000, 11000)
    ax.set_ylim(1e-8, 2e-3)
    ax.set_title("Tails (log scale)")
    ax.legend(fontsize=9)

    for a in axes:
        a.set_xlabel("index level $S_T$")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "09_densities.png"), bbox_inches="tight")
    plt.close(fig)

    # --- the kernel
    sel = (x >= lo) & (x <= hi)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.fill_between(x[sel], band_lo[sel], band_hi[sel], color=GREEN, alpha=0.2,
                    label="95% bootstrap band")
    ax.plot(x[sel], M_fhs[sel], color=GREEN, lw=2.4, label="GARCH-filtered $p$")
    ax.plot(x[sel], M_naive[sel], color=ORANGE, lw=2, ls="--",
            label="naive unconditional $p$")
    ax.axvline(F, color="grey", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_title(r"Estimated pricing kernel $\hat M = e^{-rT} q/p$")
    ax.set_ylabel(r"$\hat M(S_T)$, log scale")
    ax.legend(fontsize=9)

    ax = axes[1]
    for (label, Md), c in zip(M_drift.items(), [BLUE, GREEN, RED, PURPLE, ORANGE]):
        ax.plot(x[sel], Md[sel], lw=1.9, color=c, label=label)
    ax.axvline(F, color="grey", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_title("Sensitivity to the assumed physical drift")
    ax.set_ylabel(r"$\hat M(S_T)$, log scale")
    ax.legend(fontsize=9)

    for a in axes:
        a.set_xlabel("index level $S_T$")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "10_pricing_kernel.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
