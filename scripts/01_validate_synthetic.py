"""Part 2 -- validate the extractor against densities we already know.

Three experiments:
  A. Black-Scholes, exact prices. Confirms we recover the lognormal and
     that the error falls as O(h^2), the signature of a central second
     difference. This is a correctness test.
  B. Black-Scholes, prices perturbed by quote noise of size eps. Shows the
     error rising as eps/h^2 and the resulting U-shaped total error with an
     interior optimal spacing. This is the test that explains why Parts 3-4
     need smoothing and constraints.
  C. Heston, semi-analytic prices vs the density obtained by inverting the
     characteristic function. Confirms we recover a skewed, fat-tailed,
     genuinely non-lognormal truth.

Run:  python scripts/01_validate_synthetic.py
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

from pk import black_scholes as bs
from pk import heston as hs
from pk.bl import density_diagnostics, extract_density, l1_error, linf_error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG = os.path.join(ROOT, "doc", "figures")
RES = os.path.join(ROOT, "results")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {"figure.dpi": 150, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold"}
)
BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

# --- market setup -----------------------------------------------------------
S0, T, R, Q, SIGMA = 5000.0, 0.5, 0.04, 0.015, 0.20
K_LO, K_HI = 2800.0, 8800.0  # roughly +/- 4 sd in log space
FORWARD = S0 * np.exp((R - Q) * T)


def grid(h):
    n = int(round((K_HI - K_LO) / h))
    return K_LO + h * np.arange(n + 1)


# --- A. exact prices, truncation error only ---------------------------------
def experiment_a():
    rows = []
    for h in [200, 100, 50, 25, 20, 10, 5, 4, 2, 1]:
        K = grid(h)
        C = bs.call_price(S0, K, T, R, Q, SIGMA)
        Ki, q_hat = extract_density(K, C, R, T)
        q_true = bs.rn_density(Ki, S0, T, R, Q, SIGMA)
        d = density_diagnostics(Ki, q_hat, forward=FORWARD)
        rows.append(
            {
                "h": h,
                "n_strikes": K.size,
                "L1_error": l1_error(Ki, q_hat, q_true),
                "Linf_error": linf_error(q_hat, q_true),
                "mass": d["mass"],
                "mean_error": d["mean_error"],
                "min_density": d["min_density"],
            }
        )
    df = pd.DataFrame(rows)
    # empirical convergence order between consecutive refinements
    df["order"] = np.nan
    lg = np.log(df["Linf_error"].to_numpy())
    lh = np.log(df["h"].to_numpy())
    df.loc[1:, "order"] = np.diff(lg) / np.diff(lh)
    return df


# --- B. noisy prices, bias-variance trade-off -------------------------------
def experiment_b(eps_list=(0.005, 0.05, 0.25), n_rep=200, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for eps in eps_list:
        for h in [600, 400, 300, 200, 100, 50, 25, 20, 10, 5, 2, 1]:
            K = grid(h)
            C = bs.call_price(S0, K, T, R, Q, SIGMA)
            Ki = K[1:-1]
            q_true = bs.rn_density(Ki, S0, T, R, Q, SIGMA)
            l1s, negs = [], []
            for _ in range(n_rep):
                Cn = C + rng.normal(0.0, eps, size=C.shape)
                _, q_hat = extract_density(K, Cn, R, T)
                l1s.append(l1_error(Ki, q_hat, q_true))
                negs.append(float(np.mean(q_hat < 0)))
            rows.append(
                {
                    "eps": eps,
                    "h": h,
                    "L1_error_mean": float(np.mean(l1s)),
                    "frac_negative": float(np.mean(negs)),
                    # Predicted noise-only L1. Var of the second difference is
                    # 6 eps^2, so q_hat picks up noise with sd = e^{rT}sqrt(6)eps/h^2;
                    # E|N(0,sd)| = sd sqrt(2/pi), integrated over the strike range.
                    "theory_L1": float(
                        np.exp(R * T) * np.sqrt(6) * eps / h**2
                        * np.sqrt(2 / np.pi) * (K_HI - K_LO)
                    ),
                }
            )
    return pd.DataFrame(rows)


# --- C. Heston ---------------------------------------------------------------
HP = hs.HestonParams(v0=0.04, kappa=2.0, theta=0.045, sigma=0.55, rho=-0.75)


def experiment_c(h=25.0, k_lo=2600.0, k_hi=8000.0):
    n = int(round((k_hi - k_lo) / h))
    K = k_lo + h * np.arange(n + 1)
    C = hs.call_price(S0, K, T, R, Q, HP)
    Ki, q_hat = extract_density(K, C, R, T)
    q_true = hs.rn_density(Ki, S0, T, R, Q, HP)
    diag = density_diagnostics(Ki, q_hat, forward=FORWARD)
    diag.update(
        {
            "L1_error": l1_error(Ki, q_hat, q_true),
            "Linf_error": linf_error(q_hat, q_true),
            "h": h,
            "feller": HP.feller,
        }
    )
    return Ki, q_hat, q_true, diag


def leading_error_constant(K0=5200.0):
    """Proposition 2.4 of the theory note predicts  q_hat - q = (h^2/12) q'' + O(h^4).

    Checking the *constant*, not just the order, is a much sharper test of the
    derivation: an estimator can be second-order accurate for the wrong reason.
    """
    e = 1e-3 * K0
    dens = lambda x: bs.rn_density(x, S0, T, R, Q, SIGMA)
    predicted = (dens(K0 + e) - 2 * dens(K0) + dens(K0 - e)) / e**2 / 12.0
    rows = []
    for h in [200, 100, 50, 25]:
        K = np.array([K0 - h, K0, K0 + h])
        _, qh = extract_density(K, bs.call_price(S0, K, T, R, Q, SIGMA), R, T)
        rows.append({"h": h, "measured": (qh[0] - dens(K0)) / h**2})
    df = pd.DataFrame(rows)
    df["predicted"] = predicted
    df["rel_error"] = df["measured"] / predicted - 1.0
    return df


def heston_sanity():
    """Vol-of-vol -> 0 with v0 = theta must collapse Heston onto Black-Scholes."""
    # sigma is small but not so small that (a - d)/(a + d) loses all precision;
    # 1e-6 is actually *worse* conditioned than 1e-4 for exactly that reason.
    flat = hs.HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=1e-4, rho=0.0)
    K = np.array([4000.0, 4500.0, 5000.0, 5500.0, 6000.0])
    c_h = hs.call_price(S0, K, T, R, Q, flat)
    c_b = bs.call_price(S0, K, T, R, Q, 0.20)
    return float(np.max(np.abs(c_h - c_b)))


# --- figures -----------------------------------------------------------------
def fig_convergence(df_a, df_b):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ref = df_a["Linf_error"].iloc[0] * (df_a["h"] / df_a["h"].iloc[0]) ** 2
    ax.loglog(df_a["h"], ref, "-", color="grey", lw=6, alpha=0.45,
              label=r"$O(h^2)$ reference")
    ax.loglog(df_a["h"], df_a["Linf_error"], "o--", color=BLUE, lw=1.6,
              label="measured")
    ax.set_title("Exact prices: error falls as $h^2$")
    ax.set_xlabel("strike spacing $h$ (index points)")
    ax.set_ylabel(r"$\max_K |\hat q - q|$")
    ax.legend()

    ax = axes[1]
    for eps, c, m in zip(sorted(df_b["eps"].unique()), [BLUE, ORANGE, RED], "os^"):
        sub = df_b[df_b["eps"] == eps].sort_values("h")
        ax.loglog(sub["h"], sub["L1_error_mean"], m + "-", color=c, lw=2,
                  label=rf"$\epsilon = {eps}$")
    ax.set_title("Noisy prices: refining the grid makes it worse")
    ax.set_xlabel("strike spacing $h$ (index points)")
    ax.set_ylabel(r"mean $\int |\hat q - q|\,dK$")
    ax.legend(title="price noise sd")

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.suptitle(
        "Breeden-Litzenberger is consistent but ill-posed",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "01_convergence.png"), bbox_inches="tight")
    plt.close(fig)


def fig_bs_recovery():
    h = 10.0
    K = grid(h)
    C = bs.call_price(S0, K, T, R, Q, SIGMA)
    Ki, q_hat = extract_density(K, C, R, T)
    q_true = bs.rn_density(Ki, S0, T, R, Q, SIGMA)

    Cn = C + np.random.default_rng(1).normal(0, 0.05, size=C.shape)
    _, q_noisy = extract_density(K, Cn, R, T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    ax = axes[0]
    ax.plot(Ki, q_true, color="black", lw=3, alpha=0.35, label="true lognormal")
    ax.plot(Ki, q_hat, "--", color=BLUE, lw=1.8, label="extracted (exact prices)")
    ax.set_title(f"Recovered density, $h={h:.0f}$")
    ax.set_ylabel("$q(K)$")
    ax.legend()

    ax = axes[1]
    ax.plot(Ki, q_true, color="black", lw=3, alpha=0.35, label="true lognormal")
    ax.plot(Ki, q_noisy, color=RED, lw=0.9, label="extracted (prices + 0.05 noise)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Same grid, five cents of quote noise")
    ax.legend()

    for a in axes:
        a.set_xlabel("strike $K$")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "02_bs_recovery.png"), bbox_inches="tight")
    plt.close(fig)


def fig_heston(Ki, q_hat, q_true):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(Ki, q_true, color="black", lw=3, alpha=0.35, label="Heston (CF inversion)")
    ax.plot(Ki, q_hat, "--", color=GREEN, lw=1.8, label="extracted")
    q_ln = bs.rn_density(Ki, S0, T, R, Q, SIGMA)
    ax.plot(Ki, q_ln, ":", color=ORANGE, lw=1.6, label="lognormal, same ATM vol")
    ax.set_title("Heston density recovered")
    ax.set_xlabel("strike $K$")
    ax.set_ylabel("$q(K)$")
    ax.legend()

    ax = axes[1]
    ax.semilogy(Ki, np.maximum(q_true, 1e-12), color="black", lw=3, alpha=0.35,
                label="Heston")
    ax.semilogy(Ki, np.maximum(bs.rn_density(Ki, S0, T, R, Q, SIGMA), 1e-12), ":",
                color=ORANGE, lw=1.6, label="lognormal")
    ax.set_title("Heston vs lognormal tails (log scale)")
    ax.set_xlabel("strike $K$")
    ax.set_ylabel("$q(K)$, log scale")
    ax.legend()

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "03_heston_recovery.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    print("sanity: Heston(vol-of-vol -> 0) vs Black-Scholes, max abs price diff = "
          f"{heston_sanity():.3e}")

    df_a = experiment_a()
    df_a.to_csv(os.path.join(RES, "convergence_exact.csv"), index=False)
    print("\n--- A. exact Black-Scholes prices ---")
    print(df_a.to_string(index=False, float_format=lambda v: f"{v:11.3e}"))

    df_lc = leading_error_constant()
    df_lc.to_csv(os.path.join(RES, "leading_error_constant.csv"), index=False)
    print("\n--- A2. leading truncation constant vs Prop 2.4 ---")
    print(df_lc.to_string(index=False, float_format=lambda v: f"{v:12.5e}"))

    df_b = experiment_b()
    df_b.to_csv(os.path.join(RES, "convergence_noisy.csv"), index=False)
    print("\n--- B. noisy prices (200 replications) ---")
    print(df_b.to_string(index=False, float_format=lambda v: f"{v:11.3e}"))

    Ki, q_hat, q_true, diag = experiment_c()
    pd.DataFrame([diag]).to_csv(os.path.join(RES, "heston_recovery.csv"), index=False)
    print("\n--- C. Heston ---")
    for k, v in diag.items():
        print(f"  {k:>16s}: {v:.6g}")

    fig_convergence(df_a, df_b)
    fig_bs_recovery()
    fig_heston(Ki, q_hat, q_true)
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
