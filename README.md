# Recovering the pricing kernel from index options

Estimating $dQ/dP$ from market data, and checking whether it behaves the way
theory says it should.

> **Repository status:** the paper (`doc/paper.tex`, `doc/paper.pdf`), its
> figures and the five numbered `scripts/` are here. The `pk/` package the
> scripts import is being added next.

## Status

| Part | Content | State |
|---|---|---|
| — | **Paper** (`doc/paper.tex` → `paper.pdf`, 13pp) | **done** |
| 1 | Change of measure, Girsanov, Breeden–Litzenberger, Carr–Madan | **done** (also standalone: `doc/part1_theory.tex`) |
| 2 | **Synthetic validation of the extractor** | **done** |
| 3 | SVI surface, arbitrage constraints, real SPX chain | **done** (December slice) |
| 4 | Physical density $p$, kernel $\hat M = q/p$ | **done** |
| 5 | Variance risk premium, Newey–West, jump correction | **done** |

Part 2 is deliberately first. An estimator that hasn't been tested against a
known answer isn't an estimator.

## Layout

```
doc/paper.tex          the paper: theory, validation, empirics, limitations (13pp)
doc/part1_theory.tex   standalone theory note, 8pp (superset of the paper's Section 2)
pk/black_scholes.py   prices + exact lognormal density (ground truth 1)
pk/heston.py          characteristic function, prices, density by CF inversion (ground truth 2)
pk/bl.py              Breeden–Litzenberger extractor + diagnostics
pk/data.py            CBOE chain parsing, parity-implied forward and discount factor
pk/iv.py              Black-76 and implied-vol inversion
pk/svi.py             SVI with analytic derivatives, butterfly g, constrained fit
pk/physical.py        overlapping-return KDE and GARCH filtered historical simulation
pk/varswap.py         CBOE VIX algorithm, BKM moments, realised variance, Newey-West
scripts/01_validate_synthetic.py
scripts/02_chain_qc.py
scripts/03_svi_surface.py
scripts/04_pricing_kernel.py
scripts/05_variance_risk_premium.py
doc/figures/ results/
```

Run: `python scripts/01_validate_synthetic.py` (numpy, scipy, pandas, matplotlib, arch; see `requirements.txt`).

## What Part 2 established

Setup throughout: $S_0 = 5000$, $T = 0.5$, $r = 4\%$, $q = 1.5\%$, $\sigma = 20\%$,
strikes spanning roughly $\pm 4$ standard deviations in log space.

**A. The extractor is correct.** On exact Black–Scholes prices the recovered
density matches the lognormal, and the error falls at empirical order
$1.999 \pm 0.001$ across four decades of strike spacing — the exact signature of
a central second difference. At $h = 1$ the max pointwise error is $1.0 \times 10^{-10}$.

Sharper: Proposition 2.4 of the theory note predicts not just the order but the
constant, $\hat q - q = (h^2/12)\,q'' + O(h^4)$. The measured constant matches
the predicted one to within $4 \times 10^{-5}$ relative at $h = 25$, converging
at the predicted $O(h^2)$ rate. An estimator can be second-order accurate for
the wrong reason; this rules that out.

**B. The extractor is ill-posed, and by a predictable amount.** Perturb the
prices by i.i.d. noise of size $\epsilon$ and the error stops falling. The
variance of the second difference is $6\epsilon^2$, so the noise term in $\hat q$
has standard deviation $e^{rT}\sqrt{6}\,\epsilon / h^2$, giving a predicted
integrated error of

$$\mathbb{E}\!\int |\hat q - q| \, dK \;\approx\; e^{rT}\sqrt{\tfrac{12}{\pi}}\;\frac{\epsilon}{h^2}\,(K_{\max} - K_{\min}).$$

Measured against 200 replications this is accurate to three significant figures
wherever noise dominates (e.g. $\epsilon = 0.05$, $h = 10$: predicted 5.982,
measured 5.926). Total error is U-shaped in $h$, and the optimum moves *right*
as quotes get noisier: on the tested grid $h^\star = 100, 200, 400$ for
$\epsilon = 0.005, 0.05, 0.25$. Balancing $O(h^2)$ truncation against
$O(\epsilon/h^2)$ noise predicts $h^\star \propto \epsilon^{1/4}$; a 50-fold
increase in noise should widen the optimum by $50^{1/4} \approx 2.7$, and the
measured widening is $4$ — the right order, though the grid is too coarse to
pin the exponent. A finer sweep is a cheap improvement if the write-up wants to
claim the scaling rather than gesture at it.

At five cents of noise on a 10-point grid, **44% of the extracted density is
negative**. This is the whole argument for Parts 3–4: raw differencing of real
quotes cannot work, smoothing must happen in implied-vol space, and the
butterfly constraint is not decoration.

**C. It recovers a non-lognormal truth.** Heston ($v_0 = 0.04$, $\kappa = 2$,
$\theta = 0.045$, $\sigma_v = 0.55$, $\rho = -0.75$; Feller violated at $-0.12$,
deliberately, since real calibrations usually violate it). Prices come from
Gil-Pelaez integration of the characteristic function; the true density comes
from inverting the same CF directly, so the two paths are independent enough to
be a real test. Recovery on a 25-point grid: $L^1$ error $2.0 \times 10^{-4}$,
$L^\infty$ error $2.6 \times 10^{-7}$, no negative density. The recovered
density shows the negative skew correctly — mode above the mean, fat left tail.

Two supporting checks: Heston collapses onto Black–Scholes as vol-of-vol $\to 0$
(max price difference $6.7 \times 10^{-7}$), and the CF uses the Albrecher
"little Heston trap" form so the complex log stays on its principal branch.

**A caveat the write-up should keep.** Total mass integrates to 0.9997, not 1,
and the implied mean sits below the forward by 0.4 points. That is strike-range
truncation, not estimator error — the tails outside $[K_{\min}, K_{\max}]$ carry
real mass. With real SPX chains the truncation is far more severe, and the
Part 4 kernel is only trustworthy on the range where the density is actually
identified. Worth saying out loud rather than plotting a kernel into the wings
and hoping.

## Figures

- `01_convergence.png` — $O(h^2)$ convergence; U-shaped error under noise
- `02_bs_recovery.png` — recovered lognormal, exact vs 5c of quote noise
- `03_heston_recovery.png` — recovered Heston density, linear and log scale

## Part 3, step 1: the real chain

Snapshot: 29,858 SPX/SPXW contracts, quotes as of the 2026-09-25 close,
spot 7743.41. Three standard monthly (European, AM-settled) expiries used.

**The snapshot is internally consistent.** Forward and discount factor are
recovered from put-call parity rather than an external curve, by regressing
$C - P$ on $K$ over strikes within 5% of spot, weighted by inverse quote width.

| Expiry | $T$ | strikes | $F$ | implied $r$ | implied div. yield | parity RMSE |
|---|---|---|---|---|---|---|
| 2026-10-16 | 0.058 | 482 | 7756.6 | 4.29% | 1.33% | 0.078 pts |
| 2026-11-20 | 0.153 | 458 | 7782.6 | 4.54% | 1.25% | 0.147 pts |
| 2026-12-18 | 0.230 | 455 | 7803.3 | 4.52% | 1.17% | 0.108 pts |

The implied dividend yields near 1.2% and the flat-ish rate term structure are
not inputs — they fall out of the quotes, and they are right. Parity residuals
of 0.1 index points on a 7800 index are about 1.3 basis points.

**Raw mids violate no-arbitrage constantly.** On the December slice, 35% of
adjacent strike triples fail the butterfly (convexity) test. Monotonicity holds
everywhere; 14 strikes breach the slope bound.

**Part 2's prediction holds on live quotes.** Applying the extractor directly
to raw mids on the 5-point grid ($K \in [7160, 8000]$, median half-spread 0.65):

- predicted noise sd in $\hat q$: $3.7 \times 10^{-2}$
- peak density implied by the 13.5% ATM vol: $7.9 \times 10^{-4}$
- **noise-to-signal ratio: 47**
- fraction of extracted density negative: **39%** (predicted 49%)

So the naive estimator is not noisy, it is pure noise — the signal is a
fiftieth of the error. This is the measured justification for everything in the
rest of Part 3.

The gap between 39% and the predicted 49% is itself informative. Inverting the
measured negative fraction gives an effective independent noise of 0.028 index
points against 0.375 implied by the spread — a factor of 13 smaller. Quote
errors are therefore strongly positively correlated across adjacent strikes,
which is what one expects if market makers quote off a common fitted smile, and
the second difference cancels most of the common component. Worth reporting:
the i.i.d. noise model is conservative here, not optimistic.

## On jumps

Section 3.4 of the theory note now derives the correction. Under jumps the
log-contract price is not expected quadratic variation but

$$2e^{rT}\!\int \frac{O(K)}{K^2}dK = \mathbb{E}^Q[\text{QV}] - \tfrac{2}{3}\mathbb{E}^Q\Big[\sum J^3\Big] + O(J^4).$$

Index jumps skew negative, so the correction is positive: model-free implied
variance **overstates** expected quadratic variation, and part of any measured
variance risk premium is this jump term rather than risk compensation. The
correction is estimable from the same chain via the Bakshi–Kapadia–Madan third
moment (apply Carr–Madan to $\log^3(x/F)$). Part 5 will report both the
uncorrected figure, for comparability with the published VIX, and the corrected
one. The gap is a result rather than an error term.

## Part 3, step 2: the arbitrage-free surface

Scope is the December 2026 slice alone. Raw SVI in total variance,
$w(k) = a + b[\rho(k-m) + \sqrt{(k-m)^2 + s^2}]$, fitted under hard
constraints: $b \ge 0$, $s > 0$, $w > 0$, Lee's wing bound
$b(1+|\rho|) \le 2$, and the Gatheral–Jacquier butterfly condition
$g(k) \ge 0$ on a dense grid extending 25% beyond the data. $\rho$ is
parameterised as $\tanh(\cdot)$ so $|\rho|<1$ holds structurally. Sixteen
random starts, since the objective is not convex.

**The density is read off analytically**, never differenced:
$q(K) = g(k)\,e^{-d_2^2/2} / (K\sqrt{2\pi w})$. Verified against finite
differences of the fitted slice's own call prices to $1.7\times10^{-6}$
relative — so the closed form is right, and no market data is differentiated
at any point.

**Quote reliability filter.** 29 of 455 quotes dropped: price below \$1 or
relative spread above 15%. These sit at $k$ down to $-1.72$, roughly 26
standard deviations out at this maturity, trading near fifty cents with
spreads over 30% of price. Their implied vols are a tick-size artefact.
Fitted range is $k \in [-0.774, +0.132]$.

**Fit.** $a=-0.0227$, $b=0.1052$, $\rho=-0.0830$, $m=0.0520$, $s=0.2459$.

| Region | $n$ | RMSE | inside bid-ask |
|---|---|---|---|
| core, $\lvert k\rvert \le 0.15$ | 288 | 6.6 bp | 27% |
| left, $-0.5 \le k < -0.15$ | 104 | 13.5 bp | 22% |
| deep left, $k < -0.5$ | 34 | 77.7 bp | 0% |
| all | 426 | 23.6 bp | 24% |

The aggregate 23.6 bp is misleading on its own. In the core the fit is 6.6 bp,
and the error is concentrated entirely in the deep left wing, where raw SVI's
asymptotically *linear* total variance cannot match the observed *concave*
left wing. That is a limitation of the five-parameter family, not of the data,
and it belongs in the limitations section.

The low "inside bid-ask" figures are not a failure either: the median SPX quote
half-width here is **3.9 bp**. No five-parameter model fits inside a 3.9 bp
band across 426 strikes. Reporting the ratio of model error to quote width is
the honest framing; reporting only "24% inside" would understate the fit.

**The fitted density is a proper density.**

| diagnostic | value |
|---|---|
| total mass | 0.99997 |
| mean vs forward | −0.56 pts ($-7\times10^{-5}$ relative) |
| std | 606.4 |
| skewness | −1.99 |
| excess kurtosis | 16.0 |
| min $g$ | $>0$ everywhere |

Mass and mean are *checks*, not constraints — nothing in the SVI fit forces
them, so hitting 1 and the forward to five decimals says the fit is sane.
Strong negative skew and fat tails, as expected for SPX.

**Before and after.** Same quotes, same expiry: the naive second difference was
39% negative with noise 47× the signal; the constrained fit is a smooth,
strictly positive density that integrates to one. Figure `08` overlays them.

**Wing extrapolation.** Only $1.5\times10^{-4}$ of the total mass lies outside
the quoted strike range, so the moments above barely depend on extrapolation.
One artefact to note: past $K \approx 11{,}200$ — well beyond the last quote at
10,600 — the fitted density dips to near zero and recovers, because the
butterfly constraint binds at the edge of the padded grid. The mass involved is
$O(10^{-7})$ and affects nothing, but the density should be reported on the
quoted range and the artefact named rather than cropped silently.

## Part 4: the physical density and the kernel

Horizon: 59 NYSE sessions, 2026-09-25 to 2026-12-18, $T = 0.2301$.

**The two estimators answer different questions.** $q$ is conditional on today
— priced when the ATM implied vol was 13.5%. A KDE on historical 59-day returns
is *unconditional*: it describes a randomly chosen day since 1990, and the
average such day had an 18% vol. Dividing a conditional $q$ by an unconditional
$p$ measures the gap between today and an average day, not a pricing kernel.
So both are reported, and the gap between them is a result.

| | naive KDE | GARCH-FHS |
|---|---|---|
| conditioning | unconditional | conditional on today |
| $n$ | 9,192 overlapping | 200,000 paths |
| **effective $n$** | **156** | — |
| ann. vol of horizon return | 15.0% | 13.8% |
| skew / kurtosis | — | −0.81 / 9.4 |

The naive estimator's $n=9{,}192$ is 156 independent observations wearing a
disguise: 59-day returns sampled daily overlap by 58 of 59 days. Reporting
$n$ rather than $n/h$ is the standard way an overlapping-returns standard error
is made to look better than it is.

**GARCH(1,1)-$t$ on 9,251 daily returns:** $\alpha = 0.102$, $\beta = 0.892$,
persistence 0.994, $\nu = 6.15$. Unconditional vol 23.4%, today's conditional
vol 11.6% — so the FHS density mean-reverts upward over the horizon and lands
at 13.8%. Residuals are bootstrapped rather than drawn from the fitted $t$, so
the simulated path keeps the empirical skew and leverage effect.

One fix worth naming: the standardised residuals do **not** have mean zero in
sample, because high-variance days carry systematically signed residuals.
Bootstrapping them raw injects an uncontrolled drift on top of $\mu$. Centring
the pool makes the horizon drift exactly $\mu h$, so the drift becomes an
explicit assumption rather than a side effect of the filter. Before centring,
the base case and the "$\hat\mu$" sensitivity scenario disagreed with each
other despite nominally using the same drift.

**The risk premium is visible directly in the tails.**

| event | $Q$ | $P$ | $Q/P$ |
|---|---|---|---|
| $S_T < 7000$ (−10%) | 6.20% | 3.96% | 1.56 |
| $S_T < 7300$ (−6%) | 12.3% | 10.0% | 1.23 |
| $S_T > 8300$ (+7%) | 13.7% | 20.2% | 0.68 |
| $S_T > 8700$ (+12%) | 2.02% | 5.41% | 0.37 |

The market prices a 10% drawdown at 1.6× its historical frequency and a 12%
rally at 0.37×. That asymmetry *is* the kernel.

### The main result: the puzzle is not identified

On the identified region (central 90% of $q$, $[6893, 8518]$) the estimated
kernel is non-monotone — increasing over 41% of the region with the
GARCH-filtered $p$, 40% with the naive one. Taken at face value, that is the
pricing kernel puzzle.

It should not be taken at face value. The physical drift is the least
well-determined input in the whole exercise: the 1990–2026 sample mean is
8.41%/yr with a standard error of **2.97%/yr**, a 95% interval of
[2.6%, 14.2%]. Varying $\mu$ across that interval:

| drift scenario | $\mu$ (ann.) | fraction of region increasing |
|---|---|---|
| $\hat\mu - 2$ s.e. | 2.47% | 51.8% |
| risk-neutral drift | 3.35% | 50.2% |
| $\hat\mu$ | 8.41% | 39.9% |
| $\hat\mu + 2$ s.e. | 14.35% | **2.7%** |
| GARCH constant mean | 18.93% | 2.2% |

At the top of the confidence interval the kernel is essentially monotone
decreasing — exactly what theory predicts. At the bottom it is strongly
non-monotone. **The sign of the answer is decided by the equity premium
assumption, not by the option data.** We cannot reject monotonicity at this
sample size, and a paper that reported only the point estimate would be
claiming to have found a puzzle it has no power to detect.

This also explains the GARCH constant-mean estimate of 18.93%/yr versus the
8.41% sample mean. GARCH weights low-variance days most heavily, and
low-variance days are bull markets, so the MLE mean is biased upward as an
estimate of the equity premium. Using it — as a naive pipeline would, since it
is right there in the fitted object — would have produced a clean monotone
kernel for entirely the wrong reason.

A 95% bootstrap band over 120 resamples of the residual pool is plotted
alongside. It is wide enough to contain a monotone decreasing kernel across
most of the region, which is the same conclusion by a different route:
simulation noise alone does not pin the shape down either.

Where the historical sample never visited a region, $p$ is zero and the kernel
is undefined rather than infinite; those points return `nan` and are excluded
from the plots and the monotonicity counts.

## Part 5: implied variance, jumps, and the variance risk premium

### The derivation reproduces the index

The SPXW expiries at 28 and 35 days bracket 30, so the Section 3.3 formula can
be run on our own chain and checked against the published index. Following the
CBOE recipe exactly — forward from the strike with the smallest $|C-P|$, $K_0$
the largest strike below it, OTM options with the two-consecutive-zero-bid
stopping rule, and the $-(F/K_0-1)^2$ correction:

| | days | $F$ | $K_0$ | options | $\sigma$ |
|---|---|---|---|---|---|
| near, 2026-10-23 | 28 | 7762.46 | 7760 | 179 | 0.1456 |
| next, 2026-10-30 | 35 | 7768.29 | 7765 | 481 | 0.1551 |

**Reconstructed VIX = 14.880 against a published 14.87 — an error of +0.07%.**

This is the hinge of the project. It validates the derivation and the code
together, and it licenses using the published VIX history as a model-free
implied variance series — which is what makes a 1990–2026 time series possible
from a single option snapshot.

**Wing truncation, quantified.** The $1/K^2$ weight loads on deep OTM puts,
exactly where chains run out. Share of the variance sum by moneyness:

| $K/F$ | $<0.70$ | 0.70–0.85 | 0.85–0.95 | 0.95–1.05 | $>1.05$ |
|---|---|---|---|---|---|
| share | 4.6% | 9.1% | 19.8% | 64.3% | 2.1% |

So 4.6% of the answer comes from the region most vulnerable to truncation. Not
catastrophic, but not negligible either, and it is the right number to quote in
the limitations section rather than a sentence about the weighting.

### The jump correction is large

On the 35-day slice, refitted through the Part 3 pipeline, the risk-neutral log
return has variance 0.00242 and third central moment $-4.46\times10^{-4}$
(skewness $-3.74$). By equation (30) the correction is $-\tfrac23 \mathbb{E}[\sum J^3]$:

- log-contract implied vol: **15.90%**
- jump-corrected expected QV: **14.89%**
- correction is **12.3% of implied variance**

Scaled to the sample mean implied variance, that is 0.00537 in variance units
against a measured mean VRP of 0.01136 — **roughly 47% of the measured premium
is the jump term rather than compensation for variance risk.** Two caveats kept
with the number: it is extrapolated from one cross-section, and it is an upper
bound, since attributing the entire third moment to jumps ignores that a
leverage-driven stochastic-volatility diffusion generates negative skew too.

### The premium is real, but the naive t-statistic is nonsense

9,226 daily observations, 1990-01-02 to 2026-08-26, 21-session realised
variance. Mean implied vol 20.9%, mean subsequently realised vol 18.0%.

| specification | lags | mean | s.e. | $t$ |
|---|---|---|---|---|
| naive, no HAC | 0 | 0.01136 | 0.00053 | **21.30** |
| Newey–West, lag = window | 21 | 0.01136 | 0.00206 | 5.52 |
| Newey–West, lag = 2× window | 42 | 0.01136 | 0.00232 | 4.90 |
| Newey–West, Andrews plug-in | 196 | 0.01136 | 0.00228 | 4.99 |

The windows overlap by 20 of 21 days, so the series is autocorrelated by
construction even under the null. HAC inflates the standard error **3.9×** and
the $t$ falls from 21.3 to about 5. The premium survives — but a paper quoting
21.3 would be reporting an artefact of overlap.

### Earns steadily, loses violently — with numbers

| | |
|---|---|
| mean | 0.01136 |
| median | 0.01253 |
| skewness | **−8.38** |
| excess kurtosis | 109.2 |
| positive on | 85.9% of days |
| worst observation | −0.816 (2020-02-26) |
| worst 1% of days | **−33% of the total** |

The premium is positive on six days in seven, and the worst 1% of days subtract
a third of everything the other 99% earned. That is the short-volatility payoff
profile stated quantitatively, and it is why a $t$-test on the mean is a weak
summary: it describes the average of a distribution whose economics live in the
left tail.

**By decade**, the premium is not uniformly significant:

| period | mean | $t_{NW}$ | % positive |
|---|---|---|---|
| 1990–1999 | 0.0174 | 11.18 | 92.5% |
| 2000–2009 | 0.0095 | 1.97 | 81.5% |
| 2010–2019 | 0.0097 | 5.72 | 84.3% |
| 2020–2026 | 0.0074 | 0.93 | 84.9% |

Two of four decades fail at the 5% level, and the point estimate has more than
halved since the 1990s. Worth reporting rather than pooling into one
comfortable number.

## The paper

`doc/paper.tex` → 13 pages. Build with `pdflatex paper.tex` (run twice for
cross-references; no bibtex needed, the bibliography is inline).

Structure: introduction; theory with proofs; synthetic validation; data;
arbitrage-free surface; pricing kernel; variance risk premium; limitations;
conclusion; 18 references. Every figure and table is generated by the numbered
scripts — nothing is hand-copied.

## Reproducing everything

```bash
pip install -r requirements.txt
# put the three data files in data/ (see data/README.md)
python scripts/01_validate_synthetic.py
python scripts/02_chain_qc.py
python scripts/03_svi_surface.py
python scripts/04_pricing_kernel.py
python scripts/05_variance_risk_premium.py
cd doc && pdflatex paper.tex && pdflatex paper.tex
```

Scripts write to `figures/` and `results/`. Script 04 takes a few minutes
(GARCH fit plus 120 bootstrap replications); the rest are seconds.

## Headline numbers

| | |
|---|---|
| BL convergence order (synthetic) | 1.999 |
| leading-constant match to Prop. 2.4 | $4\times10^{-5}$ rel. |
| naive extraction, noise-to-signal on live quotes | 47 |
| naive extraction, negative mass | 39% |
| SVI density total mass | 0.99997 |
| SVI density mean vs forward | $7\times10^{-5}$ rel. |
| VIX reconstructed vs published | 14.880 vs 14.87 (+0.07%) |
| jump correction, share of implied variance | 12.3% |
| mean VRP, Newey–West $t$ | 5.52 (naive 21.30) |
| kernel monotonicity across drift 95% CI | 2.7% – 51.8% increasing |
