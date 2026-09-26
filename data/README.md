# Data

The data files are not committed (the option chain alone is about 13 MB).
Download the three files below and place them in this directory.

1. **`_SPXchain.json`: SPX / SPXW option chain.** CBOE delayed quotes, snapshot as of the
   2026-09-25 close (index at 7743.41). Download from the CBOE delayed-quotes
   page for SPX.

   **Set "Options Range" to "All", not "Near the Money".** The default
   near-the-money download drops the deep out-of-the-money strikes. Those
   strikes carry the density's tails and the $1/K^2$-weighted wing of the VIX
   sum, so the SVI fit, the kernel and the VIX reconstruction all change
   without them.

2. **`sp500_history.csv`: S&P 500 daily closes**, from 1927 onwards.

3. **`VIX_History.csv`: VIX daily closes**, from 1990 onwards.
