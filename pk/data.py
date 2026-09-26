"""Loading market data and extracting the quantities the theory needs.

Two jobs here that are easy to get wrong and quietly ruin everything
downstream:

1. Parsing the chain and pairing calls with puts at each strike.
2. Recovering the forward F and the discount factor D from the quotes
   themselves via put-call parity, rather than grabbing a Treasury yield
   off a website. The parity route is self-consistent: it returns the
   discount rate the option market is actually using, including the
   borrow and dividend assumptions embedded in it. A mismatched forward
   shifts the whole extracted density sideways and corrupts the kernel.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "load_chain",
    "load_index_history",
    "load_vix",
    "ChainSlice",
    "implied_forward",
    "build_slice",
]

_OSI = re.compile(
    r"^(?P<root>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$"
)


def load_chain(path):
    """Parse a CBOE delayed-quote JSON snapshot into a tidy frame.

    Returns (df, meta). `asof` in meta is taken from the latest option
    trade time, not the file's download timestamp: the file may be pulled
    on a Saturday, but the quotes are the previous session's close, and
    the spot must be the spot from that same moment.
    """
    with open(path) as fh:
        raw = json.load(fh)
    data = raw["data"]

    df = pd.DataFrame(data["options"])
    parts = df["option"].str.extract(_OSI)
    if parts["root"].isna().any():
        raise ValueError("unparseable OSI symbols in chain")

    df["root"] = parts["root"]
    df["cp"] = parts["cp"]
    df["strike"] = parts["strike"].astype(float) / 1000.0
    df["expiry"] = pd.to_datetime(
        "20" + parts["yy"] + parts["mm"] + parts["dd"], format="%Y%m%d"
    )
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]

    last_trade = pd.to_datetime(df["last_trade_time"], errors="coerce").max()
    meta = {
        "spot": float(data["current_price"]),
        "asof": last_trade,
        "session": last_trade.normalize(),
        "downloaded": pd.to_datetime(raw["timestamp"]),
        "iv30": float(data.get("iv30", np.nan)),
    }
    return df, meta


def load_index_history(path):
    """Yahoo-style SPX history: a three-row header, then OHLCV."""
    df = pd.read_csv(path, skiprows=[1, 2]).rename(columns={"Price": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index().astype(float)
    df["log_ret"] = np.log(df["Close"]).diff()
    return df


def load_vix(path):
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    return df.set_index("DATE").sort_index()


# --------------------------------------------------------------------------


@dataclass
class ChainSlice:
    """One expiry, cleaned, with its forward and discount factor."""

    expiry: pd.Timestamp
    T: float
    spot: float
    forward: float
    discount: float
    strikes: np.ndarray      # strikes with both a call and a put quoted
    call_mid: np.ndarray
    put_mid: np.ndarray
    call_spread: np.ndarray
    put_spread: np.ndarray
    parity_rmse: float
    n_used_for_parity: int

    @property
    def rate(self) -> float:
        return -np.log(self.discount) / self.T

    def otm(self):
        """OTM option prices: puts below the forward, calls above.

        This is what Carr-Madan wants -- the liquid side at every strike.
        """
        use_put = self.strikes < self.forward
        return np.where(use_put, self.put_mid, self.call_mid)


def implied_forward(strikes, call_mid, put_mid, spot, band=0.05, weights=None):
    """Recover (F, D) from put-call parity:  C - K P = D(F - K).

    Regress the parity spread on strike. The slope is -D and the intercept
    is D*F, so F = -intercept/slope. Restricted to strikes within `band`
    of spot, where both legs are liquid and the spread is informative --
    deep wings have one nearly worthless leg and add only noise.
    """
    K = np.asarray(strikes, float)
    y = np.asarray(call_mid, float) - np.asarray(put_mid, float)

    sel = np.abs(K / spot - 1.0) <= band
    if sel.sum() < 5:
        raise ValueError("too few near-the-money strikes for a parity fit")

    w = np.ones(sel.sum()) if weights is None else np.asarray(weights, float)[sel]
    coef = np.polyfit(K[sel], y[sel], 1, w=np.sqrt(w))
    slope, intercept = coef[0], coef[1]

    D = -slope
    F = intercept / D
    resid = y[sel] - (intercept + slope * K[sel])
    return F, D, float(np.sqrt(np.mean(resid**2))), int(sel.sum())


def build_slice(df, meta, expiry, root="SPX", parity_band=0.05,
                require_positive_bid=True):
    """Assemble one cleaned expiry slice with its forward and discount factor."""
    expiry = pd.Timestamp(expiry)
    sub = df[(df["expiry"] == expiry) & (df["root"] == root)]
    if sub.empty:
        raise ValueError(f"no {root} options for {expiry.date()}")

    calls = sub[sub["cp"] == "C"].set_index("strike").sort_index()
    puts = sub[sub["cp"] == "P"].set_index("strike").sort_index()
    K = calls.index.intersection(puts.index)

    if require_positive_bid:
        ok = (calls.loc[K, "bid"] > 0) & (puts.loc[K, "bid"] > 0)
        K = K[ok]

    K = np.asarray(K, float)
    c = calls.loc[K, "mid"].to_numpy()
    p = puts.loc[K, "mid"].to_numpy()
    cs = calls.loc[K, "spread"].to_numpy()
    ps = puts.loc[K, "spread"].to_numpy()

    # Weight the parity fit by inverse quote width: tight markets carry
    # more information about the true forward than wide ones.
    w = 1.0 / np.maximum(cs + ps, 1e-6) ** 2
    F, D, rmse, n = implied_forward(K, c, p, meta["spot"], band=parity_band, weights=w)

    T = (expiry - meta["session"]).days / 365.0

    return ChainSlice(
        expiry=expiry, T=T, spot=meta["spot"], forward=F, discount=D,
        strikes=K, call_mid=c, put_mid=p, call_spread=cs, put_spread=ps,
        parity_rmse=rmse, n_used_for_parity=n,
    )
