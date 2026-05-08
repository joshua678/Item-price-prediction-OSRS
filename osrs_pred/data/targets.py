from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import numpy as np
import pandas as pd


def compute_future_returns_pct(df: pd.DataFrame, horizon: int, price_col: str) -> np.ndarray:
    """
    r_t = (p_{t+h} - p_t) / p_t
    """
    curr = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
    nxt = pd.to_numeric(df[price_col], errors="coerce").shift(-horizon).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (nxt - curr) / curr
    return r


def make_return_targets(
    df: pd.DataFrame,
    *,
    horizons: Sequence[int],
    no_buys_col: str = "no_buys_1h",
    no_sells_col: str = "no_sells_1h",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_high: float array of future returns from avgHighPrice
      y_low:  float array of future returns from avgLowPrice

    Each array is shape (N, H), preserving the supplied horizon order.

    Invalid targets are set to NaN:
      - last `horizon` rows for each horizon
      - rows where no_buys_1h/no_sells_1h indicate missing side liquidity
    """
    horizon_values = tuple(int(h) for h in horizons)

    if not horizon_values:
        raise ValueError("horizons must not be empty.")
    if any(h <= 0 for h in horizon_values):
        raise ValueError(f"horizons must contain positive integers. Got: {horizon_values}")

    y_high = np.column_stack(
        [
            compute_future_returns_pct(df, horizon=h, price_col="avgHighPrice")
            for h in horizon_values
        ]
    )
    y_low = np.column_stack(
        [
            compute_future_returns_pct(df, horizon=h, price_col="avgLowPrice")
            for h in horizon_values
        ]
    )

    for col, h in enumerate(horizon_values):
        y_high[-h:, col] = np.nan
        y_low[-h:, col] = np.nan

    if no_buys_col in df.columns:
        nb = df[no_buys_col].astype(bool).to_numpy()
        y_high[nb, :] = np.nan
    if no_sells_col in df.columns:
        ns = df[no_sells_col].astype(bool).to_numpy()
        y_low[ns, :] = np.nan

    return y_high, y_low
