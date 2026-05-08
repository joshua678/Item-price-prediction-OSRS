from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantileEdges:
    """
    Conditional quantile edges for a single target:
      - pos_edges: edges for r > +thresh (on r itself)
      - neg_edges: edges for -r > +thresh (on magnitude)
    """
    pos_edges: np.ndarray  # shape (bins_per_side-1,)
    neg_edges: np.ndarray  # shape (bins_per_side-1,)
    bins_per_side: int
    thresh: float

    @property
    def neutral_class(self) -> int:
        return int(self.bins_per_side)


def _ensure_strictly_increasing(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float).copy()
    if edges.size == 0:
        return edges
    eps = 1e-12
    edges = np.maximum.accumulate(edges + eps * np.arange(edges.size))
    return edges


def fit_conditional_quantile_edges(
    returns_train: np.ndarray,
    thresh: float,
    bins_per_side: int = 5,
) -> QuantileEdges:
    """
    Fit conditional quantile cutpoints on TRAIN ONLY.

    returns_train: array of future returns r = (p_{t+h}-p_t)/p_t for training rows.
    thresh: deadzone threshold. |r| <= thresh is neutral and excluded from quantile fitting.
    """
    r = np.asarray(returns_train, dtype=float)
    r = r[np.isfinite(r)]

    # Up-side quantiles on r | r > thresh
    pos = r[r > thresh]
    # Down-side quantiles on (-r) | r < -thresh
    neg = (-r[r < -thresh])

    q = np.arange(1, bins_per_side) / bins_per_side  # e.g. [0.2,0.4,0.6,0.8]

    if pos.size > 0:
        pos_edges = np.quantile(pos, q)
    else:
        # fallback edges: multiples of thresh (still monotone)
        pos_edges = thresh * (np.arange(1, bins_per_side))
    if neg.size > 0:
        neg_edges = np.quantile(neg, q)
    else:
        neg_edges = thresh * (np.arange(1, bins_per_side))

    pos_edges = _ensure_strictly_increasing(pos_edges)
    neg_edges = _ensure_strictly_increasing(neg_edges)

    return QuantileEdges(
        pos_edges=np.asarray(pos_edges, dtype=float),
        neg_edges=np.asarray(neg_edges, dtype=float),
        bins_per_side=int(bins_per_side),
        thresh=float(thresh),
    )


def label_returns_with_edges(
    returns: np.ndarray,
    edges: QuantileEdges,
    *,
    ignore_idx: int,
    horizon: int,
) -> np.ndarray:
    """
    Map returns to 2*bins_per_side+1 classes:
      0..(bins_per_side-1)  : down bins (0 = biggest down)
      bins_per_side         : neutral
      (bins_per_side+1)..(2*bins_per_side): up bins (2*bins_per_side = biggest up)
    """
    r = np.asarray(returns, dtype=float)
    n = r.shape[0]
    neutral = edges.neutral_class
    labels = np.full(n, neutral, dtype=np.int64)

    finite = np.isfinite(r)

    up = finite & (r > edges.thresh)
    dn = finite & (r < -edges.thresh)

    # Up bins: smallest up -> class neutral+1, biggest up -> class neutral+bins_per_side
    if up.any():
        b_up = np.digitize(r[up], edges.pos_edges, right=True)  # 0..bins_per_side-1
        labels[up] = neutral + 1 + b_up  # (neutral+1)..(neutral+bins_per_side)

    # Down bins: smallest down -> class neutral-1, biggest down -> class 0
    if dn.any():
        mag_dn = -r[dn]
        b_dn = np.digitize(mag_dn, edges.neg_edges, right=True)  # 0..bins_per_side-1
        labels[dn] = (neutral - 1) - b_dn  # (neutral-1)..0

    if horizon > 0:
        labels[-horizon:] = ignore_idx

    return labels


def compute_future_returns(
    df: pd.DataFrame,
    horizon: int,
    price_col: str,
) -> np.ndarray:
    curr = df[price_col].astype(float).to_numpy()
    nxt = df[price_col].shift(-horizon).astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (nxt - curr) / curr
    return r


def make_quantile_labels(
    df: pd.DataFrame,
    horizon: int,
    edges_high: QuantileEdges,
    edges_low: QuantileEdges,
    *,
    ignore_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 11-class labels for both heads using pre-fit quantile edges.
    """
    r_high = compute_future_returns(df, horizon=horizon, price_col="avgHighPrice")
    r_low = compute_future_returns(df, horizon=horizon, price_col="avgLowPrice")

    y_high = label_returns_with_edges(r_high, edges_high, ignore_idx=ignore_idx, horizon=horizon)
    y_low = label_returns_with_edges(r_low, edges_low, ignore_idx=ignore_idx, horizon=horizon)

    return y_high, y_low


def class_to_direction(y: np.ndarray, neutral_class: int, ignore_idx: int) -> np.ndarray:
    """
    Map 11-class labels to {0=down, 1=neutral, 2=up}. ignore_idx stays as ignore_idx.
    """
    y = np.asarray(y)
    out = np.full_like(y, fill_value=1, dtype=np.int64)  # default neutral
    out[y == ignore_idx] = ignore_idx
    m = (y != ignore_idx)

    out[m & (y < neutral_class)] = 0
    out[m & (y > neutral_class)] = 2
    out[m & (y == neutral_class)] = 1
    return out
