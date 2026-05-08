from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class UnconditionalBaseline:
    quantiles: np.ndarray          # (Q,)
    q_values: np.ndarray           # (Q,) constant predictions
    pinball: float
    coverage: np.ndarray           # (Q,)
    cov_err_mean: float
    width_10_90: float


def unconditional_quantile_baseline(
    y_true: np.ndarray,
    quantiles: Sequence[float],
) -> UnconditionalBaseline:
    """
    Unconditional baseline: predict constant quantiles for all samples,
    fitted on the *validation* targets themselves (oracle unconditional baseline).

    y_true may contain NaNs (ignored).
    """
    qs = np.asarray(list(quantiles), dtype=float)
    y = np.asarray(y_true, dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        Q = qs.size
        return UnconditionalBaseline(
            quantiles=qs,
            q_values=np.zeros(Q, dtype=float),
            pinball=0.0,
            coverage=np.zeros(Q, dtype=float),
            cov_err_mean=0.0,
            width_10_90=0.0,
        )

    # Empirical quantiles (constant predictions)
    q_vals = np.quantile(y, qs)

    # Pinball loss: mean over quantiles per sample, then mean over samples
    # e = y - q
    e = y[:, None] - q_vals[None, :]                     # (N, Q)
    loss_q = np.maximum(qs[None, :] * e, (qs[None, :] - 1.0) * e)  # (N, Q)
    pinball = float(loss_q.mean(axis=1).mean())

    # Coverage calibration: P(y <= qhat(q))
    coverage = (y[:, None] <= q_vals[None, :]).mean(axis=0)  # (Q,)
    cov_err_mean = float(np.mean(np.abs(coverage - qs)))

    # Width of the 0.1 to 0.9 interval, if present.
    width_10_90 = 0.0
    if np.any(np.isclose(qs, 0.1)) and np.any(np.isclose(qs, 0.9)):
        i10 = int(np.where(np.isclose(qs, 0.1))[0][0])
        i90 = int(np.where(np.isclose(qs, 0.9))[0][0])
        width_10_90 = float(q_vals[i90] - q_vals[i10])

    return UnconditionalBaseline(
        quantiles=qs,
        q_values=q_vals,
        pinball=pinball,
        coverage=coverage.astype(float),
        cov_err_mean=cov_err_mean,
        width_10_90=width_10_90,
    )
