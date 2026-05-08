from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QuantileEval:
    coverage: np.ndarray  # shape (Q,)
    mean_abs_cov_err: float
    median_abs_cov_err: float
    mean_width_10_90: float


def eval_quantiles(y_true: np.ndarray, q_pred: np.ndarray, quantiles: np.ndarray) -> QuantileEval:
    """
    y_true: (N,) or (N, H)
    q_pred: (N, Q) or (N, H, Q)
    quantiles: (Q,)
    """
    y = np.asarray(y_true, dtype=float)
    qp_all = np.asarray(q_pred, dtype=float)
    Q = qp_all.shape[-1]
    qs = np.asarray(quantiles, dtype=float).reshape(1, Q)

    if qp_all.ndim == 3:
        qp_all = qp_all.reshape(-1, Q)
        y = y.reshape(-1)
    elif y.ndim > 1:
        y = y.reshape(-1)

    m = np.isfinite(y)
    if m.sum() == 0:
        return QuantileEval(
            coverage=np.zeros(Q, dtype=float),
            mean_abs_cov_err=0.0,
            median_abs_cov_err=0.0,
            mean_width_10_90=0.0,
        )

    y = y[m]
    qp = qp_all[m]

    cov = np.mean(y[:, None] <= qp, axis=0)  # (Q,)
    abs_err = np.abs(cov - qs.reshape(-1))

    # interval width (0.9 - 0.1) if available
    width = 0.0
    if Q >= 2 and (0.1 in quantiles) and (0.9 in quantiles):
        i10 = int(np.where(np.isclose(quantiles, 0.1))[0][0])
        i90 = int(np.where(np.isclose(quantiles, 0.9))[0][0])
        width = float(np.mean(qp[:, i90] - qp[:, i10]))

    return QuantileEval(
        coverage=cov,
        mean_abs_cov_err=float(np.mean(abs_err)),
        median_abs_cov_err=float(np.median(abs_err)),
        mean_width_10_90=float(width),
    )
