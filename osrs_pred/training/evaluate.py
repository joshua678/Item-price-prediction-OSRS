from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataloading import transfer_kwargs
from .metrics import eval_quantiles
from ..models.lstm import LSTMQuantileRegressor


def load_best(model: LSTMQuantileRegressor, ckpt_path: Path, device: torch.device) -> list[float]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return list(ckpt["quantiles"])


@torch.no_grad()
def predict_on_test(
    model: LSTMQuantileRegressor,
    dl_test: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yH, yL, qH, qL = [], [], [], []
    model.eval()
    to_kwargs = transfer_kwargs(device)
    for X, tH, tL, item_id in dl_test:
        X = X.to(device, **to_kwargs)
        item_id = item_id.to(device, **to_kwargs)
        outH, outL = model(X, item_id)
        yH.append(tH.numpy())
        yL.append(tL.numpy())
        qH.append(outH.detach().cpu().numpy())
        qL.append(outL.detach().cpu().numpy())
    return (
        np.concatenate(yH, axis=0),
        np.concatenate(qH, axis=0),
        np.concatenate(yL, axis=0),
        np.concatenate(qL, axis=0),
    )


def report_quantiles(head: str, y_true: np.ndarray, q_pred: np.ndarray, quantiles: list[float]) -> None:
    q = np.array(quantiles, dtype=float)
    qe = eval_quantiles(y_true, q_pred, q)

    print(f"\nQuantile report - {head}")
    print("quantiles:", q.tolist())
    print("coverage :", np.round(qe.coverage, 3).tolist())
    print(f"mean|cov-q|: {qe.mean_abs_cov_err:.3f} | median|cov-q|: {qe.median_abs_cov_err:.3f}")
    if 0.1 in quantiles and 0.9 in quantiles:
        print(f"mean width (0.9-0.1): {qe.mean_width_10_90:.5f}")


def report_quantiles_by_horizon(
    head: str,
    y_true: np.ndarray,
    q_pred: np.ndarray,
    quantiles: list[float],
    horizon_labels: list[str],
) -> None:
    report_quantiles(f"{head} all horizons", y_true, q_pred, quantiles)

    q = np.array(quantiles, dtype=float)
    y = np.asarray(y_true)
    qp = np.asarray(q_pred)
    if qp.ndim != 3:
        return
    if len(horizon_labels) != qp.shape[1]:
        horizon_labels = [f"horizon_{i}" for i in range(qp.shape[1])]

    print(f"{head} by horizon:")
    for h_idx, label in enumerate(horizon_labels):
        qe = eval_quantiles(y[:, h_idx], qp[:, h_idx, :], q)
        line = (
            f"  {label}: mean|cov-q| {qe.mean_abs_cov_err:.3f} | "
            f"median|cov-q| {qe.median_abs_cov_err:.3f}"
        )
        if 0.1 in quantiles and 0.9 in quantiles:
            line += f" | width {qe.mean_width_10_90:.5f}"
        print(line)
