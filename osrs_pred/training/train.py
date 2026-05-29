from __future__ import annotations

from contextlib import nullcontext
from collections import defaultdict
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from time import perf_counter

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baselines import unconditional_quantile_baseline
from .dataloading import make_data_loader, move_batch_to_device, transfer_kwargs
from ..config import DEFAULT_VALIDATIONS_PER_EPOCH
from ..data.dataset import PriceSequenceDataset
from ..models.lstm import LSTMQuantileRegressor
from .metrics import eval_quantiles


EPOCH_PROGRESS_UPDATE_EVERY = 25
TRAINING_LOG_NAME = "training_log.json"


def _epoch_progress_enabled() -> bool:
    v = os.getenv("OSRS_PRED_EPOCH_PROGRESS", "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


def _should_update_epoch_progress(batch_idx: int, total_batches: int) -> bool:
    return batch_idx % EPOCH_PROGRESS_UPDATE_EVERY == 0 or batch_idx == total_batches


def _validation_checkpoints(total_batches: int, validations_per_epoch: int) -> set[int]:
    if validations_per_epoch <= 0:
        return {total_batches}
    fractions = np.arange(1, validations_per_epoch + 1) / validations_per_epoch
    raw = fractions * total_batches
    return {int(np.ceil(v)) for v in raw}


def _progress_loader(dl: DataLoader, desc: str, enabled: bool):
    if not enabled:
        return nullcontext(dl)
    return tqdm(
        dl,
        total=len(dl),
        desc=desc,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        miniters=EPOCH_PROGRESS_UPDATE_EVERY,
    )


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def _baseline_to_dict(name: str, baseline) -> dict:
    return {
        "name": name,
        "quantiles": baseline.quantiles,
        "q_values": baseline.q_values,
        "pinball": baseline.pinball,
        "coverage": baseline.coverage,
        "mean_abs_coverage_error": baseline.cov_err_mean,
        "width_10_90": baseline.width_10_90,
    }


def _pinball_by_horizon_np(
    q_pred: np.ndarray,
    y_true: np.ndarray,
    quantiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_pred = np.asarray(q_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if q_pred.ndim == 2:
        q_pred = q_pred[:, None, :]
    if y_true.ndim == 1:
        y_true = y_true[:, None]

    n_horizons = y_true.shape[1]
    losses = np.full(n_horizons, np.nan, dtype=float)
    counts = np.zeros(n_horizons, dtype=np.int64)
    q = np.asarray(quantiles, dtype=float).reshape(1, -1)

    for h_idx in range(n_horizons):
        mask = np.isfinite(y_true[:, h_idx]) & np.all(np.isfinite(q_pred[:, h_idx, :]), axis=1)
        counts[h_idx] = int(mask.sum())
        if counts[h_idx] == 0:
            continue
        yv = y_true[mask, h_idx][:, None]
        qv = q_pred[mask, h_idx, :]
        e = yv - qv
        loss_q = np.maximum(q * e, (q - 1.0) * e)
        losses[h_idx] = float(loss_q.mean(axis=1).mean())

    return losses, counts


def _baseline_pinball_by_horizon(
    y_true: np.ndarray,
    quantiles: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    if y.ndim == 1:
        y = y[:, None]

    losses = np.full(y.shape[1], np.nan, dtype=float)
    counts = np.zeros(y.shape[1], dtype=np.int64)
    for h_idx in range(y.shape[1]):
        values = y[:, h_idx]
        counts[h_idx] = int(np.isfinite(values).sum())
        if counts[h_idx] == 0:
            continue
        losses[h_idx] = unconditional_quantile_baseline(values, quantiles=quantiles).pinball
    return losses, counts


def _weighted_mean_by_horizon(
    high_values: np.ndarray,
    high_counts: np.ndarray,
    low_values: np.ndarray,
    low_counts: np.ndarray,
) -> np.ndarray:
    high_values = np.asarray(high_values, dtype=float)
    low_values = np.asarray(low_values, dtype=float)
    high_counts = np.asarray(high_counts, dtype=float)
    low_counts = np.asarray(low_counts, dtype=float)
    total_counts = high_counts + low_counts
    numerator = np.nan_to_num(high_values, nan=0.0) * high_counts + np.nan_to_num(low_values, nan=0.0) * low_counts
    out = np.full_like(numerator, np.nan, dtype=float)
    valid = total_counts > 0
    out[valid] = numerator[valid] / total_counts[valid]
    return out


def _skill_by_horizon(model_pinball: np.ndarray, baseline_pinball: np.ndarray) -> np.ndarray:
    model_pinball = np.asarray(model_pinball, dtype=float)
    baseline_pinball = np.asarray(baseline_pinball, dtype=float)
    out = np.full_like(model_pinball, np.nan, dtype=float)
    valid = np.isfinite(model_pinball) & np.isfinite(baseline_pinball) & (baseline_pinball > 0)
    out[valid] = 1.0 - model_pinball[valid] / baseline_pinball[valid]
    return out


def _horizon_log_fields(
    val_by_horizon: dict[str, np.ndarray],
    *,
    baseH_pinball_by_horizon: np.ndarray,
    baseL_pinball_by_horizon: np.ndarray,
    baseline_pinball_by_horizon: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "val_high_pinball_by_horizon": val_by_horizon["high_pinball"],
        "val_low_pinball_by_horizon": val_by_horizon["low_pinball"],
        "val_pinball_by_horizon": val_by_horizon["pinball"],
        "val_high_count_by_horizon": val_by_horizon["high_count"],
        "val_low_count_by_horizon": val_by_horizon["low_count"],
        "val_count_by_horizon": val_by_horizon["count"],
        "val_high_baseline_pinball_by_horizon": baseH_pinball_by_horizon,
        "val_low_baseline_pinball_by_horizon": baseL_pinball_by_horizon,
        "val_baseline_pinball_by_horizon": baseline_pinball_by_horizon,
        "val_high_skill_vs_unconditional_by_horizon": _skill_by_horizon(
            val_by_horizon["high_pinball"],
            baseH_pinball_by_horizon,
        ),
        "val_low_skill_vs_unconditional_by_horizon": _skill_by_horizon(
            val_by_horizon["low_pinball"],
            baseL_pinball_by_horizon,
        ),
        "val_skill_vs_unconditional_by_horizon": _skill_by_horizon(
            val_by_horizon["pinball"],
            baseline_pinball_by_horizon,
        ),
    }


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_scaler(train_arrays: List[np.ndarray]) -> StandardScaler:
    scaler = StandardScaler(copy=False)
    for arr in train_arrays:
        scaler.partial_fit(arr.astype(np.float32, copy=False))
    scaler.scale_[scaler.scale_ == 0] = 1.0
    return scaler


def ensure_float32_in_place(arrays: List[np.ndarray]) -> List[np.ndarray]:
    for i, arr in enumerate(arrays):
        arrays[i] = arr.astype(np.float32, copy=False)
    return arrays


def transform_float32(scaler: StandardScaler, arrays: List[np.ndarray]) -> List[np.ndarray]:
    for i, arr in enumerate(arrays):
        arrays[i] = scaler.transform(
            arr.astype(np.float32, copy=False),
            copy=False,
        ).astype(np.float32, copy=False)
    return arrays


def count_sequences(arrays: List[np.ndarray], seq_len: int, seq_stride: int) -> int:
    stride = int(seq_stride)
    if stride <= 0:
        raise ValueError(f"seq_stride must be > 0. Got {seq_stride}.")
    total = 0
    for arr in arrays:
        n = int(arr.shape[0])
        if n > seq_len:
            total += (n - seq_len + stride - 1) // stride
    return total


def pinball_loss_sum(
    q_pred: torch.Tensor,  # (B, H, Q)
    y: torch.Tensor,  # (B, H)
    quantiles: torch.Tensor,  # (Q,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (sum_loss_over_valid_targets, count_valid_targets).
    Loss is mean over quantiles per target/horizon, then summed over valid targets.
    """
    if q_pred.ndim == 2:
        q_pred = q_pred.unsqueeze(1)
    if y.ndim == 1:
        y = y.unsqueeze(1)

    # Valid targets are finite. NaN means "ignore".
    mask = torch.isfinite(y)
    if mask.sum() == 0:
        return torch.zeros((), device=y.device), torch.zeros((), device=y.device)

    yv = y[mask].unsqueeze(1)
    qv = q_pred[mask]

    e = yv - qv
    q = quantiles.view(1, -1)
    loss_q = torch.maximum(q * e, (q - 1.0) * e)
    loss_per_sample = loss_q.mean(dim=1)
    return loss_per_sample.sum(), torch.tensor(loss_per_sample.numel(), device=y.device, dtype=torch.float32)


@torch.no_grad()
def predict_on_loader(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    *,
    progress_desc: str | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yH, yL = [], []
    qH, qL = [], []
    model.eval()
    progress_enabled = show_progress and progress_desc is not None
    to_kwargs = transfer_kwargs(device)
    with _progress_loader(dl, progress_desc or "predict", progress_enabled) as it:
        for X, tH, tL, item_id in it:
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


@torch.no_grad()
def run_full_validation(
    model: nn.Module,
    dl_val: DataLoader,
    device: torch.device,
    q_tensor: torch.Tensor,
    quantiles_np: np.ndarray,
    *,
    progress_desc: str,
    show_progress: bool,
) -> tuple[float, object, object, dict[str, np.ndarray]]:
    model.eval()
    v_sum_loss = 0.0
    v_sum_count = 0.0
    val_batches = len(dl_val)

    with _progress_loader(dl_val, f"{progress_desc} loss", show_progress) as val_iter:
        for batch_idx, batch in enumerate(val_iter, start=1):
            X, yH, yL, item_id = move_batch_to_device(batch, device)

            qH_pred, qL_pred = model(X, item_id)
            lH, cH = pinball_loss_sum(qH_pred, yH, q_tensor)
            lL, cL = pinball_loss_sum(qL_pred, yL, q_tensor)
            v_sum_loss += float(lH.detach().cpu()) + float(lL.detach().cpu())
            v_sum_count += float(cH.detach().cpu()) + float(cL.detach().cpu())

            if hasattr(val_iter, "set_postfix") and _should_update_epoch_progress(batch_idx, val_batches):
                val_iter.set_postfix(pinball=f"{v_sum_loss / max(v_sum_count, 1.0):.5f}")

    val_loss = v_sum_loss / max(v_sum_count, 1.0)
    yH_true, qH_all, yL_true, qL_all = predict_on_loader(
        model,
        dl_val,
        device=device,
        progress_desc=f"{progress_desc} calibrate",
        show_progress=show_progress,
    )
    qeH = eval_quantiles(yH_true, qH_all, quantiles_np)
    qeL = eval_quantiles(yL_true, qL_all, quantiles_np)
    high_pinball_by_horizon, high_counts_by_horizon = _pinball_by_horizon_np(qH_all, yH_true, quantiles_np)
    low_pinball_by_horizon, low_counts_by_horizon = _pinball_by_horizon_np(qL_all, yL_true, quantiles_np)
    val_by_horizon = {
        "high_pinball": high_pinball_by_horizon,
        "low_pinball": low_pinball_by_horizon,
        "pinball": _weighted_mean_by_horizon(
            high_pinball_by_horizon,
            high_counts_by_horizon,
            low_pinball_by_horizon,
            low_counts_by_horizon,
        ),
        "high_count": high_counts_by_horizon,
        "low_count": low_counts_by_horizon,
        "count": high_counts_by_horizon + low_counts_by_horizon,
    }
    return val_loss, qeH, qeL, val_by_horizon


def train_model(
    *,
    train_arrays: List[np.ndarray],
    val_arrays: List[np.ndarray],
    test_arrays: List[np.ndarray],
    train_yH: List[np.ndarray],
    val_yH: List[np.ndarray],
    test_yH: List[np.ndarray],
    train_yL: List[np.ndarray],
    val_yL: List[np.ndarray],
    test_yL: List[np.ndarray],
    train_idx: List[np.ndarray],
    val_idx: List[np.ndarray],
    test_idx: List[np.ndarray],
    feat_cols: List[str],
    n_items: int,
    n_horizons: int,
    quantiles: tuple[float, ...],
    device: torch.device,
    output_dir: Path,
    seq_len: int,
    stride: int,
    batch_size: int,
    lr: float,
    epochs: int,
    id_emb_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    trained_items: List[str],
    seed: int,
    eval_stride: int = 1,
    validations_per_epoch: int = DEFAULT_VALIDATIONS_PER_EPOCH,
    metadata: dict | None = None,
) -> tuple[LSTMQuantileRegressor, Dict[str, List[float]], StandardScaler]:
    seed_everything(seed)
    if validations_per_epoch < 0:
        raise ValueError(f"validations_per_epoch must be >= 0. Got {validations_per_epoch}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_float32_in_place(train_arrays)
    ensure_float32_in_place(val_arrays)
    ensure_float32_in_place(test_arrays)

    scaler = fit_scaler(train_arrays)
    train_arrays = transform_float32(scaler, train_arrays)
    val_arrays = transform_float32(scaler, val_arrays)
    test_arrays = transform_float32(scaler, test_arrays)

    ds_train = PriceSequenceDataset(train_arrays, train_yH, train_yL, train_idx, seq_len=seq_len, seq_stride=stride)
    ds_val = PriceSequenceDataset(val_arrays, val_yH, val_yL, val_idx, seq_len=seq_len, seq_stride=eval_stride)
    test_sequences = count_sequences(test_arrays, seq_len=seq_len, seq_stride=eval_stride)

    if len(ds_train) == 0 or len(ds_val) == 0 or test_sequences == 0:
        raise RuntimeError(
            "One or more dataset splits produced zero sequences. "
            f"Train={len(ds_train):,}, Val={len(ds_val):,}, Test={test_sequences:,}. "
            "Reduce SEQ_LEN, widen the date ranges, or lower the coverage threshold."
        )

    print(f"Train sequences: {len(ds_train):,} | Val: {len(ds_val):,} | Test: {test_sequences:,}")

    dl_train = make_data_loader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True, device=device)
    dl_val = make_data_loader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False, device=device)

    Q = len(quantiles)
    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=device)

    model = LSTMQuantileRegressor(
        n_features=len(feat_cols),
        n_items=n_items,
        n_quantiles=Q,
        n_horizons=n_horizons,
        id_emb_dim=id_emb_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    first = next(iter(dl_train))
    X0, yH0, yL0, item0 = first
    print(f"[debug] batch X shape: {tuple(X0.shape)} (B,T,F)")
    print(f"[debug] batch target shape: H={tuple(yH0.shape)} L={tuple(yL0.shape)}")
    print(f"[debug] model hidden_size: {model.lstm.hidden_size}")
    print(f"[debug] model num_layers: {model.lstm.num_layers}")
    print(f"[debug] model num params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[debug] dropout: {model.lstm.dropout}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    history: Dict[str, List[float]] = defaultdict(list)
    best_val_loss = float("inf")
    best_ckpt = None

    yH_val_all, yL_val_all = [], []
    for _, tH, tL, _ in dl_val:
        yH_val_all.append(tH.numpy())
        yL_val_all.append(tL.numpy())
    yH_val_all = np.concatenate(yH_val_all, axis=0)
    yL_val_all = np.concatenate(yL_val_all, axis=0)

    baseH = unconditional_quantile_baseline(yH_val_all, quantiles=quantiles)
    baseL = unconditional_quantile_baseline(yL_val_all, quantiles=quantiles)
    baseH_pinball_by_horizon, baseH_counts_by_horizon = _baseline_pinball_by_horizon(yH_val_all, quantiles)
    baseL_pinball_by_horizon, baseL_counts_by_horizon = _baseline_pinball_by_horizon(yL_val_all, quantiles)
    baseline_pinball_by_horizon = _weighted_mean_by_horizon(
        baseH_pinball_by_horizon,
        baseH_counts_by_horizon,
        baseL_pinball_by_horizon,
        baseL_counts_by_horizon,
    )

    print(
        "Unconditional baseline (val-fit) | "
        f"pinball H {baseH.pinball:.5f} L {baseL.pinball:.5f} | "
        f"cov_err(mean) H {baseH.cov_err_mean:.3f} L {baseL.cov_err_mean:.3f} | "
        f"width(0.1-0.9) H {baseH.width_10_90:.4f} L {baseL.width_10_90:.4f}"
    )

    baseline_pinball_mean = (baseH.pinball + baseL.pinball) / 2
    horizon_labels = [
        f"{h} step{'s' if h != 1 else ''}"
        for h in range(1, n_horizons + 1)
    ]
    if metadata and len(metadata.get("pred_horizons_steps", [])) == n_horizons:
        horizon_labels = [
            f"{h} step{'s' if h != 1 else ''}"
            for h in metadata["pred_horizons_steps"]
        ]
    training_log_path = output_dir / TRAINING_LOG_NAME
    training_log = {
        "schema_version": 1,
        "status": "running",
        "description": "Per-epoch training metrics for OSRS price quantile forecasting.",
        "metadata": metadata or {},
        "data": {
            "n_items": n_items,
            "n_features": len(feat_cols),
            "n_horizons": n_horizons,
            "n_quantiles": Q,
            "train_sequences": len(ds_train),
            "val_sequences": len(ds_val),
            "test_sequences": test_sequences,
            "train_batches": len(dl_train),
            "val_batches": len(dl_val),
            "trained_items": trained_items,
            "feature_columns": feat_cols,
        },
        "hyperparameters": {
            "seq_len": seq_len,
            "stride": stride,
            "eval_stride": eval_stride,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "validations_per_epoch": validations_per_epoch,
            "id_emb_dim": id_emb_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "seed": seed,
            "device": device,
            "weight_decay": 1e-3,
            "gradient_clip_norm": 1.0,
        },
        "quantiles": list(quantiles),
        "baselines": {
            "validation_unconditional_high": _baseline_to_dict("validation_unconditional_high", baseH),
            "validation_unconditional_low": _baseline_to_dict("validation_unconditional_low", baseL),
            "validation_unconditional_mean_pinball": baseline_pinball_mean,
            "validation_unconditional_high_pinball_by_horizon": baseH_pinball_by_horizon,
            "validation_unconditional_low_pinball_by_horizon": baseL_pinball_by_horizon,
            "validation_unconditional_pinball_by_horizon": baseline_pinball_by_horizon,
            "horizon_labels": horizon_labels,
        },
        "epochs": [],
    }
    _atomic_write_json(training_log, training_log_path)
    print(f"Training metrics JSON will be updated at: {training_log_path}")

    show_epoch_progress = _epoch_progress_enabled()
    train_batches = len(dl_train)
    checkpoint_batches = _validation_checkpoints(train_batches, validations_per_epoch)
    expected_val_checkpoints = len(checkpoint_batches) * epochs
    quantiles_np = np.array(quantiles, dtype=float)
    print(
        "Full validation checkpoints: "
        f"{len(checkpoint_batches)} per epoch, {expected_val_checkpoints} total "
        f"(set OSRS_VALIDATIONS_PER_EPOCH=0 for end-of-epoch only)."
    )
    training_log["validation_cadence"] = {
        "validations_per_epoch_requested": validations_per_epoch,
        "checkpoint_batches": sorted(checkpoint_batches),
        "checkpoints_per_epoch": len(checkpoint_batches),
        "expected_total_checkpoints": expected_val_checkpoints,
        "includes_end_of_epoch": train_batches in checkpoint_batches,
    }
    training_log["validation_checkpoints"] = []
    _atomic_write_json(training_log, training_log_path)

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_epoch0 = perf_counter()

        model.train()
        sum_loss = 0.0
        sum_count = 0.0
        last_checkpoint = None

        with _progress_loader(
            dl_train,
            f"Epoch {epoch:02d}/{epochs:02d} train",
            show_epoch_progress,
        ) as train_iter:
            for batch_idx, batch in enumerate(train_iter, start=1):
                X, yH, yL, item_id = move_batch_to_device(batch, device)

                optimiser.zero_grad(set_to_none=True)
                qH_pred, qL_pred = model(X, item_id)

                lH, cH = pinball_loss_sum(qH_pred, yH, q_tensor)
                lL, cL = pinball_loss_sum(qL_pred, yL, q_tensor)

                denom = (cH + cL).clamp_min(1.0)
                loss = (lH + lL) / denom

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

                sum_loss += float(lH.detach().cpu()) + float(lL.detach().cpu())
                sum_count += float(cH.detach().cpu()) + float(cL.detach().cpu())

                if hasattr(train_iter, "set_postfix") and _should_update_epoch_progress(batch_idx, train_batches):
                    train_iter.set_postfix(pinball=f"{sum_loss / max(sum_count, 1.0):.5f}")

                if batch_idx in checkpoint_batches:
                    train_running = sum_loss / max(sum_count, 1.0)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    checkpoint_start = perf_counter()
                    val_loss, qeH, qeL, val_by_horizon = run_full_validation(
                        model,
                        dl_val,
                        device,
                        q_tensor,
                        quantiles_np,
                        progress_desc=f"Epoch {epoch:02d}/{epochs:02d} val {batch_idx}/{train_batches}",
                        show_progress=show_epoch_progress,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    checkpoint_seconds = perf_counter() - checkpoint_start
                    skill = 1 - val_loss / baseline_pinball_mean if baseline_pinball_mean > 0 else float("nan")
                    current_lr = optimiser.param_groups[0]["lr"]
                    horizon_fields = _horizon_log_fields(
                        val_by_horizon,
                        baseH_pinball_by_horizon=baseH_pinball_by_horizon,
                        baseL_pinball_by_horizon=baseL_pinball_by_horizon,
                        baseline_pinball_by_horizon=baseline_pinball_by_horizon,
                    )
                    checkpoint = {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "train_batches": train_batches,
                        "epoch_fraction": batch_idx / train_batches,
                        "global_checkpoint": len(training_log["validation_checkpoints"]) + 1,
                        "train_pinball_running": train_running,
                        "val_pinball": val_loss,
                        "val_baseline_mean_pinball": baseline_pinball_mean,
                        "val_skill_vs_unconditional": skill,
                        "val_high_mean_abs_coverage_error": qeH.mean_abs_cov_err,
                        "val_low_mean_abs_coverage_error": qeL.mean_abs_cov_err,
                        "val_high_median_abs_coverage_error": qeH.median_abs_cov_err,
                        "val_low_median_abs_coverage_error": qeL.median_abs_cov_err,
                        "val_high_width_10_90": qeH.mean_width_10_90,
                        "val_low_width_10_90": qeL.mean_width_10_90,
                        "val_high_coverage": qeH.coverage,
                        "val_low_coverage": qeL.coverage,
                        **horizon_fields,
                        "lr": current_lr,
                        "seconds_since_epoch_start": perf_counter() - t_epoch0,
                        "validation_seconds": checkpoint_seconds,
                    }
                    training_log["validation_checkpoints"].append(checkpoint)
                    last_checkpoint = checkpoint
                    _atomic_write_json(training_log, training_log_path)
                    model.train()

        train_loss = sum_loss / max(sum_count, 1.0)

        if last_checkpoint is None:
            val_loss, qeH, qeL, val_by_horizon = run_full_validation(
                model,
                dl_val,
                device,
                q_tensor,
                quantiles_np,
                progress_desc=f"Epoch {epoch:02d}/{epochs:02d} val",
                show_progress=show_epoch_progress,
            )
            skill = 1 - val_loss / baseline_pinball_mean if baseline_pinball_mean > 0 else float("nan")
            horizon_fields = _horizon_log_fields(
                val_by_horizon,
                baseH_pinball_by_horizon=baseH_pinball_by_horizon,
                baseL_pinball_by_horizon=baseL_pinball_by_horizon,
                baseline_pinball_by_horizon=baseline_pinball_by_horizon,
            )
        else:
            val_loss = float(last_checkpoint["val_pinball"])
            skill = float(last_checkpoint["val_skill_vs_unconditional"])
            horizon_fields = {
                key: last_checkpoint[key]
                for key in (
                    "val_high_pinball_by_horizon",
                    "val_low_pinball_by_horizon",
                    "val_pinball_by_horizon",
                    "val_high_count_by_horizon",
                    "val_low_count_by_horizon",
                    "val_count_by_horizon",
                    "val_high_baseline_pinball_by_horizon",
                    "val_low_baseline_pinball_by_horizon",
                    "val_baseline_pinball_by_horizon",
                    "val_high_skill_vs_unconditional_by_horizon",
                    "val_low_skill_vs_unconditional_by_horizon",
                    "val_skill_vs_unconditional_by_horizon",
                )
            }
            qeH = SimpleNamespace(
                mean_abs_cov_err=last_checkpoint["val_high_mean_abs_coverage_error"],
                median_abs_cov_err=last_checkpoint["val_high_median_abs_coverage_error"],
                mean_width_10_90=last_checkpoint["val_high_width_10_90"],
                coverage=last_checkpoint["val_high_coverage"],
            )
            qeL = SimpleNamespace(
                mean_abs_cov_err=last_checkpoint["val_low_mean_abs_coverage_error"],
                median_abs_cov_err=last_checkpoint["val_low_median_abs_coverage_error"],
                mean_width_10_90=last_checkpoint["val_low_width_10_90"],
                coverage=last_checkpoint["val_low_coverage"],
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cov_err_mean_H"].append(qeH.mean_abs_cov_err)
        history["val_cov_err_mean_L"].append(qeL.mean_abs_cov_err)
        history["val_width_10_90_H"].append(qeH.mean_width_10_90)
        history["val_width_10_90_L"].append(qeL.mean_width_10_90)
        history["val_skill_vs_unconditional"].append(skill)
        history["val_skill_vs_unconditional_by_horizon"].append(
            horizon_fields["val_skill_vs_unconditional_by_horizon"]
        )

        current_lr = optimiser.param_groups[0]["lr"]

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_s = perf_counter() - t_epoch0
        history["lr"].append(current_lr)
        history["epoch_seconds"].append(epoch_s)

        training_log["epochs"].append(
            {
                "epoch": epoch,
                "train_pinball": train_loss,
                "val_pinball": val_loss,
                "val_baseline_mean_pinball": baseline_pinball_mean,
                "val_skill_vs_unconditional": skill,
                "val_high_mean_abs_coverage_error": qeH.mean_abs_cov_err,
                "val_low_mean_abs_coverage_error": qeL.mean_abs_cov_err,
                "val_high_median_abs_coverage_error": qeH.median_abs_cov_err,
                "val_low_median_abs_coverage_error": qeL.median_abs_cov_err,
                "val_high_width_10_90": qeH.mean_width_10_90,
                "val_low_width_10_90": qeL.mean_width_10_90,
                "val_high_coverage": qeH.coverage,
                "val_low_coverage": qeL.coverage,
                **horizon_fields,
                "lr": current_lr,
                "epoch_seconds": epoch_s,
                "best_val_pinball_so_far": min(best_val_loss, val_loss),
            }
        )
        _atomic_write_json(training_log, training_log_path)

        print(
            f"Epoch {epoch:02d} "
            f"train_pinball {train_loss:.5f} | val_pinball {val_loss:.5f} | "
            f"val_cov_err(mean) H {qeH.mean_abs_cov_err:.5f} L {qeL.mean_abs_cov_err:.5f} | "
            f"val_width(0.1-0.9) H {qeH.mean_width_10_90:.4f} L {qeL.mean_width_10_90:.4f} | "
            f"LR {current_lr:.2e} | "
            f"skill {skill:.3f} | "
            f"epoch_time {epoch_s:.2f}s"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_ckpt = {
                "state_dict": model.state_dict(),
                "quantiles": list(quantiles),
                "n_horizons": n_horizons,
                "num_layers": num_layers,
            }
            if metadata:
                best_ckpt["metadata"] = dict(metadata)
            torch.save(best_ckpt, output_dir / "model.pt")
            print("  New best model saved; val pinball improved.")

    import pickle
    with open(output_dir / "scaler.pkl", "wb") as fh:
        payload = {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "feat_cols": feat_cols,
            "seq_len": seq_len,
            "trained_items": trained_items,
            "quantiles": list(quantiles),
            "n_horizons": n_horizons,
        }
        if metadata:
            payload.update(metadata)
        pickle.dump(payload, fh)

    training_log["status"] = "completed"
    training_log["best_val_pinball"] = best_val_loss
    _atomic_write_json(training_log, training_log_path)
    print(f"\nAll artefacts saved to {output_dir}")

    return model, history, scaler
