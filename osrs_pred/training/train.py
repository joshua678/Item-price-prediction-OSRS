from __future__ import annotations

from contextlib import nullcontext
from collections import defaultdict
import os
from pathlib import Path
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
from ..data.dataset import PriceSequenceDataset
from ..models.lstm import LSTMQuantileRegressor
from .metrics import eval_quantiles


EPOCH_PROGRESS_UPDATE_EVERY = 25


def _epoch_progress_enabled() -> bool:
    v = os.getenv("OSRS_PRED_EPOCH_PROGRESS", "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


def _should_update_epoch_progress(batch_idx: int, total_batches: int) -> bool:
    return batch_idx % EPOCH_PROGRESS_UPDATE_EVERY == 0 or batch_idx == total_batches


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
    metadata: dict | None = None,
) -> tuple[LSTMQuantileRegressor, Dict[str, List[float]], StandardScaler]:
    seed_everything(seed)
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

    print(
        "Unconditional baseline (val-fit) | "
        f"pinball H {baseH.pinball:.5f} L {baseL.pinball:.5f} | "
        f"cov_err(mean) H {baseH.cov_err_mean:.3f} L {baseL.cov_err_mean:.3f} | "
        f"width(0.1-0.9) H {baseH.width_10_90:.4f} L {baseL.width_10_90:.4f}"
    )

    show_epoch_progress = _epoch_progress_enabled()
    train_batches = len(dl_train)
    val_batches = len(dl_val)

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_epoch0 = perf_counter()

        model.train()
        sum_loss = 0.0
        sum_count = 0.0

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

        train_loss = sum_loss / max(sum_count, 1.0)

        model.eval()
        v_sum_loss = 0.0
        v_sum_count = 0.0

        with torch.no_grad():
            with _progress_loader(
                dl_val,
                f"Epoch {epoch:02d}/{epochs:02d} val",
                show_epoch_progress,
            ) as val_iter:
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
            progress_desc=f"Epoch {epoch:02d}/{epochs:02d} calibrate",
            show_progress=show_epoch_progress,
        )
        qeH = eval_quantiles(yH_true, qH_all, np.array(quantiles, dtype=float))
        qeL = eval_quantiles(yL_true, qL_all, np.array(quantiles, dtype=float))

        skill = 1 - val_loss / ((baseH.pinball + baseL.pinball) / 2)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cov_err_mean_H"].append(qeH.mean_abs_cov_err)
        history["val_cov_err_mean_L"].append(qeL.mean_abs_cov_err)
        history["val_width_10_90_H"].append(qeH.mean_width_10_90)
        history["val_width_10_90_L"].append(qeL.mean_width_10_90)

        current_lr = optimiser.param_groups[0]["lr"]

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_s = perf_counter() - t_epoch0

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
    print(f"\nAll artefacts saved to {output_dir}")

    return model, history, scaler
