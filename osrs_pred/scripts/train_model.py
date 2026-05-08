from __future__ import annotations

from time import perf_counter
import os

import numpy as np
import pandas as pd
import torch

from osrs_pred.config import Config, print_cuda_info
from osrs_pred.data.io import load_and_filter
from osrs_pred.data.features import engineer_features
from osrs_pred.data.leakage import assert_no_train_label_depends_on_ge_train_end
from osrs_pred.data.targets import make_return_targets
from osrs_pred.data.dataset import PriceSequenceDataset
from osrs_pred.models.lstm import LSTMQuantileRegressor
from osrs_pred.training.dataloading import make_data_loader
from osrs_pred.training.train import train_model
from osrs_pred.training.plots import plot_history
from osrs_pred.training.evaluate import load_best, predict_on_test, report_quantiles_by_horizon


def _horizon_label(steps: int, step_minutes: int) -> str:
    minutes = steps * step_minutes
    if minutes % 60 == 0:
        return f"{steps} steps ({minutes // 60}h)"
    return f"{steps} steps ({minutes}m)"


def main() -> None:
    cfg = Config.from_env()
    print_cuda_info()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    force_cache = os.getenv("FORCE_REBUILD_CACHE", "0") == "1"
    eval_stride_steps = 1
    horizon_labels = [_horizon_label(h, cfg.step_minutes) for h in cfg.pred_horizons_steps]
    print(
        "Base data: 5m | "
        f"steps/hour={cfg.steps_per_hour} | "
        f"horizons={list(cfg.pred_horizons_steps)} 5m-steps [{', '.join(horizon_labels)}] | "
        f"seq_len={cfg.seq_len_hours}h ({cfg.seq_len_steps} steps) | "
        f"stride={cfg.stride_steps} steps ({cfg.stride_steps * cfg.step_minutes}m) | "
        f"eval_stride={eval_stride_steps} step ({eval_stride_steps * cfg.step_minutes}m) | "
        f"layers={cfg.num_layers}"
    )

    t0 = perf_counter()
    item_dfs = load_and_filter(
        input_dir=cfg.input_dir,
        start=cfg.start,
        end=cfg.end,
        freq=cfg.base_freq,
        min_cover=cfg.min_cover,
        cache_dir=cfg.output_dir / "cache" / "5m_base",
        force_rebuild=force_cache,
        show_progress=True,
        progress_desc=f"load_and_filter({cfg.base_freq})",
    )
    print(f"load 5m base: {perf_counter() - t0:.2f}s")

    if not item_dfs:
        raise RuntimeError("No 5m items loaded; cannot train.")

    H = pd.Timedelta(minutes=cfg.max_horizon_steps * cfg.step_minutes)

    train_arrays, val_arrays, test_arrays = [], [], []
    train_yH, val_yH, test_yH = [], [], []
    train_yL, val_yL, test_yL = [], [], []
    train_idx, val_idx, test_idx = [], [], []

    feat_cols = None

    for item_idx, df in enumerate(item_dfs):
        key = df["item_name"].iloc[0]

        df_feat, item_feat_cols = engineer_features(
            df,
            steps_per_hour=cfg.steps_per_hour,
        )

        if feat_cols is None:
            feat_cols = item_feat_cols
        elif item_feat_cols != feat_cols:
            raise ValueError("Feature columns differ across items; this pipeline expects identical feat_cols.")

        mask_train = (df_feat["datetime"] < (cfg.train_end - H))
        mask_val = (df_feat["datetime"] >= cfg.train_end) & (df_feat["datetime"] < (cfg.val_end - H))
        mask_test = (df_feat["datetime"] >= cfg.val_end) & (df_feat["datetime"] <= (df_feat["datetime"].max() - H))

        assert_no_train_label_depends_on_ge_train_end(
            df=df_feat,
            mask_train=mask_train,
            train_end=cfg.train_end,
            horizon_steps=cfg.max_horizon_steps,
            item_tag=f"item={key}",
        )

        yH_all, yL_all = make_return_targets(df_feat, horizons=cfg.pred_horizons_steps)

        def slice_it(mask, store_arrays, store_yH, store_yL, store_idx):
            sub = df_feat.loc[mask, feat_cols].to_numpy(dtype=np.float32, copy=True)
            store_arrays.append(sub)
            m = mask.to_numpy()
            store_yH.append(yH_all[m].astype(np.float32, copy=False))
            store_yL.append(yL_all[m].astype(np.float32, copy=False))
            store_idx.append(np.full(sub.shape[0], item_idx, dtype=np.int64))

        slice_it(mask_train, train_arrays, train_yH, train_yL, train_idx)
        slice_it(mask_val, val_arrays, val_yH, val_yL, val_idx)
        slice_it(mask_test, test_arrays, test_yH, test_yL, test_idx)

    trained_items = [df["item_name"].iloc[0] for df in item_dfs]

    model, history, _ = train_model(
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        test_arrays=test_arrays,
        train_yH=train_yH,
        val_yH=val_yH,
        test_yH=test_yH,
        train_yL=train_yL,
        val_yL=val_yL,
        test_yL=test_yL,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feat_cols=feat_cols,
        n_items=len(item_dfs),
        n_horizons=cfg.n_horizons,
        quantiles=cfg.quantiles,
        device=cfg.device,
        output_dir=cfg.output_dir,
        seq_len=cfg.seq_len_steps,
        stride=cfg.stride_steps,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        epochs=cfg.epochs,
        id_emb_dim=cfg.id_emb_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        trained_items=trained_items,
        seed=cfg.seed,
        eval_stride=eval_stride_steps,
        metadata={
            "base_freq": cfg.base_freq,
            "step_minutes": cfg.step_minutes,
            "seq_len_hours": cfg.seq_len_hours,
            "seq_len_steps": cfg.seq_len_steps,
            "stride_steps": cfg.stride_steps,
            "eval_stride_steps": eval_stride_steps,
            "pred_horizons_steps": list(cfg.pred_horizons_steps),
            "pred_horizons_minutes": list(cfg.pred_horizons_minutes),
            "max_horizon_steps": cfg.max_horizon_steps,
            "n_horizons": cfg.n_horizons,
            "num_layers": cfg.num_layers,
        },
    )

    plot_path = cfg.output_dir / "training_curves.png"
    plot_history(history, plot_path)
    print("Saved training_curves.png inside:", cfg.output_dir)

    ds_test = PriceSequenceDataset(
        test_arrays,
        test_yH,
        test_yL,
        test_idx,
        seq_len=cfg.seq_len_steps,
        seq_stride=eval_stride_steps,
    )
    dl_test = make_data_loader(ds_test, batch_size=cfg.batch_size, shuffle=False, drop_last=False, device=cfg.device)

    Q = len(cfg.quantiles)
    model_eval = LSTMQuantileRegressor(
        n_features=len(feat_cols),
        n_items=len(item_dfs),
        n_quantiles=Q,
        n_horizons=cfg.n_horizons,
        id_emb_dim=cfg.id_emb_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    quantiles_loaded = load_best(model_eval, cfg.output_dir / "model.pt", cfg.device)

    yH_true, qH_pred, yL_true, qL_pred = predict_on_test(model_eval, dl_test, cfg.device)
    report_quantiles_by_horizon("HIGH", yH_true, qH_pred, quantiles_loaded, horizon_labels)
    report_quantiles_by_horizon("LOW", yL_true, qL_pred, quantiles_loaded, horizon_labels)


if __name__ == "__main__":
    main()
