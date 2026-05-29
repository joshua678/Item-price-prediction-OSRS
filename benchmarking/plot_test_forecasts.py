from __future__ import annotations

import argparse
import csv
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from osrs_pred.config import Config
from osrs_pred.data.features import engineer_features
from osrs_pred.data.io import load_and_filter
from osrs_pred.models.lstm import LSTMQuantileRegressor


@dataclass(frozen=True)
class ModelBundle:
    model: LSTMQuantileRegressor
    quantiles: np.ndarray
    horizons: tuple[int, ...]
    feat_cols: list[str]
    trained_items: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def _parse_items(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "item"


def _closest_quantile_index(quantiles: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(quantiles - target)))


def _infer_num_layers(state_dict: dict[str, torch.Tensor]) -> int:
    layers = set()
    prefix = "lstm.weight_ih_l"
    for key in state_dict:
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            if suffix.isdigit():
                layers.add(int(suffix))
    if not layers:
        raise ValueError("Could not infer LSTM layer count from checkpoint.")
    return max(layers) + 1


def _load_model_bundle(model_dir: Path, cfg: Config, device: torch.device) -> ModelBundle:
    scaler_path = model_dir / "scaler.pkl"
    model_path = model_dir / "model.pt"

    with open(scaler_path, "rb") as fh:
        scaler_payload = pickle.load(fh)

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["state_dict"]

    feat_cols = list(scaler_payload["feat_cols"])
    trained_items = list(scaler_payload["trained_items"])
    quantiles = np.asarray(scaler_payload.get("quantiles", ckpt.get("quantiles", cfg.quantiles)), dtype=float)

    head_out = int(state_dict["head_high.weight"].shape[0])
    n_quantiles = int(quantiles.size)
    if head_out % n_quantiles != 0:
        raise ValueError(
            f"Checkpoint head size {head_out} is not divisible by {n_quantiles} quantiles."
        )
    n_horizons = head_out // n_quantiles

    horizons_raw = scaler_payload.get("pred_horizons_steps") or ckpt.get("metadata", {}).get("pred_horizons_steps")
    horizons = tuple(int(h) for h in (horizons_raw or cfg.pred_horizons_steps))
    if len(horizons) != n_horizons:
        raise ValueError(
            "Checkpoint horizon count does not match configured/saved horizons. "
            f"Checkpoint has {n_horizons}; horizons are {horizons}. "
            "Retrain the model with the current multi-horizon config or provide matching artefacts."
        )

    hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
    num_layers = int(ckpt.get("num_layers") or _infer_num_layers(state_dict))
    id_emb = state_dict.get("id_emb.weight")
    id_emb_dim = int(id_emb.shape[1]) if id_emb is not None else 0
    n_items = int(id_emb.shape[0]) if id_emb is not None else len(trained_items)

    model = LSTMQuantileRegressor(
        n_features=len(feat_cols),
        n_items=n_items,
        n_quantiles=n_quantiles,
        n_horizons=n_horizons,
        id_emb_dim=id_emb_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    scale = np.asarray(scaler_payload["scale"], dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)

    return ModelBundle(
        model=model,
        quantiles=quantiles,
        horizons=horizons,
        feat_cols=feat_cols,
        trained_items=trained_items,
        scaler_mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
        scaler_scale=scale.astype(np.float32, copy=False),
    )


def _select_positions(positions: np.ndarray, plots_per_item: int) -> np.ndarray:
    if positions.size <= plots_per_item:
        return positions
    idx = np.linspace(0, positions.size - 1, num=plots_per_item, dtype=int)
    return positions[idx]


def _iter_item_frames(
    item_dfs: Iterable[pd.DataFrame],
    trained_items: list[str],
    selected_items: set[str] | None,
    max_items: int,
) -> Iterable[tuple[int, pd.DataFrame]]:
    item_to_idx = {name: idx for idx, name in enumerate(trained_items)}
    emitted = 0

    for df in item_dfs:
        name = str(df["item_name"].iloc[0])
        if name not in item_to_idx:
            continue
        if selected_items is not None and name not in selected_items:
            continue
        yield item_to_idx[name], df
        emitted += 1
        if max_items > 0 and emitted >= max_items:
            return


def _predict_window(
    bundle: ModelBundle,
    df_feat: pd.DataFrame,
    item_idx: int,
    anchor_pos: int,
    input_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x_raw = df_feat.iloc[anchor_pos - input_steps:anchor_pos][bundle.feat_cols].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    x_scaled = (x_raw - bundle.scaler_mean) / bundle.scaler_scale

    X = torch.from_numpy(x_scaled[None, :, :]).to(device)
    item_id = torch.tensor([item_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        q_high, q_low = bundle.model(X, item_id)

    return (
        q_high.squeeze(0).detach().cpu().numpy(),
        q_low.squeeze(0).detach().cpu().numpy(),
    )


def _aggregate_step_series(
    x_steps: np.ndarray,
    values: np.ndarray,
    *,
    steps_per_bucket: int,
    aggregation: str,
    bucket_mode: str = "floor",
) -> tuple[np.ndarray, np.ndarray]:
    if bucket_mode == "ceil":
        bucket = np.ceil(x_steps / steps_per_bucket).astype(int)
    else:
        bucket = np.floor(x_steps / steps_per_bucket).astype(int)
    frame = pd.DataFrame({"bucket": bucket, "value": values})
    grouped = frame.groupby("bucket", sort=True)["value"]
    if aggregation == "last":
        out = grouped.last()
    else:
        out = grouped.mean()
    return out.index.to_numpy(dtype=float), out.to_numpy(dtype=float)


def _plot_one(
    *,
    out_path: Path,
    item_name: str,
    anchor_dt: pd.Timestamp,
    input_steps: int,
    output_steps: int,
    steps_per_hour: int,
    plot_frequency: str,
    hourly_aggregation: str,
    horizons: tuple[int, ...],
    quantiles: np.ndarray,
    df_feat: pd.DataFrame,
    anchor_pos: int,
    q_high: np.ndarray,
    q_low: np.ndarray,
) -> None:
    p10_idx = _closest_quantile_index(quantiles, 0.10)
    p25_idx = _closest_quantile_index(quantiles, 0.25)
    p50_idx = _closest_quantile_index(quantiles, 0.50)
    p75_idx = _closest_quantile_index(quantiles, 0.75)
    p90_idx = _closest_quantile_index(quantiles, 0.90)

    p10_high_ret = q_high[:, p10_idx]
    p25_high_ret = q_high[:, p25_idx]
    p50_high_ret = q_high[:, p50_idx]
    p75_high_ret = q_high[:, p75_idx]
    p90_high_ret = q_high[:, p90_idx]

    p10_low_ret = q_low[:, p10_idx]
    p25_low_ret = q_low[:, p25_idx]
    p50_low_ret = q_low[:, p50_idx]
    p75_low_ret = q_low[:, p75_idx]
    p90_low_ret = q_low[:, p90_idx]

    current_high = float(df_feat["avgHighPrice"].iloc[anchor_pos])
    current_low = float(df_feat["avgLowPrice"].iloc[anchor_pos])
    pred_p10_high = current_high * (1.0 + p10_high_ret)
    pred_p25_high = current_high * (1.0 + p25_high_ret)
    pred_p50_high = current_high * (1.0 + p50_high_ret)
    pred_p75_high = current_high * (1.0 + p75_high_ret)
    pred_p90_high = current_high * (1.0 + p90_high_ret)

    pred_p10_low = current_low * (1.0 + p10_low_ret)
    pred_p25_low = current_low * (1.0 + p25_low_ret)
    pred_p50_low = current_low * (1.0 + p50_low_ret)
    pred_p75_low = current_low * (1.0 + p75_low_ret)
    pred_p90_low = current_low * (1.0 + p90_low_ret)

    x_input = np.arange(-input_steps, 1)
    x_output = np.arange(0, output_steps + 1)
    h_x = np.asarray(horizons, dtype=int)
    horizon_mask = h_x <= output_steps
    if not np.any(horizon_mask):
        raise ValueError(f"No model horizons are <= output_steps={output_steps}. Horizons: {horizons}")
    h_x = h_x[horizon_mask]
    pred_p10_high = pred_p10_high[horizon_mask]
    pred_p25_high = pred_p25_high[horizon_mask]
    pred_p50_high = pred_p50_high[horizon_mask]
    pred_p75_high = pred_p75_high[horizon_mask]
    pred_p90_high = pred_p90_high[horizon_mask]
    pred_p10_low = pred_p10_low[horizon_mask]
    pred_p25_low = pred_p25_low[horizon_mask]
    pred_p50_low = pred_p50_low[horizon_mask]
    pred_p75_low = pred_p75_low[horizon_mask]
    pred_p90_low = pred_p90_low[horizon_mask]

    high_input = df_feat["avgHighPrice"].iloc[anchor_pos - input_steps:anchor_pos + 1].to_numpy(dtype=float)
    low_input = df_feat["avgLowPrice"].iloc[anchor_pos - input_steps:anchor_pos + 1].to_numpy(dtype=float)
    high_output = df_feat["avgHighPrice"].iloc[anchor_pos:anchor_pos + output_steps + 1].to_numpy(dtype=float)
    low_output = df_feat["avgLowPrice"].iloc[anchor_pos:anchor_pos + output_steps + 1].to_numpy(dtype=float)

    if plot_frequency == "1h":
        high_anchor = np.asarray([high_input[-1]], dtype=float)
        low_anchor = np.asarray([low_input[-1]], dtype=float)

        x_input, high_input = _aggregate_step_series(
            np.arange(-input_steps, 0),
            high_input[:-1],
            steps_per_bucket=steps_per_hour,
            aggregation=hourly_aggregation,
        )
        _, low_input = _aggregate_step_series(
            np.arange(-input_steps, 0),
            low_input[:-1],
            steps_per_bucket=steps_per_hour,
            aggregation=hourly_aggregation,
        )
        x_input = np.concatenate((x_input, [0.0]))
        high_input = np.concatenate((high_input, high_anchor))
        low_input = np.concatenate((low_input, low_anchor))

        x_output, high_output = _aggregate_step_series(
            np.arange(1, output_steps + 1),
            high_output[1:],
            steps_per_bucket=steps_per_hour,
            aggregation=hourly_aggregation,
            bucket_mode="ceil",
        )
        _, low_output = _aggregate_step_series(
            np.arange(1, output_steps + 1),
            low_output[1:],
            steps_per_bucket=steps_per_hour,
            aggregation=hourly_aggregation,
            bucket_mode="ceil",
        )
        x_output = np.concatenate(([0.0], x_output))
        high_output = np.concatenate((high_anchor, high_output))
        low_output = np.concatenate((low_anchor, low_output))

        hourly_mask = (h_x % steps_per_hour) == 0
        if not np.any(hourly_mask):
            hourly_mask = np.ones_like(h_x, dtype=bool)
        h_x = h_x[hourly_mask] / steps_per_hour
        pred_p10_high = pred_p10_high[hourly_mask]
        pred_p25_high = pred_p25_high[hourly_mask]
        pred_p50_high = pred_p50_high[hourly_mask]
        pred_p75_high = pred_p75_high[hourly_mask]
        pred_p90_high = pred_p90_high[hourly_mask]
        pred_p10_low = pred_p10_low[hourly_mask]
        pred_p25_low = pred_p25_low[hourly_mask]
        pred_p50_low = pred_p50_low[hourly_mask]
        pred_p75_low = pred_p75_low[hourly_mask]
        pred_p90_low = pred_p90_low[hourly_mask]
        x_label = "hours relative to forecast anchor"
        x_min = -input_steps / steps_per_hour
        x_max = output_steps / steps_per_hour
        title_suffix = f"hourly {hourly_aggregation}"
    else:
        x_label = "5-minute steps relative to forecast anchor"
        x_min = -input_steps
        x_max = output_steps
        title_suffix = "5m"

    h_x = np.concatenate(([0.0], h_x.astype(float, copy=False)))
    pred_p10_high = np.concatenate(([current_high], pred_p10_high))
    pred_p25_high = np.concatenate(([current_high], pred_p25_high))
    pred_p50_high = np.concatenate(([current_high], pred_p50_high))
    pred_p75_high = np.concatenate(([current_high], pred_p75_high))
    pred_p90_high = np.concatenate(([current_high], pred_p90_high))
    pred_p10_low = np.concatenate(([current_low], pred_p10_low))
    pred_p25_low = np.concatenate(([current_low], pred_p25_low))
    pred_p50_low = np.concatenate(([current_low], pred_p50_low))
    pred_p75_low = np.concatenate(([current_low], pred_p75_low))
    pred_p90_low = np.concatenate(([current_low], pred_p90_low))

    order = np.argsort(h_x)
    h_x = h_x[order]
    pred_p10_high = pred_p10_high[order]
    pred_p25_high = pred_p25_high[order]
    pred_p50_high = pred_p50_high[order]
    pred_p75_high = pred_p75_high[order]
    pred_p90_high = pred_p90_high[order]
    pred_p10_low = pred_p10_low[order]
    pred_p25_low = pred_p25_low[order]
    pred_p50_low = pred_p50_low[order]
    pred_p75_low = pred_p75_low[order]
    pred_p90_low = pred_p90_low[order]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{item_name} forecast from {anchor_dt} ({title_suffix})")

    panels = [
        (
            axes[0],
            "avgHighPrice",
            high_input,
            high_output,
            pred_p50_high,
            pred_p10_high,
            pred_p90_high,
            pred_p25_high,
            pred_p75_high,
        ),
        (
            axes[1],
            "avgLowPrice",
            low_input,
            low_output,
            pred_p50_low,
            pred_p10_low,
            pred_p90_low,
            pred_p25_low,
            pred_p75_low,
        ),
    ]
    for ax, label, hist, future, pred_p50, pred_p10, pred_p90, pred_p25, pred_p75 in panels:
        band_10_90_low = np.minimum(pred_p10, pred_p90)
        band_10_90_high = np.maximum(pred_p10, pred_p90)
        band_25_75_low = np.minimum(pred_p25, pred_p75)
        band_25_75_high = np.maximum(pred_p25, pred_p75)
        ax.plot(x_input, hist, color="#4b5563", linewidth=1.4, label="input price")
        ax.plot(x_output, future, color="#111827", linewidth=1.7, label="actual output")
        ax.fill_between(
            h_x,
            band_10_90_low,
            band_10_90_high,
            color="#60a5fa",
            alpha=0.22,
            label="p10-p90 interval",
        )
        ax.fill_between(
            h_x,
            band_25_75_low,
            band_25_75_high,
            color="#f59e0b",
            alpha=0.32,
            label="p25-p75 interval",
        )
        ax.scatter(h_x, pred_p50, color="#2563eb", s=28, zorder=3, label="predicted p50")
        ax.plot(h_x, pred_p50, color="#2563eb", linewidth=1.0, alpha=0.7)
        ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1.0)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel(x_label)
    axes[-1].set_xlim(x_min, x_max)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot trained model forecasts on the test split.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory containing model.pt and scaler.pkl.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarking") / "plots")
    parser.add_argument("--items", type=str, default=None, help="Comma-separated item names. Defaults to trained items.")
    parser.add_argument("--max-items", type=int, default=0, help="Maximum items to plot. Use 0 for all.")
    parser.add_argument("--plots-per-item", type=int, default=1)
    parser.add_argument("--input-steps", type=int, default=288)
    parser.add_argument("--output-steps", type=int, default=72)
    parser.add_argument("--plot-frequency", choices=("5m", "1h"), default="5m")
    parser.add_argument("--hourly-aggregation", choices=("mean", "last"), default="mean")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.input_steps <= 0:
        raise ValueError("--input-steps must be positive.")
    if args.output_steps <= 0:
        raise ValueError("--output-steps must be positive.")
    if args.plots_per_item <= 0:
        raise ValueError("--plots-per-item must be positive.")

    cfg = Config.from_env()
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    model_dir = args.model_dir or cfg.output_dir
    bundle = _load_model_bundle(model_dir, cfg, device)

    required_future_steps = args.output_steps
    split_margin = pd.Timedelta(minutes=required_future_steps * cfg.step_minutes)
    selected_items = _parse_items(args.items)
    plotted_horizons = tuple(h for h in bundle.horizons if h <= args.output_steps)

    item_dfs = load_and_filter(
        input_dir=cfg.input_dir,
        start=cfg.start,
        end=cfg.end,
        freq=cfg.base_freq,
        min_cover=cfg.min_cover,
        cache_dir=cfg.output_dir / "cache" / "5m_base",
        show_progress=not args.no_progress,
        progress_desc=f"benchmark_load({cfg.base_freq})",
    )
    if not item_dfs:
        raise RuntimeError("No item data loaded; cannot benchmark.")

    manifest_rows: list[dict[str, str]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for item_idx, df in _iter_item_frames(item_dfs, bundle.trained_items, selected_items, args.max_items):
        item_name = str(df["item_name"].iloc[0])
        df_feat, item_feat_cols = engineer_features(df, steps_per_hour=cfg.steps_per_hour)
        missing = [col for col in bundle.feat_cols if col not in item_feat_cols]
        if missing:
            raise ValueError(f"{item_name}: missing trained feature columns: {missing}")

        mask_test = (
            (df_feat["datetime"] >= cfg.val_end)
            & (df_feat["datetime"] <= (df_feat["datetime"].max() - split_margin))
        )
        test_positions = np.flatnonzero(mask_test.to_numpy())
        if test_positions.size:
            test_positions = test_positions[test_positions >= test_positions.min() + args.input_steps]
        if test_positions.size == 0:
            print(f"Skipping {item_name}: no test windows with {args.input_steps} input and {required_future_steps} future steps.")
            continue

        for plot_idx, anchor_pos in enumerate(_select_positions(test_positions, args.plots_per_item), start=1):
            anchor_dt = pd.Timestamp(df_feat["datetime"].iloc[anchor_pos])
            q_high, q_low = _predict_window(
                bundle,
                df_feat,
                item_idx,
                int(anchor_pos),
                args.input_steps,
                device,
            )

            out_path = (
                args.output_dir
                / f"{item_idx:03d}_{_safe_name(item_name)}__{anchor_dt:%Y%m%d_%H%M%S}__{args.plot_frequency}__{plot_idx:02d}.png"
            )
            _plot_one(
                out_path=out_path,
                item_name=item_name,
                anchor_dt=anchor_dt,
                input_steps=args.input_steps,
                output_steps=args.output_steps,
                steps_per_hour=cfg.steps_per_hour,
                plot_frequency=args.plot_frequency,
                hourly_aggregation=args.hourly_aggregation,
                horizons=bundle.horizons,
                quantiles=bundle.quantiles,
                df_feat=df_feat,
                anchor_pos=int(anchor_pos),
                q_high=q_high,
                q_low=q_low,
            )
            manifest_rows.append(
                {
                    "item": item_name,
                    "item_index": str(item_idx),
                    "anchor_datetime": str(anchor_dt),
                    "input_steps": str(args.input_steps),
                    "output_steps": str(args.output_steps),
                    "plot_frequency": args.plot_frequency,
                    "hourly_aggregation": args.hourly_aggregation,
                    "horizons": ",".join(str(h) for h in plotted_horizons),
                    "plot_path": str(out_path),
                }
            )
            print(f"Wrote {out_path}")

    manifest_path = args.output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "item",
                "item_index",
                "anchor_datetime",
                "input_steps",
                "output_steps",
                "plot_frequency",
                "hourly_aggregation",
                "horizons",
                "plot_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
