from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PLOT_DPI = 160
FIGSIZE = (9.5, 5.5)
COLORS = {
    "train": "#2563eb",
    "val": "#dc2626",
    "baseline": "#6b7280",
    "high": "#7c3aed",
    "low": "#059669",
    "neutral": "#374151",
}


def _load_log(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _sample_count(row: dict[str, Any], log: dict[str, Any]) -> float:
    batch_size = float(log.get("hyperparameters", {}).get("batch_size", 1))
    train_sequences = float(log.get("data", {}).get("train_sequences", 0))
    train_batches = float(row.get("train_batches") or log.get("data", {}).get("train_batches", 0))
    epoch = float(row.get("epoch", 1))
    batch = row.get("batch")

    if batch is not None:
        samples_this_epoch = min(float(batch) * batch_size, train_sequences)
        return max(0.0, (epoch - 1) * train_sequences + samples_this_epoch)

    if train_sequences > 0:
        return epoch * train_sequences
    if train_batches > 0:
        return epoch * train_batches * batch_size
    return epoch


def _sample_label(max_samples: float) -> tuple[float, str]:
    if max_samples >= 1_000_000:
        return 1_000_000.0, "Training samples processed (millions)"
    if max_samples >= 1_000:
        return 1_000.0, "Training samples processed (thousands)"
    return 1.0, "Training samples processed"


def _metric_rows(log: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(log.get("validation_checkpoints") or [])
    if rows:
        return rows
    return list(log.get("epochs") or [])


def _xs(rows: list[dict[str, Any]], log: dict[str, Any]) -> np.ndarray:
    return np.asarray([_sample_count(row, log) for row in rows], dtype=float)


def _ys(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=float)


def _finish_plot(fig: plt.Figure, ax: plt.Axes, out_path: Path, max_samples: float) -> None:
    scale, label = _sample_label(max_samples)
    ax.set_xlabel(label)
    ax.set_xlim(left=0, right=max_samples * 1.02)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick / scale:g}" for tick in ticks])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_online_training_pinball(log: dict[str, Any], rows: list[dict[str, Any]], out_path: Path) -> None:
    x = _xs(rows, log)
    max_x = float(np.nanmax(x)) if x.size else 1.0
    train_key = "train_pinball_running" if "train_pinball_running" in rows[0] else "train_pinball"
    train = _ys(rows, train_key)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        x,
        train,
        color=COLORS["train"],
        marker="o",
        linewidth=2,
        markersize=4,
        label="Online training pinball",
    )
    ax.set_title("Online training pinball loss")
    ax.set_ylabel("Pinball loss")
    _finish_plot(fig, ax, out_path, max_x)


def plot_validation_pinball_loss(log: dict[str, Any], rows: list[dict[str, Any]], out_path: Path) -> None:
    x = _xs(rows, log)
    max_x = float(np.nanmax(x)) if x.size else 1.0
    val = _ys(rows, "val_pinball")
    baseline = float(log.get("baselines", {}).get("validation_unconditional_mean_pinball", np.nan))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        x,
        val,
        color=COLORS["val"],
        marker="o",
        linewidth=2,
        markersize=4,
        label="Validation pinball",
    )
    if np.isfinite(baseline):
        ax.text(
            0.01,
            0.95,
            f"Validation unconditional baseline: {baseline:.5f}",
            transform=ax.transAxes,
            color=COLORS["baseline"],
            fontsize=9,
            ha="left",
            va="top",
        )

    finite_val = val[np.isfinite(val)]
    if finite_val.size:
        lo = float(np.min(finite_val))
        hi = float(np.max(finite_val))
        span = max(hi - lo, 1e-6)
        pad = span * 0.25
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_title("Validation pinball loss")
    ax.set_ylabel("Pinball loss")
    _finish_plot(fig, ax, out_path, max_x)


def plot_skill(log: dict[str, Any], rows: list[dict[str, Any]], out_path: Path) -> None:
    x = _xs(rows, log)
    max_x = float(np.nanmax(x)) if x.size else 1.0
    skill = _ys(rows, "val_skill_vs_unconditional")
    finite_skill = skill[np.isfinite(skill)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, skill, color=COLORS["val"], marker="o", linewidth=2, markersize=4, label="Validation skill")
    if finite_skill.size:
        lo = float(np.min(finite_skill))
        hi = float(np.max(finite_skill))
        span = max(hi - lo, 0.002)
        pad = span * 0.18
        ax.set_ylim(lo - pad, hi + pad)
    ax.text(
        0.01,
        0.04,
        "0 = unconditional baseline",
        transform=ax.transAxes,
        color=COLORS["baseline"],
        fontsize=9,
        ha="left",
        va="bottom",
    )
    ax.set_title("Validation skill relative to unconditional baseline")
    ax.set_ylabel("Skill score")
    _finish_plot(fig, ax, out_path, max_x)


def plot_widths(log: dict[str, Any], rows: list[dict[str, Any]], out_path: Path) -> None:
    x = _xs(rows, log)
    max_x = float(np.nanmax(x)) if x.size else 1.0
    high = _ys(rows, "val_high_width_10_90")
    low = _ys(rows, "val_low_width_10_90")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, high, color=COLORS["high"], marker="o", linewidth=2, markersize=4, label="High-price p10-p90")
    ax.plot(x, low, color=COLORS["low"], marker="o", linewidth=2, markersize=4, label="Low-price p10-p90")
    ax.set_title("Predicted interval width during training")
    ax.set_ylabel("Mean p10-p90 return interval width")
    _finish_plot(fig, ax, out_path, max_x)


def plot_calibration_error(log: dict[str, Any], rows: list[dict[str, Any]], out_path: Path) -> None:
    x = _xs(rows, log)
    max_x = float(np.nanmax(x)) if x.size else 1.0
    high = _ys(rows, "val_high_mean_abs_coverage_error")
    low = _ys(rows, "val_low_mean_abs_coverage_error")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, high, color=COLORS["high"], marker="o", linewidth=2, markersize=4, label="High-price head")
    ax.plot(x, low, color=COLORS["low"], marker="o", linewidth=2, markersize=4, label="Low-price head")
    ax.set_title("Quantile calibration error during training")
    ax.set_ylabel("Mean absolute coverage error")
    _finish_plot(fig, ax, out_path, max_x)


def plot_final_coverage(log: dict[str, Any], rows: list[dict[str, Any]], out_path: Path) -> None:
    final = rows[-1]
    quantiles = np.asarray(log.get("quantiles", []), dtype=float)
    high = np.asarray(final.get("val_high_coverage", []), dtype=float)
    low = np.asarray(final.get("val_low_coverage", []), dtype=float)
    if quantiles.size == 0 or high.size != quantiles.size or low.size != quantiles.size:
        return

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], color=COLORS["baseline"], linestyle="--", linewidth=1.6, label="Ideal calibration")
    ax.plot(quantiles, high, color=COLORS["high"], marker="o", linewidth=2, label="High-price head")
    ax.plot(quantiles, low, color=COLORS["low"], marker="o", linewidth=2, label="Low-price head")
    ax.set_title("Final validation quantile calibration")
    ax.set_xlabel("Predicted quantile")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_baseline_comparison(log: dict[str, Any], rows: list[dict[str, Any]], out_path: Path) -> None:
    final = rows[-1]
    baseline = float(log.get("baselines", {}).get("validation_unconditional_mean_pinball", np.nan))
    val = float(final.get("val_pinball", np.nan))
    if not (np.isfinite(baseline) and np.isfinite(val)):
        return

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    bars = ax.bar(
        ["Unconditional\nbaseline", "LSTM quantile\nmodel"],
        [baseline, val],
        color=[COLORS["baseline"], COLORS["val"]],
        width=0.55,
    )
    ax.set_title("Final validation pinball loss")
    ax.set_ylabel("Pinball loss")
    ax.grid(True, axis="y", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.5f}", ha="center", va="bottom")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def write_manifest(out_dir: Path, rows: list[dict[str, str]]) -> None:
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["plot", "path", "description"])
        writer.writeheader()
        writer.writerows(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate README-ready plots from prediction_model/training_log.json.")
    parser.add_argument("--log-path", type=Path, default=Path("prediction_model") / "training_log.json")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarking") / "plots" / "training_log")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    log = _load_log(args.log_path)
    rows = _metric_rows(log)
    if not rows:
        raise RuntimeError(f"No epoch or validation checkpoint rows found in {args.log_path}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        (
            "online_training_pinball",
            "online_training_pinball.png",
            "Online training pinball loss by sample count.",
            plot_online_training_pinball,
        ),
        (
            "validation_pinball",
            "validation_pinball.png",
            "Zoomed validation pinball loss by sample count.",
            plot_validation_pinball_loss,
        ),
        ("skill", "validation_skill.png", "Validation skill versus the unconditional baseline by sample count.", plot_skill),
        ("widths", "interval_widths.png", "Mean p10-p90 interval widths for high and low price heads.", plot_widths),
        ("calibration_error", "calibration_error.png", "Mean absolute quantile coverage error by sample count.", plot_calibration_error),
        ("final_coverage", "final_coverage.png", "Final validation empirical coverage versus predicted quantile.", plot_final_coverage),
        ("baseline_comparison", "baseline_comparison.png", "Final validation pinball loss compared with the unconditional baseline.", plot_baseline_comparison),
    ]

    manifest_rows: list[dict[str, str]] = []
    for plot_name, filename, description, fn in plots:
        out_path = args.output_dir / filename
        fn(log, rows, out_path)
        if out_path.exists():
            manifest_rows.append({"plot": plot_name, "path": str(out_path), "description": description})
            print(f"Wrote {out_path}")

    write_manifest(args.output_dir, manifest_rows)
    print(f"Wrote {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
