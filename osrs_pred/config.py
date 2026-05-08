"""
cd item_price_prediction_OSRS
python -m osrs_pred.scripts.train_model

PRED_HORIZONS="1,2,3" python -m osrs_pred.scripts.train_model
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re

import pandas as pd
import torch


def print_cuda_info() -> None:
    print("PyTorch built with CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")


def _env_first(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


def _parse_quantiles(s: str) -> tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    qs = tuple(float(p) for p in parts)
    if not qs:
        raise ValueError("QUANTILES must not be empty.")
    if any((q <= 0.0 or q >= 1.0) for q in qs):
        raise ValueError(f"All quantiles must be in (0,1). Got: {qs}")
    if list(qs) != sorted(qs):
        raise ValueError(f"Quantiles must be sorted ascending. Got: {qs}")
    return qs


def _parse_positive_ints(s: str, *, name: str) -> tuple[int, ...]:
    cleaned = s.strip().strip("[]()")
    parts = [p for p in re.split(r"[\s,]+", cleaned) if p]
    values = tuple(int(p) for p in parts)
    if not values:
        raise ValueError(f"{name} must not be empty.")
    if any(v <= 0 for v in values):
        raise ValueError(f"{name} values must be positive integers. Got: {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} values must be unique. Got: {values}")
    return values


DEFAULT_PRED_HORIZONS_STEPS = (1, 2, 3, 6, 12, 24, 36, 72)


@dataclass(frozen=True)
class Config:
    # Paths
    input_dir: Path
    output_dir: Path

    # Date filtering
    start: pd.Timestamp
    end: pd.Timestamp

    # Split boundaries
    train_end: pd.Timestamp
    val_end: pd.Timestamp

    # Forecast horizons, in base 5m steps.
    pred_horizons_steps: tuple[int, ...]

    # Sequence/DL
    seq_len_hours: int
    stride: int
    batch_size: int

    # Training
    lr: float
    epochs: int

    # Quantile regression
    quantiles: tuple[float, ...]

    # Data quality
    min_cover: float

    # Model params
    id_emb_dim: int
    hidden_size: int
    num_layers: int
    dropout: float

    # Other
    seed: int

    # Device
    device: torch.device

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles)

    @property
    def step_minutes(self) -> int:
        return 5

    @property
    def base_freq(self) -> str:
        return f"{self.step_minutes}min"

    @property
    def steps_per_hour(self) -> int:
        return 60 // self.step_minutes

    @property
    def n_horizons(self) -> int:
        return len(self.pred_horizons_steps)

    @property
    def max_horizon_steps(self) -> int:
        return max(self.pred_horizons_steps)

    @property
    def pred_horizons_minutes(self) -> tuple[int, ...]:
        return tuple(h * self.step_minutes for h in self.pred_horizons_steps)

    @property
    def seq_len_steps(self) -> int:
        return self.seq_len_hours * self.steps_per_hour

    @property
    def stride_steps(self) -> int:
        return self.stride

    @staticmethod
    def from_env() -> "Config":
        project_root = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
        input_dir = Path(os.getenv("INPUT_DIR", project_root / "data" / "5m"))
        output_dir = Path(os.getenv("OUTPUT_DIR", project_root / "prediction_model"))

        start = pd.Timestamp(os.getenv("START", "2025-04-01"))
        end = pd.Timestamp(os.getenv("END", "2026-03-19 23:59:59"))

        train_end = pd.Timestamp(os.getenv("TRAIN_END", "2026-01-19"))
        val_end = pd.Timestamp(os.getenv("VAL_END", "2026-02-19"))

        pred_horizons_steps = _parse_positive_ints(
            os.getenv("PRED_HORIZONS", ",".join(str(h) for h in DEFAULT_PRED_HORIZONS_STEPS)),
            name="PRED_HORIZONS",
        )

        seq_len_hours = int(os.getenv("SEQ_LEN", "24"))
        stride = int(os.getenv("STRIDE", "1"))
        batch_size = int(os.getenv("BATCH_SIZE", "512"))

        lr = float(os.getenv("LR", "5e-5"))
        epochs = int(os.getenv("EPOCHS", "3"))

        quantiles = _parse_quantiles(os.getenv("QUANTILES", "0.05,0.1,0.25,0.5,0.75,0.9,0.95"))

        min_cover = float(os.getenv("MIN_COVER", "0.50"))

        id_emb_dim = int(os.getenv("ID_EMB_DIM", "0"))
        hidden_size = int(os.getenv("HIDDEN_SIZE", str(seq_len_hours*12)))
        num_layers = int(_env_first(("NUM_LAYERS", "LSTM_LAYERS", "LSTM_NUM_LAYERS"), "2"))
        if num_layers < 1:
            raise ValueError(f"NUM_LAYERS must be >= 1. Got: {num_layers}")
        dropout = float(os.getenv("DROPOUT", "0.3"))

        seed = int(os.getenv("SEED", "1"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return Config(
            input_dir=input_dir,
            output_dir=output_dir,
            start=start,
            end=end,
            train_end=train_end,
            val_end=val_end,
            pred_horizons_steps=pred_horizons_steps,
            seq_len_hours=seq_len_hours,
            stride=stride,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            quantiles=quantiles,
            min_cover=min_cover,
            id_emb_dim=id_emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seed=seed,
            device=device,
        )
