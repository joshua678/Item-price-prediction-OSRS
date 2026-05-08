from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Dataset


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be >= 0. Got {parsed}.")
    return parsed


def transfer_kwargs(device: torch.device) -> dict[str, bool]:
    return {"non_blocking": device.type == "cuda"}


def make_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    device: torch.device,
) -> DataLoader:
    use_cuda = device.type == "cuda"
    default_workers = min(4, max(1, os.cpu_count() or 1)) if use_cuda else 0
    num_workers = _env_int("OSRS_DATALOADER_WORKERS", default_workers)

    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "pin_memory": use_cuda,
        "num_workers": num_workers,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = _env_int("OSRS_DATALOADER_PREFETCH", 2)

    return DataLoader(dataset, **kwargs)


def move_batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = transfer_kwargs(device)
    X, yH, yL, item_id = batch
    return (
        X.to(device, **kwargs),
        yH.to(device, **kwargs),
        yL.to(device, **kwargs),
        item_id.to(device, **kwargs),
    )
