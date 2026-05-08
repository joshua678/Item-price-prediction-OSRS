from __future__ import annotations

from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


class PriceSequenceDataset(Dataset):
    """
    Sequence dataset for quantile regression.
    Targets are future returns with shape (sample, horizon); NaN indicates "ignore".
    """
    def __init__(
        self,
        arrays: List[np.ndarray],
        targets_high: List[np.ndarray],
        targets_low: List[np.ndarray],
        item_idxs: List[np.ndarray],
        seq_len: int,
        seq_stride: int = 1,
    ):
        self.X = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
        self.yH = np.concatenate(targets_high, axis=0).astype(np.float32)
        self.yL = np.concatenate(targets_low, axis=0).astype(np.float32)
        if self.yH.ndim == 1:
            self.yH = self.yH[:, None]
        if self.yL.ndim == 1:
            self.yL = self.yL[:, None]
        self.item_i = np.concatenate(item_idxs, axis=0).astype(np.int64)

        self.seq_len = int(seq_len)
        self.stride = int(seq_stride)

        # Build per-item boundaries so windows never cross items
        boundaries = []
        offset = 0
        for item_idx, arr in enumerate(arrays):
            n = arr.shape[0]
            boundaries.append((offset, offset + n, item_idx))
            offset += n

        # Collect only indices where a full seq_len history exists
        valid_idx = []
        for start, end, _ in boundaries:
            valid_idx.extend(range(start + self.seq_len, end, self.stride))
        self.valid_idx = np.array(valid_idx, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.valid_idx))

    def __getitem__(self, i: int):
        idx = int(self.valid_idx[i])
        x = self.X[idx - self.seq_len : idx]
        iid = self.item_i[idx]
        yH = self.yH[idx]
        yL = self.yL[idx]
        return (
            torch.from_numpy(x),
            torch.tensor(yH, dtype=torch.float32),
            torch.tensor(yL, dtype=torch.float32),
            torch.tensor(iid, dtype=torch.long),
        )
