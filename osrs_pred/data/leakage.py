from __future__ import annotations

import numpy as np
import pandas as pd


def assert_no_train_label_depends_on_ge_train_end(
    df: pd.DataFrame,
    mask_train: pd.Series,
    train_end: pd.Timestamp,
    horizon_steps: int,
    datetime_col: str = "datetime",
    item_tag: str = "",
) -> None:
    """
    Ensures that for every training row i, the label row i+horizon_steps exists and has datetime < train_end.
    This matches positional shifting behavior even with missing hours (because you reindex to a grid).
    """
    dt = pd.to_datetime(df[datetime_col])

    if not dt.is_monotonic_increasing:
        raise AssertionError(
            f"{item_tag}: {datetime_col} must be sorted ascending before this check (got non-monotonic)."
        )

    m = np.asarray(mask_train, dtype=bool)
    train_pos = np.flatnonzero(m)
    if train_pos.size == 0:
        return

    tgt_pos = train_pos + int(horizon_steps)

    oob = tgt_pos >= len(df)
    if np.any(oob):
        bad_i = train_pos[oob][:5]
        raise AssertionError(
            f"{item_tag}: {oob.sum()} training rows have no i+h row (h={horizon_steps}). "
            f"Example train positions: {bad_i.tolist()}"
        )

    tgt_dt = dt.iloc[tgt_pos].to_numpy()
    bad = tgt_dt >= pd.Timestamp(train_end)

    if np.any(bad):
        ex_i = train_pos[bad][:5]
        ex = pd.DataFrame(
            {
                "train_dt": dt.iloc[ex_i].to_numpy(),
                "label_depends_on_dt": dt.iloc[ex_i + horizon_steps].to_numpy(),
            }
        )
        raise AssertionError(
            f"{item_tag}: Found {bad.sum()} training labels depending on datetime >= TRAIN_END "
            f"(TRAIN_END={train_end}). Examples:\n{ex.to_string(index=False)}"
        )
