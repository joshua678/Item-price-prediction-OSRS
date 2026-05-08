# osrs_pred/data/io.py
from __future__ import annotations

import os
import glob
import hashlib
import pickle
import gzip
from pathlib import Path
from typing import Optional, List, Union, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm


NA_VALUES = ["", " ", "NA", "N/A", "null", "None"]
NUM_COLS = ["timestamp", "avgHighPrice", "avgLowPrice", "highPriceVolume", "lowPriceVolume", "item_id"]

# Bump this any time you change load_and_filter logic and want to invalidate old cache.
CACHE_VERSION = "load_and_filter_v2"


def _normalize_freq(freq: str) -> str:
    return str(freq).strip().lower().replace("t", "min")


def read_price_csv(fp: Union[str, Path], usecols: Optional[Callable[[str], bool]] = None) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False, na_values=NA_VALUES, usecols=usecols)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _sig_for_file(
    fp: Union[str, Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str,
    min_cover: float,
    coverage_freq: str,
) -> str:
    """
    Fast signature: uses path + stat + params + CACHE_VERSION.
    If the CSV changes, mtime/size usually changes => new signature => recompute.
    """
    p = Path(fp)
    st = p.stat()
    payload = "|".join(
        [
            str(p.resolve()),
            str(st.st_size),
            str(st.st_mtime_ns),
            str(pd.Timestamp(start).value),
            str(pd.Timestamp(end).value),
            str(freq),
            str(coverage_freq),
            f"{min_cover:.8f}",
            CACHE_VERSION,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, stem: str, sig: str) -> Path:
    # Keep filename manageable; include stem for human readability.
    return cache_dir / f"{stem}__{sig[:16]}.pkl.gz"


def _atomic_write_gz_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _read_gz_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_and_filter(
    input_dir: Union[str, Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "h",
    min_cover: float = 0.9,
    *,
    coverage_freq: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> List[pd.DataFrame]:
    """
    Loads each CSV as an item DataFrame, snaps it to a fixed base grid, and drops
    items with insufficient coverage on that grid.

    Returns a list of base-frequency item DataFrames with a 'datetime' column and 'item_name'.

    """
    input_dir = Path(input_dir)
    freq = _normalize_freq(freq)
    coverage_freq = _normalize_freq(coverage_freq or freq)
    csv_files = sorted(glob.glob(str(input_dir / "*.csv")))

    item_dfs: List[pd.DataFrame] = []
    kept = 0

    full_idx = pd.date_range(start, end, freq=freq)
    total_expected_base = len(full_idx)
    coverage_idx = pd.date_range(
        pd.Timestamp(start).floor(coverage_freq),
        pd.Timestamp(end).floor(coverage_freq),
        freq=coverage_freq,
    )
    total_expected = len(coverage_idx)

    wanted_cols = {"datetime", "timestamp", "avgHighPrice", "avgLowPrice", "highPriceVolume", "lowPriceVolume", "item_id"}
    usecols = lambda c: c in wanted_cols

    _cache_dir = Path(cache_dir) if cache_dir is not None else None
    if _cache_dir is not None:
        _cache_dir.mkdir(parents=True, exist_ok=True)

    pbar = None
    if show_progress:
        pbar = tqdm(csv_files, desc=progress_desc or f"load_and_filter({freq})", unit="file", leave=True)
        pbar.set_postfix(loaded=kept)
        it = pbar
    else:
        it = csv_files

    for fp in it:
        p = Path(fp)
        stem = p.stem

        if use_cache and _cache_dir is not None and not force_rebuild:
            sig = _sig_for_file(fp, start, end, freq, min_cover, coverage_freq)
            cpath = _cache_path(_cache_dir, stem, sig)
            if cpath.exists():
                try:
                    cached = _read_gz_pickle(cpath)
                    if cached is None:
                        if pbar is not None:
                            pbar.set_postfix(loaded=kept)
                        continue
                    if isinstance(cached, pd.DataFrame):
                        item_dfs.append(cached)
                        kept += 1
                        if pbar is not None:
                            pbar.set_postfix(loaded=kept)
                        continue
                except Exception:
                    pass

        df = read_price_csv(fp, usecols=usecols)
        if "datetime" not in df.columns and "timestamp" not in df.columns:
            if use_cache and _cache_dir is not None:
                sig = _sig_for_file(fp, start, end, freq, min_cover, coverage_freq)
                _atomic_write_gz_pickle(None, _cache_path(_cache_dir, stem, sig))
            if pbar is not None:
                pbar.set_postfix(loaded=kept)
            continue

        if "timestamp" in df.columns and df["timestamp"].notna().any():
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
        else:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", format="ISO8601")

        df = df.dropna(subset=["datetime"])
        df = df[df["datetime"].between(start, end)].sort_values("datetime").drop_duplicates("datetime")
        if df.empty:
            if use_cache and _cache_dir is not None:
                sig = _sig_for_file(fp, start, end, freq, min_cover, coverage_freq)
                _atomic_write_gz_pickle(None, _cache_path(_cache_dir, stem, sig))
            if pbar is not None:
                pbar.set_postfix(loaded=kept)
            continue

        dt_on_grid = df["datetime"].dt.floor(freq)
        dt_on_coverage_grid = pd.Index(df["datetime"].dt.floor(coverage_freq).dropna().unique())
        idx_pos = coverage_idx.get_indexer(dt_on_coverage_grid.to_numpy())
        covered = int((idx_pos >= 0).sum())
        if total_expected > 0 and covered / total_expected < min_cover:
            if use_cache and _cache_dir is not None:
                sig = _sig_for_file(fp, start, end, freq, min_cover, coverage_freq)
                _atomic_write_gz_pickle(None, _cache_path(_cache_dir, stem, sig))
            if pbar is not None:
                pbar.set_postfix(loaded=kept)
            continue

        df["datetime"] = dt_on_grid
        df = df.set_index("datetime").reindex(full_idx)
        df.index.name = "datetime"

        if "avgHighPrice" in df.columns:
            df["avgHighPrice"] = df["avgHighPrice"].ffill()
        if "avgLowPrice" in df.columns:
            df["avgLowPrice"] = df["avgLowPrice"].ffill()
        if "highPriceVolume" in df.columns:
            df["highPriceVolume"] = df["highPriceVolume"].fillna(0)
        if "lowPriceVolume" in df.columns:
            df["lowPriceVolume"] = df["lowPriceVolume"].fillna(0)

        df["item_name"] = stem

        if "item_id" in df.columns:
            df["item_id"] = df["item_id"].ffill().bfill()

        out_df = df.reset_index() if len(df) == total_expected_base else None

        if use_cache and _cache_dir is not None:
            sig = _sig_for_file(fp, start, end, freq, min_cover, coverage_freq)
            _atomic_write_gz_pickle(out_df, _cache_path(_cache_dir, stem, sig))

        if out_df is not None:
            item_dfs.append(out_df)
            kept += 1

        if pbar is not None:
            pbar.set_postfix(loaded=kept)

    if pbar is not None:
        pbar.close()

    return item_dfs
