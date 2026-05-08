from __future__ import annotations

from typing import Tuple, List

import numpy as np
import pandas as pd


def _positive_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype(float)
    return out.where(out > 0)


def _nonnegative_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype(float)
    return out.clip(lower=0).fillna(0.0)


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    default: float = 0.0,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(num), default, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (den != 0)
    np.divide(num, den, out=out, where=valid)
    return pd.Series(out, index=numerator.index)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=2).mean()
    std = s.rolling(window, min_periods=2).std()
    return _safe_divide(s - mean, std)


def engineer_features(
    df: pd.DataFrame,
    *,
    steps_per_hour: int,
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()

    if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        if getattr(df["datetime"].dt, "tz", None) is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

    steps_5m = 1
    steps_15m = max(1, int(round(steps_per_hour / 4)))
    steps_30m = max(1, int(round(steps_per_hour / 2)))
    steps_1h = int(steps_per_hour)
    steps_3h = 3 * steps_1h
    steps_6h = 6 * steps_1h
    steps_24h = 24 * steps_1h

    high = _positive_numeric(df["avgHighPrice"])
    low = _positive_numeric(df["avgLowPrice"])
    high_vol = _nonnegative_numeric(df["highPriceVolume"])
    low_vol = _nonnegative_numeric(df["lowPriceVolume"])
    total_vol = high_vol + low_vol

    df["mid"] = (high + low) / 2.0
    log_high = np.log(high)
    log_low = np.log(low)
    log_mid = np.log(df["mid"])
    ret_5m = log_mid.diff(steps_5m)

    df["mid_return_5m"] = ret_5m
    df["high_return_5m"] = log_high.diff(steps_5m)
    df["low_return_5m"] = log_low.diff(steps_5m)
    df["mid_return_15m"] = log_mid.diff(steps_15m)
    df["high_return_15m"] = log_high.diff(steps_15m)
    df["low_return_15m"] = log_low.diff(steps_15m)
    df["mid_return_30m"] = log_mid.diff(steps_30m)
    df["mid_return_1h"] = log_mid.diff(steps_1h)
    df["high_return_1h"] = log_high.diff(steps_1h)
    df["low_return_1h"] = log_low.diff(steps_1h)
    df["mid_return_3h"] = log_mid.diff(steps_3h)
    df["mid_return_6h"] = log_mid.diff(steps_6h)
    df["high_return_6h"] = log_high.diff(steps_6h)
    df["low_return_6h"] = log_low.diff(steps_6h)

    df["volatility_30m"] = ret_5m.rolling(steps_30m, min_periods=2).std()
    df["volatility_1h"] = ret_5m.rolling(steps_1h, min_periods=2).std()
    df["volatility_6h"] = ret_5m.rolling(steps_6h, min_periods=2).std()
    df["volatility_24h"] = ret_5m.rolling(steps_24h, min_periods=2).std()
    df["rv_1h"] = ret_5m.pow(2).rolling(steps_1h, min_periods=2).sum()
    df["rv_6h"] = ret_5m.pow(2).rolling(steps_6h, min_periods=2).sum()
    df["rv_24h"] = ret_5m.pow(2).rolling(steps_24h, min_periods=2).sum()

    rolling_high_1h = high.rolling(steps_1h, min_periods=1).max()
    rolling_low_1h = low.rolling(steps_1h, min_periods=1).min()
    rolling_mid_min_1h = df["mid"].rolling(steps_1h, min_periods=1).min()
    rolling_mid_max_1h = df["mid"].rolling(steps_1h, min_periods=1).max()
    rolling_mid_min_6h = df["mid"].rolling(steps_6h, min_periods=1).min()
    rolling_mid_max_6h = df["mid"].rolling(steps_6h, min_periods=1).max()

    df["range_1h"] = np.log(rolling_high_1h) - np.log(rolling_low_1h)
    df["mid_position_1h"] = _safe_divide(
        df["mid"] - rolling_mid_min_1h,
        rolling_mid_max_1h - rolling_mid_min_1h,
        default=0.5,
    )
    df["mid_position_6h"] = _safe_divide(
        df["mid"] - rolling_mid_min_6h,
        rolling_mid_max_6h - rolling_mid_min_6h,
        default=0.5,
    )

    df["spread_pct"] = _safe_divide(high - low, df["mid"])
    df["spread_z_6h"] = _rolling_zscore(df["spread_pct"], steps_6h)

    patch_ts = pd.Timestamp("2025-05-29 10:00:00")
    df["new_tax"] = (df["datetime"] >= patch_ts).astype(int)

    df["no_buys"] = low_vol.eq(0)
    df["no_sells"] = high_vol.eq(0)

    high_vol_15m = high_vol.rolling(steps_15m, min_periods=1).sum()
    low_vol_15m = low_vol.rolling(steps_15m, min_periods=1).sum()
    high_vol_1h = high_vol.rolling(steps_1h, min_periods=1).sum()
    low_vol_1h = low_vol.rolling(steps_1h, min_periods=1).sum()
    total_vol_1h = high_vol_1h + low_vol_1h
    total_vol_6h = total_vol.rolling(steps_6h, min_periods=1).sum()

    df["no_buys_1h"] = low_vol_1h.eq(0)
    df["no_sells_1h"] = high_vol_1h.eq(0)

    df["log_volume_5m"] = np.log1p(total_vol)
    df["log_volume_15m"] = np.log1p(high_vol_15m + low_vol_15m)
    df["log_volume_1h"] = np.log1p(total_vol_1h)
    df["log_volume_6h"] = np.log1p(total_vol_6h)
    df["vol_ratio_5m"] = _safe_divide(high_vol, total_vol)
    df["vol_ratio"] = _safe_divide(high_vol_1h, total_vol_1h)
    df["volume_imbalance_5m"] = _safe_divide(high_vol - low_vol, total_vol)
    df["volume_imbalance_1h"] = _safe_divide(high_vol_1h - low_vol_1h, total_vol_1h)
    df["low_volume_return_5m"] = np.log1p(low_vol).diff(steps_5m)
    df["high_volume_return_5m"] = np.log1p(high_vol).diff(steps_5m)
    df["low_volume_return_1h"] = np.log1p(low_vol_1h).diff(steps_1h)
    df["high_volume_return_1h"] = np.log1p(high_vol_1h).diff(steps_1h)
    df["zero_volume_count_1h"] = total_vol.eq(0).rolling(steps_1h, min_periods=1).sum()
    df["active_volume_share_1h"] = total_vol.gt(0).rolling(steps_1h, min_periods=1).mean()

    vwap_num_1h = (df["mid"] * total_vol).rolling(steps_1h, min_periods=1).sum()
    vwap_den_1h = total_vol.rolling(steps_1h, min_periods=1).sum()
    vwap_1h = _safe_divide(vwap_num_1h, vwap_den_1h, default=np.nan)
    df["vwap_deviation_1h"] = log_mid - np.log(vwap_1h)

    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["weekday"] = df["datetime"].dt.dayofweek
    minute_of_day = df["hour"] * 60 + df["minute"]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["tod_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    df["tod_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    df["wday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["wday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    feat_cols = [
        "high_return_5m", "low_return_5m",
        "high_return_15m", "low_return_15m",
        "high_return_1h", "low_return_1h",
        "high_return_6h", "low_return_6h",
        "volatility_30m", "volatility_1h", "volatility_6h", "volatility_24h",
        "rv_1h", "rv_6h", "rv_24h", "range_1h",
        "mid_position_1h", "mid_position_6h", "spread_pct", "spread_z_6h",
        "log_volume_5m", "log_volume_15m", "log_volume_1h", "log_volume_6h",
        "vol_ratio_5m", "vol_ratio", "volume_imbalance_5m", "volume_imbalance_1h",
        "low_volume_return_5m", "high_volume_return_5m",
        "low_volume_return_1h", "high_volume_return_1h",
        "zero_volume_count_1h", "active_volume_share_1h", "vwap_deviation_1h",
        "hour_sin", "hour_cos", "minute_sin", "minute_cos", "tod_sin", "tod_cos",
        "wday_sin", "wday_cos", "no_buys", "no_sells", "no_buys_1h", "no_sells_1h",
        "new_tax",
    ]

    defaults = {
        "mid_position_1h": 0.5,
        "mid_position_6h": 0.5,
        "active_volume_share_1h": 0.0,
    }
    df[feat_cols] = (
        df[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    for col in feat_cols:
        df[col] = df[col].fillna(defaults.get(col, 0.0))

    return df, feat_cols
