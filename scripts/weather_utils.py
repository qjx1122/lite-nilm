# -*- coding: utf-8 -*-
"""
气象数据工具 (v5 新增)

数据源:
    Open-Meteo Historical Weather Archive API
    - 全球覆盖 (含中国)
    - 1940 年至今历史数据
    - 1 小时粒度
    - 完全免费 + 无需 API Key (非商业用途)
    - 文档: https://open-meteo.com/en/docs/historical-weather-api

三级回退机制:
    [1] 本地 CSV 缓存命中 -> 直接返回 (零网络开销)
    [2] 缓存未命中 -> 调用 API -> 落盘缓存
    [3] API 失败 -> 启用降级模式 (按经验季节气温填充)

缓存格式:
    data/weather_cache/{lat:.2f}_{lon:.2f}_{year}.csv
    列: time, temperature_2m, apparent_temperature, relative_humidity_2m

特征工程依赖:
    feature_utils.build_features() 在 v5 中会读取本模块返回的 DataFrame
"""
from __future__ import annotations
import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


# Open-Meteo Archive API 端点 (无需 API Key)
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
# 默认 HTTP 超时 (秒)
DEFAULT_TIMEOUT = 15
# 缓存目录 (相对项目根)
CACHE_DIRNAME = "weather_cache"


# ============================================================
# 1. 经验季节气温 (用于 API 完全不可用时的降级模式)
#    数据基于武汉 30 年气候平均, 仅作 fallback
# ============================================================
MONTHLY_AVG_TEMP_WUHAN = {
    1: 4.3,   2: 6.2,   3: 11.0,  4: 17.2,
    5: 22.1,  6: 26.0,  7: 28.9,  8: 28.5,
    9: 23.7, 10: 17.6, 11: 11.5, 12: 6.3,
}


# ============================================================
# 2. 核心函数: 拉取一段时期的小时级历史气温
# ============================================================
def fetch_hourly_weather(latitude: float, longitude: float,
                         start_date: str, end_date: str,
                         cache_dir: Path | str = None,
                         timeout: int = DEFAULT_TIMEOUT,
                         logger=None) -> pd.DataFrame:
    """
    拉取指定经纬度、指定日期范围的小时级气象数据

    参数:
        latitude, longitude: 经纬度 (WGS84)
        start_date, end_date: 'YYYY-MM-DD' 格式 (闭区间)
        cache_dir: 本地缓存目录, None 表示不缓存
        timeout: API 超时秒数
        logger: 可选日志器

    返回:
        DataFrame, DatetimeIndex (Asia/Shanghai), 列:
            - temperature_2m            (°C, 2m 高度气温)
            - apparent_temperature      (°C, 体感温度, 综合湿度风速)
            - relative_humidity_2m      (%, 相对湿度)
    """
    def _log(msg, level="info"):
        if logger:
            getattr(logger, level)(msg)

    cache_dir = Path(cache_dir) if cache_dir else None

    # ---- Step 1: 优先尝试缓存 ----
    if cache_dir is not None:
        cached_df = _try_load_from_cache(cache_dir, latitude, longitude,
                                         start_date, end_date)
        if cached_df is not None:
            _log(f"  [weather] 缓存命中: {cache_dir}, {len(cached_df)} 条")
            return cached_df

    # ---- Step 2: 调用 API ----
    try:
        _log(f"  [weather] 调用 Open-Meteo API: ({latitude}, {longitude}), "
             f"{start_date} ~ {end_date}")
        df = _call_openmeteo_api(latitude, longitude, start_date, end_date,
                                 timeout=timeout)
        _log(f"  [weather] API 返回 {len(df)} 条小时数据, "
             f"温度范围 [{df['temperature_2m'].min():.1f}, "
             f"{df['temperature_2m'].max():.1f}] °C")

        # 写入缓存
        if cache_dir is not None:
            _save_to_cache(df, cache_dir, latitude, longitude)
            _log(f"  [weather] 已写入缓存: {cache_dir}")
        return df

    except Exception as e:
        # ---- Step 3: 降级模式 ----
        _log(f"  [weather]⚠ API 调用失败 ({type(e).__name__}: {e}), "
             "启用降级模式 (使用经验气温)", level="warning")
        return _fallback_climatology(start_date, end_date)


def _call_openmeteo_api(latitude, longitude, start_date, end_date,
                        timeout=DEFAULT_TIMEOUT) -> pd.DataFrame:
    """实际调用 Open-Meteo API, 返回 DataFrame"""
    params = {
        "latitude":  f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m",
        "timezone": "Asia/Shanghai",
    }
    url = OPENMETEO_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "NILM-AC/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())

    if "hourly" not in data or "time" not in data["hourly"]:
        raise RuntimeError(f"API 返回异常: {data.get('reason', data)}")

    hr = data["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hr["time"]),
        "temperature_2m":       hr["temperature_2m"],
        "apparent_temperature": hr["apparent_temperature"],
        "relative_humidity_2m": hr["relative_humidity_2m"],
    }).set_index("time")
    # 强制升序 + 去重
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# ============================================================
# 3. 缓存读写
# ============================================================
def _cache_path(cache_dir: Path, latitude, longitude, year: int) -> Path:
    fname = f"{latitude:.2f}_{longitude:.2f}_{year}.csv"
    return cache_dir / fname


def _try_load_from_cache(cache_dir: Path, latitude, longitude,
                         start_date, end_date) -> pd.DataFrame | None:
    """尝试从缓存加载, 若覆盖范围足够则直接返回, 否则返回 None"""
    if not cache_dir.exists():
        return None
    sd = pd.Timestamp(start_date)
    ed = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    years = list(range(sd.year, ed.year + 1))
    parts = []
    for y in years:
        p = _cache_path(cache_dir, latitude, longitude, y)
        if not p.exists():
            return None
        parts.append(pd.read_csv(p, parse_dates=["time"], encoding="utf-8-sig"))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["time"])
    df = df.set_index("time").sort_index()
    sub = df.loc[(df.index >= sd) & (df.index <= ed)]
    # 至少要有 >50% 的覆盖率才认可缓存
    expected_hours = int((ed - sd).total_seconds() / 3600) + 1
    if len(sub) < expected_hours * 0.5:
        return None
    return sub


def _save_to_cache(df: pd.DataFrame, cache_dir: Path, latitude, longitude):
    cache_dir.mkdir(parents=True, exist_ok=True)
    years = df.index.year.unique()
    for y in years:
        sub = df.loc[df.index.year == y].reset_index()
        p = _cache_path(cache_dir, latitude, longitude, int(y))
        if p.exists():
            old = pd.read_csv(p, parse_dates=["time"], encoding="utf-8-sig")
            sub = pd.concat([old, sub], ignore_index=True) \
                    .drop_duplicates(subset=["time"]).sort_values("time")
        sub.to_csv(p, index=False, encoding="utf-8-sig")


# ============================================================
# 4. 降级模式: 基于月份经验气温填充
# ============================================================
def _fallback_climatology(start_date: str, end_date: str) -> pd.DataFrame:
    """API 完全失败时, 用月度经验气温合成小时数据 (粗略, 仅供模型不挂)"""
    idx = pd.date_range(start_date, end_date, freq="1h", inclusive="both")
    monthly_t = np.array([MONTHLY_AVG_TEMP_WUHAN[m] for m in idx.month])
    # 日内正弦波动 ±5℃ (14:00 最高, 02:00 最低)
    hour_sin = np.sin(2 * np.pi * (idx.hour - 8) / 24)
    t = monthly_t + 5 * hour_sin
    df = pd.DataFrame({
        "temperature_2m":       np.round(t, 2),
        "apparent_temperature": np.round(t + 1.5, 2),   # 体感稍高
        "relative_humidity_2m": np.full(len(idx), 70.0),  # 经验湿度
    }, index=idx)
    df.index.name = "time"
    return df


# ============================================================
# 5. 对齐到 NILM 项目使用的 15min 粒度
# ============================================================
def resample_to_15min(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    把小时级气温重采样到 15min, 与 NILM 数据对齐
    采用线性插值 (温度变化平滑, 线性插值物理合理)
    """
    if weather_df.empty:
        return weather_df
    # 先扩展到 15min, 再线性插值
    new_idx = pd.date_range(
        weather_df.index.min(),
        weather_df.index.max() + pd.Timedelta(minutes=45),
        freq="15min",
    )
    out = weather_df.reindex(weather_df.index.union(new_idx)) \
                    .interpolate(method="time", limit_direction="both")
    out = out.loc[new_idx]
    return out


# ============================================================
# 6. 工程级一站式入口 (推荐用此, 自动处理对齐+缓存+降级)
# ============================================================
def get_weather_for_period(latitude: float, longitude: float,
                           start_ts: pd.Timestamp,
                           end_ts: pd.Timestamp,
                           cache_dir: Path | str = None,
                           logger=None) -> pd.DataFrame:
    """
    一站式入口: 给定时段, 返回与 15min NILM 数据可直接 join 的温度 DataFrame
    """
    start_date = (start_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = (end_ts   + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df_h = fetch_hourly_weather(latitude, longitude, start_date, end_date,
                                cache_dir=cache_dir, logger=logger)
    df_15 = resample_to_15min(df_h)
    return df_15


# ============================================================
# 7. 季节归属 (温度驱动, 取代 v4.2 的 SEASON_MAP)
# ============================================================
def assign_season_by_temp(daily_avg_temp: np.ndarray,
                          summer_th: float = 22.0,
                          winter_th: float = 12.0) -> np.ndarray:
    """
    依据日均气温归属季节:
        >= summer_th : "summer"     (制冷主导)
        <= winter_th : "winter"     (制热主导)
        其它          : "transition" (过渡季)

    阈值参考 (中国华中气候):
        22°C: 5 月中旬升温至此, 与 6-9 月制冷工况贯通
        12°C: 11 月底降至此, 与 12-2 月制热工况贯通
    """
    t = np.asarray(daily_avg_temp, dtype=float)
    out = np.full(len(t), "transition", dtype=object)
    out[t >= summer_th] = "summer"
    out[t <= winter_th] = "winter"
    return out


# ============================================================
# 8. 简单 CLI 自检 (python -m scripts.weather_utils)
# ============================================================
if __name__ == "__main__":
    import sys
    cache = Path(__file__).resolve().parent.parent / "data" / CACHE_DIRNAME
    print(f"自检: 拉取武汉 2025-07-15 一天的小时温度 (缓存目录 {cache})")
    df = fetch_hourly_weather(30.59, 114.31, "2025-07-15", "2025-07-15",
                              cache_dir=cache)
    print(df.head(6))
    print(f"...\n气温范围: {df['temperature_2m'].min():.1f}°C ~ "
          f"{df['temperature_2m'].max():.1f}°C")
