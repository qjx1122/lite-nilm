# -*- coding: utf-8 -*-
"""
独立气象数据拉取工具 (v5)

功能:
    一次性拉取项目所需全部时段的小时级历史气温, 并写入 data/weather_cache/
    后续训练/推理直接读缓存, 无需重复请求 API

用法 (Windows):
    python scripts\fetch_weather.py

    # 指定经纬度
    python scripts\fetch_weather.py --lat 39.90 --lon 116.40

    # 指定时段
    python scripts\fetch_weather.py --start 2025-07-01 --end 2026-05-31
"""
import argparse
import pandas as pd
from pathlib import Path
from common import (BUS_CSV, BR_CSV, DATA_DIR,
                    WEATHER_LATITUDE, WEATHER_LONGITUDE, WEATHER_CITY,
                    WEATHER_CACHE_DIR, get_logger, Timer)
from weather_utils import fetch_hourly_weather
from feature_utils import load_bus_csv, load_branch_csv
from time_utils import parse_timestamps

log = get_logger("fetch_weather")


def parse_args():
    p = argparse.ArgumentParser(description="批量拉取历史气温数据")
    p.add_argument("--lat", type=float, default=WEATHER_LATITUDE)
    p.add_argument("--lon", type=float, default=WEATHER_LONGITUDE)
    p.add_argument("--start", type=str, default=None,
                   help="起始日期 YYYY-MM-DD, 默认自动推断")
    p.add_argument("--end", type=str, default=None,
                   help="结束日期 YYYY-MM-DD, 默认自动推断")
    p.add_argument("--cache-dir", type=str, default=str(WEATHER_CACHE_DIR))
    return p.parse_args()


def auto_detect_time_range():
    """从总线/分路 CSV 自动推断时间范围"""
    sd, ed = None, None
    for csv_path in [BUS_CSV, BR_CSV]:
        if not Path(csv_path).exists():
            continue
        # 流式读首尾, 不全量加载
        try:
            if "merged_bus" in str(csv_path) or "Ch1" in str(csv_path):
                df = pd.read_csv(csv_path, usecols=["event_time"],
                                 encoding="utf-8")
                df["t"] = parse_timestamps(df["event_time"])
            else:
                df = pd.read_csv(csv_path, usecols=["time"], encoding="utf-8")
                df["t"] = parse_timestamps(df["time"])
            df = df.dropna(subset=["t"])
            s, e = df["t"].min(), df["t"].max()
            sd = s if sd is None or s < sd else sd
            ed = e if ed is None or e > ed else ed
            log.info(f"  扫描 {Path(csv_path).name}: {s} ~ {e}")
        except Exception as ex:
            log.warning(f"  跳过 {csv_path}: {ex}")
    return sd, ed


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 72)
    log.info("批量拉取历史气温数据 (Open-Meteo API)")
    log.info(f"  地点    : ({args.lat}, {args.lon})  / 默认: {WEATHER_CITY}")
    log.info(f"  缓存目录: {cache_dir}")
    log.info("=" * 72)

    # ---------- 1. 确定时间范围 ----------
    if args.start and args.end:
        sd = pd.Timestamp(args.start)
        ed = pd.Timestamp(args.end)
        log.info(f"  使用命令行指定时段: {sd.date()} ~ {ed.date()}")
    else:
        log.info("  未指定时段, 从数据 CSV 自动推断:")
        sd, ed = auto_detect_time_range()
        if sd is None:
            log.error("  数据目录中找不到任何可识别的 CSV, 请用 --start/--end 指定")
            return 1
        # 前后各加 1 天缓冲
        sd = sd - pd.Timedelta(days=1)
        ed = ed + pd.Timedelta(days=1)
        log.info(f"  推断时段 (含缓冲): {sd.date()} ~ {ed.date()}")

    # ---------- 2. 按年分批拉取 ----------
    log.info("-" * 72)
    log.info("按年分批拉取 (减小单次请求体积, 提高成功率)")
    years = list(range(sd.year, ed.year + 1))
    all_dfs = []
    for y in years:
        y_sd = max(sd, pd.Timestamp(f"{y}-01-01"))
        y_ed = min(ed, pd.Timestamp(f"{y}-12-31"))
        log.info(f"\n  [Year {y}] 拉取 {y_sd.date()} ~ {y_ed.date()}")
        with Timer(f"Year {y}", log):
            df = fetch_hourly_weather(
                latitude=args.lat, longitude=args.lon,
                start_date=y_sd.strftime("%Y-%m-%d"),
                end_date=y_ed.strftime("%Y-%m-%d"),
                cache_dir=cache_dir, logger=log,
            )
            all_dfs.append(df)
            log.info(f"    返回 {len(df)} 条, "
                     f"温度 [{df['temperature_2m'].min():.1f}, "
                     f"{df['temperature_2m'].max():.1f}] °C")

    # ---------- 3. 汇总统计 ----------
    log.info("=" * 72)
    log.info("拉取完成, 汇总统计:")
    full = pd.concat(all_dfs).sort_index().drop_duplicates()
    log.info(f"  总时段: {full.index.min()} ~ {full.index.max()}")
    log.info(f"  总条数: {len(full):,} 条")
    log.info(f"  温度  : 均值 {full['temperature_2m'].mean():.1f}°C, "
             f"范围 [{full['temperature_2m'].min():.1f}, "
             f"{full['temperature_2m'].max():.1f}] °C")
    log.info(f"  湿度  : 均值 {full['relative_humidity_2m'].mean():.1f}%")
    # 按月统计
    log.info("  按月气温:")
    monthly = full.groupby(full.index.month)["temperature_2m"] \
                  .agg(["mean", "min", "max", "count"])
    for m, row in monthly.iterrows():
        log.info(f"    {m:>2}月  均值={row['mean']:>5.1f}°C  "
                 f"范围=[{row['min']:>5.1f}, {row['max']:>5.1f}]  "
                 f"n={int(row['count']):,}")

    # 缓存目录大小
    cache_files = list(cache_dir.glob("*.csv"))
    total_size = sum(f.stat().st_size for f in cache_files) / 1024
    log.info(f"\n  缓存文件: {len(cache_files)} 个, 共 {total_size:.1f} KB")
    log.info(f"  路径    : {cache_dir}")
    log.info("=" * 72)
    log.info("Done. 后续 train/inference 将自动读取缓存, 无需联网")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
