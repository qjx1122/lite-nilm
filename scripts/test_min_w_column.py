# -*- coding: utf-8 -*-
"""
analyze_on_periods min_w 列单元测试 (v13.16 追加)
=====================================================

覆盖:
  T1. 段级 CSV 含 min_w 列 (位置在 duration_min 与 mean_w 之间), 单段场景值正确
  T2. 多段场景 — 每段独立 min
  T3. daily 汇总: ON 天 min_w = 各段 min_w 的最小 (硬计算对拍)
  T4. daily 汇总: OFF 天 min_w = 全天最小 (段行沿用)
  T5. 复合列 (v13.16 target_col='p1+p2') 场景下 min_w 也正确
  T6. 空 DataFrame 兜底 (含 min_w 列名, 不崩)
  T7. compute_daily_summary 向后兼容: 老段级 CSV (无 min_w) 应给 daily.min_w = "" 不崩

运行:
  python scripts/test_min_w_column.py
退出码: 0 = 全通过, 1 = 有失败
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_on_periods import (
    compute_on_periods, compute_daily_summary, _fmt_ts,
)

PASS = 0
FAIL = 0
FAILURES = []


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {msg}")
    else:
        FAIL += 1
        FAILURES.append(msg)
        print(f"  [FAIL] {msg}")


# ============================================================
# T1. 段级 CSV 含 min_w 列 (位置正确, 单段值)
# ============================================================
print("=" * 70)
print(" T1. 段级 CSV min_w 列存在 + 位置 + 单段值")
print("=" * 70)

# 单段, 4 采样点: 30, 100, 80, 50 (min=30, max=100, mean=65)
idx = pd.date_range("2026-05-21 09:00:00", periods=4, freq="15min")
df = pd.DataFrame({
    "time": idx.strftime("%Y-%m-%d %H:%M:%S"),
    "p1": [30, 100, 80, 50],
})
periods = compute_on_periods(df, "p1", on_thr_w=10.0, split_by_day=False)

cols = periods.columns.tolist()
check("min_w" in cols, f"T1.1 段级列含 min_w (columns={cols})")
# 位置: duration_min 之后, mean_w 之前
if "min_w" in cols and "duration_min" in cols and "mean_w" in cols:
    check(cols.index("min_w") == cols.index("duration_min") + 1
          and cols.index("min_w") + 1 == cols.index("mean_w"),
          f"T1.2 min_w 位置 = duration_min 之后, mean_w 之前")

check(len(periods) == 1, f"T1.3 段数 = {len(periods)} (期望 1)")
if len(periods) == 1:
    seg = periods.iloc[0]
    check(abs(seg["min_w"] - 30.0) < 1e-3,
          f"T1.4 min_w = {seg['min_w']} (期望 30)")
    check(abs(seg["mean_w"] - 65.0) < 1e-3,
          f"T1.5 mean_w = {seg['mean_w']} (期望 65)")
    check(abs(seg["peak_w"] - 100.0) < 1e-3,
          f"T1.6 peak_w = {seg['peak_w']} (期望 100)")


# ============================================================
# T2. 多段场景 — 每段独立 min
# ============================================================
print()
print("=" * 70)
print(" T2. 多段场景, 每段独立 min")
print("=" * 70)

# 段 A: 09:00-09:45 = [20, 80, 60, 40] -> min=20, mean=50, max=80
# OFF:  10:00-10:45 = [0, 0, 0, 0]
# 段 B: 11:00-11:45 = [100, 150, 200, 120] -> min=100, mean=142.5, max=200
n = 12
idx2 = pd.date_range("2026-05-21 09:00:00", periods=n, freq="15min")
vals = [20, 80, 60, 40, 0, 0, 0, 0, 100, 150, 200, 120]
df2 = pd.DataFrame({
    "time": idx2.strftime("%Y-%m-%d %H:%M:%S"),
    "p1": vals,
})
periods2 = compute_on_periods(df2, "p1", on_thr_w=10.0, split_by_day=False)
check(len(periods2) == 2, f"T2.1 段数 = {len(periods2)} (期望 2)")
if len(periods2) == 2:
    check(periods2.iloc[0]["min_w"] == 20.0,
          f"T2.2 段 A min_w = {periods2.iloc[0]['min_w']} (期望 20)")
    check(periods2.iloc[1]["min_w"] == 100.0,
          f"T2.3 段 B min_w = {periods2.iloc[1]['min_w']} (期望 100)")


# ============================================================
# T3. daily 汇总: ON 天 min_w = 各段 min_w 的最小
# ============================================================
print()
print("=" * 70)
print(" T3. daily 汇总 ON 天: min_w = min(各段 min_w)")
print("=" * 70)

# 用 T2 的 periods (2 段: min_w=20 与 100), 期望 daily.min_w = 20
# split_by_day=True 才能走 daily
periods3 = compute_on_periods(df2, "p1", on_thr_w=10.0, split_by_day=True)
daily3 = compute_daily_summary(periods3, "p1")

check(len(daily3) == 1, f"T3.1 daily 行数 = {len(daily3)} (期望 1, 单日)")
check("min_w" in daily3.columns,
      f"T3.2 daily 含 min_w 列 (columns={daily3.columns.tolist()})")
if len(daily3) >= 1 and "min_w" in daily3.columns:
    check(daily3.iloc[0]["min_w"] == 20.0,
          f"T3.3 ON 天 min_w = {daily3.iloc[0]['min_w']} "
          f"(期望 20 = min(20, 100))")
    # min_w 位置在 last_off_time 之后, mean_w 之前
    dcols = daily3.columns.tolist()
    check(dcols.index("min_w") == dcols.index("last_off_time") + 1
          and dcols.index("min_w") + 1 == dcols.index("mean_w"),
          f"T3.4 daily min_w 位置 = last_off_time 之后, mean_w 之前")


# ============================================================
# T4. daily 汇总: OFF 天 min_w = 全天最小 (段行沿用)
# ============================================================
print()
print("=" * 70)
print(" T4. daily 汇总 OFF 天: min_w = 全天最小 (段行沿用)")
print("=" * 70)

# 构造 2 天数据: Day1 有 ON 段, Day2 全天 OFF (待机 5W)
n4 = 8 + 96  # Day1 前 2h + Day2 全天 15min * 96 = 24h
idx4a = pd.date_range("2026-05-21 08:00:00", periods=8, freq="15min")
idx4b = pd.date_range("2026-05-22 00:00:00", periods=96, freq="15min")
idx4 = idx4a.union(idx4b)

# Day1 8 点开机 60W, 后半段回 0; Day2 全天 5W 待机
vals4 = [60, 60, 60, 60, 0, 0, 0, 0] + [5] * 96
df4 = pd.DataFrame({
    "time": idx4.strftime("%Y-%m-%d %H:%M:%S"),
    "p1": vals4,
})
periods4 = compute_on_periods(df4, "p1", on_thr_w=10.0, split_by_day=True)
daily4 = compute_daily_summary(periods4, "p1")

check(len(daily4) == 2, f"T4.1 daily 2 天 (实测 {len(daily4)})")
if len(daily4) == 2:
    day1 = daily4[daily4["date"] == "2026-05-21"].iloc[0]
    day2 = daily4[daily4["date"] == "2026-05-22"].iloc[0]
    # Day1: 段内 min = 60 (60,60,60,60 全一样)
    check(day1["min_w"] == 60.0,
          f"T4.2 Day1 (ON 天) min_w = {day1['min_w']} (期望 60)")
    # Day2: OFF 天, 段行 min_w = 5, daily 沿用
    check(day2["min_w"] == 5.0,
          f"T4.3 Day2 (OFF 天) min_w = {day2['min_w']} (期望 5)")
    check(day2["n_segments"] == 0,
          f"T4.4 Day2 n_segments = {day2['n_segments']} (OFF 天期望 0)")


# ============================================================
# T5. 复合列 target_col='p1+p2' 场景下 min_w 正确
# ============================================================
print()
print("=" * 70)
print(" T5. 复合列 'p1+p2' 场景 min_w 正确")
print("=" * 70)

# 4 采样点: p1=[20,30,30,25], p2=[30,30,30,25] -> 复合=[50,60,60,50]
# min=50, mean=55, max=60
idx5 = pd.date_range("2026-05-21 09:00:00", periods=4, freq="15min")
df5 = pd.DataFrame({
    "time": idx5.strftime("%Y-%m-%d %H:%M:%S"),
    "p1": [20, 30, 30, 25],
    "p2": [30, 30, 30, 25],
})
periods5 = compute_on_periods(df5, "p1+p2", on_thr_w=10.0, split_by_day=False)
check(len(periods5) == 1, f"T5.1 复合列 ON 段数 = {len(periods5)}")
if len(periods5) == 1:
    seg = periods5.iloc[0]
    check(seg["min_w"] == 50.0,
          f"T5.2 复合列段 min_w = {seg['min_w']} (期望 50)")
    check(seg["mean_w"] == 55.0,
          f"T5.3 复合列段 mean_w = {seg['mean_w']} (期望 55)")
    check(seg["peak_w"] == 60.0,
          f"T5.4 复合列段 peak_w = {seg['peak_w']} (期望 60)")


# ============================================================
# T6. 空 DataFrame 兜底 (含 min_w 列名)
# ============================================================
print()
print("=" * 70)
print(" T6. 空 DataFrame 兜底")
print("=" * 70)

df_empty = pd.DataFrame({"time": [], "p1": []})
periods_empty = compute_on_periods(df_empty, "p1", on_thr_w=10.0,
                                    split_by_day=False)
check("min_w" in periods_empty.columns,
      f"T6.1 空 df 段级列含 min_w (columns={periods_empty.columns.tolist()})")

daily_empty = compute_daily_summary(periods_empty, "p1")
check("min_w" in daily_empty.columns,
      f"T6.2 空 periods daily 列含 min_w (columns={daily_empty.columns.tolist()})")


# ============================================================
# T7. compute_daily_summary 向后兼容: 老段级 CSV (无 min_w) 不崩
# ============================================================
print()
print("=" * 70)
print(" T7. compute_daily_summary 向后兼容 (无 min_w 老输入)")
print("=" * 70)

# 人为构造无 min_w 的 periods DataFrame (模拟老版本 CSV 读回)
legacy = pd.DataFrame([{
    "being_time": "2026/5/21 9:00:00",
    "end_time":   "2026/5/21 9:45:00",
    "p1": 1,
    "duration_min": 60.0,
    "mean_w": 65.0,
    "peak_w": 100.0,
    "energy_kwh": 0.065,
}])
try:
    daily_legacy = compute_daily_summary(legacy, "p1")
    check(len(daily_legacy) == 1,
          f"T7.1 老段级无 min_w 不崩 (daily 行数 {len(daily_legacy)})")
    if len(daily_legacy) == 1:
        # 无 min_w 时应返回 "" (CSV 空字符串) 或 NaN
        v = daily_legacy.iloc[0]["min_w"]
        check(v == "" or (isinstance(v, float) and pd.isna(v)),
              f"T7.2 老输入 daily.min_w = {v!r} (期望空字符串或 NaN)")
except Exception as e:
    check(False, f"T7 老输入抛异常: {e}")


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 70)
print(f" 汇总: 通过 {PASS} / 失败 {FAIL} / 总计 {PASS + FAIL}")
print("=" * 70)
if FAIL:
    print("失败项:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("[OK] 全部单测通过")
sys.exit(0)
