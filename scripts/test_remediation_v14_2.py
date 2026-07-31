# -*- coding: utf-8 -*-
"""
v14.2 整改模块单元测试 (power_temp_calib + time_prior + 日报 P7 列)
====================================================================

覆盖:
  T1. build_branch_temp_power_lut: 桶边界/p50 精确/min_n 丢弃小桶/NaN 温度剔除
  T2. lut_expected_power: 桶内命中/无覆盖桶 NaN/四舍五入浮点边界/stat 选择
  T3. apply_power_temp_calib: lift-only 语义 (只升不降)/gamma 数学硬对齐/
      min_gain 死区/state==0 不动/无桶跳过/无 LUT no-op
  T4. build_hourly_on_prior: 24 小时先验率精确/无样本小时=0
  T5. apply_time_prior_suppress: 双低 (低先验+低置信) 才压/高置信不动/
      先验不低不动/无先验 no-op
  T6. 日报 P7 新 4 列: is_on_day/off_day_fp/off_day_false_on_kWh/on_only_mae_w
      数值逐值硬对齐 + ON 日 off_* 为 0 + 全 OFF 日 on_only=""

运行:
  python scripts/test_remediation_v14_2.py
退出码: 0 = 全通过
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from power_temp_calib import (
    build_branch_temp_power_lut, lut_expected_power,
    apply_power_temp_calib, build_hourly_on_prior,
    apply_time_prior_suppress)
from metrics_utils import build_daily_metrics_rows

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


def make_weather(ts, temps):
    return pd.DataFrame({"temperature_2m": temps}, index=pd.DatetimeIndex(ts))


# ============================================================
# T1. build_branch_temp_power_lut
# ============================================================
print("=" * 70)
print(" T1. 功率温桶 LUT 构建: 桶边界/分位/min_n/NaN")
print("=" * 70)

# 构造: 24°C 桶 ON 功率 = [200,300,400,500]*8 轮 (32 点), 26°C 桶 = [1000]*25
ts1 = pd.date_range("2026-06-01", periods=57, freq="15min")
y1 = np.array([0] * 57, dtype=float)
temps1 = np.array([10.0] * 57)
y1[0:32] = np.tile([200, 300, 400, 500], 8)         # 32 点 ON, t=24.3
temps1[0:32] = 24.3                                  # 桶 [24,26)
y1[32:57] = 1000.0                                   # 25 点 ON, t=26.1
temps1[32:57] = 26.1                                 # 桶 [26,28)
lut = build_branch_temp_power_lut(ts1, y1, make_weather(ts1, temps1),
                                  on_thr=50.0, bin_width=2.0, min_n=20)
bins = lut["bins"]
check(set(bins.keys()) == {"24.0_26.0", "26.0_28.0"}, f"T1.1 桶键 = {sorted(bins.keys())}")
check(bins["24.0_26.0"]["p50"] == 350.0, f"T1.2 24-26 桶 p50={bins['24.0_26.0']['p50']} (期望 350)")
check(bins["24.0_26.0"]["n"] == 32, f"T1.3 24-26 桶 n={bins['24.0_26.0']['n']} (期望 32)")
check(bins["26.0_28.0"]["p50"] == 1000.0, "T1.4 26-28 桶 p50=1000")
check(abs(bins["24.0_26.0"]["p25"] - 275.0) < 1e-9, "T1.5 p25=275 精确")
check(lut["bin_width"] == 2.0 and lut["on_thr"] == 50.0, "T1.6 元数据透传")

# OFF 点不统计 + min_n 丢弃小桶
y2 = np.array([0.0] * 57)
y2[0:5] = 800.0     # 只有 5 个 ON 点 < min_n=20
lut2 = build_branch_temp_power_lut(ts1, y2, make_weather(ts1, temps1),
                                   on_thr=50.0, bin_width=2.0, min_n=20)
check(lut2["bins"] == {}, f"T1.7 桶内 n=5 < min_n=20 -> 空 LUT (实际 {lut2['bins']})")

# NaN 温度剔除
temps3 = temps1.copy()
temps3[0:32] = np.nan     # 这批 ON 点温度缺失
lut3 = build_branch_temp_power_lut(ts1, y1, make_weather(ts1, temps3),
                                   on_thr=50.0, bin_width=2.0, min_n=20)
check(list(lut3["bins"].keys()) == ["26.0_28.0"],
      f"T1.8 NaN 温度的 ON 点被剔除 (剩 {list(lut3['bins'].keys())})")

# ============================================================
# T2. lut_expected_power
# ============================================================
print("=" * 70)
print(" T2. lut_expected_power: 命中/无桶/边界/stat")
print("=" * 70)

exp = lut_expected_power(lut, np.array([25.99, 24.0, 26.0, 30.0, np.nan]), stat="p50")
check(exp[0] == 350.0 and exp[1] == 350.0, f"T2.1 24.0 与 25.99 命中 24-26 桶 (={exp[0]})")
check(exp[2] == 1000.0, "T2.2 26.0 命中下沿 26-28 桶 (半开区间 [lo,hi))")
check(np.isnan(exp[3]), "T2.3 30.0 无覆盖 -> NaN (不外推)")
check(np.isnan(exp[4]), "T2.4 NaN 温度 -> NaN")
exp90 = lut_expected_power(lut, np.array([25.0]), stat="p90")
# numpy linear 插值: 32 点排序 pos=0.9*31=27.9 -> 落在 500 段 (24-31 位), =500 非 475
check(abs(exp90[0] - 500.0) < 1e-9, f"T2.5 stat=p90 -> {exp90[0]} (numpy 分位=500)")
check(np.isnan(lut_expected_power(None, np.array([25.0]))[0]), "T2.6 LUT=None -> NaN")
check(np.isnan(lut_expected_power({"bins": {}}, np.array([25.0]))[0]), "T2.7 空 LUT -> NaN")

# ============================================================
# T3. apply_power_temp_calib: lift-only
# ============================================================
print("=" * 70)
print(" T3. 功率温桶标定: 只升不降/gamma/min_gain/state/无桶")
print("=" * 70)

# 场景: 4 点全 ON, 温度都在 24-26 桶 (P50=350), gamma=0.85 -> floor=297.5
ts3 = pd.date_range("2026-07-10", periods=4, freq="15min")
temps3b = np.array([25.0, 25.0, 25.0, 25.0])
w3 = make_weather(ts3, temps3b)
y_pred = np.array([100.0, 290.0, 500.0, 297.5])  # ①低估(升) ②死区内(297.5/290=1.0259<1.05) ③高(不降) ④恰=floor(不动)
state = np.array([1, 1, 1, 1])
y_new, info = apply_power_temp_calib(y_pred, state, ts3, w3, lut,
                                     gamma=0.85, min_gain=1.05)
check(abs(y_new[0] - 297.5) < 1e-9, f"T3.1 点① 100 -> {y_new[0]} (=0.85*350=297.5)")
check(y_new[1] == 290.0, f"T3.2 点② 死区 (297.5/290=1.026<1.05) 不动 (={y_new[1]})")
check(y_new[2] == 500.0, f"T3.3 点③ 500>floor 不降 (={y_new[2]})")
check(y_new[3] == 297.5, f"T3.4 点④ 恰等于 floor (ratio=1.0<1.05) 不动 (={y_new[3]})")
check(info["n_lifted"] == 1 and info["applied"], f"T3.5 n_lifted={info['n_lifted']} (期望 1)")
check(abs(info.get("mean_lift_w", 0) - 197.5) < 1e-9, f"T3.6 mean_lift={info.get('mean_lift_w')} (期望 197.5)")

# state==0 点不动
y_new2, info2 = apply_power_temp_calib(y_pred, np.array([0, 0, 1, 1]), ts3, w3, lut, gamma=0.85)
check(y_new2[0] == 100.0 and info2["n_lifted"] == 0, "T3.7 state=0 点不上抬")

# 无桶覆盖 -> 跳过
temps30 = np.array([30.0, 30.0, 30.0, 30.0])
y_new3, info3 = apply_power_temp_calib(y_pred, state, ts3, make_weather(ts3, temps30), lut, gamma=0.85)
check(np.array_equal(y_new3, y_pred) and info3["n_lifted"] == 0 and info3["n_no_bucket"] == 4,
      f"T3.8 30°C 无桶 -> 全跳过 (n_no_bucket={info3['n_no_bucket']})")

# LUT=None / weather=None -> no-op
y_new4, info4 = apply_power_temp_calib(y_pred, state, ts3, w3, None)
check(np.array_equal(y_new4, y_pred) and info4.get("skip") == "no_lut", "T3.9 LUT=None -> no-op")
y_new5, info5 = apply_power_temp_calib(y_pred, state, ts3, None, lut)
check(np.array_equal(y_new5, y_pred) and info5.get("skip") == "no_weather", "T3.10 weather=None -> no-op")

# ============================================================
# T4. build_hourly_on_prior
# ============================================================
print("=" * 70)
print(" T4. 时段 ON 率先验")
print("=" * 70)

ts4 = pd.DatetimeIndex(["2026-06-01 02:00", "2026-06-01 02:15",
                        "2026-06-02 02:00", "2026-06-02 14:00",
                        "2026-06-03 14:00"])
s4 = np.array([1, 0, 1, 1, 0])
prior = build_hourly_on_prior(ts4, s4)
check(len(prior) == 24, f"T4.1 24 小时键齐全 (实际 {len(prior)})")
check(abs(prior["2"] - 2 / 3) < 1e-9, f"T4.2 2 时先验={prior['2']:.4f} (=2/3)")
check(prior["14"] == 0.5, f"T4.3 14 时先验={prior['14']} (=0.5)")
check(prior["3"] == 0.0, "T4.4 无样本小时 = 0.0")

# ============================================================
# T5. apply_time_prior_suppress
# ============================================================
print("=" * 70)
print(" T5. 时段先验抑制: 双低才压")
print("=" * 70)

# prior: 2时=0.001(低), 14时=0.5(不低)
prior5 = {"2": 0.001, "14": 0.5}
ts5 = pd.DatetimeIndex(["2026-07-01 02:00", "2026-07-01 02:15",
                        "2026-07-01 14:00", "2026-07-01 03:00"])
st5 = np.array([1, 1, 1, 1])
p5 = np.array([0.6, 0.95, 0.6, 0.6])
st_new, info5b = apply_time_prior_suppress(st5, p5, ts5, prior5,
                                           low_rate=0.01, p_req=0.9)
check(st_new[0] == 0, "T5.1 2时 p=0.6 (低先验+低置信) -> 压 OFF")
check(st_new[1] == 1, "T5.2 2时 p=0.95 (高置信) -> 不动")
check(st_new[2] == 1, "T5.3 14时 (先验 0.5 不低) -> 不动")
check(st_new[3] == 1, "T5.4 3时无先验键(默认视为不在低先验集合) -> 不动")
check(info5b["n_suppressed"] == 1 and info5b["low_prior_hours"] == [2],
      f"T5.5 n_suppressed={info5b['n_suppressed']}, low_hours={info5b['low_prior_hours']}")
# 无先验 -> no-op
st_new2, info5c = apply_time_prior_suppress(st5, p5, ts5, None)
check(np.array_equal(st_new2, st5) and info5c.get("skip") == "no_prior", "T5.6 prior=None -> no-op")
# p_req 收紧 -> 不压
st_new3, _ = apply_time_prior_suppress(st5, p5, ts5, prior5, low_rate=0.01, p_req=0.5)
check(st_new3[0] == 1, "T5.7 p_req=0.5 时 p=0.6>=0.5 -> 不动 (阈值语义正确)")

# ============================================================
# T6. 日报 P7 新 4 列
# ============================================================
print("=" * 70)
print(" T6. 日报 P7 列: is_on_day/off_day_fp/off_day_false_on_kWh/on_only_mae_w")
print("=" * 70)

# 2 天: D1 全 OFF 真值, 预测 4 点 100W (FP=4, 误报 0.1kWh);
#       D2 有 ON: 真 [0,0,300,300] (2 ON), 预测 [0,0,100,500]
# 2 天: D1 全 OFF 真值, 预测 4 点 100W (FP=4, 误报 100*4*0.25/1000=0.1kWh);
#       D2: 真 ON 4 点各 300W (idx 96..99), 预测 [100,200,300,400]
#           ON 点误差 [200,100,0,100] 和=400 -> on_only_mae=100.0, 整体 MAE=400/96
#           (曾手算和=500 写反, 已用组内逐步展开实证修正)
ts6 = (pd.date_range("2026-07-01", periods=96, freq="15min")
       .append(pd.date_range("2026-07-02", periods=96, freq="15min")))
yt6 = np.zeros(192); yp6 = np.zeros(192)
yp6[0:4] = 100.0                                # D1 误报
yt6[96:100] = 300.0                             # D2 真 ON 4 点
yp6[96:100] = np.array([100, 200, 300, 400])
st6 = (yt6 >= 50).astype(int)
sp6 = (yp6 >= 50).astype(int)
rows6 = build_daily_metrics_rows(ts6, yt6, yp6, st6, sp6, split_name="test")
r61 = [r for r in rows6 if r["date"] == "2026-07-01"][0]
r62 = [r for r in rows6 if r["date"] == "2026-07-02"][0]
check(r61["is_on_day"] == 0, f"T6.1 D1 全 OFF 日 is_on_day={r61['is_on_day']} (期望 0)")
check(r61["off_day_fp"] == 4, f"T6.2 D1 off_day_fp={r61['off_day_fp']} (期望 4)")
check(abs(r61["off_day_false_on_kWh"] - 0.1) < 1e-9,
      f"T6.3 D1 误报电量={r61['off_day_false_on_kWh']} (期望 0.1 kWh)")
check(r61["on_only_mae_w"] == "", f"T6.4 D1 on_only_mae_w='{r61['on_only_mae_w']}' (期望 空串)")
check(r61["F1"] == 0, "T6.5 D1 F1=0 (P7 口径原始行为保留)")
check(r62["is_on_day"] == 1, f"T6.6 D2 ON 日 is_on_day={r62['is_on_day']} (期望 1)")
check(r62["off_day_fp"] == 0 and r62["off_day_false_on_kWh"] == 0.0,
      "T6.7 D2 ON 日 off_* 恒 0")
check(abs(r62["on_only_mae_w"] - 100.0) < 1e-9,
      f"T6.8 D2 on_only_mae_w={r62['on_only_mae_w']} (=400/4=100.0)")
check(abs(r62["MAE_W"] - 400.0 / 96.0) < 1e-9,
      f"T6.9 D2 整体 MAE={r62['MAE_W']} (=400/96≈4.167, 口径对照)")
# 既有列不受影响 (T1 house-style 回归)
keys6 = list(r61.keys())
check(keys6.index("n_samples") < keys6.index("n_bus_raw") < keys6.index("Accuracy"),
      "T6.10 旧列位置未被破坏 (n_samples < n_bus_raw < Accuracy)")
check(keys6.index("TN") < keys6.index("is_on_day"), "T6.11 新 4 列在 TN 之后")

# ============================================================
# 汇总
# ============================================================
print("=" * 70)
print(f" 汇总: 通过 {PASS} / 失败 {FAIL} / 总计 {PASS + FAIL}")
print("=" * 70)
if FAIL:
    print("[FAIL] 存在失败断言:")
    for i, m in enumerate(FAILURES, 1):
        print(f"  {i}. {m}")
    sys.exit(1)
print("[OK] 全部单测通过")
sys.exit(0)
