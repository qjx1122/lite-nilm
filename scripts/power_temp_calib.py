# -*- coding: utf-8 -*-
"""
[v14.2] 推理端可选增强: 功率温桶标定 + 时段先验抑制
====================================================

背景 (V14_RERUN_ANALYSIS.md 实测证据链):
  - P4/P5: U0778/U0789/U0800 推理期 ON 功率档位上移 +28%~+75%,
    回归头外推回缩到训练条件期望 -> 系统性低估 (预测 ON 均 = 真值 62%~78%)
  - P3: U842/U0789 边界 FP 毛刺集中在模型"拿不准但时段先验也低"的点

设计纪律 (零评估集泄漏):
  - LUT / 时段先验 全部由 03_train 用**训练窗**数据计算并随 bundle 落盘
    (bundle["branch_temp_power_lut"] / bundle["hourly_on_prior"]),
    推理端只读 bundle, 绝不在推理/评估集上调参
  - 两个能力独立配置 (time_filters.json 用户级), 默认关闭 (零回归)

能力 1: apply_power_temp_calib (功率温桶标定, lift-only)
  - 对 state==1 的点, 用"训练期同温度桶 ON 功率 P50"作为期望锚,
    floor = gamma * P50(temp); 仅当 floor/pred >= min_gain 死区比才上抬
  - 只升不降 (fix 低估不引入高估); LUT 无覆盖的温度段 -> 不调 (不外推)

能力 2: apply_time_prior_suppress (时段先验抑制)
  - 训练期某小时 ON 率先验 < low_rate 且该点 p_on < p_req 时,
    判定"时段与置信度双低" -> 压 OFF (清 FP 毛刺, 不动高置信预测)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


# ============================================================
# 训练侧: 功率温桶 LUT 构建
# ============================================================
def _temp_bin(t: float, bin_width: float) -> Optional[Tuple[float, float]]:
    """温度 -> (lo, hi) 桶; NaN -> None"""
    if t is None or not np.isfinite(t):
        return None
    lo = np.floor(t / bin_width) * bin_width
    return (float(lo), float(lo + bin_width))


def build_branch_temp_power_lut(ts_index, y, weather_df,
                                on_thr: float,
                                bin_width: float = 2.0,
                                min_n: int = 20) -> Dict[str, Any]:
    """
    用训练期逐点 (时间戳, y功率 W) + 气象温度, 构建"温桶 -> ON 功率分位" LUT。

    参数:
        ts_index:   对齐后时间戳 (15min)
        y:          目标分路功率 (W)
        weather_df: 含 temperature_2m 的气象 DataFrame (与 build_features 同口径,
                    reindex(ts, method="nearest"))
        on_thr:     ON 阈值 (W), 只对真 ON 点统计功率分布
        bin_width:  温度桶宽 (°C), 默认 2.0
        min_n:      桶内最少 ON 点数, 不足则丢弃该桶 (防小样本误导)

    返回 (pickle 安全, 键全为 str):
        {"bin_width": bw, "on_thr": on_thr, "min_n": min_n,
         "bins": {"lo_hi": {"p25":.., "p50":.., "p90":.., "on_mean":.., "n":int}}}
    """
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(ts_index)))
    y = np.asarray(y, dtype=float)
    w = weather_df.reindex(ts, method="nearest")
    temps = w["temperature_2m"].values.astype(float)

    on_mask = (y >= on_thr) & np.isfinite(temps)
    bins: Dict[str, Dict[str, Any]] = {}
    if on_mask.sum() == 0:
        return {"bin_width": bin_width, "on_thr": float(on_thr),
                "min_n": int(min_n), "bins": {}}

    df_b = pd.DataFrame({"t": temps[on_mask], "y": y[on_mask]})
    df_b["bkey"] = df_b["t"].map(
        lambda v: _temp_bin(v, bin_width))
    df_b = df_b.dropna(subset=["bkey"])
    for (lo, hi), g in df_b.groupby("bkey"):
        if len(g) < min_n:
            continue
        yv = g["y"].values
        bins[f"{lo:.1f}_{hi:.1f}"] = {
            "p25": float(np.percentile(yv, 25)),
            "p50": float(np.percentile(yv, 50)),
            "p90": float(np.percentile(yv, 90)),
            "on_mean": float(yv.mean()),
            "n": int(len(g)),
        }
    return {"bin_width": float(bin_width), "on_thr": float(on_thr),
            "min_n": int(min_n), "bins": bins}


def lut_expected_power(lut_pack: Optional[Dict[str, Any]],
                       temps: np.ndarray,
                       stat: str = "p50") -> np.ndarray:
    """
    对任意温度数组, 查 LUT 返回期望功率 (无覆盖桶 -> NaN)。
    stat ∈ {"p25", "p50", "p90", "on_mean"}
    """
    temps = np.asarray(temps, dtype=float)
    out = np.full(len(temps), np.nan, dtype=float)
    if not lut_pack or not isinstance(lut_pack, dict):
        return out
    bins = lut_pack.get("bins", {}) or {}
    bw = float(lut_pack.get("bin_width", 2.0))
    if not bins:
        return out
    lut_map = {}
    for key, rec in bins.items():
        try:
            lo_s, hi_s = key.split("_")
            lut_map[(float(lo_s), float(hi_s))] = rec
        except Exception:
            continue
    for i, t in enumerate(temps):
        b = _temp_bin(t, bw)
        if b is None:
            continue
        rec = lut_map.get((round(b[0], 6), round(b[1], 6)))
        # round 对齐浮点; 直接匹配失败时退化为线性扫描
        if rec is None:
            for (lo, hi), r2 in lut_map.items():
                if lo - 1e-6 <= t < hi - 1e-6:
                    rec = r2
                    break
        if rec is not None and stat in rec:
            out[i] = float(rec[stat])
    return out


# ============================================================
# 推理侧: 功率温桶标定 (lift-only)
# ============================================================
def apply_power_temp_calib(y_pred, state_pred, ts_index, weather_df,
                           lut_pack,
                           gamma: float = 0.85,
                           stat: str = "p50",
                           min_gain: float = 1.05,
                           logger=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    对 state==1 的点做"温桶期望功率"下限标定 (只升不降):

        floor_i = gamma * LUT[stat](temp_i)     (LUT 无覆盖 -> 跳过)
        若 floor_i / pred_i >= min_gain:  pred_i <- floor_i

    参数:
        y_pred:    推理预测功率 (W)
        state_pred: 推理预测状态 (0/1)
        ts_index:  推理时间戳
        weather_df: 推理期气象 (含 temperature_2m)
        lut_pack:  build_branch_temp_power_lut 产物 (训练侧统计)
        gamma:     收缩系数 (默认 0.85, 向训练期望收缩 15%, 保留日内形状)
        stat:     LUT 分位 (默认 p50; p90 更激进)
        min_gain: 死区比 (默认 1.05 = 差异 <5% 不动, 抗抖)
    """
    info: Dict[str, Any] = {"applied": False, "n_lifted": 0,
                            "n_on": int(np.sum(np.asarray(state_pred) == 1)),
                            "n_no_bucket": 0, "gamma": gamma, "stat": stat}
    y = np.asarray(y_pred, dtype=float).copy()
    st = np.asarray(state_pred, dtype=int)
    if lut_pack is None or not (lut_pack.get("bins") if isinstance(lut_pack, dict) else None):
        info["skip"] = "no_lut"
        return y, info
    if weather_df is None or len(weather_df) == 0:
        info["skip"] = "no_weather"
        return y, info

    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(ts_index)))
    w = weather_df.reindex(ts, method="nearest")
    temps = w["temperature_2m"].values.astype(float)
    exp = lut_expected_power(lut_pack, temps, stat=stat)
    floors = gamma * exp

    on = (st == 1)
    has_bucket = np.isfinite(floors)
    n_no_bucket = int((on & ~has_bucket).sum())
    info["n_no_bucket"] = n_no_bucket

    need = on & has_bucket & (y > 1) & (floors > 1) & ((floors / np.maximum(y, 1e-9)) >= min_gain) & (floors > y)
    y[need] = floors[need]
    info["n_lifted"] = int(need.sum())
    info["applied"] = info["n_lifted"] > 0
    if info["n_lifted"] > 0:
        info["mean_lift_w"] = float((floors[need] - np.asarray(y_pred, dtype=float)[need]).mean())
    if logger is not None:
        logger.info(f"  [v14.2 power_temp_calib] ON={info['n_on']}, 上抬 {info['n_lifted']} 步 "
                    f"(无桶覆盖 {n_no_bucket} 步), gamma={gamma}, stat={stat}, min_gain={min_gain}")
    return y, info


# ============================================================
# 训练侧: 时段 ON 率先验
# ============================================================
def build_hourly_on_prior(ts_index, s_true) -> Dict[str, float]:
    """
    训练期每小时 (0-23) ON 率先验: {str(hour): rate in [0,1]}
    """
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(ts_index)))
    st = np.asarray(s_true, dtype=int)
    prior: Dict[str, float] = {}
    hours = ts.hour.values
    for h in range(24):
        m = hours == h
        prior[str(h)] = float(st[m].mean()) if m.sum() > 0 else 0.0
    return prior


# ============================================================
# 推理侧: 时段先验抑制
# ============================================================
def apply_time_prior_suppress(state_pred, p_on, ts_index, prior,
                              low_rate: float = 0.01,
                              p_req: float = 0.9,
                              logger=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    时段先验抑制 (清边界 FP 毛刺):
        若 prior[hour_i] < low_rate 且 p_on_i < p_req -> state_i <- 0

    只压"时段先验和模型置信度双低"的点; 高置信 (p_on >= p_req) 永不动。
    """
    info: Dict[str, Any] = {"applied": False, "n_suppressed": 0,
                            "low_rate": low_rate, "p_req": p_req}
    st = np.asarray(state_pred, dtype=int).copy()
    p = np.asarray(p_on, dtype=float)
    if not prior:
        info["skip"] = "no_prior"
        return st, info
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(ts_index)))
    hours = ts.hour.values
    low_hours = {int(k) for k, v in prior.items()
                 if float(v) < low_rate}
    info["low_prior_hours"] = sorted(low_hours)
    if not low_hours:
        info["skip"] = "no_low_prior_hour"
        return st, info
    hit = (st == 1) & np.isin(hours, list(low_hours)) & (p < p_req)
    st[hit] = 0
    info["n_suppressed"] = int(hit.sum())
    info["applied"] = info["n_suppressed"] > 0
    if logger is not None:
        logger.info(f"  [v14.2 time_prior] 低先验时段={sorted(low_hours)}, "
                    f"压制 {info['n_suppressed']} 步 (p_on<{p_req})")
    return st, info


__all__ = [
    "build_branch_temp_power_lut",
    "lut_expected_power",
    "apply_power_temp_calib",
    "build_hourly_on_prior",
    "apply_time_prior_suppress",
]
