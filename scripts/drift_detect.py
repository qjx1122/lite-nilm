# -*- coding: utf-8 -*-
"""
数据漂移检测模块 (v6 新增)

机制:
    L2 在推理时自动检测两类漂移并告警
        1. 协变量漂移 (covariate shift): 总线信号分布变化
        2. 概念漂移 (concept drift): 同温度下功率分布变化 (本项目核心问题)

输出:
    drift_report.csv 含每个监测维度的:
        - 训练分布参考值
        - 推理分布观测值
        - 漂移幅度 (相对偏差 %)
        - 告警级别 (NORMAL / WARN / ALERT)

阈值约定 (可在 common.py 调整):
    相对偏差 < 10%: NORMAL  (正常波动)
    10% ~ 25%:    WARN    (轻度漂移, 建议关注)
    > 25%:        ALERT   (严重漂移, 建议重训)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


DRIFT_THRESH_WARN  = 0.10
DRIFT_THRESH_ALERT = 0.25


def detect_drift(infer_df: pd.DataFrame,
                 top_cols: list,
                 weather_df: pd.DataFrame = None,
                 temp_power_lut: dict = None,
                 logger=None) -> pd.DataFrame:
    """
    对推理数据做漂移检测, 返回结果 DataFrame
    
    参数:
        infer_df:       对齐后的推理数据 (15min, 含 top_cols)
        top_cols:       Top-25 电参量列名 (训练时确定的)
        weather_df:     推理时段气温 (可选)
        temp_power_lut: 训练阶段构造的温度-功率 LUT (可选, 用于概念漂移)
    
    返回:
        DataFrame, 每行一个监测维度
    """
    rows = []
    
    # ---- 1. 总线 Top-1 信号的均值漂移 (协变量) ----
    if top_cols and top_cols[0] in infer_df.columns:
        sig = infer_df[top_cols[0]].astype(float)
        # 训练参考: 用 LUT 全局中位数作为训练基线 (若有)
        if temp_power_lut and "__global_median__" in temp_power_lut:
            train_ref = temp_power_lut["__global_median__"]
            obs_mean  = float(sig.mean())
            dev_ratio = abs(obs_mean - train_ref) / max(train_ref, 1e-6)
            rows.append({
                "dimension": "covariate/bus_signal_mean",
                "metric": "总线 Top-1 信号均值",
                "train_ref": round(train_ref, 2),
                "infer_obs": round(obs_mean, 2),
                "drift_ratio": round(dev_ratio, 4),
                "level": _level(dev_ratio),
                "note": f"特征列 {top_cols[0]}",
            })
    
    # ---- 2. 概念漂移核心: 同温度桶下信号分布对比 ----
    if (weather_df is not None and temp_power_lut and 
            top_cols and top_cols[0] in infer_df.columns):
        sig = infer_df[top_cols[0]].astype(float).values
        temp = weather_df["temperature_2m"].reindex(
            infer_df.index, method="nearest"
        ).values
        
        # 对推理数据中出现的每个温度桶, 对比训练期望
        bin_results = []
        for key, train_v in temp_power_lut.items():
            if not isinstance(key, tuple):
                continue
            lo, hi = key
            m = (temp >= lo) & (temp <= hi)
            n = int(m.sum())
            if n < 10:
                continue
            obs_v = float(np.median(sig[m]))
            dev = abs(obs_v - train_v) / max(train_v, 1e-6)
            bin_results.append({
                "temp_lo": round(lo, 1), "temp_hi": round(hi, 1),
                "n_infer": n,
                "train_ref": round(train_v, 1),
                "infer_obs": round(obs_v, 1),
                "drift_ratio": round(dev, 4),
            })
        
        if bin_results:
            # 汇总: 加权平均漂移幅度
            bin_df = pd.DataFrame(bin_results)
            total_n = bin_df["n_infer"].sum()
            weighted_drift = (bin_df["drift_ratio"] * bin_df["n_infer"]).sum() / total_n
            max_drift = bin_df["drift_ratio"].max()
            worst = bin_df.iloc[bin_df["drift_ratio"].idxmax()]
            
            rows.append({
                "dimension": "concept/temp_power_weighted",
                "metric": "温度-功率加权漂移",
                "train_ref": "见明细",
                "infer_obs": "见明细",
                "drift_ratio": round(weighted_drift, 4),
                "level": _level(weighted_drift),
                "note": f"覆盖 {len(bin_df)} 个温度桶, {total_n} 条样本",
            })
            rows.append({
                "dimension": "concept/temp_power_worst_bin",
                "metric": f"最大漂移桶 [{worst['temp_lo']}, {worst['temp_hi']}]°C",
                "train_ref": worst["train_ref"],
                "infer_obs": worst["infer_obs"],
                "drift_ratio": round(max_drift, 4),
                "level": _level(max_drift),
                "note": f"n={int(worst['n_infer'])}",
            })
            
            # 输出最严重的 3 个温度桶
            for _, r in bin_df.nlargest(3, "drift_ratio").iterrows():
                rows.append({
                    "dimension": "concept/temp_power_detail",
                    "metric": f"温度 [{r['temp_lo']}, {r['temp_hi']}]°C",
                    "train_ref": r["train_ref"],
                    "infer_obs": r["infer_obs"],
                    "drift_ratio": round(r["drift_ratio"], 4),
                    "level": _level(r["drift_ratio"]),
                    "note": f"n={int(r['n_infer'])}",
                })
    
    # ---- 3. 时段分布漂移 (是否新数据集中在某个特殊时段) ----
    hour_dist = infer_df.index.hour.value_counts(normalize=True).sort_index()
    # 训练时通常各小时较均匀 (~1/24=4.17%), 若推理集中在少数小时则告警
    max_hour_share = float(hour_dist.max())
    if max_hour_share > 0.08:  # 单小时占比 > 8% (近 2 倍均匀)
        rows.append({
            "dimension": "time/hour_skew",
            "metric": "推理时段集中度",
            "train_ref": "~4.2%/小时",
            "infer_obs": f"{max_hour_share*100:.1f}%",
            "drift_ratio": round((max_hour_share - 1/24) / (1/24), 4),
            "level": _level(abs(max_hour_share - 1/24) / (1/24)),
            "note": "推理数据时段分布偏斜",
        })
    
    df_out = pd.DataFrame(rows)
    
    # 控制台告警
    if logger and not df_out.empty:
        n_alert = (df_out["level"] == "ALERT").sum()
        n_warn  = (df_out["level"] == "WARN").sum()
        if n_alert > 0:
            logger.warning(f"  [漂移检测] 🚨 检测到 {n_alert} 个严重漂移维度 (ALERT)")
        elif n_warn > 0:
            logger.warning(f"  [漂移检测] ⚠️ 检测到 {n_warn} 个轻度漂移维度 (WARN)")
        else:
            logger.info(f"  [漂移检测] ✓ 所有维度正常")
        # 打印每行
        for _, r in df_out.iterrows():
            icon = {"NORMAL": "✓", "WARN": "⚠", "ALERT": "🚨"}.get(r["level"], "?")
            logger.info(f"    {icon} {r['dimension']:<35} "
                       f"drift={r['drift_ratio']*100:>+6.2f}%  [{r['level']}]")
    
    return df_out


def _level(drift_ratio: float) -> str:
    """根据漂移比例分级"""
    r = abs(float(drift_ratio))
    if r >= DRIFT_THRESH_ALERT:
        return "ALERT"
    elif r >= DRIFT_THRESH_WARN:
        return "WARN"
    return "NORMAL"


# ============================================================
# CLI 自检
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from feature_utils import load_branch_csv, load_bus_csv, resample_and_align
    from weather_utils import get_weather_for_period
    from drift_features import build_temp_power_lut
    from common import get_logger
    
    log = get_logger("drift_detect", log_to_file=False)
    
    # 训练数据构造 LUT
    print("加载训练数据 + 构造 LUT...")
    br_t = load_branch_csv("../data/4206894986488-250710-260428.csv")
    bus_t, cols_t = load_bus_csv("../data/e241_800080252842_4206894986488-Ch1-20250710-260428-1.csv")
    df_t = resample_and_align(bus_t, br_t, keep_cols=cols_t)
    w_t = get_weather_for_period(30.59, 114.31, df_t.index.min(), df_t.index.max(),
                                  cache_dir=Path("../data/weather_cache"))
    corr = df_t[cols_t].corrwith(df_t["y_ac"]).abs().sort_values(ascending=False)
    top_cols = corr.head(25).index.tolist()
    lut = build_temp_power_lut(df_t, w_t, top_cols)
    print(f"LUT 桶数: {len([k for k in lut if isinstance(k, tuple)])}")
    
    # 推理数据 (5 月新数据, 预期会检测到 concept drift)
    print("\n加载推理数据 + 漂移检测...")
    br_i = load_branch_csv("../data/4206894986488-260429-0517.csv")
    bus_i, _ = load_bus_csv("../data/e241_800080252842_4206894986488-Ch1-20260429-0517.csv")
    df_i = resample_and_align(bus_i, br_i, keep_cols=cols_t)
    w_i = get_weather_for_period(30.59, 114.31, df_i.index.min(), df_i.index.max(),
                                  cache_dir=Path("../data/weather_cache"))
    
    report = detect_drift(df_i, top_cols, weather_df=w_i,
                          temp_power_lut=lut, logger=log)
    print("\n漂移检测报告:")
    print(report.to_string(index=False))
