# -*- coding: utf-8 -*-
"""
漂移感知特征工程模块 (v6 新增)

机制:
    L1 用户行为基线特征 - 让模型显式感知"近期 vs 历史"的行为差异
    
新增 5 维特征:
    1. power_recent_24h_mean   : 总线 Top-1 列近 24 小时滚动均值
    2. power_recent_7d_mean    : 总线 Top-1 列近 7 天滚动均值
    3. power_deviation_24h     : 当前 / 近 24h 均值 (偏离比)
    4. temp_power_residual     : 当前功率 - 历史同温度下的期望功率
    5. is_morning_peak         : 上午用电高峰标记 (9~11点)

设计原则:
    - 因果合规: 所有滚动统计都用 closed='left' 不引入未来信息
    - 在线可用: 推理时只需维护一个滑动窗口缓存
    - 物理可解释: 每个特征对应明确的业务概念
"""
import numpy as np
import pandas as pd


def build_drift_features(df: pd.DataFrame, top_cols: list,
                         weather_df: pd.DataFrame = None,
                         temp_power_lut: dict = None) -> pd.DataFrame:
    """
    在 df 上构造 5 维漂移感知特征
    
    参数:
        df:             对齐后的总线数据 (含 top_cols + 可选 y_ac)
        top_cols:       已选定的 Top-25 电参量列
        weather_df:     15min 粒度温度 DataFrame, 含 temperature_2m
        temp_power_lut: {temp_bin: expected_power} 温度-功率查找表 (训练阶段构造)
    
    返回:
        DataFrame, 含 5 个新列, index 与 df 一致
    """
    out = pd.DataFrame(index=df.index)
    
    # 用 Top-1 电参量列作为总线"代表功率信号"
    if top_cols and top_cols[0] in df.columns:
        signal = df[top_cols[0]].astype(float)
    else:
        signal = pd.Series(0.0, index=df.index)
    
    # ---- 1. 近 24h 滚动均值 (96 个 15min = 24h) ----
    # closed='left' 避免数据泄漏
    out["power_recent_24h_mean"] = signal.rolling(
        window=96, min_periods=4, closed="left"
    ).mean().bfill().fillna(signal.median())
    
    # ---- 2. 近 7 天滚动均值 ----
    out["power_recent_7d_mean"] = signal.rolling(
        window=672, min_periods=16, closed="left"
    ).mean().bfill().fillna(signal.median())
    
    # ---- 3. 偏离比 = 当前 / 近 24h ----
    denom = out["power_recent_24h_mean"].replace(0, np.nan)
    out["power_deviation_24h"] = (signal / denom).fillna(1.0).clip(0, 5)
    
    # ---- 4. 温度-功率残差 (核心: 直接对抗概念漂移) ----
    if weather_df is not None and temp_power_lut is not None:
        temp = weather_df["temperature_2m"].reindex(df.index, method="nearest")
        expected_pow = _lookup_temp_power(temp.values, temp_power_lut)
        # 残差 = 当前信号 - 历史同温度的期望
        # 推理时若此值显著为正, 说明用户行为偏离历史 (耗电更多)
        out["temp_power_residual"] = signal.values - expected_pow
    else:
        # 若无温度/LUT, 填 0 (退化到无此特征)
        out["temp_power_residual"] = 0.0
    
    # ---- 5. 时段标记 ----
    ts = df.index
    out["is_morning_peak"] = ((ts.hour >= 9) & (ts.hour <= 11)).astype(int)
    
    return out


def build_temp_power_lut(df_train: pd.DataFrame, weather_df: pd.DataFrame,
                         top_cols: list, n_bins: int = 20,
                         return_meta: bool = False):
    """
    在训练阶段构造"温度 -> 期望功率"查找表 (LUT)
    
    机制:
        按温度分桶, 计算每桶内总线 Top-1 信号的中位数
        作为 "历史经验下该温度对应的总线功率水平"
    
    参数:
        df_train:    训练集对齐数据
        weather_df:  对应时段的气温数据
        top_cols:    特征列名
        n_bins:      温度分桶数
        return_meta: 若 True, 额外返回每桶元数据 dict
                     {(lo,hi): {"n": int, "mean": float, "std": float,
                                "p25": float, "median": float, "p75": float,
                                "signal_col": str}}
    
    返回:
        return_meta=False (默认, 保持向后兼容):
            {(t_lo, t_hi): expected_signal}
        return_meta=True:
            (lut, meta_dict)
    """
    if not top_cols or top_cols[0] not in df_train.columns:
        return ({}, {}) if return_meta else {}
    signal_col = top_cols[0]
    signal = df_train[signal_col].astype(float)
    temp = weather_df["temperature_2m"].reindex(df_train.index,
                                                method="nearest").values
    
    # 温度分桶 (基于训练集分布的分位数, 避免空桶)
    edges = np.quantile(temp[~np.isnan(temp)], np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)  # 去重
    if len(edges) < 2:
        return ({}, {}) if return_meta else {}
    
    lut = {}
    meta = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (temp >= lo) & (temp <= hi)
        n = int(mask.sum())
        if n >= 5:  # 至少 5 条样本
            vals = signal.values[mask]
            expected = float(np.median(vals))
            lut[(float(lo), float(hi))] = expected
            if return_meta:
                meta[(float(lo), float(hi))] = {
                    "n": n,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p25": float(np.percentile(vals, 25)),
                    "median": expected,
                    "p75": float(np.percentile(vals, 75)),
                    "signal_col": signal_col,
                }
    # 兜底: 全局中位
    lut["__global_median__"] = float(np.median(signal))
    if return_meta:
        return lut, meta
    return lut


# ============================================================
# CSV 导出工具 (v13.15 新增: 温度概念漂移可视化支撑)
# ============================================================
def export_temp_power_lut_csv(lut: dict, out_path,
                              meta: dict = None,
                              logger=None) -> "pd.DataFrame":
    """
    [训练侧] 将温度桶 LUT 导出为 CSV, 便于事后核对/对比
    
    输出列:
        bin_id, temp_lo, temp_hi, temp_width, expected_signal (=median),
        n_samples, mean_signal, std_signal, p25_signal, p75_signal,
        signal_col, is_global_median
    """
    from pathlib import Path
    rows = []
    if not lut:
        df = pd.DataFrame(columns=[
            "bin_id", "temp_lo", "temp_hi", "temp_width",
            "expected_signal", "n_samples",
            "mean_signal", "std_signal", "p25_signal", "p75_signal",
            "signal_col", "is_global_median",
        ])
    else:
        bin_id = 0
        # 桶按温度升序排
        tuple_keys = sorted([k for k in lut if isinstance(k, tuple)])
        for key in tuple_keys:
            lo, hi = key
            info = (meta or {}).get(key, {})
            rows.append({
                "bin_id": bin_id,
                "temp_lo": round(float(lo), 4),
                "temp_hi": round(float(hi), 4),
                "temp_width": round(float(hi - lo), 4),
                "expected_signal": round(float(lut[key]), 3),
                "n_samples": info.get("n", ""),
                "mean_signal": round(info["mean"], 3) if "mean" in info else "",
                "std_signal": round(info["std"], 3) if "std" in info else "",
                "p25_signal": round(info["p25"], 3) if "p25" in info else "",
                "p75_signal": round(info["p75"], 3) if "p75" in info else "",
                "signal_col": info.get("signal_col", ""),
                "is_global_median": 0,
            })
            bin_id += 1
        # 追加全局中位一行, 便于查表
        if "__global_median__" in lut:
            rows.append({
                "bin_id": -1,
                "temp_lo": "", "temp_hi": "", "temp_width": "",
                "expected_signal": round(float(lut["__global_median__"]), 3),
                "n_samples": "", "mean_signal": "", "std_signal": "",
                "p25_signal": "", "p75_signal": "",
                "signal_col": "",
                "is_global_median": 1,
            })
        df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    if logger:
        n_bins = int((df["is_global_median"] == 0).sum()) if len(df) else 0
        logger.info(f"  [v13.15] 温度桶 LUT -> {out_path} "
                    f"({n_bins} 桶 + 全局中位)")
    return df


def export_temp_power_actual_vs_expected_csv(
        df: pd.DataFrame, top_cols: list,
        weather_df: pd.DataFrame, temp_power_lut: dict,
        out_path, logger=None) -> "pd.DataFrame":
    """
    [推理侧] 按训练 LUT 的桶边界, 统计推理数据每个桶的实测总线信号中位数,
    与训练期望对比, 输出漂移 CSV
    
    输出列:
        bin_id, temp_lo, temp_hi,
        train_expected_signal, infer_n_samples,
        infer_median_signal, infer_mean_signal, infer_p25_signal, infer_p75_signal,
        abs_residual (=infer_median - train_expected),
        rel_drift (=abs_residual / max(train_expected, 1e-6)),
        drift_flag  (|rel_drift|>=0.30 记为 ALERT, >=0.15 记为 WARN, 否则 OK),
        signal_col
    """
    from pathlib import Path
    rows = []
    signal_col = top_cols[0] if top_cols and top_cols[0] in df.columns else None
    if signal_col is None or not temp_power_lut or weather_df is None:
        empty = pd.DataFrame(columns=[
            "bin_id", "temp_lo", "temp_hi",
            "train_expected_signal", "infer_n_samples",
            "infer_median_signal", "infer_mean_signal",
            "infer_p25_signal", "infer_p75_signal",
            "abs_residual", "rel_drift", "drift_flag", "signal_col",
        ])
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(out_path, index=False, encoding="utf-8-sig")
        if logger:
            logger.info(f"  [v13.15] (跳过) 无 LUT/温度/信号列, 写空表 -> {out_path}")
        return empty

    signal = df[signal_col].astype(float).values
    temp = weather_df["temperature_2m"].reindex(df.index,
                                                method="nearest").values

    tuple_keys = sorted([k for k in temp_power_lut if isinstance(k, tuple)])
    for bin_id, key in enumerate(tuple_keys):
        lo, hi = key
        mask = (temp >= lo) & (temp <= hi)
        n = int(np.nansum(mask))
        train_expected = float(temp_power_lut[key])
        if n == 0:
            row = {
                "bin_id": bin_id,
                "temp_lo": round(float(lo), 4),
                "temp_hi": round(float(hi), 4),
                "train_expected_signal": round(train_expected, 3),
                "infer_n_samples": 0,
                "infer_median_signal": "",
                "infer_mean_signal": "",
                "infer_p25_signal": "",
                "infer_p75_signal": "",
                "abs_residual": "",
                "rel_drift": "",
                "drift_flag": "NO_DATA",
                "signal_col": signal_col,
            }
        else:
            vals = signal[mask]
            med = float(np.nanmedian(vals))
            abs_res = med - train_expected
            rel = abs_res / max(abs(train_expected), 1e-6)
            if abs(rel) >= 0.30:
                flag = "ALERT"
            elif abs(rel) >= 0.15:
                flag = "WARN"
            else:
                flag = "OK"
            row = {
                "bin_id": bin_id,
                "temp_lo": round(float(lo), 4),
                "temp_hi": round(float(hi), 4),
                "train_expected_signal": round(train_expected, 3),
                "infer_n_samples": n,
                "infer_median_signal": round(med, 3),
                "infer_mean_signal": round(float(np.nanmean(vals)), 3),
                "infer_p25_signal": round(float(np.nanpercentile(vals, 25)), 3),
                "infer_p75_signal": round(float(np.nanpercentile(vals, 75)), 3),
                "abs_residual": round(abs_res, 3),
                "rel_drift": round(rel, 4),
                "drift_flag": flag,
                "signal_col": signal_col,
            }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    if logger:
        alert_n = int((out_df["drift_flag"] == "ALERT").sum())
        warn_n  = int((out_df["drift_flag"] == "WARN").sum())
        nodata_n = int((out_df["drift_flag"] == "NO_DATA").sum())
        logger.info(f"  [v13.15] 温度桶 实测 vs 期望 -> {out_path} "
                    f"({len(out_df)} 桶: ALERT={alert_n}, WARN={warn_n}, "
                    f"NO_DATA={nodata_n})")
    return out_df


def _lookup_temp_power(temps: np.ndarray, lut: dict) -> np.ndarray:
    """对每个温度查 LUT, 返回历史期望功率"""
    if not lut:
        return np.zeros_like(temps, dtype=float)
    global_med = lut.get("__global_median__", 0.0)
    out = np.full(len(temps), global_med, dtype=float)
    for key, v in lut.items():
        if not isinstance(key, tuple):   # 跳过 "__global_median__" 标量
            continue
        lo, hi = key
        m = (temps >= lo) & (temps <= hi)
        out[m] = v
    return out


# ============================================================
# CLI 自检
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from feature_utils import load_branch_csv, load_bus_csv, resample_and_align
    from weather_utils import get_weather_for_period
    from pathlib import Path
    
    br = load_branch_csv("../data/4206894986488-250710-260428.csv")
    bus, cols = load_bus_csv("../data/e241_800080252842_4206894986488-Ch1-20250710-260428-1.csv")
    df = resample_and_align(bus, br, keep_cols=cols)
    w = get_weather_for_period(30.59, 114.31, df.index.min(), df.index.max(),
                                cache_dir=Path("../data/weather_cache"))
    
    # 构造 LUT
    print("构造温度-功率 LUT...")
    corr = df[cols].corrwith(df["y_ac"]).abs().sort_values(ascending=False)
    top_cols = corr.head(25).index.tolist()
    lut = build_temp_power_lut(df, w, top_cols)
    print(f"LUT 大小: {len(lut)} 个桶")
    for k, v in list(lut.items())[:10]:
        print(f"  {k}: {v:.1f}")
    
    print()
    print("构造漂移特征...")
    feats = build_drift_features(df, top_cols, weather_df=w, temp_power_lut=lut)
    print(f"特征 shape: {feats.shape}")
    print("\n前 5 行:")
    print(feats.head())
    print("\n统计:")
    print(feats.describe())
