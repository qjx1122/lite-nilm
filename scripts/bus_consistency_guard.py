# -*- coding: utf-8 -*-
"""
P0: 日级 d87 启动签名守卫 + ON 段功率标定 (v14.1.1)

U2844 2026-07 OOD 硬证据结论:
  1) 假开日 (7/02-04) 总线 d73 抬升与真 ON 日同量级 → 不能靠 bus_lift 区分 FP
  2) 真 ON 日 d87 |极值| 普遍 ≥100~200; 假开日多为 45~92
     训练期 OFF 日 d87_absmax 中位~91, ON 日中位~187; th=100 可拦大部分 OFF 且几乎不杀 ON
  3) 模型功率系统性高估 (pred/true ≈ 1.8) → 需功率下修, 不是上修
  4) 7/05 漏检: 真 ON 段 p_on 中位~0.50 < best_thr=0.79, 但当日 d87_absmax=136
     → 启动日降低阈值可召回

机制 (推理端, 无标签):
  A. 日级 d87 启动签名:
       日 |d87|_max < day_absmax_th → 全天强制 OFF (抑制无启动假开日)
  B. 启动日双阈值:
       启动日: thr_lo (默认 0.40); 非启动日: 保持原 thr (已在上游截断)
       仅对"日级通过"的天, 用 thr_lo 重判 raw p_on
  C. 功率标定 (下修为主):
       用 train_on_mean / pred_on_mean 的有界比缩放 (clip 到 [lo,hi])
       若无 train 统计, 用固定 scale (默认 0.55, 来自 U2844 残差)

默认 enabled=False; 由 time_filters.json bus_guard 或 env 打开.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_META: Dict[str, Any] = {
    "enabled": False,
    "version": "v14.1.1_d87_day_power",
    # --- d87 日级 ---
    "d87_col": "load_iden_data87",
    "day_absmax_th": 100.0,          # 日 |d87|max 门槛
    "day_min_startup_spikes": 0,     # 可选: 日 |d87|>=spike_th 次数
    "spike_th": 80.0,
    # --- 启动日阈值 ---
    "startup_thr": 0.40,             # 启动日降低后的 p_on 阈值
    "use_original_state_if_higher": True,  # 与上游 state 取并集 (不丢掉高 thr 已检出)
    # --- 形态学 (与主后处理一致, 仅在重判后轻量再滤) ---
    "post_min_on": 1,
    "post_fill_short_off": 3,
    # --- 功率标定 ---
    "power_calib_enable": True,
    "train_on_mean_w": None,         # 训练 ON 均值; None 则用 fixed_scale
    "fixed_scale": 0.55,             # 无 train 统计时的默认下修
    "scale_lo": 0.40,
    "scale_hi": 1.15,                # 允许轻微上修, 但主场景是下修
    "scale_deadzone_lo": 0.90,       # |scale-1| 很小时不调
    "scale_deadzone_hi": 1.10,
}


def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _merge_meta(user_meta: Optional[dict]) -> dict:
    meta = dict(DEFAULT_META)
    if isinstance(user_meta, dict):
        meta.update({k: v for k, v in user_meta.items()})
    return meta


def compute_daily_d87_absmax(
    bus_df: pd.DataFrame,
    d87_col: str = "load_iden_data87",
    time_col: str = "event_time",
) -> Dict[str, Dict[str, float]]:
    """返回 { 'YYYY-MM-DD': {'absmax', 'min', 'n_spikes'} }"""
    if bus_df is None or d87_col not in getattr(bus_df, "columns", []):
        return {}
    df = bus_df
    if not isinstance(df.index, pd.DatetimeIndex):
        if time_col not in df.columns:
            return {}
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], format="mixed")
        df = df.set_index(time_col)
    df = df.sort_index()
    s = pd.to_numeric(df[d87_col], errors="coerce")
    out: Dict[str, Dict[str, float]] = {}
    for day, g in s.groupby(s.index.normalize()):
        v = g.dropna().values
        if len(v) == 0:
            continue
        key = str(pd.Timestamp(day).date())
        out[key] = {
            "absmax": float(np.nanmax(np.abs(v))),
            "min": float(np.nanmin(v)),
            "n_spikes": float(np.sum(np.abs(v) >= 80.0)),
        }
    return out


def _min_duration_filter(state: np.ndarray, min_on: int = 1,
                         fill_short_off: int = 3) -> np.ndarray:
    """本地轻量形态学, 避免循环依赖."""
    s = np.asarray(state, dtype=int).copy()
    n = len(s)
    if n == 0:
        return s

    def _remove(arr, value, min_len):
        a = arr.copy()
        i = 0
        while i < n:
            if a[i] != value:
                i += 1
                continue
            j = i
            while j < n and a[j] == value:
                j += 1
            if (j - i) < min_len:
                a[i:j] = 1 - value
            i = j
        return a

    if min_on > 1:
        s = _remove(s, 1, min_on)
    if fill_short_off > 0:
        s = 1 - _remove(1 - s, 1, fill_short_off + 1)
    return s


def apply_bus_consistency_guard(
    state_pred: np.ndarray,
    y_pred: np.ndarray,
    p_on: np.ndarray,
    bus_power_15min: np.ndarray = None,   # 保留签名兼容, v14.1.1 不再依赖
    timestamps=None,
    meta: Optional[dict] = None,
    logger=None,
    bus_df: pd.DataFrame = None,
    y_pred_raw_reg: np.ndarray = None,    # 未乘 state 的回归功率, 用于重判
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    应用 d87 日级守卫 + 启动日双阈值 + 功率标定.

    bus_df: 原始 5min 总线 (含 d87), 优先用于日级签名.
    y_pred_raw_reg: 回归器原始功率 (若 None, 用 y_pred 在原 ON 位置回填).
    """
    log = logger.info if logger is not None else (lambda *a, **k: None)
    warn = logger.warning if logger is not None else (lambda *a, **k: None)

    meta = _merge_meta(meta)
    info: Dict[str, Any] = {"enabled": bool(meta.get("enabled", False)), "actions": {}}

    st = np.asarray(state_pred, dtype=int).copy()
    yp = np.asarray(y_pred, dtype=float).copy()
    po = np.asarray(p_on, dtype=float).copy()
    n = len(st)

    if n == 0 or not meta.get("enabled", False):
        info["skipped"] = "disabled_or_empty"
        return st, yp, po, info
    if timestamps is None:
        warn("  [bus_guard] 无 timestamps, 跳过日级守卫")
        info["skipped"] = "no_timestamps"
        return st, yp, po, info

    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    day_str = ts.strftime("%Y-%m-%d").to_numpy()

    # ---- 日级 d87 ----
    d87_col = str(meta.get("d87_col", "load_iden_data87"))
    day_stats = compute_daily_d87_absmax(bus_df, d87_col=d87_col) if bus_df is not None else {}
    if not day_stats:
        # 退化: 若 15min 特征里有 d87 列
        warn("  [bus_guard] 无法从 bus_df 算 d87 日统计, 尝试跳过日级抑制")
        info["actions"]["d87_days"] = 0
        startup_day = {d: True for d in np.unique(day_str)}  # 全部当启动日, 仅做阈值/功率
    else:
        day_th = float(meta.get("day_absmax_th", 100.0))
        min_spikes = int(meta.get("day_min_startup_spikes", 0))
        spike_th = float(meta.get("spike_th", 80.0))
        startup_day = {}
        for d, stt in day_stats.items():
            ok = stt["absmax"] >= day_th
            if min_spikes > 0:
                ok = ok and (stt.get("n_spikes", 0) >= min_spikes)
            startup_day[d] = bool(ok)
        # 未见统计的天保守: 不抑制
        for d in np.unique(day_str):
            startup_day.setdefault(d, True)
        info["actions"]["d87_day_stats"] = {
            d: round(day_stats[d]["absmax"], 1) for d in sorted(day_stats)
        }
        info["actions"]["startup_days"] = sorted([d for d, v in startup_day.items() if v])
        info["actions"]["force_off_days"] = sorted([d for d, v in startup_day.items() if not v])

    n_on_before = int(st.sum())
    y_mean_before = float(yp.mean())

    # 回归功率基底
    if y_pred_raw_reg is not None and len(y_pred_raw_reg) == n:
        reg = np.asarray(y_pred_raw_reg, dtype=float).copy()
    else:
        reg = yp.copy()
        # 原 OFF 位置 reg 为 0, 用 p_on 高处保留 yp
        # 无法恢复的保持 0

    thr_lo = float(meta.get("startup_thr", 0.40))
    use_union = bool(meta.get("use_original_state_if_higher", True))

    st_new = np.zeros(n, dtype=int)
    n_force_off = 0
    n_recalled = 0
    for i in range(n):
        d = day_str[i]
        if not startup_day.get(d, True):
            st_new[i] = 0
            if st[i] == 1:
                n_force_off += 1
            continue
        # 启动日: 低阈值
        hit = po[i] >= thr_lo
        if use_union:
            hit = hit or (st[i] == 1)
        st_new[i] = 1 if hit else 0
        if st_new[i] == 1 and st[i] == 0:
            n_recalled += 1

    # 形态学
    st_new = _min_duration_filter(
        st_new,
        min_on=int(meta.get("post_min_on", 1)),
        fill_short_off=int(meta.get("post_fill_short_off", 3)),
    )

    # 功率重建
    yp_new = np.zeros(n, dtype=float)
    for i in range(n):
        if st_new[i] == 1:
            # 优先 reg; 若为 0 用原 yp 或 train 均值占位
            v = reg[i] if reg[i] > 1 else yp[i]
            if v <= 1:
                v = float(meta.get("train_on_mean_w") or 600.0)
            yp_new[i] = v

    info["actions"]["fp_day_steps_forced_off"] = n_force_off
    info["actions"]["fn_steps_recalled"] = n_recalled

    # ---- 功率标定 ----
    scale = 1.0
    if meta.get("power_calib_enable", True) and st_new.sum() >= 8:
        pred_on_mean = float(yp_new[st_new == 1].mean())
        train_mean = meta.get("train_on_mean_w", None)
        if train_mean is not None and float(train_mean) > 50 and pred_on_mean > 50:
            scale = float(train_mean) / pred_on_mean
        else:
            scale = float(meta.get("fixed_scale", 0.55))
        lo, hi = float(meta["scale_lo"]), float(meta["scale_hi"])
        scale_c = float(np.clip(scale, lo, hi))
        dz_lo, dz_hi = float(meta["scale_deadzone_lo"]), float(meta["scale_deadzone_hi"])
        if scale_c < dz_lo or scale_c > dz_hi:
            yp_new[st_new == 1] *= scale_c
            info["actions"]["power_scale"] = scale_c
            info["actions"]["power_scale_raw"] = scale
        else:
            info["actions"]["power_scale"] = 1.0
            info["actions"]["power_scale_skip"] = f"raw={scale:.3f} in deadzone"
    else:
        info["actions"]["power_scale"] = 1.0

    yp_new = np.clip(yp_new, 0, None)
    yp_new[st_new == 0] = 0.0
    # p_on: 强制 OFF 日压低, 召回步抬到 thr_lo
    po_new = po.copy()
    for i in range(n):
        d = day_str[i]
        if not startup_day.get(d, True):
            po_new[i] = min(po_new[i], 0.05)
        elif st_new[i] == 1 and po_new[i] < thr_lo:
            po_new[i] = thr_lo

    info["n_on_before"] = n_on_before
    info["n_on_after"] = int(st_new.sum())
    info["y_mean_before"] = y_mean_before
    info["y_mean_after"] = float(yp_new.mean())

    log(
        f"  [bus_guard] ON {n_on_before}->{info['n_on_after']} | "
        f"force_off_steps={n_force_off}, days_off={info['actions'].get('force_off_days', [])}, "
        f"fn_recall={n_recalled}, pwr_scale={info['actions'].get('power_scale', 1.0):.3f} | "
        f"meanW {y_mean_before:.1f}->{info['y_mean_after']:.1f}"
    )
    return st_new, yp_new, po_new, info


def build_bus_power_15min_from_bus_df(
    bus_df: pd.DataFrame,
    infer_index: pd.DatetimeIndex,
    power_col: str = "load_iden_data73",
    time_col: str = "event_time",
) -> Optional[np.ndarray]:
    """兼容旧接口."""
    if bus_df is None or power_col not in bus_df.columns:
        return None
    df = bus_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if time_col not in df.columns:
            return None
        df[time_col] = pd.to_datetime(df[time_col], format="mixed")
        df = df.set_index(time_col)
    df = df.sort_index()
    s = pd.to_numeric(df[power_col], errors="coerce")
    s15 = s.resample("15min").mean()
    out = s15.reindex(pd.DatetimeIndex(infer_index), method="nearest",
                      tolerance=pd.Timedelta("8min"))
    return out.values.astype(float)


def default_meta_enabled_for_user(user_cfg: Optional[dict] = None) -> dict:
    meta = dict(DEFAULT_META)
    meta["enabled"] = True
    if isinstance(user_cfg, dict):
        bg = user_cfg.get("bus_guard") or user_cfg.get("bus_consistency_guard")
        if isinstance(bg, dict):
            meta.update(bg)
        if "bus_guard_enabled" in user_cfg:
            meta["enabled"] = bool(user_cfg["bus_guard_enabled"])
    return meta


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 4 days x 96
    n = 96 * 4
    ts = pd.date_range("2026-07-02", periods=n, freq="15min")
    # build fake 5min bus with d87
    idx5 = pd.date_range("2026-07-02", periods=n * 3, freq="5min")
    d87 = np.zeros(len(idx5))
    # day0 OFF weak d87
    d87[: 96 * 3] = rng.normal(0, 20, 96 * 3)
    d87[10] = -70
    # day2 ON strong startup
    d87[96 * 3 * 2: 96 * 3 * 3] = rng.normal(0, 20, 96 * 3)
    d87[96 * 3 * 2 + 20] = -150

    bus_df = pd.DataFrame({"load_iden_data87": d87, "load_iden_data73": 500.0}, index=idx5)
    bus_df.index.name = "event_time"
    bus_df = bus_df.reset_index()

    st = np.zeros(n, dtype=int)
    yp = np.zeros(n)
    po = np.full(n, 0.1)
    # day0 model FP
    st[:96] = 0
    st[13 * 4: 20 * 4] = 1
    yp[st == 1] = 730
    po[st == 1] = 0.9
    # day2 model miss but mid p_on
    d2 = slice(96 * 2, 96 * 3)
    po[d2] = 0.5
    po[96 * 2 + 10 * 4: 96 * 2 + 18 * 4] = 0.55
    yp[d2] = 0
    st[d2] = 0
    reg = np.full(n, 800.0)

    meta = dict(DEFAULT_META)
    meta["enabled"] = True
    meta["day_absmax_th"] = 100.0
    meta["startup_thr"] = 0.40
    meta["train_on_mean_w"] = 700.0
    st2, yp2, po2, info = apply_bus_consistency_guard(
        st, yp, po, None, ts, meta=meta, bus_df=bus_df, y_pred_raw_reg=reg,
    )
    print("info", info)
    print("day0 ON", st2[:96].sum(), "expect 0")
    print("day2 ON", st2[d2].sum(), "expect >0")
    assert st2[:96].sum() == 0, "OFF day not cleared"
    assert st2[d2].sum() >= 20, "startup day not recalled"
    print("[PASS] bus_consistency_guard v14.1.1 self-check")
