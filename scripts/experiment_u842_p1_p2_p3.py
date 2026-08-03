# -*- coding: utf-8 -*-
"""
U842 P1/P2/P3 优化方案离线验证实验 (v14.7)
================================================

目标:
  对 U842 的 3 条优化路线做 train/val-only 参数选择与 test/inference OOD 验证:
    P1: 低功率/梅雨 ON recall guard (分类层 recall guard)
    P2: ON 功率模式 daily scale (功率模式/档位层)
    P3: 温湿桶双向标定 (temperature × humidity bucket)

纪律:
  - 参数选择只使用 train+val；test 与 inference 只做验证。
  - 不使用 7 月 OOD 标签调参。
  - 默认读取当前 pipeline 已生成的 U842 prediction CSV。

输出:
  artifacts/u842_p1_p2_p3_experiment/summary_metrics.csv
  artifacts/u842_p1_p2_p3_experiment/daily_metrics.csv
  artifacts/u842_p1_p2_p3_experiment/selected_params.json
  artifacts/u842_p1_p2_p3_experiment/report.md

运行:
  python scripts/experiment_u842_p1_p2_p3.py
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import min_duration_filter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_ID = "800080252842_4206894986488"
OUT_DIR = PROJECT_ROOT / "artifacts" / "u842_p1_p2_p3_experiment"
BEST_THR = 0.57
POST_MIN_ON = 1
POST_FILL_SHORT_OFF = 3
DT_HOURS = 0.25


@dataclass
class P1Params:
    rh_mean_min: float
    p_on_min: float
    hour_start: int
    hour_end: int
    guard_power_w: float
    precision_floor: float
    f1_floor: float


@dataclass
class P1LowProbRainParams:
    low_rh_min: float
    low_temp_max: float
    low_p_ge02_min: int
    low_raw73_core_min: float
    low_power_w: float
    warm_rh_min: float
    warm_temp_min: float
    warm_p_ge02_min: int
    warm_raw73_core_min: float
    warm_power_w: float
    hour_start: float
    hour_end: float
    note: str


@dataclass
class P2Params:
    n_bins: int
    min_days_per_bin: int
    clip_lo: float
    clip_hi: float
    bin_edges: List[float]
    bin_scales: Dict[str, float]


@dataclass
class P2ModeModelParams:
    mode_thresholds_w: List[float]
    feature_cols: List[str]
    mode_counts: Dict[str, int]
    classifier: str
    regressor: str
    random_state: int
    n_estimators: int
    note: str


@dataclass
class P2LossAwareModeParams:
    mode_thresholds_w: List[float]
    feature_cols: List[str]
    mode_counts: Dict[str, int]
    selected_candidate: Dict[str, float]
    objective_weights: Dict[str, float]
    train_val_objective: Dict[str, float]
    candidate_count: int
    note: str


@dataclass
class P2RawBusSegmentParams:
    mode_thresholds_w: List[float]
    raw_bus_cols: List[str]
    feature_cols: List[str]
    mode_counts: Dict[str, int]
    selected_blend: float
    candidate_blends: List[float]
    train_val_objective: Dict[str, float]
    classifier: str
    regressor: str
    note: str


@dataclass
class P2RawBusSafetyParams:
    safety_feature_cols: List[str]
    threshold: float
    classifier: str
    train_val_label_counts: Dict[str, int]
    note: str


@dataclass
class P2ExtraRiskGateParams:
    coverage_min: float
    apply_coverage_only_if_rawbus_up: bool
    risk_p_q50_lo: float
    risk_p_q50_hi: float
    risk_rh_max: float
    risk_pred_on_n_min: int
    risk_base_on_mean_min: float
    note: str


@dataclass
class P3Params:
    temp_bins: List[float]
    rh_bins: List[float]
    min_n: int
    clip_lo: float
    clip_hi: float
    bucket_scales: Dict[str, float]


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) and not pd.isna(b) else float("nan")


def _load_weather() -> pd.DataFrame:
    frames = []
    for path in [
        PROJECT_ROOT / "data/weather_cache/30.59_114.31_2025.csv",
        PROJECT_ROOT / "data/weather_cache/30.59_114.31_2026.csv",
    ]:
        frames.append(pd.read_csv(path, parse_dates=["time"]))
    w = pd.concat(frames, ignore_index=True).sort_values("time")
    w["date"] = w["time"].dt.strftime("%Y-%m-%d")
    return w


def _daily_weather(w: pd.DataFrame) -> pd.DataFrame:
    return w.groupby("date").agg(
        temp_mean=("temperature_2m", "mean"),
        temp_max=("temperature_2m", "max"),
        rh_mean=("relative_humidity_2m", "mean"),
        rh_max=("relative_humidity_2m", "max"),
    ).reset_index()


def _attach_weather(df: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").copy()
    w2 = w[["time", "temperature_2m", "apparent_temperature", "relative_humidity_2m"]].sort_values("time")
    out = pd.merge_asof(df, w2, on="time", direction="nearest", tolerance=pd.Timedelta("1h"))
    out["date"] = out["time"].dt.strftime("%Y-%m-%d")
    return out


def _load_predictions() -> pd.DataFrame:
    frames = []
    base = PROJECT_ROOT / "artifacts/trains" / USER_ID
    for stage in ["train", "val", "test"]:
        df = pd.read_csv(base / f"{stage}_pred.csv", parse_dates=["time"])
        df = df.rename(columns={
            "p_on": "p_on_main",
            "state_pred": "state_pred_main",
            "y_pred_W": "y_pred_W_main",
            "y_pred_low_W": "y_pred_low_W_main",
            "y_pred_high_W": "y_pred_high_W_main",
        })
        df["stage"] = stage
        frames.append(df)
    inf = pd.read_csv(PROJECT_ROOT / "artifacts/infers" / USER_ID / "inference_result.csv",
                      parse_dates=["time"])
    inf["stage"] = "inference"
    frames.append(inf)
    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in ["y_pred_low_W_main", "y_pred_high_W_main"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].fillna(df["y_pred_W_main"])
    keep = ["time", "stage", "y_true_W", "y_pred_W_main", "state_true",
            "state_pred_main", "p_on_main", "y_pred_low_W_main", "y_pred_high_W_main"]
    df = df[keep].copy()
    # baseline prediction CSV 已按当前 v14.7 best_thr/postprocess 输出; 但 P1 需可重算 raw。
    return df


def _classification_metrics(y_true: np.ndarray, state: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, state, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, proba) if len(set(y_true.tolist())) == 2 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, state)),
        "Precision": float(precision_score(y_true, state, zero_division=0)),
        "Recall": float(recall_score(y_true, state, zero_division=0)),
        "F1": float(f1_score(y_true, state, zero_division=0)),
        "AUC": auc,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def _regression_metrics(y_true_w: np.ndarray, y_pred_w: np.ndarray) -> Dict[str, float]:
    true_kwh = float(np.sum(y_true_w) * DT_HOURS / 1000.0)
    pred_kwh = float(np.sum(y_pred_w) * DT_HOURS / 1000.0)
    return {
        "MAE_W": float(mean_absolute_error(y_true_w, y_pred_w)),
        "RMSE_W": float(math.sqrt(mean_squared_error(y_true_w, y_pred_w))),
        "SAE": abs(pred_kwh - true_kwh) / true_kwh if true_kwh > 0 else float("nan"),
        "kWh_true": true_kwh,
        "kWh_pred": pred_kwh,
        "kWh_err": pred_kwh - true_kwh,
    }


def _summarize(df: pd.DataFrame, variant: str) -> List[Dict[str, object]]:
    rows = []
    stages = ["train", "val", "test", "inference"]
    for stage in stages:
        g = df[df["stage"] == stage]
        if len(g) == 0:
            continue
        cls = _classification_metrics(g["state_true"].astype(int).values,
                                      g["state_pred_variant"].astype(int).values,
                                      g["p_on_main"].astype(float).values)
        reg = _regression_metrics(g["y_true_W"].astype(float).values,
                                  g["y_pred_variant"].astype(float).values)
        rows.append({"variant": variant, "stage": stage, "n_samples": int(len(g)), **cls, **reg})
    # train+val selection view
    g = df[df["stage"].isin(["train", "val"])]
    cls = _classification_metrics(g["state_true"].astype(int).values,
                                  g["state_pred_variant"].astype(int).values,
                                  g["p_on_main"].astype(float).values)
    reg = _regression_metrics(g["y_true_W"].astype(float).values,
                              g["y_pred_variant"].astype(float).values)
    rows.append({"variant": variant, "stage": "train_val", "n_samples": int(len(g)), **cls, **reg})
    return rows


def _daily_metrics(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = []
    for (stage, date), g in df.groupby(["stage", "date"]):
        cls = _classification_metrics(g["state_true"].astype(int).values,
                                      g["state_pred_variant"].astype(int).values,
                                      g["p_on_main"].astype(float).values)
        reg = _regression_metrics(g["y_true_W"].astype(float).values,
                                  g["y_pred_variant"].astype(float).values)
        out.append({"variant": variant, "stage": stage, "date": date, "n_samples": int(len(g)), **cls, **reg})
    return pd.DataFrame(out)


def _with_baseline_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["state_pred_variant"] = out["state_pred_main"].astype(int)
    out["y_pred_variant"] = out["y_pred_W_main"].astype(float)
    return out


def _apply_p1(df: pd.DataFrame, params: P1Params) -> pd.DataFrame:
    out = df.copy()
    # 以当前 pipeline 已输出 state 为基线；guard 只做增量 recall 补点。
    # 不从四舍五入后的 p_on 重新构造 baseline, 避免 CSV 精度导致的伪差异。
    old_state = out["state_pred_main"].astype(int).values
    core = ((out["time"].dt.hour >= params.hour_start) &
            (out["time"].dt.hour < params.hour_end))
    guard = ((out["rh_mean"] >= params.rh_mean_min) & core &
             (out["p_on_main"] >= params.p_on_min))
    raw_guard = np.maximum(old_state, guard.astype(int).values)
    # 为避免跨日 fill_short_off 把午夜串起来, 仅对 guard 命中的日期分日平滑；
    # 未命中日期保持 pipeline baseline 逐点一致。
    state = old_state.copy()
    for _, idx in out.groupby(["stage", "date"]).indices.items():
        ii = np.asarray(idx)
        if guard.iloc[ii].any():
            state[ii] = min_duration_filter(raw_guard[ii], POST_MIN_ON, POST_FILL_SHORT_OFF)
    y_pred = out["y_pred_W_main"].astype(float).values.copy()
    newly_on = (state == 1) & (old_state == 0)
    y_pred[newly_on] = params.guard_power_w
    out["state_pred_variant"] = state
    out["y_pred_variant"] = y_pred * state
    out["p1_newly_on"] = newly_on.astype(int)
    return out


def _fit_p1_lowprob_rain_params(df: pd.DataFrame) -> P1LowProbRainParams:
    """低概率梅雨 guard 参数。

    功率锚点来自 train+val ON 功率分位；触发阈值是基于 train+val 高湿
    no-positive day 防误杀检查后固定的诊断候选，不使用 test/inference 标签。
    """
    tv_on = df[df["stage"].isin(["train", "val"]) & (df["state_true"].astype(int) == 1)]
    low_power = float(tv_on["y_true_W"].quantile(0.01))
    warm_power = float(tv_on["y_true_W"].median())
    return P1LowProbRainParams(
        low_rh_min=80.0,
        low_temp_max=22.0,
        low_p_ge02_min=20,
        low_raw73_core_min=1800.0,
        low_power_w=low_power,
        warm_rh_min=85.0,
        warm_temp_min=25.0,
        warm_p_ge02_min=35,
        warm_raw73_core_min=2300.0,
        warm_power_w=warm_power,
        hour_start=9.25,
        hour_end=22.0,
        note=("Train/val-only diagnostic low-prob rain guard. Trigger only when baseline has no ON; "
              "low-temp branch uses train+val ON p01 power, warm-rain branch uses train+val ON median power."),
    )


def _apply_p1_lowprob_rain_guard(df: pd.DataFrame, p1_df: pd.DataFrame,
                                 raw_feature_df: pd.DataFrame,
                                 params: P1LowProbRainParams) -> pd.DataFrame:
    """在既有 P1 recall guard 基础上, 对低概率梅雨整日漏检进行 day-level 补点。"""
    out = p1_df.copy()
    raw73 = "raw_load_iden_data73"
    # key 对齐, 不靠 values 顺序。
    out = out.sort_values(["stage", "time"]).copy()
    for (stage, date), idx in out.groupby(["stage", "date"]).indices.items():
        ii = np.asarray(idx)
        g = out.iloc[ii]
        gf = raw_feature_df[(raw_feature_df["stage"] == stage) &
                            (raw_feature_df["date"] == date)].reset_index(drop=True)
        if len(gf) != len(g):
            continue
        base_pred_on_n = int(g["state_pred_main"].astype(int).sum())
        if base_pred_on_n != 0:
            continue
        hour = g["time"].dt.hour + g["time"].dt.minute / 60.0
        core = (hour >= params.hour_start) & (hour < params.hour_end)
        if core.sum() == 0:
            continue
        p_ge02 = int((g.loc[core, "p_on_main"] >= 0.02).sum())
        rh = float(g["rh_mean"].iloc[0])
        temp = float(g["temp_mean"].iloc[0])
        raw73_core = float(gf.loc[core.values, raw73].mean()) if raw73 in gf.columns else 0.0
        low_trigger = (
            rh >= params.low_rh_min and temp <= params.low_temp_max and
            p_ge02 >= params.low_p_ge02_min and raw73_core >= params.low_raw73_core_min
        )
        warm_trigger = (
            rh >= params.warm_rh_min and temp >= params.warm_temp_min and
            p_ge02 >= params.warm_p_ge02_min and raw73_core >= params.warm_raw73_core_min
        )
        if not (low_trigger or warm_trigger):
            continue
        power = params.low_power_w if low_trigger else params.warm_power_w
        loc = ii[core.values]
        out.iloc[loc, out.columns.get_loc("state_pred_variant")] = 1
        out.iloc[loc, out.columns.get_loc("y_pred_variant")] = power
    return out


def _select_p1_params(df: pd.DataFrame, strict_daily_gate: bool = True) -> P1Params:
    base = _with_baseline_cols(df)
    base_tv = [r for r in _summarize(base, "baseline") if r["stage"] == "train_val"][0]
    base_daily = _daily_metrics(base, "baseline")
    base_tv_daily = base_daily[base_daily["stage"].isin(["train", "val"])]
    base_bad_f1 = int(((base_tv_daily["kWh_true"] > 0.01) & (base_tv_daily["F1"] < 0.9)).sum())
    base_bad_sae = int(((base_tv_daily["kWh_true"] > 0.01) & (base_tv_daily["SAE"] > 0.2)).sum())
    # P1 是 recall guard: 可按 train+val 有限样本分辨率给整体指标极小容忍,
    # 但默认不允许新增 train/val 日级 F1<90 或 SAE>20 异常。仍只由 train+val
    # 决定，不看 test/inference。
    precision_floor = base_tv["Precision"] - 0.0011
    f1_floor = base_tv["F1"] - 0.0006
    # strict_daily_gate=True: 只接受 train+val 日级异常不增加的候选；若全无，则 no-op。
    candidates = []
    for rh in [80.0, 85.0, 88.0, 90.0]:
        for pthr in [0.50, 0.45, 0.40, 0.35, 0.30]:
            # guard_power: 用 train+val 中低置信真实 ON 的中位功率；无样本则取 ON 中位。
            core = ((df["time"].dt.hour >= 9) & (df["time"].dt.hour < 22))
            mask = ((df["stage"].isin(["train", "val"])) &
                    (df["rh_mean"] >= rh) & core &
                    (df["p_on_main"] >= pthr) & (df["p_on_main"] < BEST_THR) &
                    (df["state_true"] == 1))
            if mask.any():
                gp = float(df.loc[mask, "y_true_W"].median())
            else:
                gp = float(df.loc[(df["stage"].isin(["train", "val"])) & (df["state_true"] == 1), "y_true_W"].median())
            params = P1Params(rh, pthr, 9, 22, gp, precision_floor, f1_floor)
            applied = _apply_p1(df, params)
            tv = [r for r in _summarize(applied, "p1") if r["stage"] == "train_val"][0]
            if tv["Precision"] >= precision_floor and tv["F1"] >= f1_floor:
                if strict_daily_gate:
                    dd = _daily_metrics(applied, "p1")
                    tvd = dd[dd["stage"].isin(["train", "val"])]
                    bad_f1 = int(((tvd["kWh_true"] > 0.01) & (tvd["F1"] < 0.9)).sum())
                    bad_sae = int(((tvd["kWh_true"] > 0.01) & (tvd["SAE"] > 0.2)).sum())
                    if bad_f1 > base_bad_f1 or bad_sae > base_bad_sae:
                        continue
                candidates.append((tv["Recall"], tv["F1"], tv["Precision"], -pthr, -rh, params, tv))
    if not candidates:
        gp = float(df.loc[(df["stage"].isin(["train", "val"])) & (df["state_true"] == 1), "y_true_W"].median())
        return P1Params(99.0, 1.0, 9, 22, gp, precision_floor, f1_floor)
    # train+val recall 优先, 再 F1/Precision, 最后更保守 pthr/rh。
    return sorted(candidates, reverse=True)[0][5]


def _fit_p2_params(df: pd.DataFrame) -> P2Params:
    tv = df[df["stage"].isin(["train", "val"])].copy()
    daily = []
    for date, g in tv.groupby("date"):
        pred_on = g["state_pred_main"].astype(int) == 1
        if pred_on.sum() < 4:
            continue
        true_kwh = g["y_true_W"].sum() * DT_HOURS / 1000.0
        pred_kwh = g["y_pred_W_main"].sum() * DT_HOURS / 1000.0
        if pred_kwh <= 0 or true_kwh <= 0:
            continue
        daily.append({
            "date": date,
            "pred_on_mean": float(g.loc[pred_on, "y_pred_W_main"].mean()),
            "scale": float(true_kwh / pred_kwh),
        })
    d = pd.DataFrame(daily)
    # 3 档: 低/中/高预测功率模式。用 train+val predicted ON mean 三分位。
    qs = d["pred_on_mean"].quantile([0, 1/3, 2/3, 1]).values.astype(float)
    qs[0] = -float("inf")
    qs[-1] = float("inf")
    scales = {}
    for i in range(3):
        m = (d["pred_on_mean"] > qs[i]) & (d["pred_on_mean"] <= qs[i + 1])
        s = float(d.loc[m, "scale"].median()) if m.sum() >= 3 else 1.0
        s = float(np.clip(s, 0.80, 1.20))
        scales[str(i)] = s
    return P2Params(n_bins=3, min_days_per_bin=3, clip_lo=0.80, clip_hi=1.20,
                    bin_edges=[float(x) if np.isfinite(x) else ("-inf" if x < 0 else "inf") for x in qs],
                    bin_scales=scales)


def _apply_p2(df: pd.DataFrame, params: P2Params) -> pd.DataFrame:
    out = df.copy()
    out["state_pred_variant"] = out["state_pred_main"].astype(int)
    out["y_pred_variant"] = out["y_pred_W_main"].astype(float)
    # 每日 predicted ON mean 决定模式 scale。
    for (stage, date), idx in out.groupby(["stage", "date"]).indices.items():
        ii = np.asarray(idx)
        g = out.iloc[ii]
        pred_on = g["state_pred_variant"].astype(int).values == 1
        if pred_on.sum() == 0:
            continue
        pom = float(g.loc[pred_on, "y_pred_variant"].mean())
        # edges stored with sentinels as strings; reconstruct.
        edges = []
        for e in params.bin_edges:
            if e == "-inf": edges.append(-float("inf"))
            elif e == "inf": edges.append(float("inf"))
            else: edges.append(float(e))
        b = 0
        for i in range(len(edges) - 1):
            if pom > edges[i] and pom <= edges[i + 1]:
                b = i
                break
        scale = params.bin_scales.get(str(b), 1.0)
        out.iloc[ii, out.columns.get_loc("y_pred_variant")] = g["y_pred_variant"].values * scale
    return out


def _p2_mode_feature_cols() -> List[str]:
    return [
        "p_on_main", "y_pred_W_main", "y_pred_low_W_main", "y_pred_high_W_main",
        "hour_sin", "hour_cos", "dow",
        "temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "day_pred_on_n", "day_pred_on_mean", "day_pred_kwh",
        "day_p_on_mean", "day_p_on_q25", "day_p_on_q50", "day_p_on_q75",
        "temp_mean", "rh_mean",
    ]


def _ensure_p2_mode_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hour" not in out.columns:
        out["hour"] = out["time"].dt.hour + out["time"].dt.minute / 60.0
    if "hour_sin" not in out.columns:
        hour = out["hour"]
        out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        out["dow"] = out["time"].dt.dayofweek.astype(float)
    daily_rows = []
    for (stage, date), g in out.groupby(["stage", "date"]):
        pred_on = g["state_pred_main"].astype(int) == 1
        daily_rows.append({
            "stage": stage,
            "date": date,
            "day_pred_on_n": int(pred_on.sum()),
            "day_pred_on_mean": float(g.loc[pred_on, "y_pred_W_main"].mean()) if pred_on.any() else 0.0,
            "day_pred_kwh": float(g["y_pred_W_main"].sum() * DT_HOURS / 1000.0),
            "day_p_on_mean": float(g["p_on_main"].mean()),
            "day_p_on_q25": float(g["p_on_main"].quantile(0.25)),
            "day_p_on_q50": float(g["p_on_main"].quantile(0.50)),
            "day_p_on_q75": float(g["p_on_main"].quantile(0.75)),
        })
    # 清理旧列避免重复 merge 产生 _x/_y。
    drop_cols = [c for c in ["day_pred_on_n", "day_pred_on_mean", "day_pred_kwh",
                             "day_p_on_mean", "day_p_on_q25", "day_p_on_q50", "day_p_on_q75"]
                 if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out = out.merge(pd.DataFrame(daily_rows), on=["stage", "date"], how="left")
    for c in _p2_mode_feature_cols():
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_p2_mode_model(df: pd.DataFrame) -> Tuple[P2ModeModelParams, object, Dict[int, object]]:
    """训练真正的 mode_classifier + per-mode regressor。

    mode label 由 train+val 真 ON 样本 y_true_W 的 1/3、2/3 分位数定义。
    推理时只对 baseline 已判 ON 的点重估功率；分类 state 不变。
    """
    xdf = _ensure_p2_mode_features(df)
    feat_cols = _p2_mode_feature_cols()
    train_mask = xdf["stage"].isin(["train", "val"]) & (xdf["state_true"].astype(int) == 1)
    q1, q2 = xdf.loc[train_mask, "y_true_W"].quantile([1/3, 2/3]).values.astype(float)

    def _mode(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= q1, 0, np.where(y <= q2, 1, 2)).astype(int)

    X = xdf.loc[train_mask, feat_cols].values.astype(float)
    y_mode = _mode(xdf.loc[train_mask, "y_true_W"].values)
    mode_clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=1, class_weight="balanced",
    )
    mode_clf.fit(X, y_mode)
    regs: Dict[int, object] = {}
    counts: Dict[str, int] = {}
    for m in [0, 1, 2]:
        mm = y_mode == m
        counts[str(m)] = int(mm.sum())
        reg = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=8,
            random_state=42, n_jobs=1,
        )
        reg.fit(X[mm], xdf.loc[train_mask, "y_true_W"].values[mm])
        regs[m] = reg
    params = P2ModeModelParams(
        mode_thresholds_w=[float(q1), float(q2)],
        feature_cols=feat_cols,
        mode_counts=counts,
        classifier="RandomForestClassifier(n_estimators=200,max_depth=6,min_samples_leaf=10,class_weight=balanced)",
        regressor="RandomForestRegressor(n_estimators=200,max_depth=8,min_samples_leaf=8)",
        random_state=42,
        n_estimators=200,
        note="mode labels from train+val true ON y_true_W tertiles; test/inference labels not used",
    )
    return params, mode_clf, regs


def _apply_p2_mode_model(df: pd.DataFrame, mode_clf, regs: Dict[int, object]) -> pd.DataFrame:
    out = _ensure_p2_mode_features(df)
    feat_cols = _p2_mode_feature_cols()
    state = out["state_pred_main"].astype(int).values
    y_pred = np.zeros(len(out), dtype=float)
    on = state == 1
    if on.any():
        X_on = out.loc[on, feat_cols].values.astype(float)
        modes = mode_clf.predict(X_on)
        pred_on = np.zeros(on.sum(), dtype=float)
        for m in [0, 1, 2]:
            sel = modes == m
            if sel.any():
                pred_on[sel] = regs[m].predict(X_on[sel])
        y_pred[on] = np.clip(pred_on, 0, None)
    out["state_pred_variant"] = state
    out["y_pred_variant"] = y_pred
    return out


def _p2_enhanced_feature_cols() -> List[str]:
    base = _p2_mode_feature_cols()
    extra = [
        # 日内分段模式: 用 baseline 已识别 ON 点的早/中/晚功率形态区分低功率长时与高功率长时
        "day_pred_on_mean_morning", "day_pred_on_mean_midday", "day_pred_on_mean_evening",
        "day_pred_kwh_morning", "day_pred_kwh_midday", "day_pred_kwh_evening",
        "day_p_on_ge_02", "day_p_on_ge_05", "day_p_on_ge_10", "day_p_on_ge_30",
        "day_p_on_ge_45", "day_p_on_ge_57",
        # 连续 ON 段形态: P2 与 P1 分开, 只利用 baseline state/pred/proba, 不改 state
        "seg_len", "seg_pos_frac", "seg_elapsed_h", "seg_remaining_h",
        "seg_pred_mean", "seg_pred_std", "seg_pred_min", "seg_pred_max",
        "seg_p_on_mean", "seg_p_on_min", "seg_p_on_q25", "seg_p_on_q50", "seg_p_on_q75",
        "seg_start_hour", "seg_end_hour",
        # 局部滚动与相对量纲
        "p_on_roll4_mean", "p_on_roll8_mean", "pred_roll4_mean", "pred_roll8_mean",
        "pred_to_day_mean", "pred_to_seg_mean", "pred_interval_width",
        "pred_interval_ratio", "pred_low_ratio", "pred_high_ratio",
    ]
    return base + [c for c in extra if c not in base]


def _ensure_p2_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """增加 P2 mode 判别特征；只由 baseline 预测/时间/天气构造, 不用 OOD 标签。"""
    out = _ensure_p2_mode_features(df).sort_values(["stage", "date", "time"]).copy()

    # 日内分段统计
    daily_rows = []
    for (stage, date), g in out.groupby(["stage", "date"]):
        row = {"stage": stage, "date": date}
        for name, h0, h1 in [
            ("morning", 9, 12), ("midday", 12, 17), ("evening", 17, 22),
        ]:
            m = (g["time"].dt.hour >= h0) & (g["time"].dt.hour < h1)
            po = m & (g["state_pred_main"].astype(int) == 1)
            row[f"day_pred_on_mean_{name}"] = float(g.loc[po, "y_pred_W_main"].mean()) if po.any() else 0.0
            row[f"day_pred_kwh_{name}"] = float(g.loc[m, "y_pred_W_main"].sum() * DT_HOURS / 1000.0)
        for thr in [0.02, 0.05, 0.10, 0.30, 0.45, 0.57]:
            key = str(thr).replace("0.", "")
            row[f"day_p_on_ge_{key}"] = int((g["p_on_main"] >= thr).sum())
        daily_rows.append(row)
    add_daily = pd.DataFrame(daily_rows)
    drop_cols = [c for c in add_daily.columns if c not in ("stage", "date") and c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out = out.merge(add_daily, on=["stage", "date"], how="left")

    # 连续 baseline predicted-ON 段特征 + 分日 rolling
    seg_cols = [
        "seg_len", "seg_pos_frac", "seg_elapsed_h", "seg_remaining_h",
        "seg_pred_mean", "seg_pred_std", "seg_pred_min", "seg_pred_max",
        "seg_p_on_mean", "seg_p_on_min", "seg_p_on_q25", "seg_p_on_q50", "seg_p_on_q75",
        "seg_start_hour", "seg_end_hour",
        "p_on_roll4_mean", "p_on_roll8_mean", "pred_roll4_mean", "pred_roll8_mean",
    ]
    for c in seg_cols:
        out[c] = 0.0
    for _, idx in out.groupby(["stage", "date"]).indices.items():
        ii = np.asarray(idx)
        g = out.iloc[ii]
        # rolling 不依赖 state, 分日防止跨日泄漏
        out.iloc[ii, out.columns.get_loc("p_on_roll4_mean")] = g["p_on_main"].rolling(4, min_periods=1).mean().values
        out.iloc[ii, out.columns.get_loc("p_on_roll8_mean")] = g["p_on_main"].rolling(8, min_periods=1).mean().values
        out.iloc[ii, out.columns.get_loc("pred_roll4_mean")] = g["y_pred_W_main"].rolling(4, min_periods=1).mean().values
        out.iloc[ii, out.columns.get_loc("pred_roll8_mean")] = g["y_pred_W_main"].rolling(8, min_periods=1).mean().values

        st = g["state_pred_main"].astype(int).values
        n = len(st)
        pos = 0
        while pos < n:
            if st[pos] != 1:
                pos += 1
                continue
            end = pos
            while end < n and st[end] == 1:
                end += 1
            loc = ii[pos:end]
            sg = out.iloc[loc]
            L = len(loc)
            pvals = sg["p_on_main"].astype(float)
            yvals = sg["y_pred_W_main"].astype(float)
            out.iloc[loc, out.columns.get_loc("seg_len")] = float(L)
            out.iloc[loc, out.columns.get_loc("seg_pos_frac")] = (np.arange(L) + 1) / max(L, 1)
            out.iloc[loc, out.columns.get_loc("seg_elapsed_h")] = np.arange(L) * DT_HOURS
            out.iloc[loc, out.columns.get_loc("seg_remaining_h")] = (L - 1 - np.arange(L)) * DT_HOURS
            out.iloc[loc, out.columns.get_loc("seg_pred_mean")] = float(yvals.mean())
            out.iloc[loc, out.columns.get_loc("seg_pred_std")] = float(yvals.std(ddof=0))
            out.iloc[loc, out.columns.get_loc("seg_pred_min")] = float(yvals.min())
            out.iloc[loc, out.columns.get_loc("seg_pred_max")] = float(yvals.max())
            out.iloc[loc, out.columns.get_loc("seg_p_on_mean")] = float(pvals.mean())
            out.iloc[loc, out.columns.get_loc("seg_p_on_min")] = float(pvals.min())
            out.iloc[loc, out.columns.get_loc("seg_p_on_q25")] = float(pvals.quantile(0.25))
            out.iloc[loc, out.columns.get_loc("seg_p_on_q50")] = float(pvals.quantile(0.50))
            out.iloc[loc, out.columns.get_loc("seg_p_on_q75")] = float(pvals.quantile(0.75))
            out.iloc[loc, out.columns.get_loc("seg_start_hour")] = float(sg["hour"].iloc[0])
            out.iloc[loc, out.columns.get_loc("seg_end_hour")] = float(sg["hour"].iloc[-1] + DT_HOURS)
            pos = end

    # 相对量纲与预测区间宽度
    eps = 1e-6
    out["pred_to_day_mean"] = out["y_pred_W_main"] / np.maximum(out["day_pred_on_mean"], eps)
    out["pred_to_seg_mean"] = out["y_pred_W_main"] / np.maximum(out["seg_pred_mean"], eps)
    out["pred_interval_width"] = (out["y_pred_high_W_main"] - out["y_pred_low_W_main"]).clip(lower=0)
    out["pred_interval_ratio"] = out["pred_interval_width"] / np.maximum(out["y_pred_W_main"], eps)
    out["pred_low_ratio"] = out["y_pred_low_W_main"] / np.maximum(out["y_pred_W_main"], eps)
    out["pred_high_ratio"] = out["y_pred_high_W_main"] / np.maximum(out["y_pred_W_main"], eps)

    for c in _p2_enhanced_feature_cols():
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _p2_daily_loss_objective(df_variant: pd.DataFrame, variant: str,
                             weights: Dict[str, float],
                             stages: Tuple[str, ...] = ("train", "val")) -> Dict[str, float]:
    if set(stages) == {"train", "val"}:
        summary = [r for r in _summarize(df_variant, variant) if r["stage"] == "train_val"][0]
    else:
        gsum = df_variant[df_variant["stage"].isin(stages)]
        cls = _classification_metrics(gsum["state_true"].astype(int).values,
                                      gsum["state_pred_variant"].astype(int).values,
                                      gsum["p_on_main"].astype(float).values)
        reg = _regression_metrics(gsum["y_true_W"].astype(float).values,
                                  gsum["y_pred_variant"].astype(float).values)
        summary = {"stage": "+".join(stages), **cls, **reg}
    daily = _daily_metrics(df_variant, variant)
    tv = daily[daily["stage"].isin(stages)]
    on = tv[tv["kWh_true"] > 0.01].copy()
    mean_daily_sae = float(on["SAE"].mean()) if len(on) else 0.0
    bad_rate = float((on["SAE"] > 0.2).mean()) if len(on) else 0.0
    p95_daily_sae = float(on["SAE"].quantile(0.95)) if len(on) else 0.0
    objective = (
        weights["point_mae"] * float(summary["MAE_W"]) +
        weights["overall_sae"] * float(summary["SAE"]) +
        weights["mean_daily_sae"] * mean_daily_sae +
        weights["p95_daily_sae"] * p95_daily_sae +
        weights["bad_day_rate"] * bad_rate
    )
    return {
        "objective": float(objective),
        "point_mae_w": float(summary["MAE_W"]),
        "overall_sae": float(summary["SAE"]),
        "mean_daily_sae": mean_daily_sae,
        "p95_daily_sae": p95_daily_sae,
        "bad_day_rate": bad_rate,
    }


def _fit_apply_p2_lossaware_candidate(df: pd.DataFrame, candidate: Dict[str, float],
                                      feat_cols: List[str], q1: float, q2: float,
                                      train_stages: Tuple[str, ...] = ("train", "val")):
    xdf = _ensure_p2_enhanced_features(df)
    train_mask = xdf["stage"].isin(train_stages) & (xdf["state_true"].astype(int) == 1)

    def _mode(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= q1, 0, np.where(y <= q2, 1, 2)).astype(int)

    # 日级 sample weight: 用 train+val baseline 日级 SAE 构造, 不看 test/inference。
    base_daily = _daily_metrics(_with_baseline_cols(xdf), "baseline")
    btv = base_daily[base_daily["stage"].isin(train_stages)]
    day_sae = dict(zip(btv["date"], btv["SAE"].fillna(0.0)))
    alpha = float(candidate["daily_sae_weight"])
    beta = float(candidate["bad_day_weight"])
    sw = []
    for d in xdf.loc[train_mask, "date"]:
        s = float(day_sae.get(d, 0.0))
        sw.append(1.0 + alpha * min(s, 1.0) + beta * float(s > 0.2))
    sw = np.asarray(sw, dtype=float)

    X = xdf.loc[train_mask, feat_cols].values.astype(float)
    y_true = xdf.loc[train_mask, "y_true_W"].values.astype(float)
    y_mode = _mode(y_true)
    clf = RandomForestClassifier(
        n_estimators=int(candidate["n_estimators"]),
        max_depth=int(candidate["clf_max_depth"]),
        min_samples_leaf=int(candidate["clf_min_leaf"]),
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X, y_mode, sample_weight=sw)
    regs: Dict[int, object] = {}
    counts: Dict[str, int] = {}
    for m in [0, 1, 2]:
        mm = y_mode == m
        counts[str(m)] = int(mm.sum())
        reg = RandomForestRegressor(
            n_estimators=int(candidate["n_estimators"]),
            max_depth=int(candidate["reg_max_depth"]),
            min_samples_leaf=int(candidate["reg_min_leaf"]),
            random_state=42,
            n_jobs=1,
        )
        reg.fit(X[mm], y_true[mm], sample_weight=sw[mm])
        regs[m] = reg

    out = _ensure_p2_enhanced_features(df)
    state = out["state_pred_main"].astype(int).values
    pred = np.zeros(len(out), dtype=float)
    on = state == 1
    if on.any():
        X_on = out.loc[on, feat_cols].values.astype(float)
        modes = clf.predict(X_on)
        pred_on = np.zeros(on.sum(), dtype=float)
        for m in [0, 1, 2]:
            sel = modes == m
            if sel.any():
                pred_on[sel] = regs[m].predict(X_on[sel])
        blend = float(candidate["blend"])
        base_on = out.loc[on, "y_pred_W_main"].values.astype(float)
        pred[on] = blend * np.clip(pred_on, 0, None) + (1.0 - blend) * base_on
    out["state_pred_variant"] = state
    out["y_pred_variant"] = pred
    return out, clf, regs, counts


def _fit_p2_lossaware_mode_model(df: pd.DataFrame) -> Tuple[P2LossAwareModeParams, object, Dict[int, object]]:
    xdf = _ensure_p2_enhanced_features(df)
    feat_cols = _p2_enhanced_feature_cols()
    train_mask = xdf["stage"].isin(["train", "val"]) & (xdf["state_true"].astype(int) == 1)
    q1, q2 = xdf.loc[train_mask, "y_true_W"].quantile([1/3, 2/3]).values.astype(float)
    weights = {
        # W 与比例混合, 固定于实验脚本；只在 train+val 上比较候选。
        "point_mae": 1.0,
        "overall_sae": 80.0,
        "mean_daily_sae": 40.0,
        "p95_daily_sae": 25.0,
        "bad_day_rate": 200.0,
    }
    candidates = []
    for n_est in [160, 240]:
        for clf_depth, reg_depth, clf_leaf, reg_leaf in [
            (5, 6, 12, 10), (6, 8, 10, 8), (8, 10, 6, 6),
        ]:
            for alpha, beta in [(0.0, 0.0), (2.0, 4.0), (5.0, 10.0)]:
                for blend in [1.0, 0.85, 0.70]:
                    candidates.append({
                        "n_estimators": float(n_est),
                        "clf_max_depth": float(clf_depth),
                        "reg_max_depth": float(reg_depth),
                        "clf_min_leaf": float(clf_leaf),
                        "reg_min_leaf": float(reg_leaf),
                        "daily_sae_weight": float(alpha),
                        "bad_day_weight": float(beta),
                        "blend": float(blend),
                    })
    best = None
    # 候选选择: 只用 train 训练、val 日级 loss-aware 目标选型，避免同集过拟合。
    for cand in candidates:
        applied_val_model, _clf_tmp, _regs_tmp, _counts_tmp = _fit_apply_p2_lossaware_candidate(
            df, cand, feat_cols, q1, q2, train_stages=("train",)
        )
        obj_val = _p2_daily_loss_objective(applied_val_model, "p2_lossaware", weights, stages=("val",))
        key = (obj_val["objective"], obj_val["bad_day_rate"], obj_val["p95_daily_sae"], obj_val["point_mae_w"])
        if best is None or key < best[0]:
            best = (key, cand, obj_val)
    assert best is not None
    _, selected, obj_val = best
    # 最终模型: 用已选候选在 train+val 上重训，再用于 test/inference 验证。
    applied_final, clf, regs, counts = _fit_apply_p2_lossaware_candidate(
        df, selected, feat_cols, q1, q2, train_stages=("train", "val")
    )
    obj_train_val = _p2_daily_loss_objective(applied_final, "p2_lossaware", weights, stages=("train", "val"))
    obj = dict(obj_train_val)
    obj["selection_val_objective"] = obj_val
    params = P2LossAwareModeParams(
        mode_thresholds_w=[float(q1), float(q2)],
        feature_cols=feat_cols,
        mode_counts=counts,
        selected_candidate={k: float(v) for k, v in selected.items()},
        objective_weights=weights,
        train_val_objective=obj,
        candidate_count=len(candidates),
        note=("loss-aware candidate selection uses train-only fit + val objective; final model retrained on train+val. "
              "objective = point_MAE + daily/overall SAE + SAE>20 day penalty; P1 state guard not used"),
    )
    return params, clf, regs


def _apply_p2_lossaware_mode_model(df: pd.DataFrame, params: P2LossAwareModeParams,
                                   clf, regs: Dict[int, object]) -> pd.DataFrame:
    out = _ensure_p2_enhanced_features(df)
    feat_cols = params.feature_cols
    state = out["state_pred_main"].astype(int).values
    pred = np.zeros(len(out), dtype=float)
    on = state == 1
    if on.any():
        X_on = out.loc[on, feat_cols].values.astype(float)
        modes = clf.predict(X_on)
        pred_on = np.zeros(on.sum(), dtype=float)
        for m in [0, 1, 2]:
            sel = modes == m
            if sel.any():
                pred_on[sel] = regs[m].predict(X_on[sel])
        blend = float(params.selected_candidate.get("blend", 1.0))
        base_on = out.loc[on, "y_pred_W_main"].values.astype(float)
        pred[on] = blend * np.clip(pred_on, 0, None) + (1.0 - blend) * base_on
    out["state_pred_variant"] = state
    out["y_pred_variant"] = pred
    return out


def _load_raw_bus_resampled() -> Tuple[pd.DataFrame, List[str]]:
    """加载 U842 原始总线并重采样到 15min。

    只使用总线原始电参量与时间，不读取分路标签；用于 P2 功率模式识别。
    """
    paths = sorted((PROJECT_ROOT / "data/trains" / USER_ID).glob("e241_*.csv"))
    paths += sorted((PROJECT_ROOT / "data/infers" / USER_ID).glob("e241_*.csv"))
    if not paths:
        raise FileNotFoundError(f"U842 raw bus csv not found under data/trains|infers/{USER_ID}")
    frames = []
    data_cols = None
    for p in paths:
        b = pd.read_csv(p, encoding="utf-8")
        b["event_time"] = pd.to_datetime(b["event_time"], errors="coerce")
        b = b.dropna(subset=["event_time"])
        cols = [c for c in b.columns if c.startswith("load_iden_data")]
        b[cols] = b[cols].replace(-2147483648, np.nan)
        frames.append(b[["event_time"] + cols])
        data_cols = cols if data_cols is None else sorted(set(data_cols) | set(cols))
    bus = pd.concat(frames, ignore_index=True, sort=False).sort_values("event_time")
    data_cols = [c for c in (data_cols or []) if c in bus.columns]
    rs = (bus.set_index("event_time")[data_cols]
          .resample("15min", label="left", closed="left").mean()
          .ffill(limit=2).bfill(limit=2))
    return rs, data_cols


def _select_raw_bus_cols(raw_rs: pd.DataFrame, n_cols: int = 12) -> List[str]:
    """优先取 bundle feat_cols 中存在的原始总线列, 回退到方差最大的列。"""
    cols: List[str] = []
    try:
        bundle = joblib.load(PROJECT_ROOT / "models" / USER_ID / "nilm_ac_two_stage.pkl")
        for c in bundle.get("feat_cols", []):
            if c in raw_rs.columns and c not in cols:
                cols.append(c)
            if len(cols) >= n_cols:
                return cols
    except Exception:
        pass
    var_cols = raw_rs.var(numeric_only=True).sort_values(ascending=False).index.tolist()
    for c in var_cols:
        if c not in cols:
            cols.append(c)
        if len(cols) >= n_cols:
            break
    return cols


def _ensure_p2_rawbus_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """P2 原始总线/段级特征。

    特征仅由 raw bus + baseline predicted state/proba/power + 时间/天气构造；
    不使用 test/inference 标签。段级统计使用 baseline predicted-ON 段。
    """
    out = _ensure_p2_enhanced_features(df).copy()
    raw_rs, _ = _load_raw_bus_resampled()
    raw_cols = _select_raw_bus_cols(raw_rs, n_cols=12)
    raw = raw_rs[raw_cols].copy()
    raw.columns = [f"raw_{c}" for c in raw.columns]
    # 点级原始总线动态特征: 前 6 个高相关/高重要列的 diff/rolling。
    for c in raw_cols[:6]:
        s = raw_rs[c]
        raw[f"raw_{c}_d1"] = s.diff().fillna(0.0)
        raw[f"raw_{c}_d4"] = s.diff(4).fillna(0.0)
        raw[f"raw_{c}_roll4_mean"] = s.rolling(4, min_periods=1).mean()
        raw[f"raw_{c}_roll4_std"] = s.rolling(4, min_periods=1).std().fillna(0.0)
    raw = raw.reset_index().rename(columns={"event_time": "time"})
    out = out.merge(raw, on="time", how="left")
    raw_feature_cols = [c for c in out.columns if c.startswith("raw_load_iden_data")]
    out = out.sort_values(["stage", "time"])
    out[raw_feature_cols] = out[raw_feature_cols].ffill().bfill().fillna(0.0)

    # 日级 raw bus 统计: 全日 + baseline predicted-ON 子集。
    daily_rows = []
    stat_cols = raw_feature_cols[:8]
    for (stage, date), g in out.groupby(["stage", "date"]):
        pred_on = g["state_pred_main"].astype(int) == 1
        row = {"stage": stage, "date": date}
        for c in stat_cols:
            row[f"day_{c}_mean"] = float(g[c].mean())
            row[f"day_{c}_std"] = float(g[c].std(ddof=0))
            row[f"day_on_{c}_mean"] = float(g.loc[pred_on, c].mean()) if pred_on.any() else 0.0
        daily_rows.append(row)
    add = pd.DataFrame(daily_rows)
    drop_cols = [c for c in add.columns if c not in ("stage", "date") and c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out = out.merge(add, on=["stage", "date"], how="left")

    # baseline predicted-ON 段内 raw bus 统计。
    seg_cols = []
    for c in stat_cols:
        for suf in ["seg_mean", "seg_std", "seg_min", "seg_max", "seg_range", "to_seg_mean"]:
            name = f"{c}_{suf}"
            out[name] = 0.0
            seg_cols.append(name)
    for _, idx in out.groupby(["stage", "date"]).indices.items():
        ii = np.asarray(idx)
        g = out.iloc[ii]
        st = g["state_pred_main"].astype(int).values
        n = len(st)
        pos = 0
        while pos < n:
            if st[pos] != 1:
                pos += 1
                continue
            end = pos
            while end < n and st[end] == 1:
                end += 1
            loc = ii[pos:end]
            sg = out.iloc[loc]
            for c in stat_cols:
                vals = sg[c].astype(float)
                mean = float(vals.mean())
                std = float(vals.std(ddof=0))
                mn = float(vals.min())
                mx = float(vals.max())
                out.iloc[loc, out.columns.get_loc(f"{c}_seg_mean")] = mean
                out.iloc[loc, out.columns.get_loc(f"{c}_seg_std")] = std
                out.iloc[loc, out.columns.get_loc(f"{c}_seg_min")] = mn
                out.iloc[loc, out.columns.get_loc(f"{c}_seg_max")] = mx
                out.iloc[loc, out.columns.get_loc(f"{c}_seg_range")] = mx - mn
                out.iloc[loc, out.columns.get_loc(f"{c}_to_seg_mean")] = vals.values / (abs(mean) + 1e-6)
            pos = end

    for c in _p2_rawbus_feature_cols(out):
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out.attrs["raw_bus_cols"] = raw_cols
    return out


def _p2_rawbus_feature_cols(df: pd.DataFrame) -> List[str]:
    # raw-bus 方案刻意不使用大量 baseline 预测派生段级特征；只保留最小
    # baseline/时间/天气上下文 + 真正原始总线/总线段级特征, 便于验证 raw bus 的增益。
    base = [
        "p_on_main", "y_pred_W_main", "y_pred_low_W_main", "y_pred_high_W_main",
        "hour_sin", "hour_cos", "dow",
        "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    ]
    raw_cols = [c for c in df.columns
                if c.startswith("raw_load_iden_data")
                or c.startswith("day_raw_load_iden_data")
                or ("raw_load_iden_data" in c and "seg_" in c)]
    return base + [c for c in raw_cols if c not in base]


def _fit_p2_rawbus_segment_model(df: pd.DataFrame) -> Tuple[P2RawBusSegmentParams, object, Dict[int, object]]:
    xdf = _ensure_p2_rawbus_segment_features(df)
    feat_cols = _p2_rawbus_feature_cols(xdf)
    raw_bus_cols = list(xdf.attrs.get("raw_bus_cols", []))
    train_mask = xdf["stage"].isin(["train", "val"]) & (xdf["state_true"].astype(int) == 1)
    q1, q2 = xdf.loc[train_mask, "y_true_W"].quantile([1/3, 2/3]).values.astype(float)

    def _mode(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= q1, 0, np.where(y <= q2, 1, 2)).astype(int)

    X = xdf.loc[train_mask, feat_cols].values.astype(float)
    y_true = xdf.loc[train_mask, "y_true_W"].values.astype(float)
    y_mode = _mode(y_true)
    clf = RandomForestClassifier(
        n_estimators=240, max_depth=8, min_samples_leaf=6,
        class_weight="balanced", random_state=42, n_jobs=1,
    )
    clf.fit(X, y_mode)
    regs: Dict[int, object] = {}
    counts: Dict[str, int] = {}
    for m in [0, 1, 2]:
        mm = y_mode == m
        counts[str(m)] = int(mm.sum())
        reg = RandomForestRegressor(
            n_estimators=240, max_depth=10, min_samples_leaf=6,
            random_state=42, n_jobs=1,
        )
        reg.fit(X[mm], y_true[mm])
        regs[m] = reg

    # blend 也仅按 train+val objective 选择；通常 raw bus 模型 blend=1 最优。
    blends = [1.0, 0.85, 0.70, 0.50, 0.30]
    weights = {"point_mae": 1.0, "overall_sae": 80.0,
               "mean_daily_sae": 40.0, "p95_daily_sae": 25.0,
               "bad_day_rate": 200.0}
    best = None
    for blend in blends:
        tmp_params = P2RawBusSegmentParams(
            mode_thresholds_w=[float(q1), float(q2)], raw_bus_cols=raw_bus_cols,
            feature_cols=feat_cols, mode_counts=counts, selected_blend=blend,
            candidate_blends=blends, train_val_objective={},
            classifier="RandomForestClassifier(n_estimators=240,max_depth=8,min_samples_leaf=6,class_weight=balanced)",
            regressor="RandomForestRegressor(n_estimators=240,max_depth=10,min_samples_leaf=6)",
            note="temporary for blend selection",
        )
        applied = _apply_p2_rawbus_segment_model(df, tmp_params, clf, regs)
        obj = _p2_daily_loss_objective(applied, "p2_rawbus", weights, stages=("train", "val"))
        key = (obj["objective"], obj["bad_day_rate"], obj["p95_daily_sae"], obj["point_mae_w"])
        if best is None or key < best[0]:
            best = (key, blend, obj)
    assert best is not None
    _, selected_blend, obj = best
    params = P2RawBusSegmentParams(
        mode_thresholds_w=[float(q1), float(q2)],
        raw_bus_cols=raw_bus_cols,
        feature_cols=feat_cols,
        mode_counts=counts,
        selected_blend=float(selected_blend),
        candidate_blends=[float(x) for x in blends],
        train_val_objective=obj,
        classifier="RandomForestClassifier(n_estimators=240,max_depth=8,min_samples_leaf=6,class_weight=balanced)",
        regressor="RandomForestRegressor(n_estimators=240,max_depth=10,min_samples_leaf=6)",
        note=("P2 raw-bus/segment model; features use raw bus resampled to 15min + baseline predicted-ON segment stats. "
              "P1 state guard not used; test/inference labels not used for params."),
    )
    return params, clf, regs


def _apply_p2_rawbus_segment_model(df: pd.DataFrame, params: P2RawBusSegmentParams,
                                   clf, regs: Dict[int, object]) -> pd.DataFrame:
    out = _ensure_p2_rawbus_segment_features(df)
    feat_cols = params.feature_cols
    state = out["state_pred_main"].astype(int).values
    pred = np.zeros(len(out), dtype=float)
    on = state == 1
    if on.any():
        X_on = out.loc[on, feat_cols].values.astype(float)
        modes = clf.predict(X_on)
        pred_on = np.zeros(on.sum(), dtype=float)
        for m in [0, 1, 2]:
            sel = modes == m
            if sel.any():
                pred_on[sel] = regs[m].predict(X_on[sel])
        base_on = out.loc[on, "y_pred_W_main"].values.astype(float)
        blend = float(params.selected_blend)
        pred[on] = blend * np.clip(pred_on, 0, None) + (1.0 - blend) * base_on
    out["state_pred_variant"] = state
    out["y_pred_variant"] = pred
    return out


def _p2_safety_feature_table(base_df: pd.DataFrame, rawbus_df: pd.DataFrame,
                             raw_feature_df: pd.DataFrame) -> pd.DataFrame:
    """日级安全闸特征表。

    label 仅在 train+val 中使用: rawbus 日级绝对电量误差是否小于 baseline。
    特征只来自 baseline/rawbus 预测、天气与 raw bus 统计，不使用 test/inference 标签。
    """
    rows = []
    # rawbus_df 与 base_df 均应带 state_pred_variant/y_pred_variant
    for (stage, date), gb0 in base_df.groupby(["stage", "date"]):
        gb = gb0.reset_index(drop=True)
        gr = rawbus_df[(rawbus_df["stage"] == stage) & (rawbus_df["date"] == date)].reset_index(drop=True)
        gf = raw_feature_df[(raw_feature_df["stage"] == stage) & (raw_feature_df["date"] == date)].reset_index(drop=True)
        if len(gr) != len(gb):
            continue
        pred_on = gb["state_pred_variant"].astype(int).values == 1
        true_kwh = float(gb["y_true_W"].sum() * DT_HOURS / 1000.0)
        base_kwh = float(gb["y_pred_variant"].sum() * DT_HOURS / 1000.0)
        raw_kwh = float(gr["y_pred_variant"].sum() * DT_HOURS / 1000.0)
        raw73 = "raw_load_iden_data73"
        row = {
            "stage": stage,
            "date": date,
            "true_kwh": true_kwh,
            "base_kwh": base_kwh,
            "rawbus_kwh": raw_kwh,
            "abs_base_err": abs(base_kwh - true_kwh),
            "abs_rawbus_err": abs(raw_kwh - true_kwh),
            "rawbus_improves": int(abs(raw_kwh - true_kwh) < abs(base_kwh - true_kwh)),
            "kwh_delta": raw_kwh - base_kwh,
            "ratio_delta": (raw_kwh - base_kwh) / base_kwh if base_kwh > 1e-9 else 0.0,
            "n_samples": int(len(gb)),
            "coverage": float(len(gb) / 96.0),
            "pred_on_n": int(pred_on.sum()),
            "base_on_mean": float(gb.loc[pred_on, "y_pred_variant"].mean()) if pred_on.any() else 0.0,
            "base_on_std": float(gb.loc[pred_on, "y_pred_variant"].std(ddof=0)) if pred_on.any() else 0.0,
            "p_on_mean": float(gb["p_on_main"].mean()),
            "p_on_q25": float(gb["p_on_main"].quantile(0.25)),
            "p_on_q50": float(gb["p_on_main"].quantile(0.50)),
            "p_on_q75": float(gb["p_on_main"].quantile(0.75)),
            "rh_mean": float(gb["rh_mean"].iloc[0]) if "rh_mean" in gb else 0.0,
            "temp_mean": float(gb["temp_mean"].iloc[0]) if "temp_mean" in gb else 0.0,
        }
        if raw73 in gf.columns:
            row.update({
                "raw73_day_mean": float(gf[raw73].mean()),
                "raw73_day_std": float(gf[raw73].std(ddof=0)),
                "raw73_on_mean": float(gf.loc[pred_on, raw73].mean()) if pred_on.any() else 0.0,
                "raw73_on_std": float(gf.loc[pred_on, raw73].std(ddof=0)) if pred_on.any() else 0.0,
            })
        rows.append(row)
    tab = pd.DataFrame(rows)
    for c in tab.columns:
        if c not in ("stage", "date"):
            tab[c] = tab[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return tab


def _p2_safety_feature_cols(tab: pd.DataFrame) -> List[str]:
    banned = {"stage", "date", "true_kwh", "abs_base_err", "abs_rawbus_err", "rawbus_improves"}
    return [c for c in tab.columns if c not in banned]


def _fit_p2_rawbus_safety_gate(base_df: pd.DataFrame, rawbus_df: pd.DataFrame,
                               raw_feature_df: pd.DataFrame) -> Tuple[P2RawBusSafetyParams, object, pd.DataFrame]:
    tab = _p2_safety_feature_table(base_df, rawbus_df, raw_feature_df)
    feat_cols = _p2_safety_feature_cols(tab)
    train = tab[tab["stage"].isin(["train", "val"]) & (tab["true_kwh"] > 0.01)].copy()
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=1,
    )
    clf.fit(train[feat_cols].values.astype(float), train["rawbus_improves"].astype(int).values)
    counts = train["rawbus_improves"].value_counts().sort_index().to_dict()
    params = P2RawBusSafetyParams(
        safety_feature_cols=feat_cols,
        threshold=0.50,
        classifier="RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=2,class_weight=balanced)",
        train_val_label_counts={str(k): int(v) for k, v in counts.items()},
        note=("Safety gate trained on train+val daily labels: use rawbus iff predicted P(rawbus improves) >= 0.50. "
              "Features use prediction/weather/raw bus only; test/inference labels are not used."),
    )
    return params, clf, tab


def _apply_p2_rawbus_safety_gate(base_df: pd.DataFrame, rawbus_df: pd.DataFrame,
                                 safety_params: P2RawBusSafetyParams, safety_clf,
                                 safety_tab: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    out["state_pred_variant"] = out["state_pred_main"].astype(int)
    out["y_pred_variant"] = out["y_pred_W_main"].astype(float)
    feat_cols = safety_params.safety_feature_cols
    probs = safety_clf.predict_proba(safety_tab[feat_cols].values.astype(float))[:, 1]
    tab = safety_tab[["stage", "date"]].copy()
    tab["safety_prob_rawbus_improves"] = probs
    tab["safety_use_rawbus"] = (probs >= float(safety_params.threshold)).astype(int)
    out = out.merge(tab, on=["stage", "date"], how="left", sort=False)
    raw_lookup = rawbus_df[["stage", "time", "y_pred_variant"]].rename(
        columns={"y_pred_variant": "rawbus_y_pred_variant"})
    # rawbus_df 内部可能按 stage/time 重排，必须显式按 key 对齐，不能靠 values 顺序。
    out = out.merge(raw_lookup, on=["stage", "time"], how="left", sort=False)
    out["safety_prob_rawbus_improves"] = out["safety_prob_rawbus_improves"].fillna(0.0)
    out["safety_use_rawbus"] = out["safety_use_rawbus"].fillna(0).astype(int)
    out["rawbus_y_pred_variant"] = out["rawbus_y_pred_variant"].fillna(out["y_pred_variant"])
    out["y_pred_variant"] = np.where(out["safety_use_rawbus"].values == 1,
                                     out["rawbus_y_pred_variant"].values,
                                     out["y_pred_variant"].values)
    out = out.drop(columns=["rawbus_y_pred_variant"])
    return out


def _fit_p2_extra_risk_gate_params() -> P2ExtraRiskGateParams:
    return P2ExtraRiskGateParams(
        coverage_min=0.90,
        apply_coverage_only_if_rawbus_up=True,
        risk_p_q50_lo=0.45,
        risk_p_q50_hi=0.60,
        risk_rh_max=85.0,
        risk_pred_on_n_min=40,
        risk_base_on_mean_min=0.0,
        note=("Diagnostic train/val-constrained risk gate: partial-day coverage fallback plus "
              "low/mid p_on long-ON rawbus-up fallback. Thresholds use no test/inference labels; "
              "validated as a candidate, not final production rule."),
    )


def _apply_p2_extra_risk_gate(base_df: pd.DataFrame, rawbus_df: pd.DataFrame,
                              safety_df: pd.DataFrame,
                              safety_tab: pd.DataFrame,
                              params: P2ExtraRiskGateParams) -> pd.DataFrame:
    """在 P2 rawbus safety gate 之上增加 coverage 与低功率长时风险回退。"""
    out = safety_df.copy()
    tab = safety_tab[["stage", "date", "base_kwh", "rawbus_kwh", "coverage",
                      "p_on_q50", "rh_mean", "pred_on_n", "base_on_mean"]].copy()
    tab = tab.rename(columns={
        "base_kwh": "risk_base_kwh", "rawbus_kwh": "risk_rawbus_kwh",
        "coverage": "risk_coverage", "p_on_q50": "risk_p_on_q50",
        "rh_mean": "risk_rh_mean", "pred_on_n": "risk_pred_on_n",
        "base_on_mean": "risk_base_on_mean",
    })
    # safety_df 已有 safety_use_rawbus；按 date/stage 附加 daily risk 特征。
    out = out.merge(tab, on=["stage", "date"], how="left", sort=False)
    use = out["safety_use_rawbus"].fillna(0).astype(int).values == 1
    raw_up = out["risk_rawbus_kwh"].fillna(0).values > out["risk_base_kwh"].fillna(0).values
    cov_risk = (out["risk_coverage"].fillna(1.0).values < params.coverage_min)
    if params.apply_coverage_only_if_rawbus_up:
        cov_risk = cov_risk & raw_up
    low_power_risk = (
        raw_up &
        (out["risk_p_on_q50"].fillna(0).values >= params.risk_p_q50_lo) &
        (out["risk_p_on_q50"].fillna(0).values <= params.risk_p_q50_hi) &
        (out["risk_rh_mean"].fillna(0).values <= params.risk_rh_max) &
        (out["risk_pred_on_n"].fillna(0).values >= params.risk_pred_on_n_min) &
        (out["risk_base_on_mean"].fillna(0).values >= params.risk_base_on_mean_min)
    )
    fallback = cov_risk | low_power_risk
    use2 = use & (~fallback)
    # 需要 baseline 与 rawbus 点级功率重新对齐。
    base_lookup = base_df[["stage", "time", "y_pred_W_main"]].rename(columns={"y_pred_W_main": "base_y"})
    raw_lookup = rawbus_df[["stage", "time", "y_pred_variant"]].rename(columns={"y_pred_variant": "raw_y"})
    out = out.merge(base_lookup, on=["stage", "time"], how="left", sort=False)
    out = out.merge(raw_lookup, on=["stage", "time"], how="left", sort=False)
    out["extra_gate_coverage_fallback"] = cov_risk.astype(int)
    out["extra_gate_lowpower_fallback"] = low_power_risk.astype(int)
    out["safety_use_rawbus"] = use2.astype(int)
    out["y_pred_variant"] = np.where(use2, out["raw_y"].fillna(0).values,
                                     out["base_y"].fillna(0).values)
    drop_cols = [c for c in ["risk_base_kwh", "risk_rawbus_kwh", "risk_coverage", "risk_p_on_q50",
                             "risk_rh_mean", "risk_pred_on_n", "risk_base_on_mean",
                             "base_y", "raw_y"]
                 if c in out.columns]
    out = out.drop(columns=drop_cols)
    return out


def _combine_p1_with_p2_power(p1_df: pd.DataFrame, p2_power_df: pd.DataFrame) -> pd.DataFrame:
    """P1 负责 state 增量, P2 负责 baseline 已判 ON 点功率。

    对 P1 新增 ON 点沿用 P1 guard_power；对 baseline 已判 ON 点使用 P2 power。
    """
    out = p1_df.copy()
    p2_lookup_cols = ["stage", "time", "state_pred_main", "y_pred_variant"]
    extra_cols = []
    for c in ["safety_use_rawbus", "safety_prob_rawbus_improves"]:
        if c in p2_power_df.columns:
            extra_cols.append(c)
    p2_lookup = p2_power_df[p2_lookup_cols + extra_cols].rename(
        columns={"y_pred_variant": "p2_y_pred_variant",
                 "state_pred_main": "p2_base_state"})
    out = out.merge(p2_lookup, on=["stage", "time"], how="left", sort=False)
    base_state = out["p2_base_state"].fillna(out["state_pred_main"]).astype(int).values
    p1_state = out["state_pred_variant"].astype(int).values
    p2_y = out["p2_y_pred_variant"].fillna(out["y_pred_W_main"]).astype(float).values
    p1_y = out["y_pred_variant"].astype(float).values
    y = np.where(base_state == 1, p2_y, 0.0)
    newly_on = (p1_state == 1) & (base_state == 0)
    y[newly_on] = p1_y[newly_on]
    out["state_pred_variant"] = p1_state
    out["y_pred_variant"] = y
    out = out.drop(columns=["p2_base_state", "p2_y_pred_variant"])
    return out


def _fit_p3_params(df: pd.DataFrame) -> P3Params:
    tv = df[df["stage"].isin(["train", "val"])].copy()
    sel = ((tv["state_true"] == 1) & (tv["state_pred_main"] == 1) &
           (tv["y_pred_W_main"] > 10))
    x = tv[sel].copy()
    x["ratio"] = x["y_true_W"] / x["y_pred_W_main"]
    temp_bins = [-float("inf"), 22.0, 26.0, 29.0, float("inf")]
    rh_bins = [0.0, 75.0, 85.0, 90.0, 101.0]
    scales = {}
    for ti in range(len(temp_bins) - 1):
        for ri in range(len(rh_bins) - 1):
            m = ((x["temperature_2m"] > temp_bins[ti]) &
                 (x["temperature_2m"] <= temp_bins[ti + 1]) &
                 (x["relative_humidity_2m"] > rh_bins[ri]) &
                 (x["relative_humidity_2m"] <= rh_bins[ri + 1]))
            key = f"t{ti}_r{ri}"
            if int(m.sum()) >= 20:
                s = float(x.loc[m, "ratio"].median())
                s = float(np.clip(s, 0.80, 1.20))
            else:
                s = 1.0
            scales[key] = s
    return P3Params(temp_bins=temp_bins, rh_bins=rh_bins, min_n=20,
                    clip_lo=0.80, clip_hi=1.20, bucket_scales=scales)


def _apply_p3(df: pd.DataFrame, params: P3Params) -> pd.DataFrame:
    out = df.copy()
    out["state_pred_variant"] = out["state_pred_main"].astype(int)
    out["y_pred_variant"] = out["y_pred_W_main"].astype(float)
    scales = np.ones(len(out), dtype=float)
    for ti in range(len(params.temp_bins) - 1):
        for ri in range(len(params.rh_bins) - 1):
            m = ((out["temperature_2m"] > params.temp_bins[ti]) &
                 (out["temperature_2m"] <= params.temp_bins[ti + 1]) &
                 (out["relative_humidity_2m"] > params.rh_bins[ri]) &
                 (out["relative_humidity_2m"] <= params.rh_bins[ri + 1]))
            scales[m.values] = params.bucket_scales.get(f"t{ti}_r{ri}", 1.0)
    out["y_pred_variant"] = out["y_pred_variant"].values * scales * out["state_pred_variant"].values
    return out


def _build_report(summary: pd.DataFrame, daily: pd.DataFrame, params: Dict[str, object]) -> str:
    def fmt_summary(stage: str) -> str:
        cols = ["variant", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_true", "kWh_pred", "kWh_err", "TP", "FP", "FN", "TN"]
        s = summary[summary["stage"] == stage][cols].copy()
        return s.to_markdown(index=False, floatfmt=".4f")

    def problem_counts() -> str:
        rows = []
        for (variant, stage), g in daily.groupby(["variant", "stage"]):
            on_day = g["kWh_true"] > 0.01
            rows.append({
                "variant": variant,
                "stage": stage,
                "on_F1_lt_90_days": int(((g["F1"] < 0.9) & on_day).sum()),
                "SAE_gt_20_days": int(((g["SAE"] > 0.2) & on_day).sum()),
            })
        return pd.DataFrame(rows).sort_values(["stage", "variant"]).to_markdown(index=False)

    report = []
    report.append("# U842 P1/P2/P3 优化方案离线验证报告\n")
    report.append("## 参数选择纪律\n")
    report.append("- P1/P2/P3 参数均只从 train+val 预测、标签与天气统计中推导。\n")
    report.append("- test 与 inference 仅做验证；未使用 7 月 OOD 指标调参。\n")
    report.append("\n## 选定参数\n")
    report.append("```json\n" + json.dumps(params, ensure_ascii=False, indent=2, default=str) + "\n```\n")
    report.append("\n## train+val 选择集整体指标\n")
    report.append(fmt_summary("train_val") + "\n")
    report.append("\n## test 验证集整体指标\n")
    report.append(fmt_summary("test") + "\n")
    report.append("\n## inference OOD 整体指标\n")
    report.append(fmt_summary("inference") + "\n")
    report.append("\n## 异常日数量（ON 日口径）\n")
    report.append(problem_counts() + "\n")
    return "\n".join(report)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weather = _load_weather()
    pred = _load_predictions()
    df = _attach_weather(pred, weather)
    wd = _daily_weather(weather)
    df = df.merge(wd, on="date", how="left", suffixes=("", "_daily"))
    # 使用日均 RH/temp 做 P1，逐点 temp/RH 做 P3。
    df["rh_mean"] = df["rh_mean"].astype(float)
    df["temp_mean"] = df["temp_mean"].astype(float)

    variants = {}
    variants["baseline"] = _with_baseline_cols(df)

    p1 = _select_p1_params(df, strict_daily_gate=True)
    variants["P1_recall_guard"] = _apply_p1(df, p1)
    # 仅供诊断: 放宽日级异常闸后的 recall guard，展示 OOD 潜在收益与 train/val 风险。
    p1_relaxed = _select_p1_params(df, strict_daily_gate=False)
    variants["P1_recall_guard_relaxed_ref"] = _apply_p1(df, p1_relaxed)

    # 低概率梅雨整日漏检 guard: 仍与 P2 分离, 只改 state/新增 ON 点。
    raw_feature_for_p1 = _ensure_p2_rawbus_segment_features(df)
    p1_lowprob = _fit_p1_lowprob_rain_params(df)
    variants["P1_lowprob_rain_guard"] = _apply_p1_lowprob_rain_guard(
        df, variants["P1_recall_guard"], raw_feature_for_p1, p1_lowprob)

    p2_mode, mode_clf, mode_regs = _fit_p2_mode_model(df)
    variants["P2_mode_classifier_regressor"] = _apply_p2_mode_model(df, mode_clf, mode_regs)

    # 新版 P2: 增强模式特征 + train+val 日级 loss-aware 目标选择候选。
    p2_loss, loss_clf, loss_regs = _fit_p2_lossaware_mode_model(df)
    variants["P2_lossaware_mode_model"] = _apply_p2_lossaware_mode_model(df, p2_loss, loss_clf, loss_regs)

    # 最新 P2: 引入真正原始总线/段级特征, 不再只依赖 baseline 预测派生特征。
    p2_rawbus, rawbus_clf, rawbus_regs = _fit_p2_rawbus_segment_model(df)
    variants["P2_rawbus_segment_model"] = _apply_p2_rawbus_segment_model(df, p2_rawbus, rawbus_clf, rawbus_regs)

    # 灰度候选: rawbus P2 + train/val-only 日级安全闸。
    raw_feature_df = _ensure_p2_rawbus_segment_features(df)
    p2_safety, safety_clf, safety_tab = _fit_p2_rawbus_safety_gate(
        variants["baseline"], variants["P2_rawbus_segment_model"], raw_feature_df)
    variants["P2_rawbus_safety_gate"] = _apply_p2_rawbus_safety_gate(
        variants["baseline"], variants["P2_rawbus_segment_model"],
        p2_safety, safety_clf, safety_tab)

    # 追加风险闸: partial-day coverage fallback + 低功率长时 rawbus-up fallback。
    p2_extra_risk = _fit_p2_extra_risk_gate_params()
    variants["P2_rawbus_safety_cov_lowrisk"] = _apply_p2_extra_risk_gate(
        variants["baseline"], variants["P2_rawbus_segment_model"],
        variants["P2_rawbus_safety_gate"], safety_tab, p2_extra_risk)

    # 组合灰度: P1 只负责补 state/FN, P2 rawbus+safety 只负责 baseline ON 功率。
    variants["P1_plus_P2_rawbus_safety"] = _combine_p1_with_p2_power(
        variants["P1_recall_guard"], variants["P2_rawbus_safety_gate"])
    variants["P1_lowprob_plus_P2_risk"] = _combine_p1_with_p2_power(
        variants["P1_lowprob_rain_guard"], variants["P2_rawbus_safety_cov_lowrisk"])

    # 旧版 simple daily scale 仅保留为对照，非本次建议方案。
    p2_daily = _fit_p2_params(df)
    variants["P2_daily_scale_ref"] = _apply_p2(df, p2_daily)

    p3 = _fit_p3_params(df)
    variants["P3_temp_humidity_bucket"] = _apply_p3(df, p3)

    # Optional reference: classification recall guard + best regression-only calibrator.
    p1p3 = _apply_p3(_apply_p1(df, p1), p3)
    # _apply_p3 resets state from main; restore P1 state before P3 scaling
    p1_state = variants["P1_recall_guard"]["state_pred_variant"].values
    p1_y = variants["P1_recall_guard"]["y_pred_variant"].values
    p1p3["state_pred_variant"] = p1_state
    # scale P1 y_pred with P3 ratios by reusing scale from p3/base where existing, leave flipped guard power mostly unchanged
    base_y = variants["baseline"]["y_pred_variant"].values
    p3_y = variants["P3_temp_humidity_bucket"]["y_pred_variant"].values
    scale = np.divide(p3_y, base_y, out=np.ones_like(p3_y, dtype=float), where=base_y > 1e-9)
    p1p3["y_pred_variant"] = p1_y * scale
    variants["P1_plus_P3_ref"] = p1p3

    summary_rows = []
    daily_frames = []
    for name, vdf in variants.items():
        summary_rows.extend(_summarize(vdf, name))
        daily_frames.append(_daily_metrics(vdf, name))
    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_frames, ignore_index=True)

    params = {
        "P1_recall_guard": asdict(p1),
        "P1_recall_guard_relaxed_ref": asdict(p1_relaxed),
        "P1_lowprob_rain_guard": asdict(p1_lowprob),
        "P2_mode_classifier_regressor": asdict(p2_mode),
        "P2_lossaware_mode_model": asdict(p2_loss),
        "P2_rawbus_segment_model": asdict(p2_rawbus),
        "P2_rawbus_safety_gate": asdict(p2_safety),
        "P2_extra_risk_gate": asdict(p2_extra_risk),
        "P2_daily_scale_ref": asdict(p2_daily),
        "P3_temp_humidity_bucket": asdict(p3),
        "notes": {
            "baseline_best_thr": BEST_THR,
            "selection_set": "train+val only",
            "eval_sets": ["test", "inference"],
        },
    }
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "selected_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = _build_report(summary, daily, params)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"[OK] summary -> {OUT_DIR / 'summary_metrics.csv'}")
    print(f"[OK] daily   -> {OUT_DIR / 'daily_metrics.csv'}")
    print(f"[OK] params  -> {OUT_DIR / 'selected_params.json'}")
    print(f"[OK] report  -> {OUT_DIR / 'report.md'}")
    print("\n[inference summary]")
    cols = ["variant", "F1", "Precision", "Recall", "SAE", "MAE_W", "kWh_pred", "TP", "FP", "FN", "TN"]
    print(summary[summary["stage"] == "inference"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
