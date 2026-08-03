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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
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
    if "hour_sin" not in out.columns:
        hour = out["time"].dt.hour + out["time"].dt.minute / 60.0
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

    p2_mode, mode_clf, mode_regs = _fit_p2_mode_model(df)
    variants["P2_mode_classifier_regressor"] = _apply_p2_mode_model(df, mode_clf, mode_regs)

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
        "P2_mode_classifier_regressor": asdict(p2_mode),
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
