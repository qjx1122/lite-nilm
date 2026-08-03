# -*- coding: utf-8 -*-
"""
推理后处理 - v2 优化
- 最小持续时长过滤: 单点 ON 视为误报, 至少连续 N 个时段才确认开机
- 单点 OFF 平滑: 中间夹一个 OFF 视为压缩机短暂停歇, 填充为 ON
"""
import numpy as np


def min_duration_filter(state: np.ndarray, min_on: int = 2,
                        fill_short_off: int = 1) -> np.ndarray:
    """
    形态学滤波:
        1) 先把 "孤立 ON 短脉冲" (长度 < min_on) 视为误报, 置 0
        2) 再把 "孤立 OFF 短缺口" (长度 <= fill_short_off) 视为压缩机短歇, 填 1
    参数:
        state: 0/1 序列
        min_on: 连续 ON 段最少持续点数 (15min/点, 默认 2 = 30 分钟)
        fill_short_off: 连续 OFF 段长度 <= 此值时填回 ON, 默认 1
    """
    s = np.asarray(state, dtype=int).copy()
    if len(s) == 0:
        return s

    # ---- 步骤 1: 去除短 ON ----
    s = _remove_short_runs(s, value=1, min_len=min_on)
    # ---- 步骤 2: 填短 OFF (压缩机喘息) ----
    if fill_short_off > 0:
        # 反转值后再调用同函数
        s = 1 - _remove_short_runs(1 - s, value=1, min_len=fill_short_off + 1)
    return s


def _remove_short_runs(s: np.ndarray, value: int, min_len: int) -> np.ndarray:
    """删除长度 < min_len 的指定值连续段, 替换为 1-value"""
    s = s.copy()
    n = len(s)
    i = 0
    while i < n:
        if s[i] != value:
            i += 1
            continue
        j = i
        while j < n and s[j] == value:
            j += 1
        run_len = j - i
        if run_len < min_len:
            s[i:j] = 1 - value
        i = j
    return s


def apply_postprocess(state_pred: np.ndarray, p_reg: np.ndarray,
                      min_on: int = 2, fill_short_off: int = 1):
    """
    完整后处理: 状态形态学过滤 + 功率门控
    返回: (state_filt, y_pred_filt)
    """
    state_filt = min_duration_filter(state_pred,
                                     min_on=min_on,
                                     fill_short_off=fill_short_off)
    y_pred_filt = state_filt * np.clip(p_reg, 0, None)
    return state_filt, y_pred_filt


def _default_threshold_stability_tol(n_samples: int) -> float:
    """
    v14.7 阈值稳定化容忍带。

    背景: F_beta 在有限验证集上是阶梯函数, 单个边界样本的概率微漂移就可能
    让最优阈值跨平台跳档。容忍带仅由验证集样本数决定, 不使用推理/7月评估集。

    取值: 约 two-sample resolution, 并限制在 [1e-4, 2e-3]。
    - U842 val n=1536 -> 0.00130, 可覆盖 1 个 TP/FP/FN 量级的非显著差异。
    - 小验证集不让容忍带无限放大, 避免过度改写 precision-oriented 用户。
    """
    n = max(int(n_samples), 1)
    return float(min(2e-3, max(1e-4, 2.0 / n)))


def search_best_threshold(p_scores, y_true_state, beta: float = 0.5,
                          thr_grid=None, min_on: int = 2,
                          fill_short_off: int = 1,
                          stable_tiebreak: bool = True,
                          stability_tol: float | None = None) -> dict:
    """
    在 val 集上搜索阈值, 应用同样的后处理。

    基础目标仍是 F_beta 最大。v14.7 起增加有限验证集稳定化:
      1) 先找 raw_best_fbeta / raw_best_thr;
      2) 在 raw_best_fbeta - tol 以内视为统计上近似同分;
      3) beta >= 1 时在近似同分候选中偏向更高 Recall, 再比 Precision;
         beta < 1 时偏向更高 Precision, 再比 Recall;
      4) 若 P/R 仍相同, beta >= 1 取较低阈值, beta < 1 取较高阈值。

    该规则只依赖验证集, 用于降低 Windows/Linux、LightGBM/OpenMP 等概率微漂移
    对 best_thr 的放大效应；不会读取推理集或 7月 OOD 标签。

    返回:
        {
          best_thr,              # 稳定化后采用的阈值
          best_fbeta,            # 向后兼容: raw 最优 F_beta
          selected_fbeta,        # best_thr 对应 F_beta
          raw_best_thr, raw_best_fbeta,
          threshold_stability_tol,
          curve(list[dict])
        }
    """
    from sklearn.metrics import (confusion_matrix, precision_score,
                                 recall_score, f1_score)
    if thr_grid is None:
        thr_grid = np.round(np.arange(0.02, 0.96, 0.01), 3)

    y_arr = np.asarray(y_true_state, dtype=int)
    curve = []
    raw_best_row = None
    fbeta_key = f"f{beta}"

    for thr in thr_grid:
        st = (p_scores >= thr).astype(int)
        st = min_duration_filter(st, min_on=min_on,
                                 fill_short_off=fill_short_off)
        p = precision_score(y_arr, st, zero_division=0)
        r = recall_score(y_arr, st, zero_division=0)
        f1 = f1_score(y_arr, st, zero_division=0)
        if (p + r) == 0 or (beta**2 * p + r) == 0:
            fbeta = 0.0
        else:
            fbeta = (1 + beta**2) * p * r / (beta**2 * p + r)
        tn, fp, fn, tp = confusion_matrix(y_arr, st, labels=[0, 1]).ravel()
        row = {
            "threshold": float(thr),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            fbeta_key: float(fbeta),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        curve.append(row)
        # 保留旧行为: raw best 若完全同分, 取扫描顺序中的第一个(低阈值)。
        if raw_best_row is None or fbeta > raw_best_row[fbeta_key]:
            raw_best_row = row

    if raw_best_row is None:
        # 空 grid 的防御性回退, 保持旧默认阈值语义。
        return {
            "best_thr": 0.5,
            "best_fbeta": 0.0,
            "selected_fbeta": 0.0,
            "raw_best_thr": 0.5,
            "raw_best_fbeta": 0.0,
            "threshold_stability_tol": 0.0,
            "threshold_selection_policy": "empty_grid_fallback",
            "beta": beta,
            "curve": curve,
            "post_min_on": min_on,
            "post_fill_short_off": fill_short_off,
        }

    tol = (_default_threshold_stability_tol(len(y_arr))
           if stability_tol is None else float(stability_tol))
    tol = max(0.0, tol)

    selected = raw_best_row
    policy = "raw_fbeta_argmax"
    candidate_count = 1
    if stable_tiebreak:
        floor = raw_best_row[fbeta_key] - tol
        candidates = [row for row in curve if row[fbeta_key] >= floor]
        candidate_count = len(candidates)
        if beta >= 1.0:
            # Recall 优先: 降低近似同分阈值跳高导致的 OOD FN/kWh 低估风险。
            selected = sorted(
                candidates,
                key=lambda row: (
                    -row["recall"],
                    -row["precision"],
                    -row[fbeta_key],
                    row["threshold"],
                ),
            )[0]
            policy = "stable_recall_first_within_val_tolerance"
        else:
            # beta<1 是 precision-oriented 口径, 不反向改写业务偏好。
            selected = sorted(
                candidates,
                key=lambda row: (
                    -row["precision"],
                    -row["recall"],
                    -row[fbeta_key],
                    -row["threshold"],
                ),
            )[0]
            policy = "stable_precision_first_within_val_tolerance"

    return {
        "best_thr": float(selected["threshold"]),
        # 向后兼容: best_fbeta 仍表示原始 argmax F_beta。
        "best_fbeta": float(raw_best_row[fbeta_key]),
        "selected_fbeta": float(selected[fbeta_key]),
        "raw_best_thr": float(raw_best_row["threshold"]),
        "raw_best_fbeta": float(raw_best_row[fbeta_key]),
        "selected_precision": float(selected["precision"]),
        "selected_recall": float(selected["recall"]),
        "raw_best_precision": float(raw_best_row["precision"]),
        "raw_best_recall": float(raw_best_row["recall"]),
        "threshold_stability_tol": float(tol if stable_tiebreak else 0.0),
        "threshold_candidate_count": int(candidate_count),
        "threshold_selection_policy": policy,
        "beta": beta,
        "curve": curve,
        "post_min_on": min_on,
        "post_fill_short_off": fill_short_off,
    }


