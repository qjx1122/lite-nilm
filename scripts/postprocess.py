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


def search_best_threshold(p_scores, y_true_state, beta: float = 0.5,
                          thr_grid=None, min_on: int = 2,
                          fill_short_off: int = 1) -> dict:
    """
    在 val 集上搜索最佳阈值 (F_beta 最大), 应用同样的后处理。
    beta < 1 -> 更看重 Precision (减少误报)
    beta > 1 -> 更看重 Recall
    返回:
        {best_thr, best_fbeta, curve(list[dict])}
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    if thr_grid is None:
        thr_grid = np.round(np.arange(0.02, 0.96, 0.01), 3)

    best_thr, best_fbeta = 0.5, -1.0
    curve = []
    for thr in thr_grid:
        st = (p_scores >= thr).astype(int)
        st = min_duration_filter(st, min_on=min_on,
                                 fill_short_off=fill_short_off)
        p = precision_score(y_true_state, st, zero_division=0)
        r = recall_score(y_true_state, st, zero_division=0)
        f1 = f1_score(y_true_state, st, zero_division=0)
        if (p + r) == 0 or (beta**2 * p + r) == 0:
            fbeta = 0.0
        else:
            fbeta = (1 + beta**2) * p * r / (beta**2 * p + r)
        curve.append({
            "threshold": float(thr),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            f"f{beta}": float(fbeta),
        })
        if fbeta > best_fbeta:
            best_fbeta, best_thr = fbeta, float(thr)
    return {
        "best_thr": best_thr,
        "best_fbeta": best_fbeta,
        "beta": beta,
        "curve": curve,
        "post_min_on": min_on,
        "post_fill_short_off": fill_short_off,
    }


