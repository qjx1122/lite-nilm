# -*- coding: utf-8 -*-
"""
样本重加权工具 (v3 优化)

机制:
    1) 对训练集 ON 样本的功率分布做直方图密度估计
    2) 样本权重 = 1 / max(density, eps)
    3) 归一化使总权重 = n_samples (保持梯度规模一致)
效果:
    - 稀有功率值 (如 400W 弱负荷) 权重提升
    - 主峰功率值 (如 700W 制冷) 权重适度降低
    - 缓解模型对主峰区域的过拟合 / 对稀疏区域的欠拟合
"""
import numpy as np


def compute_inverse_density_weights(y: np.ndarray, n_bins: int = 20,
                                    clip_quantile: float = 0.95,
                                    eps: float = 1e-3,
                                    floor_ratio: float = 0.1) -> np.ndarray:
    """
    计算逆密度权重
    参数:
        y: 训练集 ON 功率序列 (W), 必须 > 0
        n_bins: 直方图分箱数 (15~30 较合理)
        clip_quantile: 权重上限分位点 (防止极端值权重失控)
        eps: 密度下限, 防止除零
        floor_ratio: 最小权重 = 平均权重 × floor_ratio (避免某些样本被完全忽略)
    返回:
        weights: 与 y 等长, 归一化后均值 = 1.0
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return np.array([])

    # 1) 直方图 (固定边界, 覆盖全量值域)
    y_min, y_max = float(y.min()), float(y.max())
    if y_max <= y_min:
        return np.ones_like(y)
    edges = np.linspace(y_min, y_max + 1e-6, n_bins + 1)
    hist, _ = np.histogram(y, bins=edges)
    bin_idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    density = hist[bin_idx].astype(float)

    # 2) 逆密度
    w = 1.0 / np.maximum(density, eps)

    # 3) 上限裁剪 (避免某一稀有 bin 内权重爆炸)
    w_cap = np.quantile(w, clip_quantile)
    w = np.minimum(w, w_cap)

    # 4) 下限保护
    mean_w = w.mean()
    w = np.maximum(w, mean_w * floor_ratio)

    # 5) 归一化, 使权重均值 = 1.0 (等同于无权时的总损失规模)
    w = w / w.mean()
    return w


def summarize_weights(y: np.ndarray, w: np.ndarray, n_bins: int = 5) -> dict:
    """统计权重在不同功率区间的分布, 用于日志输出"""
    y = np.asarray(y); w = np.asarray(w)
    if len(y) == 0:
        return {}
    qs = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    out = {
        "n_samples": int(len(y)),
        "weight_min": float(w.min()),
        "weight_max": float(w.max()),
        "weight_mean": float(w.mean()),
        "weight_std": float(w.std()),
        "bins_by_power": [],
    }
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        m = (y >= lo) & (y <= hi if i == n_bins - 1 else y < hi)
        if m.any():
            out["bins_by_power"].append({
                "power_range_W": f"[{lo:.0f}, {hi:.0f}]",
                "n": int(m.sum()),
                "avg_weight": float(w[m].mean()),
            })
    return out
