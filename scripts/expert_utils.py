# -*- coding: utf-8 -*-
"""
季节分层建模工具 (v4.2 月份路由 + v5 温度驱动路由)

机制:
    1) v4.2 (按月份, 仍保留作 fallback):
        SEASON_MAP[month] -> 'summer' / 'transition' / 'winter'
    2) v5 (按日均温度, 推荐):
        weather_utils.assign_season_by_temp(daily_avg_t)
    3) SeasonalRegressorBundle: 训练/推理 3 个季节专家
    4) 样本不足兜底: 任一 expert ON 样本 < MIN_ON_FOR_EXPERT 时, 回退到全局模型
"""
import numpy as np
import pandas as pd


# ============================================================
# v4.2 季节映射 (按月份, 保留作为 fallback)
# ============================================================
SEASON_MAP = {
    1: "winter", 2: "winter", 12: "winter",
    5: "summer", 6: "summer", 7: "summer", 8: "summer", 9: "summer",
    3: "transition", 4: "transition",
    10: "transition", 11: "transition",
}
SEASON_LABELS = ["summer", "transition", "winter"]

# 每个 expert 训练所需的最小 ON 样本数
MIN_ON_FOR_EXPERT = 50


def assign_season(timestamps,
                  daily_avg_temp: np.ndarray = None,
                  use_temperature: bool = False,
                  summer_th: float = 22.0,
                  winter_th: float = 12.0) -> np.ndarray:
    """
    统一季节归属入口 (v5 升级)

    参数:
        timestamps:       时间戳序列
        daily_avg_temp:   每个时间戳对应的"当日平均温度", 若用温度路由必填
        use_temperature:  False=按月份 (v4.2 兼容), True=按温度 (v5 推荐)
        summer_th:        日均 >= 此值视为 summer  (默认 22°C)
        winter_th:        日均 <= 此值视为 winter  (默认 12°C)
    返回:
        np.ndarray of {"summer", "transition", "winter"}
    """
    ts = pd.DatetimeIndex(timestamps)
    if use_temperature:
        if daily_avg_temp is None:
            raise ValueError("use_temperature=True 时必须提供 daily_avg_temp")
        t = np.asarray(daily_avg_temp, dtype=float)
        if len(t) != len(ts):
            raise ValueError(f"daily_avg_temp 长度 {len(t)} 与 timestamps {len(ts)} 不匹配")
        out = np.full(len(t), "transition", dtype=object)
        out[t >= summer_th] = "summer"
        out[t <= winter_th] = "winter"
        return out
    else:
        # v4.2 兼容模式: 按月份硬路由
        return np.array([SEASON_MAP[m] for m in ts.month])


# ============================================================
# 季节分层回归 Bundle (v4 起结构稳定, v5 仅改变 season 输入来源)
# ============================================================
class SeasonalRegressorBundle:
    """
    季节分层条件功率回归 (含 P10/P50/P90 分位)
    - 每个 expert 是独立的 GradientBoostingRegressor (quantile loss)
    - 共享同一套特征
    - 训练: 对每个 season, 在 (该季节 + ON) 子集上训练
    - 推理: 按 season 标签路由, 若该 season 在训练时缺失, 回退到 fallback 全局模型
    """

    def __init__(self,
                 gbr_factory,
                 quantiles=(0.10, 0.50, 0.90),
                 min_on_for_expert: int = MIN_ON_FOR_EXPERT,
                 logger=None):
        self.gbr_factory = gbr_factory
        self.quantiles = tuple(quantiles)
        self.min_on_for_expert = min_on_for_expert
        self.experts = {}        # {season: {alpha: GBR}}
        self.fallback = {}       # {alpha: GBR}  全局兜底
        self.train_stats = {}    # {season: {n_on, y_mean, y_std}}
        self.log = logger

    # ---------- 训练 ----------
    def fit(self, X_train, y_train, state_train, season_labels_train,
            sample_weight=None):
        """
        season_labels_train: 与 X_train 等长的季节标签数组 (np.ndarray of str)
                             由外部根据 use_temperature 设定
        """
        seasons = np.asarray(season_labels_train)
        mask_on_all = state_train == 1

        # ---- 1. 训练 fallback 全局模型 ----
        if self.log:
            self.log.info("  [Expert/Fallback] 训练全局兜底回归器")
        for q in self.quantiles:
            gbr = self.gbr_factory(q)
            sw = sample_weight[mask_on_all] if sample_weight is not None else None
            gbr.fit(X_train[mask_on_all], y_train[mask_on_all], sample_weight=sw)
            self.fallback[q] = gbr

        # ---- 2. 逐 season 训练 expert ----
        for sea in SEASON_LABELS:
            mask_sea_on = (seasons == sea) & mask_on_all
            n_on = int(mask_sea_on.sum())
            if n_on >= self.min_on_for_expert:
                y_sub = y_train[mask_sea_on]
                self.train_stats[sea] = {
                    "n_on": n_on,
                    "y_mean": float(y_sub.mean()),
                    "y_std":  float(y_sub.std()),
                    "y_median": float(np.median(y_sub)),
                }
                self.experts[sea] = {}
                for q in self.quantiles:
                    gbr = self.gbr_factory(q)
                    sw = sample_weight[mask_sea_on] if sample_weight is not None else None
                    gbr.fit(X_train[mask_sea_on], y_train[mask_sea_on],
                            sample_weight=sw)
                    self.experts[sea][q] = gbr
                if self.log:
                    self.log.info(f"  [Expert/{sea:<10}] ON 样本 {n_on}, "
                                  f"y均值={y_sub.mean():.1f}W, "
                                  f"中位={np.median(y_sub):.1f}W, "
                                  f"std={y_sub.std():.1f}W  [训练成功]")
            else:
                if self.log:
                    self.log.warning(
                        f"  [Expert/{sea:<10}] ON 样本仅 {n_on} (< "
                        f"{self.min_on_for_expert}), 跳过, 推理时将回退到 fallback")
        return self

    # ---------- 推理 ----------
    def predict(self, X, season_labels, alpha=0.5) -> np.ndarray:
        """
        season_labels: 与 X 等长的季节标签数组
        """
        n = len(X)
        out = np.zeros(n, dtype=float)
        seasons = np.asarray(season_labels)
        for sea in SEASON_LABELS:
            mask = seasons == sea
            if not mask.any():
                continue
            if sea in self.experts and alpha in self.experts[sea]:
                model = self.experts[sea][alpha]
            else:
                model = self.fallback[alpha]
            out[mask] = model.predict(X[mask])
        return out

    def predict_all_quantiles(self, X, season_labels) -> dict:
        """同时返回 P10/P50/P90"""
        return {q: self.predict(X, season_labels, alpha=q) for q in self.quantiles}

    # ---------- 持久化 ----------
    def strip_for_save(self):
        """清除不可 pickle 的成员 (闭包工厂 + logger). 保存前必调用."""
        self.gbr_factory = None
        self.log = None
        return self

    # ---------- 元数据 ----------
    def expert_summary(self) -> list:
        rows = []
        for sea in SEASON_LABELS:
            row = {"season": sea}
            if sea in self.train_stats:
                row.update(self.train_stats[sea])
                row["status"] = "trained"
            else:
                row["status"] = "fallback"
                row["n_on"] = 0
            rows.append(row)
        return rows


def diagnose_seasonal_distribution(season_labels, y, state, logger=None):
    """训练前诊断: 各季节的 ON 样本数 + 功率分布"""
    seasons = np.asarray(season_labels)
    stats = []
    for sea in SEASON_LABELS:
        m = (seasons == sea) & (state == 1)
        n = int(m.sum())
        if n > 0:
            y_sub = y[m]
            stats.append({
                "season": sea, "n_on": n,
                "mean_W":  float(y_sub.mean()),
                "median_W": float(np.median(y_sub)),
                "std_W":   float(y_sub.std()),
                "min_W":   float(y_sub.min()),
                "max_W":   float(y_sub.max()),
            })
        else:
            stats.append({"season": sea, "n_on": 0,
                          "mean_W": None, "median_W": None, "std_W": None,
                          "min_W": None, "max_W": None})
    if logger:
        logger.info("  季节分布诊断:")
        for s in stats:
            if s["n_on"] > 0:
                logger.info(f"    {s['season']:<11} ON={s['n_on']:<5} "
                            f"均值={s['mean_W']:>6.1f}W "
                            f"中位={s['median_W']:>6.1f}W "
                            f"std={s['std_W']:>6.1f}W "
                            f"范围=[{s['min_W']:.0f}, {s['max_W']:.0f}]")
            else:
                logger.info(f"    {s['season']:<11} ON=0    (无样本, 推理时回退)")
    return stats
