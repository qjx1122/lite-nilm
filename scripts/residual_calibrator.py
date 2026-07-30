# -*- coding: utf-8 -*-
"""
L4 残差校正层 (v6 新增, 治本方案)

机制:
    主模型预测 y_pred_raw 后, 再过一个轻量 GBDT 学习残差:
        delta_hat = g(temp, hour, recent_signal, season_id, y_pred_raw)
        y_final   = y_pred_raw + delta_hat
    
    g 在 val 集上学习 (不污染主模型训练), 仅对 ON 段做加性校正,
    并加 |delta| < 2*MAE_main 的硬限幅, 防止过校正反弹.

设计原则:
    - 算法: 50 棵 GBDT (轻量, 不易过拟合)
    - 仅 5 维输入 (温度/小时/近期信号/季节/原预测)
    - 限幅: 最大校正量 = min(2 * train_MAE, 150W)
    - 业务开关: bundle["use_residual_calib"] 控制是否启用
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


CALIB_HARD_CAP_W = 150.0   # 校正量绝对上限 (防止过校正)


class ResidualCalibrator:
    """残差校正器: 输入 5 维特征, 输出加性校正量"""

    SEASON_TO_ID = {"summer": 0, "transition": 1, "winter": 2}

    def __init__(self, n_estimators: int = 50, max_depth: int = 3,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.gbr = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            random_state=random_state, loss="huber",
        )
        self.cap_w = CALIB_HARD_CAP_W
        self.train_mae = None
        self.feat_names = ["temp_2m", "hour", "recent_signal",
                           "season_id", "y_pred_raw"]
        self._trained = False

    def _build_features(self, y_pred_raw: np.ndarray,
                        timestamps,
                        weather_df: pd.DataFrame,
                        recent_signal: np.ndarray,
                        season_labels: np.ndarray) -> np.ndarray:
        """构造 5 维校正特征"""
        n = len(y_pred_raw)
        ts = pd.DatetimeIndex(timestamps)
        # 温度
        if weather_df is not None and "temperature_2m" in weather_df.columns:
            temp = weather_df["temperature_2m"].reindex(ts, method="nearest").values
        else:
            temp = np.full(n, 20.0)
        # 小时
        hour = ts.hour.values
        # 近期信号 (训练时由外部传入, 推理时同样传入)
        recent = np.asarray(recent_signal, dtype=float) if recent_signal is not None \
                 else np.zeros(n)
        # 季节 ID
        sid = np.array([self.SEASON_TO_ID.get(s, 1) for s in season_labels])
        # 原始预测
        y_raw = np.asarray(y_pred_raw, dtype=float)
        X = np.column_stack([temp, hour, recent, sid, y_raw]).astype(np.float32)
        return X

    def fit(self, y_pred_raw: np.ndarray, y_true: np.ndarray,
            state_pred: np.ndarray,
            timestamps, weather_df: pd.DataFrame,
            recent_signal: np.ndarray,
            season_labels: np.ndarray, logger=None):
        """
        在 val 集上学习: 仅用 ON 时段(state_pred==1)且 y_true 已知的样本
        """
        mask = (state_pred == 1) & np.isfinite(y_true) & (y_true > 0)
        n_on = int(mask.sum())
        if n_on < 30:
            if logger:
                logger.warning(f"  [L4] val ON 样本仅 {n_on} (<30), 跳过 calibrator 训练")
            self._trained = False
            return self

        X = self._build_features(y_pred_raw, timestamps,
                                 weather_df, recent_signal, season_labels)
        delta = y_true - y_pred_raw   # 真实残差
        self.gbr.fit(X[mask], delta[mask])
        self.train_mae = float(np.abs(delta[mask]).mean())
        # 自适应限幅: min(2 * 训练 MAE, 全局上限)
        self.cap_w = min(2 * self.train_mae, CALIB_HARD_CAP_W)
        self._trained = True

        if logger:
            pred_delta = self.gbr.predict(X[mask])
            logger.info(f"  [L4] 校正器训练完成: n_on={n_on}, "
                        f"训练残差均值={delta[mask].mean():+.1f}W, "
                        f"中位={np.median(delta[mask]):+.1f}W")
            logger.info(f"  [L4]   校正器输出: mean={pred_delta.mean():+.1f}W, "
                        f"std={pred_delta.std():.1f}W, "
                        f"限幅±{self.cap_w:.0f}W")
        return self

    def predict_delta(self, y_pred_raw: np.ndarray,
                      timestamps, weather_df: pd.DataFrame,
                      recent_signal: np.ndarray,
                      season_labels: np.ndarray,
                      state_pred: np.ndarray = None) -> np.ndarray:
        """预测校正量 (仅 ON 时段返回非零, OFF 时段返回 0)"""
        if not self._trained:
            return np.zeros_like(y_pred_raw, dtype=float)
        X = self._build_features(y_pred_raw, timestamps,
                                 weather_df, recent_signal, season_labels)
        delta = self.gbr.predict(X)
        # 限幅
        delta = np.clip(delta, -self.cap_w, self.cap_w)
        # 仅 ON 时段应用
        if state_pred is not None:
            delta = delta * (np.asarray(state_pred) == 1).astype(float)
        return delta

    def apply(self, y_pred_raw: np.ndarray,
              timestamps, weather_df: pd.DataFrame,
              recent_signal: np.ndarray,
              season_labels: np.ndarray,
              state_pred: np.ndarray = None) -> np.ndarray:
        """应用校正, 返回 y_final = y_pred_raw + delta_hat (≥ 0)"""
        delta = self.predict_delta(y_pred_raw, timestamps, weather_df,
                                   recent_signal, season_labels, state_pred)
        return np.clip(y_pred_raw + delta, 0, None)

    def strip_for_save(self):
        """清除不可 pickle 的成员"""
        return self


# ============================================================
# L5 多模型动态切换 (按推理时整体漂移级别切换)
# ============================================================
class ModelSwitcher:
    """
    根据漂移检测结果动态选择推理模型

    v6.9 策略 (整体切换, 不逐样本; 主模型权重区分 "L4 启用 / 未启用"):
        NORMAL : 主模型 (v6, 精度优先)
        WARN   : L4 启用 -> 0.75 主 + 0.25 fb;   L4 未启用 -> 0.6 主 + 0.4 fb
        ALERT  : L4 启用 -> 0.5 主 + 0.5 fb;     L4 未启用 -> 0.0 主 + 1.0 fb (旧行为)

    v6.9 设计依据 (5 月推理硬证据):
        L4 单独启用时, SAE 从 8.77% -> 3.08% (-65%); 若 L5 在 ALERT 模式下
        把主权重压到 0, 会把 L4 的 SAE 校正收益全部丢失。新策略在保证 MAE
        稳定的前提下, 让 L4 的电量校正能力保留约一半。
    """

    def __init__(self, main_bundle, fallback_bundle=None, logger=None):
        self.main = main_bundle
        self.fallback = fallback_bundle    # 通常是 v4.2
        self.log = logger

    def decide(self, drift_report, calib_active: bool = False) -> dict:
        """
        根据 drift_report (DataFrame) 决定切换策略

        参数:
            drift_report : 漂移检测报告 DataFrame (含 level/dimension/drift_ratio)
            calib_active : 当前推理是否已成功应用 L4 残差校正 (v6.9 新增)
                          True  -> 主模型已被校正, 在 OOD 上更可靠, 保留更多权重
                          False -> 主模型未校正 (退化为 v6.7 行为)

        返回: {"mode": "NORMAL/WARN/ALERT", "main_weight": 0~1,
              "use_fallback": bool, "calib_active": bool, ...}
        """
        # 从 common.py 读取阈值常量 (避免硬编码, 便于线上调参)
        from common import (L5_ALERT_N_ALERT, L5_ALERT_MAX_CONCEPT_DRIFT,
                            L5_WARN_N_ALERT, L5_WARN_N_WARN,
                            L5_WARN_MAX_CONCEPT_DRIFT,
                            L5_MAIN_WEIGHT_ALERT_WITH_L4,
                            L5_MAIN_WEIGHT_ALERT_WITHOUT_L4,
                            L5_MAIN_WEIGHT_WARN_WITH_L4,
                            L5_MAIN_WEIGHT_WARN_WITHOUT_L4)

        if drift_report is None or len(drift_report) == 0:
            return {"mode": "NORMAL", "main_weight": 1.0, "use_fallback": False,
                    "calib_active": bool(calib_active),
                    "reason": "无漂移报告"}

        # 按 ALERT > WARN > NORMAL 优先级取最严重的 level
        levels = drift_report["level"].value_counts().to_dict()
        n_alert = levels.get("ALERT", 0)
        n_warn  = levels.get("WARN", 0)

        # 排除"covariate/bus_signal_mean"这种总线粗粒度的, 优先看 concept drift
        concept_rows = drift_report[
            drift_report["dimension"].str.startswith("concept/")
        ]
        if len(concept_rows) > 0:
            max_concept_drift = float(concept_rows["drift_ratio"].abs().max())
        else:
            max_concept_drift = 0.0

        if n_alert >= L5_ALERT_N_ALERT or \
           max_concept_drift >= L5_ALERT_MAX_CONCEPT_DRIFT:
            mode = "ALERT"
            if self.fallback is not None:
                main_w = (L5_MAIN_WEIGHT_ALERT_WITH_L4 if calib_active
                          else L5_MAIN_WEIGHT_ALERT_WITHOUT_L4)
            else:
                main_w = 1.0
            use_fb = self.fallback is not None and main_w < 1.0
            reason = (f"严重漂移: {n_alert} 个 ALERT, 最大概念漂移 "
                      f"{max_concept_drift*100:.1f}% "
                      f"({'L4 已启用, 主权重保留' if calib_active else 'L4 未启用'})")
        elif n_alert >= L5_WARN_N_ALERT or n_warn >= L5_WARN_N_WARN or \
             max_concept_drift >= L5_WARN_MAX_CONCEPT_DRIFT:
            mode = "WARN"
            if self.fallback is not None:
                main_w = (L5_MAIN_WEIGHT_WARN_WITH_L4 if calib_active
                          else L5_MAIN_WEIGHT_WARN_WITHOUT_L4)
            else:
                main_w = 1.0
            use_fb = self.fallback is not None and main_w < 1.0
            reason = (f"轻度漂移: {n_alert} ALERT + {n_warn} WARN, "
                      f"最大概念漂移 {max_concept_drift*100:.1f}% "
                      f"({'L4 已启用' if calib_active else 'L4 未启用'})")
        else:
            mode = "NORMAL"
            main_w = 1.0
            use_fb = False
            reason = f"无显著漂移 (max concept drift {max_concept_drift*100:.1f}%)"

        decision = {
            "mode": mode, "main_weight": main_w, "use_fallback": use_fb,
            "calib_active": bool(calib_active),
            "reason": reason,
            "drift_alert_count": int(n_alert),
            "drift_warn_count": int(n_warn),
            "max_concept_drift": round(max_concept_drift, 4),
        }
        if self.log:
            icon = {"NORMAL": "✓", "WARN": "⚠️", "ALERT": "🚨"}.get(mode, "?")
            calib_tag = "L4✓" if calib_active else "L4✗"
            self.log.info(f"  [L5/ModelSwitcher] {icon} 决策: mode={mode}, "
                          f"主模型权重={main_w}, 启用 fallback={use_fb}, "
                          f"{calib_tag}")
            self.log.info(f"  [L5/ModelSwitcher]   理由: {reason}")
        return decision

    @staticmethod
    def blend(y_main: np.ndarray, y_fallback: np.ndarray,
              main_weight: float) -> np.ndarray:
        """加权融合两个模型的预测"""
        w = float(main_weight)
        if y_fallback is None or w >= 1.0:
            return y_main
        if w <= 0.0:
            return y_fallback
        return w * y_main + (1 - w) * y_fallback


# ============================================================
# CLI 自检
# ============================================================
if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    print("ResidualCalibrator 自检:")
    rc = ResidualCalibrator(n_estimators=30)
    n = 200
    y_pred = np.random.uniform(400, 800, n)
    y_true = y_pred + np.random.normal(50, 30, n)   # 模拟系统性偏低 50W
    state = np.ones(n, dtype=int)
    ts = pd.date_range("2026-05-01", periods=n, freq="15min")
    seasons = np.array(["summer"] * n)
    recent = np.full(n, 600.0)
    rc.fit(y_pred, y_true, state, ts, None, recent, seasons)
    delta = rc.predict_delta(y_pred, ts, None, recent, seasons, state)
    print(f"  原 MAE: {np.abs(y_pred - y_true).mean():.1f}W")
    print(f"  校正后 MAE: {np.abs(y_pred + delta - y_true).mean():.1f}W")
    print(f"  delta 均值: {delta.mean():+.1f}W (期望约 +50W)")
    
    print("\nModelSwitcher 自检:")
    sw = ModelSwitcher(main_bundle={}, fallback_bundle={})
    # 模拟漂移报告
    dr1 = pd.DataFrame([{"dimension": "concept/x", "drift_ratio": 0.05, "level": "NORMAL"}])
    dr2 = pd.DataFrame([{"dimension": "concept/x", "drift_ratio": 0.30, "level": "ALERT"},
                        {"dimension": "concept/y", "drift_ratio": 0.40, "level": "ALERT"},
                        {"dimension": "concept/z", "drift_ratio": 0.60, "level": "ALERT"}])
    for tag, dr in [("低漂移", dr1), ("高漂移", dr2)]:
        d = sw.decide(dr)
        print(f"  {tag}: {d}")
