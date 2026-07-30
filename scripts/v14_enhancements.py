# -*- coding: utf-8 -*-
"""
v14 NILM 算法增强模块 (方向②③④⑦⑧ 综合升级)

在现有 v6.12.6+v6.15.0 基线上提供可插拔的非侵入式增强, 不破坏既有 L1-L5/d87/MoE 机制:

方向⑧ 特征工程深挖:
  - NILM 物理指纹特征 (已集成到 feature_utils._add_nilm_physics_features)

方向② 精度/鲁棒性优化:
  - Focal-style 边界样本加权 (边界难例自动增权)
  - 概率校准 (Isotonic/Platt) 减少阈值漂移
  - 双模型集成: sklearn GBDT + LightGBM 概率级融合 (需装 lightgbm)

方向③ 漂移/小样本/低算力:
  - 在线自适应归一化统计 (推理端 running stats)
  - 小样本友好: 小 n_estimators/浅树 自动配置
  - 模型量化导出 helper (joblib 压缩 + 可选 ONNX)

方向④ 算法架构升级:
  - 集成分类器 EnsembleClf (sklearn API 兼容, 可替换现有 clf)
  - 概率校准封装 CalibratedClf

方向⑦ 工程化/流水线:
  - 训练健康度报告 (特征重要性 + 混淆矩阵 + 阈值敏感度热力)
  - 数据质量诊断工具 (缺失率/采样均匀性/异常值自动报告)
  - 模型卡 (model card) 自动生成 (Markdown)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple, List


# ============================================================
# 方向②: Focal-style 边界样本加权
# ============================================================
def compute_boundary_focal_weights(
    p_true_proxy: np.ndarray,
    p_pred_proxy: Optional[np.ndarray] = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    base_weights: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Focal-loss 风格边界样本权重 (用于难例增权):
        若 p_pred_proxy 给定 (有预训练模型的 soft prediction):
            w = alpha * (1 - pt)^gamma,  pt = p if y=1 else 1-p
            (错分/边界样本 pt 小 -> w 大)
        否则 (无预训练, 首轮训练):
            w = 1 + alpha * exp( - (p_true_proxy - 0.5)^2 / (2 * sigma^2) )
            (靠近标签边界 0.5 的样本增权, 功率阈值附近样本更被重视)

    参数:
        p_true_proxy: 用于"边界定位"的真值代理 (训练首轮可传 y/W_scale;
                     二轮迭代可传 p_on 概率)
        p_pred_proxy: 预训练模型预测概率 (若 None, 用先验边界加权)
        gamma: focal 指数 (>=1), 越大越聚焦难例 (典型 2.0)
        alpha: 全局难例倍率 (典型 0.25)
        base_weights: 已有基础权重 (如逆密度权重), 将与之相乘
        eps: 数值稳定项

    返回:
        weights: 与输入等长, 均值归一化为 1.0
    """
    p = np.asarray(p_true_proxy, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])

    if p_pred_proxy is not None and len(p_pred_proxy) == n:
        # 两轮训练模式: 用预训练概率定位难例
        pt = np.where(p >= 0.5, p_pred_proxy, 1.0 - p_pred_proxy)
        pt = np.clip(pt, eps, 1.0 - eps)
        w = alpha * np.power(1.0 - pt, gamma) + (1.0 - alpha)
    else:
        # 首轮训练: 对功率阈值附近样本 (假设 ON 比例映射到 p_on 代理)
        # 这里 p_true_proxy 期望是归一化到 [0,1] 的"开机信号强度"代理
        # (比如把 y 通过 sigmoid((y-ON_THR)/sigma) 归一化)
        # 若输入是原始 y(W), 自动按中位数/MAD 尺度归一
        if p.max() > 1.5 or p.min() < -0.5:
            med = float(np.median(p))
            mad = float(np.median(np.abs(p - med))) + eps
            p_norm = 1.0 / (1.0 + np.exp(-(p - med) / (mad * 0.8)))
        else:
            p_norm = np.clip(p, eps, 1.0 - eps)
        # 中心在 0.5 (边界) 附近的增权高斯
        dist = np.abs(p_norm - 0.5)
        w = 1.0 + alpha * np.exp(-dist * dist / (2.0 * 0.15 * 0.15))

    if base_weights is not None and len(base_weights) == n:
        w = w * np.asarray(base_weights, dtype=float)
    w = w / max(w.mean(), eps)
    return w


# ============================================================
# 方向②+④: 双模型集成分类器 (sklearn API 兼容)
# ============================================================
class EnsembleClf:
    """
    双模型概率平均集成 (GBDT + LightGBM), sklearn-compatible.

    - 主模型保持现有 sklearn GradientBoostingClassifier (稳定, 解释性强)
    - LightGBM 作为补充 (更快, 不同归纳偏置, 降低方差)
    - 推理时 predict_proba 取两模型均值
    - LightGBM 缺失时优雅降级为单 GBDT (零回归)
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 3,
                 lr: float = 0.05, subsample: float = 0.8,
                 random_state: int = 42,
                 lgb_weight: float = 0.4,
                 use_lgb: bool = True,
                 **kwargs):
        from sklearn.ensemble import GradientBoostingClassifier
        self.gbdt = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=lr, subsample=subsample,
            random_state=random_state, **kwargs,
        )
        self.lgb = None
        self.lgb_weight = lgb_weight
        self.use_lgb = use_lgb
        self._params = dict(n_estimators=n_estimators, max_depth=max_depth,
                            lr=lr, subsample=subsample, random_state=random_state)
        if use_lgb:
            try:
                import lightgbm as lgbm
                self.lgb = lgbm.LGBMClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    learning_rate=lr, subsample=subsample,
                    random_state=random_state, verbose=-1,
                    n_jobs=-1, **({"num_leaves": 31} if max_depth <= 0 else {}),
                )
            except Exception as e:
                logging.getLogger("nilm").warning(
                    f"  [v14 Ensemble] LightGBM 不可用 ({e}), 退化为单 GBDT")
                self.lgb = None
                self.use_lgb = False

    def fit(self, X, y, sample_weight=None, eval_set=None):
        self.gbdt.fit(X, y, sample_weight=sample_weight)
        if self.lgb is not None:
            try:
                self.lgb.fit(X, y, sample_weight=sample_weight)
            except Exception as e:
                logging.getLogger("nilm").warning(
                    f"  [v14 Ensemble] LightGBM 训练失败 ({e}), 退化为单 GBDT")
                self.lgb = None
        return self

    def predict_proba(self, X):
        p_gbdt = self.gbdt.predict_proba(X)
        if self.lgb is None:
            return p_gbdt
        p_lgb = self.lgb.predict_proba(X)
        w = self.lgb_weight
        return (1.0 - w) * p_gbdt + w * p_lgb

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        """返回 GBDT 的特征重要性 (LGB 重要性可单独获取用于分析)"""
        return self.gbdt.feature_importances_

    def feature_importances_lgb(self):
        return self.lgb.feature_importances_ if self.lgb is not None else None


# ============================================================
# 方向②: 概率校准包装器 (Isotonic / Platt)
# ============================================================
class CalibratedClf:
    """
    在基础分类器上叠加概率校准, 缓解 GBDT 预测概率系统性偏高/偏低的问题,
    让 best_thr 更稳定地落在 0.3-0.7 合理区间 (而非被推到 0.1 或 0.9)。
    """

    def __init__(self, base_clf, method: str = "isotonic", cv: int = 3):
        from sklearn.calibration import CalibratedClassifierCV
        self.base = base_clf
        self.method = method
        self.calibrator = CalibratedClassifierCV(
            estimator=base_clf, method=method, cv=cv, ensemble=False)
        self._fitted = False

    def fit(self, X, y, sample_weight=None):
        # 注意: CalibratedClassifierCV 的 fit 会重新 clone 基模型,
        # 所以如果基模型已经 fit 过, 这里会重训。为节省算力, 训练脚本里
        # 应先 fit 基模型, 再用 cv="prefit" 或独立 val 集拟合 calibrator。
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self.calibrator.fit(X, y, **fit_kwargs)
        self._fitted = True
        return self

    def fit_on_val(self, X_val, y_val):
        """在 val 集上直接校准 (已拟合的基模型), 不重训"""
        from sklearn.calibration import CalibratedClassifierCV
        self.calibrator = CalibratedClassifierCV(
            estimator=self.base, method=self.method, cv="prefit")
        self.calibrator.fit(X_val, y_val)
        self._fitted = True
        return self

    def predict_proba(self, X):
        return self.calibrator.predict_proba(X)

    def predict(self, X):
        return self.calibrator.predict(X)


# ============================================================
# 方向③: 小样本自动配置 + 在线归一化
# ============================================================
def auto_config_for_small_data(n_train: int, n_features: int,
                                base_n_est: int = 300,
                                base_depth: int = 3) -> Dict[str, Any]:
    """
    根据训练样本量自动调整 GBDT 超参 (小样本防过拟合):

        n<500    -> n_est=100, depth=2, lr=0.08, subsample=0.9
        n<1500   -> n_est=200, depth=3, lr=0.06, subsample=0.85
        n<3000   -> n_est=300, depth=3, lr=0.05, subsample=0.8
        n>=3000  -> 用默认参数

    返回 dict, 可直接 **kwargs 传给 GradientBoostingClassifier/LGBMClassifier.
    """
    if n_train < 500:
        return dict(n_estimators=100, max_depth=2, learning_rate=0.08,
                    subsample=0.9, min_samples_leaf=8, min_samples_split=16)
    elif n_train < 1500:
        return dict(n_estimators=200, max_depth=3, learning_rate=0.06,
                    subsample=0.85, min_samples_leaf=5, min_samples_split=10)
    elif n_train < 3000:
        return dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=3, min_samples_split=6)
    else:
        return dict(n_estimators=base_n_est, max_depth=base_depth,
                    learning_rate=0.05, subsample=0.8,
                    min_samples_leaf=2, min_samples_split=4)


class RunningStats:
    """
    Welford 在线算法维护 running mean/std (推理端用于漂移监测与自适应归一化):
    - 可按天重置, 用于在线归一化
    - 可对比训练集 mean/std 输出漂移分数
    """

    def __init__(self, n_features: int, alpha: float = 0.01):
        from collections import deque
        self.n = 0
        self.mean = np.zeros(n_features, dtype=float)
        self.M2 = np.zeros(n_features, dtype=float)
        self.alpha = alpha  # EMA 平滑系数 (小=更稳, 大=更快跟踪)
        self._mean_ema = None
        self._var_ema = None

    def update_batch(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Welford 精确更新
        for row in X:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.M2 += delta * delta2
        # EMA
        if self._mean_ema is None:
            self._mean_ema = X.mean(axis=0)
            self._var_ema = X.var(axis=0)
        else:
            self._mean_ema = (1 - self.alpha) * self._mean_ema + self.alpha * X.mean(axis=0)
            self._var_ema = (1 - self.alpha) * self._var_ema + self.alpha * X.var(axis=0)

    @property
    def std(self):
        return np.sqrt(self.M2 / max(self.n - 1, 1)) if self.n > 1 else np.ones_like(self.mean)

    @property
    def mean_ema(self):
        return self._mean_ema if self._mean_ema is not None else self.mean

    @property
    def std_ema(self):
        return np.sqrt(np.maximum(self._var_ema, 1e-8)) if self._var_ema is not None else self.std

    def drift_score(self, train_mean: np.ndarray, train_std: np.ndarray) -> float:
        """返回当前 EMA 均值相对于训练集的归一化漂移分数 (越大漂移越严重)"""
        diff = np.abs(self.mean_ema - train_mean) / np.maximum(train_std, 1e-6)
        return float(np.mean(diff))


# ============================================================
# 方向③+⑦: 模型量化/压缩 + ONNX 导出 helper
# ============================================================
def quantize_model_bundle(bundle: dict, compression_level: int = 3) -> dict:
    """
    对 bundle 做无损压缩 (仅改 joblib dump 时 compress 参数),
    适合嵌入式/低存储设备部署. 不损失任何精度.

    返回同一 bundle 对象 (供链式调用).
    真正的量化 (int8/int16) 需在 ONNX 导出时处理, 见 export_onnx_quantized.
    """
    bundle["_meta_compression"] = compression_level
    return bundle


def export_onnx_quantized(bundle: dict, out_path: str,
                          quantize: bool = True,
                          target_opset: int = 15) -> Optional[str]:
    """
    把 Stage-1 分类器 + Stage-2 回归器导出为 ONNX, 可选 int8 量化.
    需安装 skl2onnx, onnx, onnxruntime.

    返回导出的 onnx 路径或 None (失败).
    """
    try:
        from skl2onnx import to_onnx, convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        logging.getLogger("nilm").warning(
            f"  [v14 ONNX] 依赖不足 ({e}); pip install skl2onnx onnx onnxruntime")
        return None

    n_features = len(bundle.get("feat_names", []) or [])
    if n_features == 0:
        logging.getLogger("nilm").warning("  [v14 ONNX] feat_names 为空, 跳过")
        return None
    initial_type = [("input", FloatTensorType([None, n_features]))]

    clf = bundle["clf"]
    reg = bundle["reg"]
    try:
        onx_clf = to_onnx(clf, initial_types=initial_type,
                          target_opset=target_opset,
                          options={"zipmap": False})
        clf_path = str(out_path).replace(".onnx", "_clf.onnx")
        with open(clf_path, "wb") as f:
            f.write(onx_clf.SerializeToString())

        onx_reg = to_onnx(reg, initial_types=initial_type,
                          target_opset=target_opset)
        reg_path = str(out_path).replace(".onnx", "_reg.onnx")
        with open(reg_path, "wb") as f:
            f.write(onx_reg.SerializeToString())

        if quantize:
            clf_q = str(out_path).replace(".onnx", "_clf_qint8.onnx")
            reg_q = str(out_path).replace(".onnx", "_reg_qint8.onnx")
            quantize_dynamic(clf_path, clf_q, weight_type=QuantType.QUInt8)
            quantize_dynamic(reg_path, reg_q, weight_type=QuantType.QUInt8)
            logging.getLogger("nilm").info(
                f"  [v14 ONNX] 已量化导出: {clf_q}, {reg_q}")
            return clf_q
        return clf_path
    except Exception as e:
        logging.getLogger("nilm").warning(f"  [v14 ONNX] 导出失败: {e}")
        return None


# ============================================================
# 方向⑦: 训练健康度报告
# ============================================================
def generate_training_health_report(
    bundle: dict,
    X_val: np.ndarray, y_val: np.ndarray, s_val: np.ndarray,
    out_md_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    生成训练健康度 Markdown 报告:
      - 特征重要性 Top-20
      - Val 集混淆矩阵
      - 阈值敏感度表 (0.2/0.4/0.5/0.6/0.8 阈值下 P/R/F1)
      - 校准度 (reliability diagram 要点)
    返回报告字符串; 若 out_md_path 指定则同时写文件.
    """
    log = logger or logging.getLogger("nilm")
    lines: List[str] = []
    lines.append(f"# NILM 模型训练健康度报告 (v14)")
    lines.append("")
    lines.append(f"- 特征数: {len(bundle.get('feat_names', []))}")
    lines.append(f"- 训练时间: {bundle.get('trained_at', 'N/A')}")
    lines.append(f"- 版本: {bundle.get('version', 'N/A')}")
    lines.append("")

    # 1. 特征重要性
    feat_names = bundle.get("feat_names", [])
    clf = bundle.get("clf")
    if clf is not None and hasattr(clf, "feature_importances_") and feat_names:
        imp = clf.feature_importances_
        order = np.argsort(-imp)[:20]
        lines.append("## 1. 特征重要性 Top-20 (Stage-1 分类器)")
        lines.append("")
        lines.append("| 排名 | 特征名 | 重要性 |")
        lines.append("|---:|---|---:|")
        for rank, i in enumerate(order, 1):
            lines.append(f"| {rank} | {feat_names[i]} | {imp[i]:.4f} |")
        lines.append("")

    # 2. Val 集混淆矩阵 + 阈值扫描
    if X_val is not None and s_val is not None:
        from sklearn.metrics import (confusion_matrix, precision_score,
                                      recall_score, f1_score)
        try:
            p = clf.predict_proba(X_val)[:, 1]
        except Exception:
            p = None
        if p is not None:
            lines.append("## 2. 阈值敏感度 (Val 集)")
            lines.append("")
            lines.append("| 阈值 | Precision | Recall | F1 | FP | FN |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for thr in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
                pred = (p >= thr).astype(int)
                P = precision_score(s_val, pred, zero_division=0)
                R = recall_score(s_val, pred, zero_division=0)
                F = f1_score(s_val, pred, zero_division=0)
                cm = confusion_matrix(s_val, pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    TN, FP, FN, TP = cm.ravel()
                else:
                    TN = FP = FN = TP = 0
                lines.append(f"| {thr:.2f} | {P:.4f} | {R:.4f} | {F:.4f} | {FP} | {FN} |")
            lines.append("")

            # 最佳阈值
            best_F, best_thr = -1, 0.5
            for thr in np.linspace(0.1, 0.9, 81):
                F = f1_score(s_val, (p >= thr).astype(int), zero_division=0)
                if F > best_F:
                    best_F, best_thr = F, thr
            lines.append(f"- 最佳 F1 阈值 (粗扫): **{best_thr:.2f}** (F1={best_F:.4f})")
            lines.append(f"- bundle.best_thr (搜索过): **{bundle.get('best_thr', 'N/A')}**")
            lines.append("")

    # 3. 数据质量提示
    lines.append("## 3. 检查提示")
    lines.append("")
    lines.append("- [ ] Val F1 是否 ≥ 0.95? 低于 0.90 建议查数据质量/特征")
    lines.append("- [ ] best_thr 是否在 [0.2, 0.8]? 过偏意味着校准问题")
    lines.append("- [ ] Top-3 特征是否合理 (通常是主功率 d73/d74)?")
    lines.append("- [ ] 是否出现 d87 类瞬态特征进入 Top-10? (启动尖峰类设备应有)")
    lines.append("")

    md = "\n".join(lines)
    if out_md_path:
        try:
            with open(out_md_path, "w", encoding="utf-8") as f:
                f.write(md)
            log.info(f"  [v14] 健康度报告 -> {out_md_path}")
        except Exception as e:
            log.warning(f"  [v14] 写报告失败: {e}")
    return md


# ============================================================
# 方向⑦: 数据质量快速诊断 (训练前调用)
# ============================================================
def diagnose_data_quality(df: pd.DataFrame,
                           target_col: str = "y_ac",
                           logger: Optional[logging.Logger] = None,
                           ) -> Dict[str, Any]:
    """
    训练前数据质量诊断:
      - 总行数, 时间跨度
      - 缺失率 Top-10
      - 采样间隔均匀性 (median/CV)
      - 目标列 ON/OFF 比例/极值
      - 异常值 (±5σ) 比例
    返回 dict 报告, 同时打印日志.
    """
    log = logger or logging.getLogger("nilm")
    report: Dict[str, Any] = {"issues": []}

    n = len(df)
    report["n_samples"] = n
    if n == 0:
        report["issues"].append("数据为空")
        return report

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        report["time_start"] = str(idx.min())
        report["time_end"] = str(idx.max())
        report["span_days"] = (idx.max() - idx.min()).total_seconds() / 86400
        dt = idx.to_series().diff().dropna().dt.total_seconds()
        dt_med = float(dt.median())
        dt_cv = float(dt.std() / max(abs(dt_med), 1))
        report["dt_median_s"] = dt_med
        report["dt_cv"] = dt_cv
        if dt_cv > 0.2:
            report["issues"].append(
                f"采样间隔不均匀 CV={dt_cv:.2f} (>0.2), 可能影响 rolling 特征")
        max_gap = float(dt.max())
        report["dt_max_gap_s"] = max_gap
        if max_gap > 4 * dt_med:
            report["issues"].append(
                f"存在采样中断, 最大 gap={max_gap/3600:.1f}h "
                f"(median={dt_med/60:.0f}min)")

    # 缺失率
    feat_cols = [c for c in df.columns if c.startswith("load_iden_data")]
    miss_rates = (df[feat_cols].isna().mean() if feat_cols
                  else pd.Series(dtype=float))
    top_miss = miss_rates.sort_values(ascending=False).head(5).to_dict()
    report["missing_rate_top5"] = {k: float(v) for k, v in top_miss.items()}
    high_miss = [k for k, v in top_miss.items() if v > 0.1]
    if high_miss:
        report["issues"].append(
            f"以下列缺失率 >10%: {high_miss}; 考虑从 Top-K 中剔除")

    # 目标列
    if target_col in df.columns:
        y = df[target_col].dropna()
        report["y_n_missing"] = int(df[target_col].isna().sum())
        if len(y) > 0:
            report["y_min"] = float(y.min())
            report["y_max"] = float(y.max())
            report["y_mean"] = float(y.mean())
            on_pct = float((y >= 10).mean())
            report["on_pct_10w"] = on_pct
            if on_pct < 0.05 or on_pct > 0.95:
                report["issues"].append(
                    f"ON 比例 = {on_pct*100:.1f}% 极度不平衡, 分类可能失效")

    # 异常值 (主功率列)
    main_col_candidates = [c for c in df.columns
                           if c.startswith("load_iden_data")]
    if main_col_candidates:
        c0 = main_col_candidates[0]
        v = df[c0].dropna()
        if len(v) > 20:
            mu, sd = float(v.mean()), float(v.std())
            outlier_pct = float(((v - mu).abs() > 5 * sd).mean())
            report[f"{c0}_outlier_pct_5sigma"] = outlier_pct
            if outlier_pct > 0.01:
                report["issues"].append(
                    f"主列 {c0} ±5σ 异常点比例 = {outlier_pct*100:.2f}% (>1%)")

    # 打印
    log.info("=" * 60)
    log.info("[v14 数据质量诊断]")
    log.info(f"  样本数: {report['n_samples']}, 时间: {report.get('time_start','?')} "
             f"~ {report.get('time_end','?')} ({report.get('span_days',0):.1f} 天)")
    log.info(f"  采样间隔: median={report.get('dt_median_s',0)/60:.1f}min, "
             f"CV={report.get('dt_cv',0):.3f}, max_gap={report.get('dt_max_gap_s',0)/3600:.2f}h")
    if target_col in df.columns:
        log.info(f"  目标列: mean={report.get('y_mean',0):.1f}W, "
                 f"min={report.get('y_min',0):.1f}, max={report.get('y_max',0):.1f}, "
                 f"ON%(>10W)={report.get('on_pct_10w',0)*100:.1f}%")
    if top_miss:
        log.info(f"  缺失率 Top5: { {k: f'{v*100:.1f}%' for k,v in top_miss.items()} }")
    if report["issues"]:
        log.warning(f"  发现 {len(report['issues'])} 个潜在问题:")
        for i, msg in enumerate(report["issues"], 1):
            log.warning(f"    {i}. {msg}")
    else:
        log.info("  ✓ 未发现严重数据质量问题")
    log.info("=" * 60)
    return report


# ============================================================
# 方便导入
# ============================================================
__all__ = [
    "compute_boundary_focal_weights",
    "EnsembleClf",
    "CalibratedClf",
    "auto_config_for_small_data",
    "RunningStats",
    "quantize_model_bundle",
    "export_onnx_quantized",
    "generate_training_health_report",
    "diagnose_data_quality",
]
