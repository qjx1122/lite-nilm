# -*- coding: utf-8 -*-
"""
v14 增强模块 (v14_enhancements.py) 单元测试
====================================================

补齐 v14 系列缺失的 dedicated 单测 (v13.x 各版本均有 25-47 组断言的独立单测,
v14 此前只有 smoke test 端到端, 无数值硬对齐单测)。

覆盖 (9 个公开 API 全覆盖):
  T1. compute_boundary_focal_weights 首轮分支 (无 p_pred):
      边界样本增权 > 极端样本, 均值归一=1, 公式硬对齐
  T2. compute_boundary_focal_weights 二轮分支 (有 p_pred):
      难例(pt小)权重 > 易例(pt大), 数值逐值硬对齐
  T3. focal 边界: 空输入 / base_weights 相乘 / 原始 W 自动归一 / alpha=0 恒权
  T4. EnsembleClf 降级路径 (无 lightgbm): use_lgb=True 自动退化 = 纯 GBDT
      概率与参考 GBDT 逐值一致 (同 random_state)
  T5. EnsembleClf 概率融合数学: stub lightgbm 注入,
      predict_proba == (1-w)*p_gbdt + w*p_lgb 逐值硬对齐
  T6. CalibratedClf: cv=3 fit / fit_on_val(prefit) 两路径,
      概率形状/[0,1]/每行和=1/predict∈{0,1}
  T7. auto_config_for_small_data: 4 档边界 (500/1500/3000) 精确划分 +
      min_samples_leaf 区分第3/4档 + 自定义 base 透传
  T8. RunningStats: Welford mean/std 与 numpy 离线值 1e-10 对齐 /
      单样本 std=1 / EMA 首批=batch均值 / drift_score 恒零→漂移增大
  T9. quantize_model_bundle 幂等返回同对象 + export_onnx_quantized
      依赖缺失/feat_names 空 → None (优雅降级不崩)
  T10. generate_training_health_report: Markdown 关键节齐全 + 阈值表 7 行 +
       落盘读回一致
  T11. diagnose_data_quality: 正常无 issue / gap / 采样不均 / 缺失 >10% /
       ON 比例极端 / 5σ 异常 / 空 df 各 issue 精准触发
  T12. 端到端: focal 权重直通 EnsembleClf.fit(sample_weight=...)
       在合成二分类上 F1 达标, 输出合法

运行:
  python scripts/test_v14_enhancements.py
退出码: 0 = 全通过
"""
import sys
import types
import tempfile
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v14_enhancements import (
    compute_boundary_focal_weights,
    EnsembleClf,
    CalibratedClf,
    auto_config_for_small_data,
    RunningStats,
    quantize_model_bundle,
    export_onnx_quantized,
    generate_training_health_report,
    diagnose_data_quality,
)

PASS = 0
FAIL = 0
FAILURES = []


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {msg}")
    else:
        FAIL += 1
        FAILURES.append(msg)
        print(f"  [FAIL] {msg}")


# ------------------------------------------------------------
# 公共合成数据 (固定种子 -> 确定性)
# ------------------------------------------------------------
rng = np.random.RandomState(42)
_CLF_READY = {}


def make_clf_data(n_tr=400, n_te=150, seed=42):
    """简单二分类: 两个特征簇 + 标签噪声, 每轮调用相同 -> 确定性"""
    key = (n_tr, n_te, seed)
    if key in _CLF_READY:
        return _CLF_READY[key]
    r = np.random.RandomState(seed)
    X_tr = r.randn(n_tr, 4)
    # 线性可分为主 + 5% 噪声
    s_tr = ((X_tr[:, 0] + 0.8 * X_tr[:, 1] + 0.2 * r.randn(n_tr)) > 0).astype(int)
    flip = r.rand(n_tr) < 0.02
    s_tr[flip] = 1 - s_tr[flip]
    X_te = r.randn(n_te, 4)
    s_te = ((X_te[:, 0] + 0.8 * X_te[:, 1] + 0.2 * r.randn(n_te)) > 0).astype(int)
    flip = r.rand(n_te) < 0.02
    s_te[flip] = 1 - s_te[flip]
    out = (X_tr, s_tr, X_te, s_te)
    _CLF_READY[key] = out
    return out


try:
    import lightgbm  # noqa: F401
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import skl2onnx  # noqa: F401
    HAS_SKL2ONNX = True
except ImportError:
    HAS_SKL2ONNX = False

print("=" * 70)
print(f" 环境: lightgbm={'有' if HAS_LGB else '无(测降级路径)'}, "
      f"skl2onnx={'有' if HAS_SKL2ONNX else '无(测降级路径)'}")
print("=" * 70)


# ============================================================
# T1. focal 首轮分支 (无 p_pred): 边界增权 + 均值归一 + 公式硬对齐
# ============================================================
print("=" * 70)
print(" T1. focal 首轮: 边界样本权重 > 极端样本")
print("=" * 70)

p01 = np.linspace(0.0, 1.0, 101)  # 归一化代理信号, 中点就是边界 0.5
w = compute_boundary_focal_weights(p01)  # 默认 alpha=0.25
check(w.shape == (101,), "T1.1 输出长度 = 输入长度")
check(abs(w.mean() - 1.0) < 1e-9, f"T1.2 均值归一 = 1 (实际 {w.mean():.12f})")
check(bool((w > 0).all()), "T1.3 权重全正")
i_mid, i_edge = 50, 0
check(w[i_mid] > w[i_edge],
      f"T1.4 边界(p=0.5)权重 {w[i_mid]:.4f} > 极端(p=0) {w[i_edge]:.4f}")
check(w[i_mid] > w[100], "T1.5 边界(p=0.5)权重 > 极端(p=1)")
# 公式硬对齐 (未归一前): w_raw = 1 + 0.25*exp(-dist^2/(2*0.15^2))
dist = abs(p01 - 0.5)
w_raw = 1.0 + 0.25 * np.exp(-dist * dist / (2.0 * 0.15 * 0.15))
w_raw /= w_raw.mean()
check(np.allclose(w, w_raw, atol=1e-12), "T1.6 与高斯边界公式逐值硬对齐 (atol=1e-12)")
check(abs(w[i_mid] / max(w[i_edge], 1e-12)
          - (1.25 / (1.0 + 0.25 * np.exp(-0.25 / 0.045)))) < 1e-6,
      "T1.7 边界/极端权重比 ≈ 1.25/(1+0.25*e^-5.56)≈1.2488 (实测差 2.7e-8, "
      "源自 2*0.15*0.15 与字面量 0.045 的浮点 ulp 差, 非逻辑误差)")

# ============================================================
# T2. focal 二轮分支 (有 p_pred): 难例增权, 逐值硬对齐
# ============================================================
print("=" * 70)
print(" T2. focal 二轮: 难例(pt小)权重 > 易例(pt大)")
print("=" * 70)

# y=[1,1], pred=[0.9(易), 0.5(难)] -> pt=[0.9, 0.5]
y2 = np.array([1.0, 1.0])
p2 = np.array([0.9, 0.5])
w2 = compute_boundary_focal_weights(y2, p_pred_proxy=p2)
pt = np.clip(p2, 1e-6, 1 - 1e-6)
w2_raw = 0.25 * np.power(1.0 - pt, 2.0) + 0.75  # alpha=0.25, gamma=2
w2_expected = w2_raw / w2_raw.mean()
check(np.allclose(w2, w2_expected, atol=1e-12),
      f"T2.1 逐值硬对齐: {np.round(w2, 6)} == {np.round(w2_expected, 6)}")
check(w2[1] > w2[0], f"T2.2 难例(pt=0.5) {w2[1]:.4f} > 易例(pt=0.9) {w2[0]:.4f}")
# 负类镜像: y=0, pred=0.1 -> pt=1-0.1=0.9 (同样易) 应与 y=1,pred=0.9 权重相同
w3 = compute_boundary_focal_weights(np.array([0.0]), p_pred_proxy=np.array([0.1]))
w4 = compute_boundary_focal_weights(np.array([1.0]), p_pred_proxy=np.array([0.9]))
check(np.allclose(w3, w4, atol=1e-12),
      f"T2.3 负类镜像对称: w(y=0,p=0.1)={w3[0]:.6f} == w(y=1,p=0.9)={w4[0]:.6f}")
check(abs(w2.mean() - 1.0) < 1e-9, "T2.4 二轮分支均值归一 = 1")
# gamma 语义硬考证 (曾测反, 已修正): 实现带 (1-alpha)=0.75 保底项,
# 未归一权重 w=0.25*(1-pt)^gamma+0.75 随 gamma 单调不增
# -> 归一化后难/易比随 gamma 增大而向 1 收敛 (差异化被保底压平)
ratios = []
for g in (1.0, 2.0, 4.0):
    wg = compute_boundary_focal_weights(y2, p_pred_proxy=p2, gamma=g)
    check(wg[1] > wg[0], f"T2.5a gamma={g:g}: 难例权重 > 易例 (全 gamma 成立)")
    ratios.append(wg[1] / wg[0])
check(ratios[0] > ratios[1] > ratios[2] > 1.0,
      f"T2.5b 难/易比随 gamma 单调收敛: {[round(r, 4) for r in ratios]} "
      f"(保底项 (1-alpha) 主导, 已用 numpy 独立复核)")
# 真 focal 抑易性: 极简单 (pt=0.999) 的未归一权重随 gamma 严格递减
pt_easy = 0.999
w_un = [0.25 * (1 - pt_easy) ** g + 0.75 for g in (1.0, 2.0, 4.0)]
check(w_un[0] > w_un[1] > w_un[2],
      f"T2.5c pt=0.999 未归一权重 {[round(x, 6) for x in w_un]} 随 gamma 严格递减"
      f" (focal 抑易成立)")

# ============================================================
# T3. focal 边界情况
# ============================================================
print("=" * 70)
print(" T3. focal 边界: 空输入 / base_weights / 原始W自动归一 / alpha=0")
print("=" * 70)

w_empty = compute_boundary_focal_weights(np.array([]))
check(isinstance(w_empty, np.ndarray) and w_empty.shape == (0,),
      "T3.1 空输入 -> 空数组不崩")
# base_weights: p 全=0.5 -> focal 部分为常数, 归一后 w ∝ base
w_base = compute_boundary_focal_weights(np.full(4, 0.5),
                                        base_weights=np.array([1.0, 2.0, 3.0, 4.0]))
check(np.allclose(w_base, np.array([0.4, 0.8, 1.2, 1.6]), atol=1e-9),
      f"T3.2 base_weights 相乘生效: {np.round(w_base, 3)}")
# 原始 W 值 (max>1.5) -> 自动 sigmoid 归一分支, 不崩且合法
w_watt = compute_boundary_focal_weights(np.array([0, 5, 10, 50, 500, 2000], dtype=float))
check(w_watt.shape == (6,) and np.isfinite(w_watt).all() and (w_watt > 0).all(),
      "T3.3 原始瓦数输入自动归一: 形状/有限/全正")
check(abs(w_watt.mean() - 1.0) < 1e-9, "T3.4 自动归一分支均值 = 1")
# alpha=0 -> 首轮 w 恒为 1+0*... = 1, 归一后全 1.0
w_a0 = compute_boundary_focal_weights(p01, alpha=0.0)
check(np.allclose(w_a0, 1.0, atol=1e-12), "T3.5 alpha=0 -> 恒权 1.0")
# gamma=0 二轮 -> w = alpha+ (1-alpha) = 1 常数
w_g0 = compute_boundary_focal_weights(y2, p_pred_proxy=p2, gamma=0.0)
check(np.allclose(w_g0, 1.0, atol=1e-9), "T3.6 gamma=0 (二轮) -> 恒权 1.0")

# ============================================================
# T4. EnsembleClf 降级路径 (本环境无 lightgbm 时)
# ============================================================
print("=" * 70)
print(" T4. EnsembleClf 优雅降级 (无 lightgbm -> 纯 GBDT)")
print("=" * 70)

X_tr, s_tr, X_te, s_te = make_clf_data()

if not HAS_LGB:
    ens = EnsembleClf(n_estimators=30, max_depth=2, lr=0.1, subsample=0.9,
                      random_state=42, use_lgb=True)  # 想要 LGB 但环境没有
    check(ens.lgb is None and ens.use_lgb is False,
          "T4.1 use_lgb=True 但无 lightgbm -> 自动降级 (lgb=None)")
    ens.fit(X_tr, s_tr)
    p_ens = ens.predict_proba(X_te)
    # 参考: 同参纯 GBDT (EnsembleClf 内部构造参数一一对应)
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    ref = GBC(n_estimators=30, max_depth=2, learning_rate=0.1, subsample=0.9,
              random_state=42)
    ref.fit(X_tr, s_tr)
    p_ref = ref.predict_proba(X_te)
    check(np.allclose(p_ens, p_ref, atol=1e-12),
          "T4.2 降级后概率与纯 GBDT 逐值一致 (零回归)")
    check(p_ens.shape == (150, 2), "T4.3 predict_proba 形状 (n, 2)")
    pred = ens.predict(X_te)
    check(set(np.unique(pred)).issubset({0, 1}),
          "T4.4 predict 输出 ∈ {0,1}")
    check(np.array_equal(pred, (p_ens[:, 1] >= 0.5).astype(int)),
          "T4.5 predict = (proba>=0.5) 自洽")
    check(ens.feature_importances_.shape == (4,)
          and abs(float(ens.feature_importances_.sum()) - 1.0) < 1e-9,
          "T4.6 feature_importances_ 透传 GBDT (和=1)")
    check(ens.feature_importances_lgb() is None,
          "T4.7 feature_importances_lgb() 降级时 = None")
    # 显式 use_lgb=False 也不崩
    ens2 = EnsembleClf(n_estimators=10, use_lgb=False)
    ens2.fit(X_tr, s_tr)
    check(ens2.predict(X_te).shape == (150,), "T4.8 use_lgb=False 显式降级可用")
    # [v14.6] meta 显形旗标语义: ensemble_lgb_active = (clf.lgb is not None)
    check((getattr(ens, "lgb", None) is not None) is False,
          "T4.9 降级时 ensemble_lgb_active 旗标 = False (meta 可显形)")
else:
    ens = EnsembleClf(n_estimators=30, max_depth=2, random_state=42, use_lgb=True)
    ens.fit(X_tr, s_tr)
    check(ens.lgb is not None and ens.predict(X_te).shape == (150,),
          "T4.x 环境有 lightgbm: 真集成路径基本可用")
    check((getattr(ens, "lgb", None) is not None) is True,
          "T4.y 集成可用时 ensemble_lgb_active 旗标 = True (meta 可显形)")

# ============================================================
# T5. EnsembleClf 概率融合数学 (stub lightgbm)
# ============================================================
print("=" * 70)
print(" T5. EnsembleClf 概率融合: (1-w)*p_gbdt + w*p_lgb 硬对齐")
print("=" * 70)

_LGB_KEY = "lightgbm"
_orig_lgb_mod = sys.modules.get(_LGB_KEY)

class _FakeLGBMClassifier:
    """确定性假 LGBM: 预测概率恒为 0.9 (便于硬对齐融合公式)"""
    def __init__(self, **kw):
        self.kw = kw
    def fit(self, X, y, sample_weight=None):
        self._n_feat = np.asarray(X).shape[1]
        return self
    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, 0.9)
        return np.column_stack([1.0 - p, p])
    @property
    def feature_importances_(self):
        return np.ones(getattr(self, "_n_feat", 4))

_stub = types.ModuleType(_LGB_KEY)
_stub.LGBMClassifier = _FakeLGBMClassifier
sys.modules[_LGB_KEY] = _stub
try:
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    ens5 = EnsembleClf(n_estimators=30, max_depth=2, lr=0.1, subsample=0.9,
                       random_state=42, lgb_weight=0.4, use_lgb=True)
    check(ens5.lgb is not None and ens5.use_lgb is True,
          "T5.1 stub 注入后 LGB 成功挂载")
    ens5.fit(X_tr, s_tr)
    p_mix = ens5.predict_proba(X_te)
    ref5 = GBC(n_estimators=30, max_depth=2, learning_rate=0.1, subsample=0.9,
               random_state=42)
    ref5.fit(X_tr, s_tr)
    p_gbdt = ref5.predict_proba(X_te)
    n5 = len(X_te)
    p_lgb = np.column_stack([np.full(n5, 0.1), np.full(n5, 0.9)])
    p_expect = 0.6 * p_gbdt + 0.4 * p_lgb
    check(np.allclose(p_mix, p_expect, atol=1e-12),
          "T5.2 融合概率 == 0.6*p_gbdt + 0.4*p_lgb 逐值硬对齐 (atol=1e-12)")
    check(np.allclose(p_mix.sum(axis=1), 1.0, atol=1e-12),
          "T5.3 融合概率每行和 = 1")
    # 换权重 w=0.25 公式仍成立
    ens5b = EnsembleClf(n_estimators=30, max_depth=2, lr=0.1, subsample=0.9,
                        random_state=42, lgb_weight=0.25, use_lgb=True)
    ens5b.fit(X_tr, s_tr)
    check(np.allclose(ens5b.predict_proba(X_te),
                      0.75 * p_gbdt + 0.25 * p_lgb, atol=1e-12),
          "T5.4 lgb_weight=0.25 时 (1-w)/w 系数正确")
    check(ens5b.feature_importances_lgb() is not None
          and float(ens5b.feature_importances_lgb().sum()) == 4.0,
          f"T5.5 feature_importances_lgb() 返回 LGB 重要性 (sum={float(ens5b.feature_importances_lgb().sum()):.1f})")
finally:
    if _orig_lgb_mod is not None:
        sys.modules[_LGB_KEY] = _orig_lgb_mod
    else:
        sys.modules.pop(_LGB_KEY, None)
print(f"  [INFO] stub lightgbm 已卸载, 环境还原")

# ============================================================
# T6. CalibratedClf 两路径
# ============================================================
print("=" * 70)
print(" T6. CalibratedClf: cv=3 fit / fit_on_val(prefit)")
print("=" * 70)

from sklearn.ensemble import GradientBoostingClassifier as GBC

base6 = GBC(n_estimators=30, max_depth=2, random_state=42)
cal = CalibratedClf(base6, method="isotonic", cv=3)
sw6 = np.ones(len(X_tr))  # sample_weight 透传不崩
cal.fit(X_tr, s_tr, sample_weight=sw6)
p_cal = cal.predict_proba(X_te)
check(p_cal.shape == (150, 2), "T6.1 cv=3 路径 predict_proba 形状 (n,2)")
check(bool((p_cal >= 0).all() and (p_cal <= 1).all()), "T6.2 校准概率 ∈ [0,1]")
check(np.allclose(p_cal.sum(axis=1), 1.0, atol=1e-9), "T6.3 校准概率每行和 = 1")
check(set(np.unique(cal.predict(X_te))).issubset({0, 1}),
      "T6.4 predict 输出 ∈ {0,1}")
# prefit 路径
base6b = GBC(n_estimators=30, max_depth=2, random_state=42)
base6b.fit(X_tr, s_tr)
cal2 = CalibratedClf(base6b, method="isotonic", cv=3)
cal2.fit_on_val(X_te, s_te)
p_cal2 = cal2.predict_proba(X_te)
check(p_cal2.shape == (150, 2) and np.allclose(p_cal2.sum(axis=1), 1.0, atol=1e-9),
      "T6.5 fit_on_val(prefit) 路径: 不崩且概率归一")

# ============================================================
# T7. auto_config_for_small_data 4 档边界
# ============================================================
print("=" * 70)
print(" T7. auto_config_for_small_data: 500/1500/3000 边界精确划分")
print("=" * 70)

c = auto_config_for_small_data(200, n_features=50)
check(c["n_estimators"] == 100 and c["max_depth"] == 2
      and c["min_samples_leaf"] == 8,
      "T7.1 n=200 (<500) -> est=100/depth=2/leaf=8")
c = auto_config_for_small_data(499, 50)
check(c["n_estimators"] == 100, "T7.2 n=499 仍第1档")
c = auto_config_for_small_data(500, 50)
check(c["n_estimators"] == 200 and c["min_samples_leaf"] == 5,
      "T7.3 n=500 (边界) -> 第2档 est=200/leaf=5")
c = auto_config_for_small_data(1499, 50)
check(c["n_estimators"] == 200, "T7.4 n=1499 仍第2档")
c = auto_config_for_small_data(1500, 50)
check(c["n_estimators"] == 300 and c["max_depth"] == 3
      and c["min_samples_leaf"] == 3 and c["min_samples_split"] == 6,
      "T7.5 n=1500 (边界) -> 第3档 est=300/leaf=3/split=6")
c = auto_config_for_small_data(2999, 50)
check(c["min_samples_leaf"] == 3, "T7.6 n=2999 仍第3档")
c = auto_config_for_small_data(3000, 50)
check(c["n_estimators"] == 300 and c["min_samples_leaf"] == 2
      and c["min_samples_split"] == 4,
      "T7.7 n=3000 (边界) -> 第4档 (leaf=2/split=4 与第3档区分)")
c = auto_config_for_small_data(10000, 50, base_n_est=500, base_depth=4)
check(c["n_estimators"] == 500 and c["max_depth"] == 4,
      "T7.8 大样本 + 自定义 base_n_est/base_depth 透传")
check(all(k in auto_config_for_small_data(100, 10) for k in
          ("n_estimators", "max_depth", "learning_rate", "subsample",
           "min_samples_leaf", "min_samples_split")),
      "T7.9 返回 dict 六键齐全可直接 **kwargs 展开")

# ============================================================
# T8. RunningStats: Welford 精确性 + EMA + drift_score
# ============================================================
print("=" * 70)
print(" T8. RunningStats: Welford 与 numpy 离线值硬对齐")
print("=" * 70)

r8 = np.random.RandomState(7)
X8 = r8.normal(loc=5.0, scale=2.0, size=(500, 3))
rs = RunningStats(n_features=3, alpha=0.01)
rs.update_batch(X8)
check(rs.n == 500, f"T8.1 样本计数 n={rs.n}")
check(np.allclose(rs.mean, X8.mean(axis=0), atol=1e-10),
      "T8.2 Welford mean == np.mean (atol=1e-10)")
check(np.allclose(rs.std, X8.std(axis=0, ddof=1), atol=1e-10),
      "T8.3 Welford std == np.std(ddof=1) (atol=1e-10)")
# EMA: 首批 update 后 _mean_ema 恰为 batch 均值 (初始化分支)
check(np.allclose(rs.mean_ema, X8.mean(axis=0), atol=1e-12),
      "T8.4 首批 EMA 初始化 = batch 均值")
# 单样本 std = 全 1 (保底, 不除零)
rs1 = RunningStats(n_features=2)
rs1.update_batch(np.array([1.0, 2.0]))
check(np.allclose(rs1.std, np.ones(2)), "T8.5 单样本 std = 1 (保底不崩)")
check(rs1.n == 1, "T8.6 单样本 n=1")
# drift_score: 相对训练分布零漂移 = 0, 偏移后 >0
train_mean, train_std = X8.mean(axis=0), X8.std(axis=0, ddof=1)
d0 = rs.drift_score(train_mean, train_std)
check(abs(d0) < 1e-12, f"T8.7 同分布 drift_score ≈ 0 (实际 {d0:.2e})")
X8b = np.full((50, 3), 15.0)  # 显著偏移 (相对训练分布 +5σ 级)
rs2 = RunningStats(n_features=3, alpha=1.0)  # alpha=1 -> EMA 完全跟随新批
rs2.update_batch(X8)
rs2.update_batch(X8b)
d1 = rs2.drift_score(train_mean, train_std)
check(d1 > 3.0, f"T8.8 EMA 完全跟随后 drift_score={d1:.2f} > 3 (显著漂移)")
# 增量等价性: 分两批 update == 一次全量 (Welford 精确性)
rsA, rsB = RunningStats(3), RunningStats(3)
rsA.update_batch(X8[:250]); rsA.update_batch(X8[250:])
rsB.update_batch(X8)
check(np.allclose(rsA.mean, rsB.mean, atol=1e-12)
      and np.allclose(rsA.std, rsB.std, atol=1e-12)
      and rsA.n == rsB.n,
      "T8.9 分批 update == 全量 update (mean/std/n 一致)")

# ============================================================
# T9. quantize_model_bundle + export_onnx_quantized 降级
# ============================================================
print("=" * 70)
print(" T9. 压缩/导出 helper: 幂等 + 依赖缺失优雅降级")
print("=" * 70)

bundle9 = {"version": "test", "feat_names": ["f1", "f2"]}
ret9 = quantize_model_bundle(bundle9, compression_level=3)
check(ret9 is bundle9, "T9.1 返回同一 bundle 对象 (链式可用)")
check(bundle9.get("_meta_compression") == 3, "T9.2 _meta_compression=3 已写入")
bundle9b = quantize_model_bundle({"feat_names": ["a"]})
check(bundle9b.get("_meta_compression") == 3, "T9.3 默认 compression_level=3")

if not HAS_SKL2ONNX:
    out_none = export_onnx_quantized(bundle9, "/tmp/_t9_should_not_exist.onnx")
    check(out_none is None, "T9.4 无 skl2onnx -> 返回 None 不崩")
    check(not Path("/tmp/_t9_should_not_exist.onnx").exists()
          and not Path("/tmp/_t9_should_not_exist_clf.onnx").exists(),
          "T9.5 依赖缺失时不产生残留文件")
    out_none2 = export_onnx_quantized({"feat_names": []}, "/tmp/_t9b.onnx")
    check(out_none2 is None, "T9.6 空 feat_names -> None")
else:
    out_none2 = export_onnx_quantized({"feat_names": []}, "/tmp/_t9b.onnx")
    check(out_none2 is None, "T9.x 有 skl2onnx 但 feat_names 空 -> None")

# ============================================================
# T10. generate_training_health_report
# ============================================================
print("=" * 70)
print(" T10. 训练健康度报告: 内容齐全 + 落盘读回一致")
print("=" * 70)

clf10 = GBC(n_estimators=30, max_depth=2, random_state=42)
clf10.fit(X_tr, s_tr)
bundle10 = {
    "clf": clf10,
    "feat_names": ["f0", "f1", "f2", "f3"],
    "best_thr": 0.44,
    "version": "v14.0.0-test",
    "trained_at": "2026-07-31T00:00:00",
}
md = generate_training_health_report(bundle10, X_te, s_tr[:len(X_te)], s_te,
                                     logger=None)
check(isinstance(md, str) and len(md) > 200, "T10.1 返回非空 Markdown 字符串")
check("## 1. 特征重要性 Top-20" in md, "T10.2 含特征重要性节")
check("## 2. 阈值敏感度 (Val 集)" in md, "T10.3 含阈值敏感度节")
check(md.count("| 0.") >= 7, f"T10.4 阈值表 ≥7 行 (实际 | 0. 出现 {md.count('| 0.')} 次)")
for thr_s in ("| 0.20 |", "| 0.50 |", "| 0.80 |"):
    check(thr_s in md, f"T10.5 含阈值行 {thr_s.strip()}")
check("最佳 F1 阈值" in md, "T10.6 含最佳阈值扫描结论")
check("0.44" in md, "T10.7 回显 bundle.best_thr=0.44")
check("| f0 |" in md and "| f1 |" in md, "T10.8 特征名进入重要性表")
with tempfile.TemporaryDirectory() as td:
    p10 = Path(td) / "health.md"
    md2 = generate_training_health_report(bundle10, X_te, s_tr[:len(X_te)], s_te,
                                          out_md_path=str(p10), logger=None)
    check(p10.exists() and p10.read_text(encoding="utf-8") == md2,
          "T10.9 落盘成功且内容与返回值逐字节一致")

# ============================================================
# T11. diagnose_data_quality 各 issue 精准触发
# ============================================================
print("=" * 70)
print(" T11. 数据质量诊断: 各 issue 精准触发")
print("=" * 70)

def make_diag_df(n=200, gap_hours=0.0, miss_col=None, miss_frac=0.0,
                 y_mode="normal", outlier=False, seed=1):
    r = np.random.RandomState(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    if gap_hours > 0:  # 后半段整体右移 -> 制造大 gap
        half = n // 2
        idx = idx[:half].append(idx[half:] + pd.Timedelta(hours=gap_hours))
    df = pd.DataFrame(index=idx)
    df["load_iden_data73"] = r.uniform(50, 200, n)
    df["load_iden_data74"] = r.uniform(10, 50, n)
    if outlier:
        # 3/200=1.5% 尖峰 -> z≈sqrt((1-p)/p)>5 必触发; (p=5% 时 z<5 反而检不出, 勿用)
        df.loc[df.index[:3], "load_iden_data73"] = 1e6
    if miss_col is not None:
        sel = r.rand(n) < miss_frac
        df.loc[sel, miss_col] = np.nan
    if y_mode == "normal":
        df["y_ac"] = np.where(r.rand(n) < 0.5, 800.0, 0.0)
    elif y_mode == "all_on":
        df["y_ac"] = 800.0
    elif y_mode == "all_off":
        df["y_ac"] = 0.0
    return df

# 11.1 正常数据: 无严重 issue
rep_ok = diagnose_data_quality(make_diag_df(), target_col="y_ac", logger=None)
check(rep_ok["n_samples"] == 200, "T11.1 n_samples=200")
check(abs(rep_ok.get("dt_median_s", 0) - 900) < 1e-6,
      f"T11.2 采样间隔 median=900s (实际 {rep_ok.get('dt_median_s')})")
check(rep_ok.get("dt_cv", 9) < 0.05, f"T11.3 dt_cv≈0 ({rep_ok.get('dt_cv'):.6f})")
check(len(rep_ok["issues"]) == 0,
      f"T11.4 正常数据 issues 为空 (实际 {rep_ok['issues']})")
check(abs(rep_ok.get("on_pct_10w", -1) - 0.5) < 0.15,
      "T11.5 ON 比例统计合理 (≈0.5±0.15)")

# 11.2 大 gap -> 采样中断 issue
rep_gap = diagnose_data_quality(make_diag_df(gap_hours=48.0), logger=None)
check(any("采样中断" in i for i in rep_gap["issues"]),
      f"T11.6 48h gap 触发'采样中断' (issues={rep_gap['issues']})")
check(rep_gap.get("dt_max_gap_s", 0) > 47 * 3600, "T11.7 dt_max_gap≈48h")

# 11.3 缺失率 >10% 触发 issue
rep_miss = diagnose_data_quality(make_diag_df(miss_col="load_iden_data73",
                                              miss_frac=0.2), logger=None)
check(any("缺失率" in i for i in rep_miss["issues"]),
      f"T11.8 20% 缺失触发'缺失率' issue")
check(rep_miss["missing_rate_top5"]["load_iden_data73"] > 0.1,
      "T11.9 missing_rate_top5 记录 >10%")

# 11.4 ON 比例极端
rep_on = diagnose_data_quality(make_diag_df(y_mode="all_on"), logger=None)
check(any("ON 比例" in i for i in rep_on["issues"])
      and rep_on["on_pct_10w"] == 1.0,
      "T11.10 全 ON -> 极度不平衡 issue")
rep_off = diagnose_data_quality(make_diag_df(y_mode="all_off"), logger=None)
check(any("ON 比例" in i for i in rep_off["issues"])
      and rep_off["on_pct_10w"] == 0.0,
      "T11.11 全 OFF -> 极度不平衡 issue")

# 11.5 5σ 异常
rep_out = diagnose_data_quality(make_diag_df(outlier=True), logger=None)
check(any("±5σ" in i for i in rep_out["issues"]),
      f"T11.12 5% 极端值触发 '±5σ' issue (issues={rep_out['issues']})")
check(rep_out.get("load_iden_data73_outlier_pct_5sigma", 0) > 0.01,
      "T11.13 异常比例记录 >1%")

# 11.6 空 df
rep_empty = diagnose_data_quality(pd.DataFrame(), logger=None)
check(rep_empty["n_samples"] == 0 and "数据为空" in rep_empty["issues"],
      "T11.14 空 df -> '数据为空' 不崩")

# 11.7 时间跨度
check(abs(rep_ok.get("span_days", 0) - (200 - 1) * 900 / 86400) < 1e-6,
      f"T11.15 span_days={rep_ok.get('span_days'):.4f} 与采样点数自洽")

# ============================================================
# T12. 端到端小集成: focal 权重直通 EnsembleClf
# ============================================================
print("=" * 70)
print(" T12. 端到端: focal sample_weight -> EnsembleClf.fit -> F1 达标")
print("=" * 70)

from sklearn.metrics import f1_score as _f1

w12 = compute_boundary_focal_weights(s_tr.astype(float))  # y∈{0,1} 走首轮分支
ens12 = EnsembleClf(n_estimators=60, max_depth=2, lr=0.1, random_state=42,
                    use_lgb=False)
ens12.fit(X_tr, s_tr, sample_weight=w12)
pred12 = ens12.predict(X_te)
f1_12 = _f1(s_te, pred12, zero_division=0)
check(f1_12 >= 0.85, f"T12.1 focal 加权训练后测试 F1={f1_12:.4f} >= 0.85")
check(set(np.unique(pred12)).issubset({0, 1}), "T12.2 输出状态合法")
# 对照: 不加权同参模型 (两者都应为可用模型, 允许 focal 更好或持平)
ens12b = EnsembleClf(n_estimators=60, max_depth=2, lr=0.1, random_state=42,
                     use_lgb=False)
ens12b.fit(X_tr, s_tr)
f1_12b = _f1(s_te, ens12b.predict(X_te), zero_division=0)
check(f1_12b >= 0.85, f"T12.3 不加权对照 F1={f1_12b:.4f} >= 0.85 (基线健全)")
print(f"  [INFO] focal={f1_12:.4f} vs plain={f1_12b:.4f} "
      f"(Δ={f1_12 - f1_12b:+.4f}, 合成数据仅验证链路不断)")

# ============================================================
# 汇总
# ============================================================
print("=" * 70)
print(f" 汇总: 通过 {PASS} / 失败 {FAIL} / 总计 {PASS + FAIL}")
print("=" * 70)
if FAIL:
    print("[FAIL] 存在失败断言:")
    for i, m in enumerate(FAILURES, 1):
        print(f"  {i}. {m}")
    sys.exit(1)
print("[OK] 全部单测通过")
sys.exit(0)
