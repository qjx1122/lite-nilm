# -*- coding: utf-8 -*-
"""
postprocess.search_best_threshold 阈值稳定化单元测试 (v14.7)
====================================================================

背景:
  U842 Windows/沙盒同为 EnsembleClf+LightGBM active, 但验证集概率微漂移导致
  raw best_thr 在 0.57 vs 0.74 间跳档；推理集大量真实 ON 位于该概率带内,
  小阈值漂移被 OOD 指标放大为 FN/kWh_pred 大差异。

修复:
  search_best_threshold 在 raw_best_fbeta 的验证集 one/two-sample 容忍带内做
  稳定化 tie-break；beta>=1 时优先 Recall, beta<1 时优先 Precision。
  容忍带只由 val 样本数推导, 不使用推理/7月评估集。

覆盖:
  T1. beta=1: raw 最优为高阈值, 但低阈值在容忍带内且 Recall 更高 -> 选低阈值
  T2. stable_tiebreak=False 保留旧 raw argmax 行为
  T3. beta<1: 近似同分时保持 precision-oriented, 选高阈值
  T4. 完全同分平台阶: P/R/F 均相同 -> beta>=1 选左端低阈值
  T5. curve 输出追加混淆矩阵列 + raw/selected 元数据自洽

运行:
  python scripts/test_postprocess_threshold_stability.py
退出码: 0 = 全通过
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess import search_best_threshold  # noqa: E402

PASS = 0
FAIL = 0
FAILURES = []


def check(cond, name, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}  {detail}")


def make_near_tie_data():
    """构造 n=1000 的两阈值近似同分样本。

    thr=0.70: TP=498 FP=0 FN=2  -> Precision 更高, raw F1 略优
    thr=0.50: TP=500 FP=3 FN=0  -> Recall 更高, F1 仅低约 9.87e-4
    默认 tol=min(0.002,2/n)=0.002, 因此两者视为统计近似同分。
    """
    y = np.r_[np.ones(500, dtype=int), np.zeros(500, dtype=int)]
    p = np.r_[
        np.full(498, 0.80),  # positives kept by both thresholds
        np.full(2, 0.60),    # positives only kept by low threshold
        np.full(3, 0.60),    # negatives only admitted by low threshold
        np.full(497, 0.10),
    ]
    return p, y


print("=" * 70)
print(" postprocess.search_best_threshold v14.7 阈值稳定化测试")
print("=" * 70)

p, y = make_near_tie_data()
grid = np.array([0.50, 0.70])

res = search_best_threshold(p, y, beta=1.0, thr_grid=grid,
                            min_on=1, fill_short_off=0)
check(res["raw_best_thr"] == 0.70,
      "T1.1 raw argmax 原本为高阈值 0.70",
      str(res))
check(res["best_thr"] == 0.50,
      "T1.2 beta=1 稳定化在近似同分内优先 Recall -> 低阈值 0.50",
      str(res))
check(res["selected_recall"] > res["raw_best_recall"],
      "T1.3 selected Recall 高于 raw best")
check(0.0 < res["raw_best_fbeta"] - res["selected_fbeta"] <= res["threshold_stability_tol"],
      "T1.4 selected F1 低于 raw best 但处于容忍带内",
      f"raw={res['raw_best_fbeta']}, selected={res['selected_fbeta']}, tol={res['threshold_stability_tol']}")

res_raw = search_best_threshold(p, y, beta=1.0, thr_grid=grid,
                                min_on=1, fill_short_off=0,
                                stable_tiebreak=False)
check(res_raw["best_thr"] == 0.70 and res_raw["threshold_stability_tol"] == 0.0,
      "T2 stable_tiebreak=False 保留旧 raw argmax")

res_p = search_best_threshold(p, y, beta=0.5, thr_grid=grid,
                              min_on=1, fill_short_off=0)
check(res_p["best_thr"] == 0.70,
      "T3 beta<1 保持 precision-oriented, 近似同分选高精度高阈值",
      str(res_p))

# 两个阈值预测完全相同: P/R/F 完全平台阶, beta>=1 选左端低阈值。
y2 = np.r_[np.ones(10, dtype=int), np.zeros(10, dtype=int)]
p2 = np.r_[np.full(10, 0.80), np.full(10, 0.20)]
res_plateau = search_best_threshold(p2, y2, beta=1.0,
                                    thr_grid=np.array([0.50, 0.60]),
                                    min_on=1, fill_short_off=0)
check(res_plateau["best_thr"] == 0.50 and res_plateau["raw_best_thr"] == 0.50,
      "T4 完全同分平台阶取左端低阈值")

first_row = res["curve"][0]
check(all(k in first_row for k in ["tn", "fp", "fn", "tp"]),
      "T5.1 threshold_curve 追加 tn/fp/fn/tp 列")
check(res["threshold_selection_policy"] == "stable_recall_first_within_val_tolerance"
      and res["threshold_candidate_count"] == 2,
      "T5.2 threshold selection 元数据自洽")
check(res["post_min_on"] == 1 and res["post_fill_short_off"] == 0,
      "T5.3 后处理参数透传")

print("=" * 70)
print(f"[SUMMARY] PASS={PASS}, FAIL={FAIL}")
if FAIL:
    for msg in FAILURES:
        print(f"  - {msg}")
    sys.exit(1)
print("[OK] postprocess threshold stability tests passed")
