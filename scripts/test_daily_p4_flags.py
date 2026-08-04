# -*- coding: utf-8 -*-
"""
[v14.8] daily metrics P4 监控字段单测
======================================

覆盖:
  - coverage_samples_96 / bus_coverage_288 / branch_coverage_96
  - partial_day
  - no_positive_day
  - f1_lt_90_on / sae_gt_20_on 只在 ON 日触发
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import build_daily_metrics_rows  # noqa: E402

PASS = 0
FAIL = 0


def check(cond, name, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


print("=" * 70)
print(" daily P4 flags tests")
print("=" * 70)

# Day1: ON day, 4 samples, poor F1 and high SAE.
# Day2: no-positive day, perfect OFF, should not be reported as F1 failure.
ts = pd.to_datetime([
    "2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:30", "2026-01-01 00:45",
    "2026-01-02 00:00", "2026-01-02 00:15", "2026-01-02 00:30", "2026-01-02 00:45",
])
y_true = np.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y_pred = np.array([0.0, 0.0, 400.0, 400.0, 0.0, 0.0, 0.0, 0.0])
s_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
s_pred = np.array([0, 0, 1, 1, 0, 0, 0, 0])
rows = build_daily_metrics_rows(
    ts, y_true, y_pred, s_true, s_pred,
    split_name="unit", model_name="main", sample_period_h=0.25,
    bus_daily_counts={"2026-01-01": 144, "2026-01-02": 288},
    branch_daily_counts={"2026-01-01": 48, "2026-01-02": 96},
)
by_date = {r["date"]: r for r in rows}
d1 = by_date["2026-01-01"]
d2 = by_date["2026-01-02"]

check("coverage_samples_96" in d1 and "no_positive_day" in d1 and "sae_gt_20_on" in d1,
      "T1 P4 字段存在")
check(abs(d1["coverage_samples_96"] - round(4/96, 6)) < 1e-12, "T2.1 n_samples coverage = 4/96")
check(d1["bus_coverage_288"] == 0.5, "T2.2 bus coverage = 144/288")
check(d1["branch_coverage_96"] == 0.5, "T2.3 branch coverage = 48/96")
check(d1["partial_day"] == 1, "T2.4 partial_day=1 for 4/96 samples")
check(d1["is_on_day"] == 1 and d1["f1_lt_90_on"] == 1 and d1["sae_gt_20_on"] == 1,
      "T3 ON 日异常 flags 触发")
check(d2["no_positive_day"] == 1 and d2["f1_lt_90_on"] == 0 and d2["sae_gt_20_on"] == 0,
      "T4 no-positive day 不触发 ON 日异常 flags")
check(d2["partial_day"] == 1 and d2["bus_coverage_288"] == 1.0 and d2["branch_coverage_96"] == 1.0,
      "T5 原始采集满采但对齐样本少: raw coverage=1, sample partial=1")

print("=" * 70)
print(f"[SUMMARY] PASS={PASS}, FAIL={FAIL}")
if FAIL:
    sys.exit(1)
print("[OK] daily P4 flags tests passed")
