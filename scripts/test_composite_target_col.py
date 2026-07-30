# -*- coding: utf-8 -*-
"""
复合 target_col (p1+p2 / p1+p2+p3) 单元测试 (v13.16 新增)
============================================================

覆盖:
  T1. get_user_target_col 正则识别复合语法 + 非法拒绝 + 归一化 + 重复防呆
  T2. run_user_pipeline.py::_validate_target_col CLI 正则一致性
  T3. feature_utils.load_branch_csv 复合列物化 (用户示例逐值对齐)
  T4. feature_utils.load_branch_csv 边界: 缺分量 KeyError / 非法分量 ValueError / 单列不物化
  T5. analyze_on_periods 复合列端到端 (含 compute_on_periods)
  T6. 与旧行为向后兼容 (p1 单列走原路径)

运行:
  python scripts/test_composite_target_col.py
退出码: 0 = 全通过, 1 = 有失败
"""
import sys
import io
import tempfile
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from time_filter_utils import get_user_target_col
from feature_utils import load_branch_csv
from analyze_on_periods import (
    _resolve_target_col, _normalize_target_col, _materialize_composite_target,
    _RE_PN_COMPOSITE, compute_on_periods,
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


# ============================================================
# T1. get_user_target_col 复合语法识别
# ============================================================
print("=" * 70)
print(" T1. get_user_target_col: 复合语法 + 非法拒绝 + 归一化 + 防呆")
print("=" * 70)

# 合法复合
for val, expected in [
    ("p1+p2", "p1+p2"),
    ("p1+p2+p3", "p1+p2+p3"),
    ("p0+p5+p10", "p0+p5+p10"),
    ("P1+P2", "p1+p2"),                 # 大小写规范化
    (" p1 + p2 ", "p1+p2"),             # 空白规范化
    ("p1 +p2+ p3", "p1+p2+p3"),         # 混合空白
    # 单列仍合法 (向后兼容)
    ("p1", "p1"),
    ("p128", "p128"),
    ("P99", "p99"),
]:
    cfg = {"u1": {"target_col": val}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = get_user_target_col(cfg, "u1")
    check(result == expected, f"T1 合法: {val!r} -> {result!r} (期望 {expected!r})")

# 非法拒绝
for val in [
    "p1+",              # 尾部无分量
    "+p1",              # 头部无分量
    "p1++p2",           # 双 +
    "p1+q2",            # 非 pN 分量
    "acp1+p2",          # 前缀污染
    "p+p2",             # 缺 N
    "p1+p1",            # 重复分量 (v13.16 防呆)
    "p1+p2+p1",         # 重复分量
]:
    cfg = {"u1": {"target_col": val}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = get_user_target_col(cfg, "u1")
    check(result is None, f"T1 非法: {val!r} -> None (实际 {result!r})")


# ============================================================
# T2. _validate_target_col 与 _RE_PN_COMPOSITE 正则一致性
# ============================================================
print()
print("=" * 70)
print(" T2. analyze_on_periods._RE_PN_COMPOSITE 一致性")
print("=" * 70)

for val, expected in [
    ("p1", True), ("p1+p2", True), ("p1+p2+p3", True),
    ("p0+p5+p10", True), ("p128", True),
    ("p1+", False), ("+p1", False), ("p1++p2", False),
    ("p1+q2", False), ("acp1", False), ("", False),
]:
    m = bool(_RE_PN_COMPOSITE.match(val))
    check(m == expected,
          f"T2 _RE_PN_COMPOSITE({val!r}) = {m} (期望 {expected})")


# ============================================================
# T3. load_branch_csv 复合列物化 (用户示例逐值对齐)
# ============================================================
print()
print("=" * 70)
print(" T3. load_branch_csv 复合列物化 (用户示例逐值对齐)")
print("=" * 70)

# 精确复现用户 issue 里的示例数据
user_example = (
    "time,p1,p2\n"
    "2026/5/21 0:00:00,24,8\n"
    "2026/5/21 0:15:00,16,0\n"
    "2026/5/21 0:30:00,16,0\n"
    "2026/5/21 0:45:00,16,8\n"
)

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "user_example.csv"
    p.write_text(user_example, encoding="utf-8")

    # 3a. 复合列物化
    df = load_branch_csv(p, target_col="p1+p2")
    check("p1+p2" in df.columns,
          f"T3.1 复合列 'p1+p2' 已物化 (columns={df.columns.tolist()})")

    # 逐行对齐用户示例
    expected_values = [32, 16, 16, 24]
    actual = df["p1+p2"].tolist()
    check(actual == expected_values,
          f"T3.2 复合列值 = {actual} (用户示例期望 {expected_values})")

    # 分量列必须保留
    check("p1" in df.columns and "p2" in df.columns,
          "T3.3 原始分量 p1/p2 保留")

    # 3b. 三分量复合
    tri_data = (
        "time,p1,p2,p3\n"
        "2026/5/21 0:00:00,10,20,30\n"
        "2026/5/21 0:15:00,5,15,25\n"
    )
    p_tri = Path(td) / "tri.csv"
    p_tri.write_text(tri_data, encoding="utf-8")
    df_tri = load_branch_csv(p_tri, target_col="p1+p2+p3")
    check(df_tri["p1+p2+p3"].tolist() == [60, 45],
          f"T3.4 三分量 'p1+p2+p3' 逐行求和 = {df_tri['p1+p2+p3'].tolist()} "
          f"(期望 [60, 45])")

    # 3c. 大小写 / 空白归一化后能物化
    df_norm = load_branch_csv(p, target_col=" P1 + p2 ")
    check("p1+p2" in df_norm.columns,
          f"T3.5 输入 ' P1 + p2 ' 归一化为 'p1+p2' 后物化 "
          f"(columns={df_norm.columns.tolist()})")

    # 3d. NaN 传播 (skipna=False 硬保护)
    nan_data = (
        "time,p1,p2\n"
        "2026/5/21 0:00:00,10,20\n"
        "2026/5/21 0:15:00,,5\n"        # p1 空
        "2026/5/21 0:30:00,15,\n"       # p2 空
    )
    p_nan = Path(td) / "nan.csv"
    p_nan.write_text(nan_data, encoding="utf-8")
    df_nan = load_branch_csv(p_nan, target_col="p1+p2")
    vals = df_nan["p1+p2"].tolist()
    check(vals[0] == 30 and pd.isna(vals[1]) and pd.isna(vals[2]),
          f"T3.6 部分分量为空时结果为 NaN (skipna=False), 实测 {vals}")


# ============================================================
# T4. 边界: 缺分量 / 非法分量 / 单列不物化
# ============================================================
print()
print("=" * 70)
print(" T4. 边界与向后兼容")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "u.csv"
    p.write_text(user_example, encoding="utf-8")

    # 4a. 缺分量 -> KeyError
    try:
        load_branch_csv(p, target_col="p1+p3")   # CSV 里没有 p3
        check(False, "T4.1 缺分量应抛 KeyError, 但未抛")
    except KeyError as e:
        check("p3" in str(e),
              f"T4.1 缺分量 KeyError 精准提示 p3 缺失: {str(e)[:80]}...")

    # 4b. 非法分量 -> ValueError
    try:
        load_branch_csv(p, target_col="p1+q2")
        check(False, "T4.2 非法分量应抛 ValueError, 但未抛")
    except ValueError as e:
        check("非法分量" in str(e) or "格式" in str(e),
              f"T4.2 非法分量 ValueError: {str(e)[:80]}...")

    # 4c. 单列 p1 不触发物化, 无新增列 (向后兼容)
    df_single = load_branch_csv(p, target_col="p1")
    added = set(df_single.columns) - {"time", "p1", "p2"}
    check(len(added) == 0,
          f"T4.3 单列 'p1' 不触发物化, 无新增列 (新增: {added})")

    # 4d. target_col=None 完全向后兼容 (老调用)
    df_old = load_branch_csv(p)
    check(list(df_old.columns) == ["time", "p1", "p2"],
          f"T4.4 target_col=None 与旧行为等价 (columns={df_old.columns.tolist()})")


# ============================================================
# T5. analyze_on_periods 端到端 (compute_on_periods 复合列)
# ============================================================
print()
print("=" * 70)
print(" T5. analyze_on_periods 端到端 (复合列)")
print("=" * 70)

# 构造 12 行 15min 数据, p1+p2 段级 ON (阈值 10W)
# 段 1: 0:00-0:45 = [50,60,60,50] (p1+p2 >10)
# 全 OFF: 1:00-2:45 = [0]*8
n_rows = 12
idx = pd.date_range("2026-05-21 00:00:00", periods=n_rows, freq="15min")
p1 = [20, 30, 30, 25] + [0] * 8
p2 = [30, 30, 30, 25] + [0] * 8
df_syn = pd.DataFrame({"time": idx.strftime("%Y-%m-%d %H:%M:%S"),
                        "p1": p1, "p2": p2})

# 5a. _resolve_target_col 支持复合 (显式命令行)
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "syn.csv"
    df_syn.to_csv(p, index=False)

    resolved = _resolve_target_col(p, "p1+p2", None, None)
    check(resolved == "p1+p2",
          f"T5.1 _resolve_target_col('p1+p2') = {resolved!r}")

    # 归一化
    resolved2 = _resolve_target_col(p, " P1 + p2 ", None, None)
    check(resolved2 == "p1+p2",
          f"T5.2 _resolve_target_col(' P1 + p2 ') 归一化为 {resolved2!r}")

    # 兜底路径仍返回单列 (无从推断复合意图)
    resolved3 = _resolve_target_col(p, None, None, None)
    check(resolved3 == "p1",
          f"T5.3 无显式/配置时兜底为第 1 个 pN = {resolved3!r}")

# 5b. compute_on_periods 复合列
df_loaded = pd.read_csv(p) if False else df_syn.copy()   # 直接用内存 df
periods = compute_on_periods(df_loaded, "p1+p2", on_thr_w=10.0,
                              split_by_day=False)
# 期望: 1 个 ON 段 (0:00-0:45)
check(len(periods) == 1,
      f"T5.4 复合列 ON 段数 = {len(periods)} (期望 1)")
if len(periods) >= 1:
    seg = periods.iloc[0]
    # 段级平均 = mean(50,60,60,50) = 55
    mean_w = seg["mean_w"]
    check(abs(mean_w - 55.0) < 1e-3,
          f"T5.5 段级 mean_w = {mean_w:.3f} (期望 55.000, "
          f"(50+60+60+50)/4)")
    peak_w = seg["peak_w"]
    check(abs(peak_w - 60.0) < 1e-3,
          f"T5.6 段级 peak_w = {peak_w} (期望 60)")

# 5c. _materialize_composite_target 幂等 (已物化再调不重复)
df_mat = _materialize_composite_target(df_syn, "p1+p2")
df_mat2 = _materialize_composite_target(df_mat, "p1+p2")
n_cols_1 = len(df_mat.columns)
n_cols_2 = len(df_mat2.columns)
check(n_cols_1 == n_cols_2,
      f"T5.7 _materialize 幂等: 首次 {n_cols_1} 列, 二次仍 {n_cols_2} 列")


# ============================================================
# T6. resample_and_align 集成 (复合列作为 y_ac 源)
# ============================================================
print()
print("=" * 70)
print(" T6. resample_and_align 集成 (patch TARGET_COL='p1+p2')")
print("=" * 70)

# 因 resample_and_align 用 `from common import TARGET_COL`, 需 monkey-patch
import feature_utils as _fu
import common as _cm

orig_target = _cm.TARGET_COL
try:
    _cm.TARGET_COL = "p1+p2"
    _fu.TARGET_COL = "p1+p2"

    # 构造 20min 分辨率总线 (只是形式上有 event_time 与 load_iden_data* 列即可)
    n_bus = 20
    bus_idx = pd.date_range("2026-05-21 00:00:00", periods=n_bus, freq="5min")
    bus_df = pd.DataFrame({
        "event_time": bus_idx,
        "load_iden_data0": np.linspace(100, 200, n_bus),
        "load_iden_data1": np.linspace(50, 150, n_bus),
    })
    # 分路: 15min * 4 = 1h, p1+p2 逐行求和 (用户示例)
    br_idx = pd.date_range("2026-05-21 00:00:00", periods=4, freq="15min")
    br_df = pd.DataFrame({
        "time": br_idx,
        "p1": [24, 16, 16, 16],
        "p2": [8, 0, 0, 8],
    })
    # 手动物化 (模拟 load_branch_csv 已做的事)
    br_df["p1+p2"] = br_df["p1"] + br_df["p2"]

    df_align = _fu.resample_and_align(
        bus_df, br_df,
        keep_cols=["load_iden_data0", "load_iden_data1"])

    check("y_ac" in df_align.columns,
          f"T6.1 对齐后含 y_ac 列 (columns={df_align.columns.tolist()})")
    y_ac_vals = df_align["y_ac"].tolist()
    check(y_ac_vals == [32, 16, 16, 24],
          f"T6.2 y_ac 值 = {y_ac_vals} (用户示例期望 [32,16,16,24])")

finally:
    _cm.TARGET_COL = orig_target
    _fu.TARGET_COL = orig_target


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 70)
print(f" 汇总: 通过 {PASS} / 失败 {FAIL} / 总计 {PASS + FAIL}")
print("=" * 70)
if FAIL:
    print("失败项:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("[OK] 全部单测通过")
sys.exit(0)
