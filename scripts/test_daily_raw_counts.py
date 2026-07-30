# -*- coding: utf-8 -*-
"""
daily metrics n_bus_raw / n_branch_raw 列单元测试 (v13.16 追加)
==================================================================

覆盖:
  T1. build_daily_metrics_rows 输出行含 n_bus_raw / n_branch_raw 字段 (位置在 n_samples 之后)
  T2. 传入 bus_daily_counts + branch_daily_counts 时值正确, 缺失日期 = 0
  T3. 不传 (None) 时两列 = "" 空字符串, 向后兼容
  T4. compute_raw_daily_counts: 从 CSV 按天 group 计数正确 (无 time_filter)
  T5. compute_raw_daily_counts: 应用 time_filter_spec 后过滤生效
  T6. compute_raw_daily_counts: CSV 不存在 / 缺时间列 → 返回 {} 不崩
  T7. 端到端: rows 里 n_bus_raw 和 n_branch_raw 与 compute_raw_daily_counts 一致
  T8. save_daily_metrics_csv 落盘后新列可读回

运行:
  python scripts/test_daily_raw_counts.py
退出码: 0 = 全通过
"""
import sys
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_utils import (build_daily_metrics_rows, save_daily_metrics_csv,
                           compute_raw_daily_counts)

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
# 构造 2 天 15min 合成数据: 96 pts/day
# ============================================================
def make_synth():
    n = 96 * 2   # 2 天
    idx = pd.date_range("2026-05-21 00:00:00", periods=n, freq="15min")
    y_true = np.array([100.0] * n)
    y_pred = np.array([95.0] * n)
    s_true = np.ones(n, dtype=int)
    s_pred = np.ones(n, dtype=int)
    return idx, y_true, y_pred, s_true, s_pred


# ============================================================
# T1. 输出行含 n_bus_raw / n_branch_raw 字段, 位置正确
# ============================================================
print("=" * 70)
print(" T1. build_daily_metrics_rows 输出含新字段 + 位置")
print("=" * 70)

idx, yt, yp, st, sp = make_synth()
bus_cnt = {"2026-05-21": 288, "2026-05-22": 250}
br_cnt  = {"2026-05-21": 96,  "2026-05-22": 90}

rows = build_daily_metrics_rows(
    idx, yt, yp, st, sp, split_name="test",
    bus_daily_counts=bus_cnt, branch_daily_counts=br_cnt)

check(len(rows) == 2, f"T1.1 行数 = {len(rows)} (期望 2 天)")
if rows:
    keys = list(rows[0].keys())
    check("n_bus_raw" in keys, f"T1.2 含 n_bus_raw 字段 (keys={keys[:8]}...)")
    check("n_branch_raw" in keys, "T1.3 含 n_branch_raw 字段")
    # 位置校验: 在 n_samples 之后, Accuracy 之前
    if "n_samples" in keys and "Accuracy" in keys:
        i_ns = keys.index("n_samples")
        i_br = keys.index("n_bus_raw")
        i_bc = keys.index("n_branch_raw")
        i_ac = keys.index("Accuracy")
        check(i_ns < i_br < i_bc < i_ac,
              f"T1.4 位置: n_samples ({i_ns}) < n_bus_raw ({i_br}) < "
              f"n_branch_raw ({i_bc}) < Accuracy ({i_ac})")


# ============================================================
# T2. 传入 counts 时值正确, 缺失日期 = 0
# ============================================================
print()
print("=" * 70)
print(" T2. 传入 counts 值正确 + 缺失日期 = 0")
print("=" * 70)

bus_cnt_partial = {"2026-05-21": 288}  # 只有 21 日
rows2 = build_daily_metrics_rows(
    idx, yt, yp, st, sp, split_name="test",
    bus_daily_counts=bus_cnt_partial, branch_daily_counts=br_cnt)
r21 = [r for r in rows2 if r["date"] == "2026-05-21"][0]
r22 = [r for r in rows2 if r["date"] == "2026-05-22"][0]
check(r21["n_bus_raw"] == 288, f"T2.1 5-21 n_bus_raw = {r21['n_bus_raw']} (期望 288)")
check(r22["n_bus_raw"] == 0,   f"T2.2 5-22 (缺配置) n_bus_raw = {r22['n_bus_raw']} (期望 0)")
check(r21["n_branch_raw"] == 96,
      f"T2.3 5-21 n_branch_raw = {r21['n_branch_raw']} (期望 96)")
check(r22["n_branch_raw"] == 90,
      f"T2.4 5-22 n_branch_raw = {r22['n_branch_raw']} (期望 90)")


# ============================================================
# T3. 不传 (None) 时两列 = "" 空字符串
# ============================================================
print()
print("=" * 70)
print(" T3. 不传 counts 时两列 = '' (向后兼容)")
print("=" * 70)

rows3 = build_daily_metrics_rows(idx, yt, yp, st, sp, split_name="test")
r = rows3[0]
check(r["n_bus_raw"] == "",
      f"T3.1 不传时 n_bus_raw = {r['n_bus_raw']!r} (期望 '')")
check(r["n_branch_raw"] == "",
      f"T3.2 不传时 n_branch_raw = {r['n_branch_raw']!r} (期望 '')")


# ============================================================
# T4. compute_raw_daily_counts: 无 time_filter
# ============================================================
print()
print("=" * 70)
print(" T4. compute_raw_daily_counts 基础计数")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    # 造总线 CSV: event_time, 3 天 5min = 288*3 = 864 行, 但故意让 5-22 只有 200 行
    rows_bus = []
    # Day 1: 5-21, 288 pts (完整)
    d1 = pd.date_range("2026-05-21 00:00:00", periods=288, freq="5min")
    for t in d1:
        rows_bus.append({"event_time": t.strftime("%Y/%m/%d %H:%M:%S"),
                         "load_iden_data0": 100})
    # Day 2: 5-22, 只有 200 pts
    d2 = pd.date_range("2026-05-22 00:00:00", periods=200, freq="5min")
    for t in d2:
        rows_bus.append({"event_time": t.strftime("%Y/%m/%d %H:%M:%S"),
                         "load_iden_data0": 100})
    # Day 3: 5-23, 288 pts
    d3 = pd.date_range("2026-05-23 00:00:00", periods=288, freq="5min")
    for t in d3:
        rows_bus.append({"event_time": t.strftime("%Y/%m/%d %H:%M:%S"),
                         "load_iden_data0": 100})
    p_bus = Path(td) / "bus.csv"
    pd.DataFrame(rows_bus).to_csv(p_bus, index=False, encoding="utf-8")

    counts = compute_raw_daily_counts(p_bus, "event_time")
    check(counts.get("2026-05-21") == 288,
          f"T4.1 5-21 count = {counts.get('2026-05-21')} (期望 288)")
    check(counts.get("2026-05-22") == 200,
          f"T4.2 5-22 count = {counts.get('2026-05-22')} (期望 200)")
    check(counts.get("2026-05-23") == 288,
          f"T4.3 5-23 count = {counts.get('2026-05-23')} (期望 288)")
    check(len(counts) == 3, f"T4.4 覆盖天数 = {len(counts)} (期望 3)")


    # ============================================================
    # T5. time_filter_spec 应用后过滤生效
    # ============================================================
    print()
    print("=" * 70)
    print(" T5. compute_raw_daily_counts 应用 time_filter_spec")
    print("=" * 70)

    # spec: exclude 5-22 全天 → 结果里应无 5-22
    spec = json.dumps({"exclude": [["2026-05-22", "2026-05-22"]]})
    counts_f = compute_raw_daily_counts(p_bus, "event_time",
                                         time_filter_spec=spec)
    check("2026-05-22" not in counts_f,
          f"T5.1 exclude 5-22 后不含该日 (keys={list(counts_f.keys())})")
    check(counts_f.get("2026-05-21") == 288,
          f"T5.2 5-21 保留 = {counts_f.get('2026-05-21')}")
    check(counts_f.get("2026-05-23") == 288,
          f"T5.3 5-23 保留 = {counts_f.get('2026-05-23')}")


# ============================================================
# T6. CSV 不存在 / 缺时间列 兜底
# ============================================================
print()
print("=" * 70)
print(" T6. 兜底路径不崩")
print("=" * 70)

fake = "/tmp/_never_exists_xxx.csv"
c = compute_raw_daily_counts(fake, "event_time")
check(c == {}, f"T6.1 CSV 不存在 → 返回空字典 (实际 {c})")

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "no_time_col.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    c = compute_raw_daily_counts(p, "event_time")
    check(c == {}, f"T6.2 缺时间列 → 返回空字典 (实际 {c})")


# ============================================================
# T7. 端到端: build_daily_metrics_rows + compute_raw_daily_counts 联动
# ============================================================
print()
print("=" * 70)
print(" T7. 端到端联动: rows 数字与 counts 数字一致")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    # 2 天分路 15min 数据: 5-21 96 pts, 5-22 50 pts (残缺)
    rows_br = []
    for t in pd.date_range("2026-05-21 00:00:00", periods=96, freq="15min"):
        rows_br.append({"time": t.strftime("%Y/%m/%d %H:%M:%S"), "p1": 100})
    for t in pd.date_range("2026-05-22 00:00:00", periods=50, freq="15min"):
        rows_br.append({"time": t.strftime("%Y/%m/%d %H:%M:%S"), "p1": 100})
    p_br = Path(td) / "br.csv"
    pd.DataFrame(rows_br).to_csv(p_br, index=False, encoding="utf-8")

    br_counts = compute_raw_daily_counts(p_br, "time")
    check(br_counts.get("2026-05-21") == 96, "T7.1 分路 5-21=96")
    check(br_counts.get("2026-05-22") == 50, "T7.2 分路 5-22=50")

    idx7 = pd.date_range("2026-05-21 00:00:00", periods=192, freq="15min")
    yt7 = np.array([100.0] * 192)
    rows7 = build_daily_metrics_rows(
        idx7, yt7, yt7 * 0.95, np.ones(192, int), np.ones(192, int),
        split_name="test",
        bus_daily_counts={}, branch_daily_counts=br_counts)
    r21 = [r for r in rows7 if r["date"] == "2026-05-21"][0]
    r22 = [r for r in rows7 if r["date"] == "2026-05-22"][0]
    check(r21["n_branch_raw"] == 96,
          f"T7.3 daily rows 5-21 n_branch_raw = {r21['n_branch_raw']} (期望 96)")
    check(r22["n_branch_raw"] == 50,
          f"T7.4 daily rows 5-22 n_branch_raw = {r22['n_branch_raw']} (期望 50)")
    # 空 bus 字典 → 每天 = 0
    check(r21["n_bus_raw"] == 0 and r22["n_bus_raw"] == 0,
          f"T7.5 空 bus 字典时 n_bus_raw = 0 for all days")


# ============================================================
# T8. save_daily_metrics_csv 落盘后新列可读回
# ============================================================
print()
print("=" * 70)
print(" T8. save_daily_metrics_csv 落盘 + 读回")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    p_out = Path(td) / "daily.csv"
    save_daily_metrics_csv(rows, p_out)
    check(p_out.exists(), "T8.1 CSV 已生成")
    dd = pd.read_csv(p_out, encoding="utf-8-sig")
    check("n_bus_raw" in dd.columns,
          f"T8.2 读回 CSV 含 n_bus_raw (列: {dd.columns.tolist()[:8]}...)")
    check("n_branch_raw" in dd.columns, "T8.3 读回 CSV 含 n_branch_raw")
    check(int(dd[dd["date"] == "2026-05-21"]["n_bus_raw"].iloc[0]) == 288,
          "T8.4 读回值一致 5-21 n_bus_raw = 288")


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
