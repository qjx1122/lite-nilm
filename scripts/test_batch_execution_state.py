# -*- coding: utf-8 -*-
"""
批量执行状态 CSV + 断点续跑 单元测试 (v13.17)
====================================================

覆盖:
  T1. _load_execution_state: 文件不存在 → 返回空表, 有列头, 不崩
  T2. _load_execution_state: 文件损坏 (缺关键列) → 返回空表 + WARN
  T3. _load_execution_state: 老格式 (缺新列) → 补齐列名, 不崩 (向后兼容)
  T4. _load_execution_state: 正常读取 → 行数/内容正确
  T5. _get_completed_users: retry_failed=True (默认) → 只跳 ok/soft_skip
  T6. _get_completed_users: retry_failed=False → 也跳 fail
  T7. _upsert_execution_state: 首次写入 → 文件生成, 列齐全, 值正确
  T8. _upsert_execution_state: 同 user_id 再写 → 覆盖旧行 (upsert 语义)
  T9. _upsert_execution_state: 原子性 → 中断时 .tmp 不会污染主 CSV
  T10. 端到端: 写 3 用户 → 崩溃恢复 → resume 只跑剩下的
  T11. resume 处理 fail: retry_failed=True 时 fail 会被重跑
  T12. CSV 格式硬校验: 9 列名 / utf-8-sig / 可读回

运行:
  python scripts/test_batch_execution_state.py
退出码: 0 = 全通过
"""
import sys
import tempfile
import os
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 直接 import 目标模块的函数 (从 run_batch_users.py)
from run_batch_users import (
    _load_execution_state, _get_completed_users, _upsert_execution_state,
    _execution_state_path, _EXECUTION_STATE_CSV_NAME, _EXECUTION_STATE_COLS,
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
# T1. 文件不存在
# ============================================================
print("=" * 70)
print(" T1. _load_execution_state: 文件不存在 → 空表")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    df = _load_execution_state(td)
    check(len(df) == 0, f"T1.1 空表 (行数 {len(df)})")
    check(list(df.columns) == _EXECUTION_STATE_COLS,
          f"T1.2 列头齐全: {list(df.columns)}")


# ============================================================
# T2. 文件损坏 (缺关键列)
# ============================================================
print()
print("=" * 70)
print(" T2. _load_execution_state: 文件损坏 → 空表 (自动降级)")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / _EXECUTION_STATE_CSV_NAME
    p.write_text("some,random,columns\na,b,c\n", encoding="utf-8-sig")  # 缺 user_id/status
    df = _load_execution_state(td)
    check(len(df) == 0, "T2.1 损坏文件 → 空表")
    check(list(df.columns) == _EXECUTION_STATE_COLS, "T2.2 列头齐全")


# ============================================================
# T3. 老格式 (缺新列, 但有 user_id/status)
# ============================================================
print()
print("=" * 70)
print(" T3. _load_execution_state: 老格式 → 补齐列 (向后兼容)")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / _EXECUTION_STATE_CSV_NAME
    # 老格式: 只有 user_id 和 status
    p.write_text("user_id,status\nuser_A,ok\nuser_B,fail\n", encoding="utf-8-sig")
    df = _load_execution_state(td)
    check(len(df) == 2, f"T3.1 读到 2 行 (实际 {len(df)})")
    check(all(c in df.columns for c in _EXECUTION_STATE_COLS),
          f"T3.2 补齐所有 {len(_EXECUTION_STATE_COLS)} 列")


# ============================================================
# T4. 正常读取
# ============================================================
print()
print("=" * 70)
print(" T4. _load_execution_state: 正常读取")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    # 用 upsert 写入 3 行, 再读回
    _upsert_execution_state(td, {"user_id": "u1", "status": "ok", "success": True})
    _upsert_execution_state(td, {"user_id": "u2", "status": "soft_skip", "success": False})
    _upsert_execution_state(td, {"user_id": "u3", "status": "fail", "success": False})
    df = _load_execution_state(td)
    check(len(df) == 3, f"T4.1 3 行 (实际 {len(df)})")
    check(set(df["user_id"]) == {"u1", "u2", "u3"}, "T4.2 3 个 user_id")
    check(df[df["user_id"] == "u1"].iloc[0]["status"] == "ok", "T4.3 u1 status=ok")


# ============================================================
# T5/T6. _get_completed_users: retry_failed 分支
# ============================================================
print()
print("=" * 70)
print(" T5/T6. _get_completed_users retry_failed 分支")
print("=" * 70)

df_mix = pd.DataFrame([
    {"user_id": "u_ok", "status": "ok"},
    {"user_id": "u_soft", "status": "soft_skip"},
    {"user_id": "u_fail", "status": "fail"},
])
# T5: 默认 (retry_failed=True) → 只跳 ok+soft_skip
completed_5 = _get_completed_users(df_mix, retry_failed=True)
check(completed_5 == {"u_ok", "u_soft"},
      f"T5.1 retry_failed=True → {completed_5} (期望 {{u_ok, u_soft}})")
# T6: retry_failed=False → 也跳 fail
completed_6 = _get_completed_users(df_mix, retry_failed=False)
check(completed_6 == {"u_ok", "u_soft", "u_fail"},
      f"T6.1 retry_failed=False → {completed_6} (期望 3 项全含)")

# 空 df 边界
check(_get_completed_users(pd.DataFrame(columns=["user_id", "status"])) == set(),
      "T5.2 空 df → 空 set")
check(_get_completed_users(None) == set(), "T5.3 None → 空 set")


# ============================================================
# T7. _upsert_execution_state: 首次写入
# ============================================================
print()
print("=" * 70)
print(" T7. _upsert_execution_state: 首次写入")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    _upsert_execution_state(td, {
        "user_id": "user_X",
        "status": "ok",
        "success": True,
        "started_at": "2026-07-15 10:00:00",
        "finished_at": "2026-07-15 10:01:00",
        "duration_s": 60.0,
        "message": "成功",
        "target_col": "p1",
        "run_id": "20260715_100000",
    })
    p = _execution_state_path(td)
    check(p.exists(), f"T7.1 CSV 已生成 {p.name}")
    df = pd.read_csv(p, encoding="utf-8-sig")
    check(len(df) == 1, f"T7.2 1 行 (实际 {len(df)})")
    r = df.iloc[0]
    check(r["user_id"] == "user_X", f"T7.3 user_id 正确")
    check(r["status"] == "ok", f"T7.4 status=ok")
    check(bool(r["success"]) is True, f"T7.5 success=True (实际 {r['success']!r})")
    check(float(r["duration_s"]) == 60.0, f"T7.6 duration_s=60")
    check(r["message"] == "成功", f"T7.7 中文 message 正常 (utf-8-sig)")
    check(list(df.columns) == _EXECUTION_STATE_COLS,
          f"T7.8 列顺序稳定: {list(df.columns)}")


# ============================================================
# T8. upsert 覆盖 (同 user_id 二次写入)
# ============================================================
print()
print("=" * 70)
print(" T8. _upsert_execution_state: 同 user_id 覆盖")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    _upsert_execution_state(td, {"user_id": "user_X", "status": "fail", "success": False,
                                    "message": "第一次失败"})
    _upsert_execution_state(td, {"user_id": "user_X", "status": "ok", "success": True,
                                    "message": "第二次成功"})
    df = pd.read_csv(_execution_state_path(td), encoding="utf-8-sig")
    check(len(df) == 1, f"T8.1 覆盖后仍 1 行 (upsert, 实际 {len(df)})")
    check(df.iloc[0]["status"] == "ok", "T8.2 最新 status=ok (覆盖旧 fail)")
    check(df.iloc[0]["message"] == "第二次成功", "T8.3 message 已更新")


# ============================================================
# T9. 原子性: .tmp 不污染主 CSV
# ============================================================
print()
print("=" * 70)
print(" T9. 原子写: 主 CSV 完整 (无 .tmp 残留)")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    _upsert_execution_state(td, {"user_id": "u1", "status": "ok"})
    p = _execution_state_path(td)
    tmp = p.with_suffix(p.suffix + ".tmp")
    check(p.exists(), "T9.1 主 CSV 存在")
    check(not tmp.exists(), f"T9.2 .tmp 已清理 (无 {tmp.name} 残留)")
    # 读回验证完整
    df = pd.read_csv(p, encoding="utf-8-sig")
    check(len(df) == 1, "T9.3 读回完整")


# ============================================================
# T10. 端到端: 崩溃恢复
# ============================================================
print()
print("=" * 70)
print(" T10. 端到端: 5 用户跑 3 个后崩溃 → resume 只跑剩 2 个")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    # 模拟第 1 次跑: 5 用户中前 3 个跑完 (2 ok + 1 fail) 就崩溃
    _upsert_execution_state(td, {"user_id": "u1", "status": "ok",  "success": True})
    _upsert_execution_state(td, {"user_id": "u2", "status": "ok",  "success": True})
    _upsert_execution_state(td, {"user_id": "u3", "status": "fail","success": False})
    # 崩溃 (u4, u5 未跑)

    # 第 2 次: resume 加载
    df = _load_execution_state(td)
    check(len(df) == 3, "T10.1 崩溃后状态文件仍有 3 行")

    # 默认 retry_failed=True: 应跳 u1, u2 (u3 fail 会重跑)
    all_users = {"u1", "u2", "u3", "u4", "u5"}
    completed = _get_completed_users(df, retry_failed=True)
    to_run = all_users - completed
    check(to_run == {"u3", "u4", "u5"},
          f"T10.2 retry_failed=True 待跑 = {to_run} (期望 u3+u4+u5)")

    # retry_failed=False: 也跳 u3
    completed2 = _get_completed_users(df, retry_failed=False)
    to_run2 = all_users - completed2
    check(to_run2 == {"u4", "u5"},
          f"T10.3 retry_failed=False 待跑 = {to_run2} (期望 u4+u5)")


# ============================================================
# T11. resume 重跑 fail 用户后, 状态被更新为 ok
# ============================================================
print()
print("=" * 70)
print(" T11. resume 覆盖: fail → ok 更新")
print("=" * 70)

with tempfile.TemporaryDirectory() as td:
    _upsert_execution_state(td, {"user_id": "u3", "status": "fail", "success": False,
                                    "message": "第 1 次 fail"})
    # 模拟 resume 后重跑成功
    _upsert_execution_state(td, {"user_id": "u3", "status": "ok", "success": True,
                                    "message": "重跑成功"})
    df = _load_execution_state(td)
    r = df[df["user_id"] == "u3"].iloc[0]
    check(r["status"] == "ok", "T11.1 u3 状态更新为 ok")
    check(bool(r["success"]) is True, "T11.2 success=True")
    # 再检查完成集合含 u3
    completed = _get_completed_users(df)
    check("u3" in completed, "T11.3 重跑后 u3 进入已完成集合")


# ============================================================
# T12. CSV 格式硬校验
# ============================================================
print()
print("=" * 70)
print(" T12. CSV 格式硬校验 (9 列 / utf-8-sig / 可读回)")
print("=" * 70)

check(len(_EXECUTION_STATE_COLS) == 9,
      f"T12.1 状态 CSV 恰 9 列 (实际 {len(_EXECUTION_STATE_COLS)})")
expected = ["user_id", "status", "success", "started_at", "finished_at",
            "duration_s", "message", "target_col", "run_id"]
check(_EXECUTION_STATE_COLS == expected,
      f"T12.2 列名/顺序: {_EXECUTION_STATE_COLS}")

with tempfile.TemporaryDirectory() as td:
    _upsert_execution_state(td, {"user_id": "中文用户_测试",
                                    "status": "ok",
                                    "message": "含中文 & 特殊字符 |, \""})
    df = pd.read_csv(_execution_state_path(td), encoding="utf-8-sig")
    check(df.iloc[0]["user_id"] == "中文用户_测试", "T12.3 utf-8-sig 中文 user_id 正常")
    check("中文" in str(df.iloc[0]["message"]), "T12.4 中文 message 正常")


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
