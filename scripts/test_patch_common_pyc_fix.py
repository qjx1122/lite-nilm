# -*- coding: utf-8 -*-
"""
patch_common / restore_common 陈旧 .pyc 竞态修复 单元测试 (v14.6)
====================================================================

背景 (2026-08-03 实证, U2844 事故):
  CPython 校验 .pyc 只比 (source mtime 秒级, source size)。
  run_user_pipeline.main() 在 patch_common 之前会先
  `from common import ON_THR_W` (无 --common-overrides 用户必经),
  若该 import 重编译出的 pyc 与随后 patch_common 的写入落在同一秒
  (p1->p2 为同尺寸编辑), 02_align_and_feat.py 等子进程会读到
  **陈旧** TARGET_COL —— U2844 (配置 target_col=p2) 批跑实测按 p1
  产出 (对齐日志 峰值892W/零样本72.3% 为 p1 指纹; p2 应为 899W/77.2%)。

修复: patch_common / restore_common 写入后 _purge_common_pyc()
      删除 scripts/__pycache__/common.*.pyc。

覆盖:
  T1. patch_common('p2'): 磁盘写入生效 + .bak 建立 + common.*.pyc 被清除
  T2. 同秒碰撞构造: patch 后强制 common.py mtime 与既有 pyc 记录秒对齐,
      子进程 import 仍必须读到新值 'p2' (无修复时读到 'p1' = 回归)
  T3. composite patch_common('p1+p2'): 子进程读到 'p1+p2'
  T4. restore_common: 恢复原文 + 删除 .bak + 子进程读到 'p1'
  T5. _purge_common_pyc 幂等: 反复调用/无 __pycache__ 目录均不炸

运行:
  python scripts/test_patch_common_pyc_fix.py
退出码: 0 = 全通过
"""
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_user_pipeline import patch_common, restore_common, _purge_common_pyc

PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent
COMMON = SCRIPTS / "common.py"
BAK = SCRIPTS / "common.py.bak"

PASS = 0
FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}  {detail}")


def subprocess_target():
    """子进程 import common 返回 TARGET_COL (模拟 02 子进程视角)"""
    out = subprocess.run(
        [PY, "-c",
         "import sys; sys.path.insert(0, r'%s'); import common; print(common.TARGET_COL)"
         % str(SCRIPTS)],
        capture_output=True, text=True, cwd=str(SCRIPTS.parent))
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def fresh_common_pyc():
    """确保当前 common.py 被编译成 pyc (返回 pyc 头里的 source_mtime)"""
    p = subprocess_target()  # 触发编译
    pycs = list((SCRIPTS / "__pycache__").glob("common.cpython-*.pyc"))
    if pycs:
        hdr = pycs[0].read_bytes()[:16]
        return p, struct.unpack("<I", hdr[8:12])[0]
    return p, None


def pyc_files():
    d = SCRIPTS / "__pycache__"
    return list(d.glob("common.*.pyc")) if d.is_dir() else []


try:
    # ---------- 预备: 确认原始状态 ----------
    orig = COMMON.read_text(encoding="utf-8")
    assert 'TARGET_COL = "p1"' in orig, "前置: common.py 应处于原始 p1 状态"

    # ---------- T1: patch_common 基本功能 + pyc 清除 ----------
    print("T1. patch_common('p2') 基本功能")
    fresh_common_pyc()          # 先造出 pyc (模拟 in-process import 重编译)
    assert pyc_files(), "前置: 应存在 common pyc"
    patch_common("p2")
    check("T1.1 磁盘 TARGET_COL = p2",
          'TARGET_COL = "p2"' in COMMON.read_text(encoding="utf-8"))
    check("T1.2 .bak 已建立", BAK.exists())
    check("T1.3 common.*.pyc 已被清除", not pyc_files())

    # ---------- T2: 同秒同尺寸碰撞 ----------
    print("T2. 同秒碰撞: mtime 对齐后子进程仍须读到 p2")
    m = int(time.time())
    os.utime(COMMON, (m, m))   # 模拟 patch 写入与既有 pyc 记录同秒
    time.sleep(0.05)
    val = subprocess_target()
    check("T2.1 子进程读到 p2", val == "p2", f"读到 {val!r} (无修复时为 'p1')")

    # ---------- T3: composite 目标 ----------
    print("T3. patch_common('p1+p2')")
    patch_common("p1+p2")      # bak 已存在时从 bak 重放替换
    val = subprocess_target()
    check("T3.1 子进程读到 p1+p2", val == "p1+p2", f"读到 {val!r}")

    # ---------- T4: restore_common ----------
    print("T4. restore_common 恢复原文")
    restore_common()
    check("T4.1 磁盘恢复 p1", COMMON.read_text(encoding="utf-8") == orig)
    check("T4.2 .bak 已删除", not BAK.exists())
    val = subprocess_target()
    check("T4.3 子进程读到 p1", val == "p1", f"读到 {val!r}")

    # ---------- T5: _purge_common_pyc 幂等 ----------
    print("T5. _purge_common_pyc 幂等")
    try:
        _purge_common_pyc(); _purge_common_pyc()
        check("T5.1 重复调用不炸", True)
    except Exception as e:
        check("T5.1 重复调用不炸", False, repr(e))

finally:
    # 无论如何恢复现场
    if BAK.exists():
        restore_common()
    else:
        COMMON.write_text(COMMON.read_text(encoding="utf-8"), encoding="utf-8")
    _purge_common_pyc()

print(f"\n{'='*56}")
print(f"结果: {PASS} 通过 / {FAIL} 失败")
if FAILURES:
    for f_ in FAILURES:
        print("  [FAIL]", f_)
sys.exit(0 if FAIL == 0 else 1)
