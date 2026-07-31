# -*- coding: utf-8 -*-
"""
[v14.3] 特征黑名单 + 训推特征对齐机制 单测

覆盖:
  A. parse_exclude_features_env  — env 解析契约 (8 组)
  B. align_features_to_bundle    — 推理侧对齐 (9 组)
  C. 03_train 黑名单 drop 语义    — 通过真实 env + 函数组合复现 (4 组)

运行: python scripts/test_feature_blacklist_v143.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from feature_utils import parse_exclude_features_env, align_features_to_bundle

PASS, FAIL = 0, []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
    else:
        FAIL.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}: {detail}")


class _Log:
    def __init__(self):
        self.infos, self.warns = [], []
    def info(self, m): self.infos.append(m)
    def warning(self, m): self.warns.append(m)


print("=" * 70)
print("A. parse_exclude_features_env")
print("=" * 70)
check("A1 空串 -> []", parse_exclude_features_env("") == [])
check("A2 None -> []", parse_exclude_features_env(None) == [])
check("A3 单值", parse_exclude_features_env("dow") == ["dow"])
check("A4 多值+空白", parse_exclude_features_env(" dow , is_weekend ") == ["dow", "is_weekend"])
check("A5 尾逗号/空项丢弃", parse_exclude_features_env("dow,,") == ["dow"])
check("A6 全空白串 -> []", parse_exclude_features_env("   ") == [])
check("A7 中文逗号兼容", parse_exclude_features_env("dow，is_weekend") == ["dow", "is_weekend"],
      str(parse_exclude_features_env("dow，is_weekend")))
check("A8 不 split 特征名内的下划线", parse_exclude_features_env("temp_day_max,dow") == ["temp_day_max", "dow"])

print("\n" + "=" * 70)
print("B. align_features_to_bundle")
print("=" * 70)
X = pd.DataFrame({"a": [1., 2.], "b": [3., 4.], "c": [5., 6.]})
b_ok = {"feat_names": ["a", "b", "c"]}
b_extra = {"feat_names": ["b", "a"]}          # 训练剔除了 c, 且列序不同
b_missing = {"feat_names": ["a", "b", "c", "zzz"]}
lg = _Log()

out = align_features_to_bundle(X, b_ok, logger=lg, ctx="T")
check("B1 全同 -> 值不变", out.values.tolist() == X.values.tolist())
check("B2 全同 -> 列序不变", list(out.columns) == ["a", "b", "c"])
check("B3 全同 -> 无 WARN 无 INFO", lg.infos == [] and lg.warns == [])

lg2 = _Log()
out2 = align_features_to_bundle(X, b_extra, logger=lg2, ctx="T")
check("B4 多列剔除(drop c)", list(out2.columns) == ["b", "a"])
check("B5 多余剔除日志", any("多出列" in m for m in lg2.infos), str(lg2.infos))
res_video = out2.values.tolist()
check("B6 剔除后数值正确", res_video == [[3., 1.], [4., 2.]], str(res_video))

lg3 = _Log()
out3 = align_features_to_bundle(X, b_missing, logger=lg3, ctx="T")
check("B7 缺失列补 0", out3["zzz"].tolist() == [0.0, 0.0])
check("B8 缺失 WARN", any("缺失" in m for m in lg3.warns), str(lg3.warns))

check("B9 bundle 无 feat_names -> 直通", align_features_to_bundle(X, {}) is X)

print("\n" + "=" * 70)
print("C. 黑名单端到端语义 (env -> drop -> bundle -> align 复现 03/05 契约)")
print("=" * 70)
# 复现 03_train.py 内部逻辑: env 解析 -> 命中剔除 -> feat_names 记录
Xfull = pd.DataFrame({
    "load_iden_data1": [1., 2., 3.], "hour": [9., 10., 11.],
    "dow": [0., 1., 2.], "sin_doy": [0.1, 0.2, 0.3],
})
os.environ["NILM_EXCLUDE_FEATURES"] = "dow,is_weekend"
excl = parse_exclude_features_env(os.environ.get("NILM_EXCLUDE_FEATURES", ""))
hit = [c for c in excl if c in Xfull.columns]
miss = [c for c in excl if c not in Xfull.columns]
Xtr = Xfull.drop(columns=hit)
check("C1 命中列被剔除", hit == ["dow"] and "dow" not in Xtr.columns)
check("C2 未命中不影响", miss == ["is_weekend"] and "hour" in Xtr.columns)
feat_names = Xtr.columns.tolist()
# 推理侧: build_features 仍然产生 dow (无黑名单概念) -> align 必须剔除
Xinf = pd.DataFrame({
    "load_iden_data1": [4., 5.], "hour": [12., 13.],
    "dow": [3., 3.], "sin_doy": [0.4, 0.5],
})
lg4 = _Log()
Xinf_al = align_features_to_bundle(Xinf, {"feat_names": feat_names}, logger=lg4, ctx="C")
check("C3 推理侧 dow 被对齐剔除", list(Xinf_al.columns) == feat_names, str(list(Xinf_al.columns)))
check("C4 对齐后行数/数值不丢", Xinf_al.values.tolist() == [[4., 12., 0.4], [5., 13., 0.5]],
      str(Xinf_al.values.tolist()))

del os.environ["NILM_EXCLUDE_FEATURES"]

print("\n" + "=" * 70)
print(f"结果: {PASS} PASS / {PASS + len(FAIL)} 总")
if FAIL:
    print("失败明细:")
    for f_ in FAIL:
        print(f"  [FAIL] {f_}")
    sys.exit(1)
print("[OK] 全部单测通过")
