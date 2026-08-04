# 5用户 P0→P4 优先级优化验证实验报告

> 基于当前版本 5 用户批跑结果（`logs/_batch/batch_run_20260804_070114.log`），按上一轮建议的 P0→P4 顺序做验证。  
> 实验脚本：`scripts/experiment_5user_priority_remediation.py`  
> 实验产物：`artifacts/five_user_priority_remediation/`  
> 本轮代码修改：P4 日报监控字段已进入 `scripts/metrics_utils.py`，并已通过 5 用户 force-retrain 批跑验证，`inference_daily_metrics.csv` 现包含 `coverage_samples_96 / bus_coverage_288 / branch_coverage_96 / partial_day / no_positive_day / f1_lt_90_on / sae_gt_20_on`。  
> 重要说明：P0/P2/P3 中的 branch-off / low-true guards 仍使用 inference branch 标签作为**诊断上限**，不是可直接生产的规则；生产化必须用 train/val-only 的 bus/branch 一致性信号选型。

---

## 1. 顺序优化定义

| 阶段 | variant | 动作 | 目的 |
|---|---|---|---|
| B0 | `B0_baseline_current` | 当前批跑主路径 | 对照基线 |
| P0 | `P0_branch_off_U0789_U0800` | 对 U0789/U0800 做 oracle branch-off：真实 OFF 点强制关断 | 量化电路一致性/force-off 守卫上限 |
| P1 | `P1_add_U842_focused` | 加入 U842 P0-P3 focused 优化版本 | 验证 U842 专项修复对整体的贡献 |
| P2 | `P2_add_U0778_off_guard` | 对 U0778 做 oracle branch-off | 量化 U0778 OFF 虚块修复上限 |
| P3 | `P3_add_U2844_lowtrue_fp_guard` | 对 U2844 低真值高 FP 日移除 FP 点 | 量化 U2844 low-true/high-FP 守卫收益 |
| P4 | `P4_reporting_flags` | 增加 coverage/no-positive/off-FP 日级标记 | 报表监控增强，不改变指标 |

---

## 2. ALL 用户池化 inference 指标

| variant | F1 | Precision | Recall | SAE | MAE_W | kWh_true | kWh_pred | kWh_err | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 baseline | 0.849 | 0.774 | 0.941 | 10.08% | 145.98 | 1113.10 | 1225.33 | +112.23 | 6004 | 1750 | 378 | 8568 |
| P0 U0789/U0800 branch-off | 0.951 | 0.961 | 0.941 | 7.59% | 99.73 | 1113.10 | 1028.60 | -84.51 | 6004 | 241 | 378 | 10077 |
| P1 + U842 focused | 0.965 | 0.961 | 0.968 | 4.49% | 92.83 | 1113.10 | 1063.08 | -50.02 | 6180 | 250 | 202 | 10068 |
| P2 + U0778 off guard | 0.977 | 0.987 | 0.968 | 6.46% | 87.69 | 1113.10 | 1041.16 | -71.94 | 6180 | 84 | 202 | 10234 |
| P3 + U2844 lowtrue-FP guard | 0.980 | 0.993 | 0.968 | 7.09% | 86.03 | 1113.10 | 1034.23 | -78.87 | 6180 | 46 | 202 | 10272 |
| P4 reporting flags | 0.980 | 0.993 | 0.968 | 7.09% | 86.03 | 1113.10 | 1034.23 | -78.87 | 6180 | 46 | 202 | 10272 |

### 2.1 主要结论

- P0 是最大分类收益来源：FP 从 1750 降到 241，说明 U0789/U0800 的主要问题确实是 OFF/非目标点误报虚块。
- P1 加入 U842 focused 修复后，FN 从 378 降到 202，Recall 从 0.941 提升到 0.968。
- P2/P3 继续压 FP：最终 FP=46，Precision=0.993。
- 但最终总量 SAE 从 10.08% 到 7.09%，不是单调改善；原因是 P0/P2/P3 oracle force-off 移除大量预测电量后，部分用户从高估转为低估。说明总量 SAE 不能单独作为判断依据，必须结合日级和用户级指标。

---

## 3. 分用户最终 P4 指标

| user | F1 | Precision | Recall | SAE | MAE_W | kWh_true | kWh_pred | kWh_err | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U842 | 0.979 | 0.977 | 0.981 | 0.74% | 66.58 | 224.92 | 226.57 | +1.65 | 1377 | 32 | 26 | 1345 |
| U2844 | 0.918 | 0.980 | 0.864 | 5.77% | 60.82 | 127.34 | 134.68 | +7.34 | 680 | 14 | 107 | 1695 |
| U0778 | 0.996 | 1.000 | 0.991 | 9.23% | 118.00 | 247.94 | 225.07 | -22.88 | 1570 | 0 | 14 | 1872 |
| U0789 | 0.998 | 1.000 | 0.995 | 10.10% | 157.28 | 449.11 | 403.76 | -45.36 | 1491 | 0 | 7 | 2342 |
| U0800 | 0.978 | 1.000 | 0.957 | 30.79% | 21.31 | 63.79 | 44.15 | -19.64 | 1062 | 0 | 48 | 3018 |

---

## 4. 每用户优化效果与问题分析

## 4.1 U842 — P1 focused 修复有效

| 指标 | baseline | P4 |
|---|---:|---:|
| F1 | 0.914 | 0.979 |
| Recall | 0.856 | 0.981 |
| FN | 202 | 26 |
| SAE | 14.60% | 0.74% |
| MAE_W | 108.07 | 66.58 |

原因：P0-P3 focused guards 修复了 6/8、6/21、7/5、7/6 等梅雨/尾段 FN 与功率锚问题。

剩余问题：6/19、6/27 低功率长时 ON baseline 高估仍是未生产化问题；已有低功率长时识别器验证表明当前特征不足，需要补样本或非空调基线分解。

方案：将 U842 P0-P3 guards 纳入 train/val-only 泛化选型；6/19/6/27 继续做低功率长时 cap/anchor + 更强风险识别器。

---

## 4.2 U2844 — low-true high-FP 守卫有效，但 Recall 未修

| 指标 | baseline | P4 |
|---|---:|---:|
| F1 | 0.895 | 0.918 |
| Precision | 0.929 | 0.980 |
| Recall | 0.864 | 0.864 |
| FP | 52 | 14 |
| FN | 107 | 107 |
| SAE | 11.21% | 5.77% |

原因：P3 对 U2844 低真值高 FP 日移除 false-positive 点，显著提高 precision 和 SAE；但 6/19、6/22、7/5 等 FN/Recall 问题仍未解决。

方案：下一步做 U2844 专属 recall guard / bus_guard 滑窗锚，重点修 6月 OOD FN；低真值高 FP 日继续用 bus/branch 一致性闸，但生产参数需 train/val 选型。

---

## 4.3 U0778 — OFF 虚块可修，但高温低估暴露

| 指标 | baseline | P4 |
|---|---:|---:|
| F1 | 0.946 | 0.996 |
| Precision | 0.904 | 1.000 |
| FP | 166 | 0 |
| SAE | 0.38% | 9.23% |
| kWh_err | -0.95 | -22.88 |

原因：P2 oracle branch-off 移除了 OFF/非目标点 FP，分类变得几乎完美；但此前总量 SAE 低是“FP 高估”和“ON 日低估”互相抵消。移除 FP 后，高温/ON 日低估暴露。

方案：生产上先做 OFF 日一致性守卫，再单独做高温/档位上抬；不能只看总量 SAE。高温参数需从 train/val 或训练侧统计推导。

---

## 4.4 U0789 — P0 证明电路一致性守卫收益巨大

| 指标 | baseline | P4 |
|---|---:|---:|
| F1 | 0.799 | 0.998 |
| Precision | 0.668 | 1.000 |
| FP | 742 | 0 |
| SAE | 26.55% | 10.10% |
| kWh_err | +119.26 | -45.36 |

原因：U0789 的主问题是大量 OFF/非目标点被预测为 ON，属于电路归属/双开状态/非目标负荷混淆。P0 oracle branch-off 直接移除 FP 后，分类几乎完美。

方案：最高优先级上线候选是 P-CE1 电路一致性守卫 / branch absence force-off。但不能用 inference 标签，需构造 train/val 可用的 bus/branch 一致性信号，并加防误杀门。功率层仍有 10.1% 低估，需要后续双状态 per-mode regressor。

---

## 4.5 U0800 — FP 可清零，但功率低估严重

| 指标 | baseline | P4 |
|---|---:|---:|
| F1 | 0.723 | 0.978 |
| Precision | 0.581 | 1.000 |
| FP | 767 | 0 |
| SAE | 19.57% | 30.79% |
| kWh_err | +12.48 | -19.64 |

原因：U0800 baseline 大量 OFF 日虚块导致 FP=767；P0 清除 FP 后分类大幅改善，但原先高估抵消了 ON 日低估。清除 FP 后，真实 ON 日功率低估成为主问题。

方案：短期需要 force-off/一致性守卫 + 高温档位上抬同时做。U0800 训练样本少，必须补充 OFF 日和高温 ON 日；否则只做 force-off 会导致总量低估。

---

## 5. P4 报表增强

新增日级监控字段：

```text
coverage = n_samples / 96
no_positive_day = TP+FP+FN == 0
off_day_fp = kWh_true<=0.01 and FP>0
f1_lt_90_on
sae_gt_20_on
```

目的：避免把 no-positive day 的 F1=0 误解为失败，并把 partial-day / OFF FP 单独分桶。

---

## 6. 生产化优先级

| 优先级 | 动作 | 用户 |
|---|---|---|
| P0 | 训练/验证通用电路一致性 / force-off 守卫 | U0789、U0800，兼顾 U0778 |
| P1 | 将 U842 P0-P3 focused guards 泛化选型 | U842 |
| P2 | OFF 虚块守卫后做高温/档位补偿 | U0778、U0800 |
| P3 | bus_guard 滑窗锚 + recall guard | U2844 |
| P4 | coverage/no-positive/off-FP 日报字段 | 全用户 |

---

## 7. 结论

本轮 P0→P4 依次验证证明：

1. **通用电路一致性/force-off 是 5 用户当前最大收益项**，尤其 U0789/U0800；
2. **U842 focused guards 离线收益显著**，但还需泛化选型；
3. **U0778/U0800 的总量 SAE 具有抵消假象**，清除 FP 后会暴露 ON 日低估；
4. **U2844 的 FP 可部分修，但 Recall 问题仍需单独 guard**；
5. **P4 报表字段是必要基础设施**，否则无法区分 no-positive / partial-day / OFF FP。

最终诊断上限：

```text
5用户 pooled F1: 0.849 -> 0.980
pooled Precision: 0.774 -> 0.993
pooled FP: 1750 -> 46
pooled MAE: 145.98W -> 86.03W
```

但 pooled SAE 从 10.08% 到 7.09%，不是最大幅改善，原因是多个用户的 FP 高估被清除后，ON 日低估暴露。因此后续必须把“分类一致性守卫”和“功率档位补偿”成对设计。
