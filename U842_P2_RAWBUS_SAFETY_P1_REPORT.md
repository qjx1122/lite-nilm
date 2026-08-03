# U842 P2 rawbus + train/val-only 安全闸 + P1 recall guard 组合灰度验证报告

> 更新说明：后续已完成低功率长时风险闸、partial-day coverage 闸、P1低概率梅雨guard 组合验证，详见 `U842_P2_RISK_COVERAGE_P1_LOWPROB_REPORT.md`。本报告保留上一阶段安全闸+P1组合证据。
>
> 用户要求：继续完成 `P2 rawbus + train/val-only 安全闸 + P1 recall guard` 组合灰度验证。  
> 实验脚本：`scripts/experiment_u842_p1_p2_p3.py`  
> 产物目录：`artifacts/u842_p1_p2_p3_experiment/`  
> 基线：v14.7 U842 当前沙盒链，`best_thr=0.57`，`ensemble_lgb_active=True`。  
> 参数纪律：安全闸只用 train+val 日级标签训练；test/inference/7月只验证，不参与调参。

---

## 1. 本轮验证目标

上一轮 `P2_rawbus_segment_model` 已证明 raw bus/segment 特征有效：

```text
inference SAE: 14.60% -> 7.49%
inference MAE: 108.07W -> 93.22W
SAE>20% ON日: 14 -> 11
```

但仍有风险：

```text
6/19、6/27 低功率长时 ON 高估加重
7/11 从 SAE 15.3% 恶化到 21.9%
6/8、7/6 分类整日漏检无效
```

本轮验证两个新增方案：

| variant | 目标 |
|---|---|
| `P2_rawbus_safety_gate` | 对 `P2_rawbus_segment_model` 增加 train/val-only 日级安全闸，判断当天是否采用 rawbus 输出，否则回退 baseline 功率。 |
| `P1_plus_P2_rawbus_safety` | P1 负责补 recall/FN；P2 rawbus+safety 负责 baseline 已判 ON 点功率。 |

---

## 2. 安全闸设计

### 2.1 标签定义，仅用于 train+val

对每个 train/val ON 日，计算：

```text
base_abs_err = abs(base_kWh_pred - kWh_true)
raw_abs_err  = abs(rawbus_kWh_pred - kWh_true)
label = 1 if raw_abs_err < base_abs_err else 0
```

train+val 标签分布：

| label | 含义 | 天数 |
|---:|---|---:|
| 0 | rawbus 不优于 baseline | 15 |
| 1 | rawbus 优于 baseline | 35 |

未使用 test/inference 标签训练安全闸。

### 2.2 安全闸模型

```text
RandomForestClassifier(
  n_estimators=200,
  max_depth=4,
  min_samples_leaf=2,
  class_weight='balanced',
  random_state=42,
  n_jobs=1
)
```

采用固定阈值：

```text
use_rawbus = P(rawbus_improves) >= 0.50
```

### 2.3 安全闸特征

特征只来自：

```text
baseline 日级预测
rawbus 日级预测
P1/P2 前可获得的天气
原始总线 raw load_iden_data73 日级/ON段统计
coverage / n_samples / baseline pred_on 分布
```

核心字段：

```text
base_kwh, rawbus_kwh, kwh_delta, ratio_delta
n_samples, coverage, pred_on_n
base_on_mean, base_on_std
p_on_mean, p_on_q25/q50/q75
rh_mean, temp_mean
raw73_day_mean/std, raw73_on_mean/std
```

---

## 3. P1 与 P2 分开处理的组合逻辑

组合 variant：`P1_plus_P2_rawbus_safety`

规则：

```text
1. baseline 已判 ON 的点：
   使用 P2_rawbus_safety_gate 的功率输出。

2. P1 新增 ON 的点：
   使用 P1 recall guard 的 guard_power_w。

3. OFF 点：
   仍为 0。
```

因此归因仍然分开：

```text
P1：只改变 state，减少 FN。
P2：只改变 baseline 已判 ON 点功率。
```

---

## 4. 整体指标对比

### 4.1 test

| variant | F1 | SAE | MAE_W | RMSE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9885 | 0.0362 | 27.67 | 79.36 | 91.798 | 88.472 | -3.325 | 514/2/10/914 |
| P2 rawbus | 0.9885 | 0.0249 | 21.95 | 71.81 | 91.798 | 89.509 | -2.289 | 514/2/10/914 |
| P2 rawbus + safety | 0.9885 | 0.0280 | 22.50 | 72.06 | 91.798 | 89.230 | -2.567 | 514/2/10/914 |
| P1 + P2 rawbus + safety | 0.9885 | 0.0280 | 22.50 | 72.06 | 91.798 | 89.230 | -2.567 | 514/2/10/914 |

解释：

- P1 在 test 不触发，因此 `P2 rawbus+safety` 与 `P1+P2 rawbus+safety` 相同。
- 安全闸比 rawbus 纯模型略保守，test SAE 2.49% -> 2.80%，但仍显著优于 baseline 3.62%。

### 4.2 inference OOD

| variant | F1 | Precision | Recall | SAE | MAE_W | RMSE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9144 | 0.9812 | 0.8560 | 0.1460 | 108.07 | 213.41 | 224.919 | 192.088 | -32.832 | 1201/23/202/1354 |
| P1 recall guard | 0.9172 | 0.9813 | 0.8610 | 0.1407 | 106.35 | 209.14 | 224.919 | 193.280 | -31.639 | 1208/23/195/1354 |
| P2 rawbus | 0.9144 | 0.9812 | 0.8560 | 0.0749 | 93.22 | 202.44 | 224.919 | 208.074 | -16.845 | 1201/23/202/1354 |
| P2 rawbus + safety | 0.9144 | 0.9812 | 0.8560 | 0.0799 | 93.39 | 202.22 | 224.919 | 206.959 | -17.960 | 1201/23/202/1354 |
| **P1 + P2 rawbus + safety** | **0.9172** | **0.9813** | **0.8610** | **0.0745** | **91.67** | **197.71** | **224.919** | **208.152** | **-16.768** | **1208/23/195/1354** |

关键结论：

```text
P1 + P2 rawbus + safety 是当前组合中最均衡的灰度候选：
F1 提升、FN 下降、SAE 低于 rawbus 纯模型、MAE 最低。
```

相对 baseline：

```text
F1:       0.9144 -> 0.9172
Recall:   0.8560 -> 0.8610
FN:       202 -> 195
SAE:      14.60% -> 7.45%
MAE_W:    108.07 -> 91.67
kWh_pred: 192.09 -> 208.15
kWh_err:  -32.83 -> -16.77
```

---

## 5. 异常日数量

ON 日口径：

| variant | inference F1<90% | inference SAE>20% | test F1<90% | test SAE>20% |
|---|---:|---:|---:|---:|
| baseline | 5 | 14 | 0 | 2 |
| P1 recall guard | 5 | 14 | 0 | 2 |
| P2 rawbus | 5 | 11 | 0 | 2 |
| P2 rawbus + safety | 5 | 10 | 0 | 2 |
| **P1 + P2 rawbus + safety** | **5** | **9** | **0** | **2** |

解释：

- F1<90% 天数不变，因为 P1 当前只修复 7/3、7/4 等非 F1<90 日的边界 FN；6/8、6/9、6/21、7/5、7/6 仍未全部修复。
- SAE>20% 天数从 14 降到 9，是本轮组合的主要收益。

---

## 6. 安全闸在 inference 上的决策

安全闸主要回退了：

| date | base kWh | rawbus kWh | safety use rawbus | 说明 |
|---|---:|---:|---:|---|
| 2026-06-28 | 9.363 | 9.936 | False | 回退 baseline；此日 rawbus 原本改善，但安全闸保守。 |
| 2026-07-11 | 9.387 | 9.929 | False | 回退 baseline，避免从 SAE 15.3% 恶化到 21.9%。 |

安全闸未能拦住：

| date | 问题 |
|---|---|
| 2026-06-19 | rawbus 将低功率长时 ON 高估从 266.6% 恶化到 274.3%，安全闸仍放行。 |
| 2026-06-27 | rawbus 将 SAE 42.0% 恶化到 58.8%，安全闸仍放行。 |
| 2026-06-05 | partial-day 高估加重，安全闸仍放行。 |

说明：安全闸有效但不充分，仍需低功率长时 ON 专门防护。

---

## 7. 关键异常日明细

### 7.1 明显改善

| date | baseline SAE | P2 rawbus SAE | P1+P2 safety SAE | 说明 |
|---|---:|---:|---:|---|
| 2026-06-09 | 31.7% | 2.7% | 2.7% | 低功率高估基本修复。 |
| 2026-06-18 | 24.1% | 7.5% | 7.5% | 高功率低估大幅缓解。 |
| 2026-06-26 | 33.3% | 10.3% | 10.3% | 高功率低估大幅缓解。 |
| 2026-07-01 | 37.7% | 21.3% | 21.3% | 接近 20%，但仍略超标。 |
| 2026-07-03 | 35.4% | 20.6% | 19.1% | P1 补回 1 个 FN，组合后跌破 20%。 |
| 2026-07-04 | 41.8% | 30.7% | 21.1% | P1 补回 6 个 FN，组合后显著改善但仍略超 20%。 |

### 7.2 仍恶化/未解决

| date | baseline SAE | P1+P2 safety SAE | 问题 |
|---|---:|---:|---|
| 2026-06-05 | 30.8% | 37.7% | partial-day / 低功率日高估加重。 |
| 2026-06-19 | 266.6% | 274.3% | 低功率长时 ON 仍被高估，且安全闸未拦截。 |
| 2026-06-27 | 42.0% | 58.8% | 低/中功率长时 ON 高估加重。 |
| 2026-06-08 | 100.0% | 100.0% | 分类整日漏检，P2 无法作用。 |
| 2026-07-06 | 100.0% | 100.0% | 分类整日漏检，P2 无法作用。 |

---

## 8. 是否建议灰度

### 8.1 推荐结论

**建议作为灰度候选继续验证，不建议直接全量上线。**

理由：

[OK] 收益明确：

```text
SAE: 14.60% -> 7.45%
MAE: 108.07W -> 91.67W
SAE>20% ON日: 14 -> 9
FN: 202 -> 195
```

[WARN] 风险仍明确：

```text
6/19、6/27 低功率长时 ON 高估加重
6/05 partial-day 高估加重
6/08、7/06 分类整日漏检无效
```

### 8.2 下一步安全闸增强方向

当前安全闸只拦住了 7/11，未拦住 6/19/6/27。下一步应增加 train/val-only 低功率长时 ON 风险闸：

```text
若 rawbus 相对 baseline 上抬，且 baseline pred_on_mean 已偏高，
同时 day_p_on 分布/ON段 raw bus 形态接近 train+val 中“高估风险”模式，
则回退 baseline 或降低 blend。
```

还应增加 partial-day 闸：

```text
coverage < 0.9 且 rawbus 上抬时，回退 baseline 或低 blend。
```

这些阈值仍只能从 train/val 选。

---

## 9. 最终结论

本轮组合验证完成后，当前 U842 最佳灰度候选为：

```text
P1_plus_P2_rawbus_safety
```

它相比 baseline：

```text
F1: 0.9144 -> 0.9172
SAE: 14.60% -> 7.45%
MAE: 108.07W -> 91.67W
SAE>20% ON日: 14 -> 9
```

但它仍不是最终可全量上线版本。必须继续补充：

```text
低功率长时 ON 风险闸
partial-day coverage 闸
6/8、7/6 分类侧补样本或 P1 低概率梅雨 guard
```
