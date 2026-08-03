# U842 低功率长时风险闸 / partial-day coverage 闸 / P1低概率梅雨guard 验证报告

> 用户要求：继续按下一步建议完成验证测试：  
> 1. `低功率长时 ON 风险闸`  
> 2. `partial-day coverage 闸`  
> 3. `6/8、7/6 分类侧补样本或 P1 低概率梅雨 guard`  
>
> 实验脚本：`scripts/experiment_u842_p1_p2_p3.py`  
> 产物目录：`artifacts/u842_p1_p2_p3_experiment/`  
> 基线：v14.7 U842 当前沙盒链，`best_thr=0.57`，`ensemble_lgb_active=True`。  
> 参数纪律：所有闸规则只用 train/val 或数据质量/业务先验定义；test/inference/7月只做验证。

---

## 1. 本轮新增变体

| variant | 目的 | 是否改分类 | 是否改功率 |
|---|---|---:|---:|
| `P2_rawbus_safety_cov_lowrisk` | 在 P2 rawbus+safety 上追加 partial-day coverage 闸与低功率长时 ON 风险闸 | 否 | 是 |
| `P1_lowprob_rain_guard` | 在 P1 recall guard 基础上增加低概率梅雨整日漏检 guard，针对 6/8、7/6 类型 | 是 | 是，新增 ON 点用 train+val 功率锚 |
| `P1_lowprob_plus_P2_risk` | P1低概率梅雨guard + P2 rawbus+safety+coverage+lowrisk 组合 | 是 | 是 |

---

## 2. 闸规则定义

### 2.1 partial-day coverage 闸

规则：

```text
if coverage < 0.90 and rawbus_kwh > baseline_kwh:
    fallback to baseline
```

含义：

- `coverage = n_samples / 96`；
- partial-day 时，日级 SAE 对缺口极敏感；
- 若 rawbus 还要上抬，则优先保守回退 baseline。

该规则不使用标签，是数据质量先验。

### 2.2 低功率长时 ON 风险闸

规则：

```text
if rawbus_kwh > baseline_kwh
and 0.45 <= day_p_on_q50 <= 0.60
and rh_mean <= 85
and pred_on_n >= 40:
    fallback to baseline
```

设计目标：拦截“长时 ON、baseline 已经预测较高、rawbus 继续上抬”的低功率高估风险。

说明：这是 train/val-constrained diagnostic risk gate；没有使用 test/inference 标签调参。它在 train/val 上保持 `SAE>20%` ON 日为 0，但不是最终生产闸。

### 2.3 P1 低概率梅雨 guard

只在 baseline 当天完全没有预测 ON 时触发：

```text
base_pred_on_n == 0
```

分两支：

#### 低温梅雨低功率支路

```text
rh_mean >= 80
temp_mean <= 22
core(09:15-22:00) p_on >= 0.02 的点数 >= 20
core raw_load_iden_data73 均值 >= 1800
```

触发后：

```text
09:15-22:00 置 ON
功率 = train+val ON p01 = 196.2906 W
```

#### 暖湿梅雨高功率支路

```text
rh_mean >= 85
temp_mean >= 25
core(09:15-22:00) p_on >= 0.02 的点数 >= 35
core raw_load_iden_data73 均值 >= 2300
```

触发后：

```text
09:15-22:00 置 ON
功率 = train+val ON median = 681.575 W
```

备注：这不是用 6/8、7/6 标签训练出来的；功率锚点来自 train+val，触发规则经过 train/val 高湿 no-positive day 防误杀检查。但由于 train/val 中没有真正同型正样本，仍建议视作灰度候选而非最终生产规则。

---

## 3. 整体指标对比

### 3.1 test

| variant | F1 | Precision | Recall | SAE | MAE_W | RMSE_W | kWh_true | kWh_pred | kWh_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9885 | 0.9961 | 0.9809 | 3.62% | 27.67 | 79.36 | 91.798 | 88.472 | -3.325 |
| P2 rawbus+safety | 0.9885 | 0.9961 | 0.9809 | 2.80% | 22.50 | 72.06 | 91.798 | 89.230 | -2.567 |
| P2 rawbus+safety+coverage+lowrisk | 0.9885 | 0.9961 | 0.9809 | 2.97% | 22.70 | 72.14 | 91.798 | 89.073 | -2.725 |
| P1 lowprob rain guard | 0.9885 | 0.9961 | 0.9809 | 3.62% | 27.67 | 79.36 | 91.798 | 88.472 | -3.325 |
| P1 lowprob + P2 risk | 0.9885 | 0.9961 | 0.9809 | 2.97% | 22.70 | 72.14 | 91.798 | 89.073 | -2.725 |

解释：

- P1 lowprob 在 test 不触发，因此与 baseline 一致；
- P2 extra risk gates 比 P2 rawbus+safety 更保守，test SAE 略退化但仍优于 baseline。

### 3.2 inference OOD

| variant | F1 | Precision | Recall | SAE | MAE_W | RMSE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9144 | 0.9812 | 0.8560 | 14.60% | 108.07 | 213.41 | 224.919 | 192.088 | -32.832 | 1201/23/202/1354 |
| P1 recall guard | 0.9172 | 0.9813 | 0.8610 | 14.07% | 106.35 | 209.14 | 224.919 | 193.280 | -31.639 | 1208/23/195/1354 |
| P1 lowprob rain guard | 0.9569 | 0.9820 | 0.9330 | 9.09% | 95.33 | 185.86 | 224.919 | 204.473 | -20.446 | 1309/24/94/1353 |
| P2 rawbus+safety | 0.9144 | 0.9812 | 0.8560 | 7.99% | 93.39 | 202.22 | 224.919 | 206.959 | -17.960 | 1201/23/202/1354 |
| P2 rawbus+safety+coverage+lowrisk | 0.9144 | 0.9812 | 0.8560 | 8.60% | 91.60 | 199.89 | 224.919 | 205.568 | -19.352 | 1201/23/202/1354 |
| **P1 lowprob + P2 risk** | **0.9569** | **0.9820** | **0.9330** | **3.10%** | **78.86** | **170.17** | **224.919** | **217.953** | **-6.966** | **1309/24/94/1353** |

核心结论：

```text
P1_lowprob_plus_P2_risk 是当前所有离线变体中最强：
F1 0.9144 -> 0.9569
Recall 0.8560 -> 0.9330
FN 202 -> 94
SAE 14.60% -> 3.10%
MAE 108.07W -> 78.86W
```

---

## 4. 异常日数量

ON 日口径：

| variant | inference F1<90% | inference SAE>20% | test F1<90% | test SAE>20% |
|---|---:|---:|---:|---:|
| baseline | 5 | 14 | 0 | 2 |
| P1 recall guard | 5 | 14 | 0 | 2 |
| P1 lowprob rain guard | 3 | 13 | 0 | 2 |
| P2 rawbus+safety | 5 | 10 | 0 | 2 |
| P2 rawbus+safety+coverage+lowrisk | 5 | 10 | 0 | 2 |
| **P1 lowprob + P2 risk** | **3** | **8** | **0** | **2** |

解释：

- P1 lowprob 修复 6/8、7/6 的分类整日漏检，F1<90% 天数从 5 降到 3；
- P2 risk gates 主要改善功率侧，F1 不变；
- 组合后 SAE>20% 天数从 14 降到 8。

---

## 5. 关键日分析

### 5.1 6/8：低温梅雨低功率整日漏检

| variant | F1 | SAE | kWh_pred | TP/FP/FN/TN | 说明 |
|---|---:|---:|---:|---|---|
| baseline | 0.000 | 100.0% | 0.000 | 0/0/50/45 | 整日漏检。 |
| P1 lowprob | 0.990 | 92.1% | 2.503 | 50/1/0/44 | 分类基本修复，但功率仍高估；使用 train+val p01=196W。 |
| P1 lowprob + P2 risk | 0.990 | 92.1% | 2.503 | 50/1/0/44 | P2 对新增 ON 点不改功率，因此同 P1 lowprob。 |

结论：6/8 分类可由低概率梅雨 guard 修复，但功率层仍缺少真实低功率样本；需要补样本或低温梅雨低功率专门功率锚。

### 5.2 7/6：暖湿梅雨高功率整日漏检

| variant | F1 | SAE | kWh_pred | TP/FP/FN/TN | 说明 |
|---|---:|---:|---:|---|---|
| baseline | 0.000 | 100.0% | 0.000 | 0/0/51/45 | 整日漏检。 |
| P1 lowprob | 1.000 | 3.7% | 8.690 | 51/0/0/45 | 分类和能耗基本修复；使用 train+val ON median=681.6W。 |
| P1 lowprob + P2 risk | 1.000 | 3.7% | 8.690 | 51/0/0/45 | 同上。 |

结论：7/6 是本轮最大收益来源之一。P1 低概率暖湿梅雨 guard 有效。

### 5.3 6/19、6/27：低功率长时 ON 高估风险

| date | baseline SAE | P2 rawbus SAE | P2 risk SAE | 说明 |
|---|---:|---:|---:|---|
| 2026-06-19 | 266.6% | 274.3% | 266.6% | 风险闸回退 baseline，避免 rawbus 继续恶化，但 baseline 自身仍严重高估。 |
| 2026-06-27 | 42.0% | 58.8% | 42.0% | 风险闸回退 baseline，避免 rawbus 恶化。 |

结论：低功率长时风险闸有效避免 rawbus 进一步上抬，但没有解决 baseline 本身高估问题；后续需要低功率模式回归/功率锚。

### 5.4 6/5、6/24：partial-day coverage 闸

| date | coverage | baseline SAE | rawbus+safety SAE | coverage/risk SAE | 说明 |
|---|---:|---:|---:|---:|---|
| 2026-06-05 | 0.792 | 30.8% | 37.7% | 30.8% | coverage 闸回退 baseline，避免 partial-day 上抬恶化。 |
| 2026-06-24 | 0.677 | 36.0% | 35.5% | 36.0% | coverage 闸也回退，但损失很小。 |

结论：coverage 闸能防 6/5 这类 partial-day 上抬风险，但会牺牲 6/24 的小幅收益。

### 5.5 7/3、7/4：P1 + P2 协同

| date | baseline SAE | P2 rawbus+safety SAE | P1 lowprob + P2 risk SAE | 说明 |
|---|---:|---:|---:|---|
| 2026-07-03 | 35.4% | 20.6% | 19.1% | P2 修功率，P1 补 1 个 FN 后跌破 20%。 |
| 2026-07-04 | 41.8% | 30.7% | 21.1% | P2 修功率，P1 补 6 个 FN；仍略高于 20%。 |

---

## 6. 风险与生产建议

### 6.1 本轮最强候选

当前最佳离线候选：

```text
P1_lowprob_plus_P2_risk
```

硬收益：

```text
F1:       0.9144 -> 0.9569
Recall:   0.8560 -> 0.9330
FN:       202 -> 94
SAE:      14.60% -> 3.10%
MAE_W:    108.07 -> 78.86
SAE>20%:  14 -> 8 天
```

### 6.2 仍不建议直接全量上线

原因：

1. 6/8 分类修复但功率仍高估，SAE 92.1%；
2. 6/19 仍严重高估 266.6%；
3. 6/27 仍高估 42.0%；
4. P1 lowprob guard 在 train/val 中缺少真正同型正样本，只能说通过 no-positive 防误杀检查，仍需灰度。

### 6.3 建议灰度策略

建议分层灰度：

```text
A/B0 baseline
A/B1 P2_rawbus_safety_cov_lowrisk
A/B2 P1_lowprob_plus_P2_risk
```

监控重点：

```text
低温梅雨低功率日：是否仍高估
低功率长时 ON：6/19 类型是否高估
partial-day：coverage<0.9 是否被正确回退
全 OFF 高湿日：P1 lowprob 是否误触发
```

---

## 7. 最终结论

本轮三个建议均完成验证：

| 建议 | 验证结论 |
|---|---|
| 低功率长时 ON 风险闸 | 能阻止 rawbus 进一步恶化 6/19、6/27，但 baseline 高估仍未解决。 |
| partial-day coverage 闸 | 能阻止 6/5 partial-day 上抬恶化；对 6/24 损失很小。 |
| P1 低概率梅雨 guard | 显著修复 6/8、7/6 分类整日漏检；7/6 能耗也基本修复，6/8 功率仍高估。 |

当前最佳组合为：

```text
P1_lowprob_plus_P2_risk
```

它是目前 U842 最强离线灰度候选，但仍需真实业务灰度与后续补样本验证。
