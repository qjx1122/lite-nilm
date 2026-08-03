# U842 P2 日级 loss-aware mode model 验证报告

> 用户要求：  
> 1. 给 P2 加入日级 loss-aware 训练目标；  
> 2. 增加模式判别特征；  
> 3. 与 P1 分开处理；  
> 4. 继续遵守约束。  
>
> 实验脚本：`scripts/experiment_u842_p1_p2_p3.py`  
> 产物目录：`artifacts/u842_p1_p2_p3_experiment/`  
> 基线：v14.7 U842 当前沙盒链，`best_thr=0.57`，`ensemble_lgb_active=True`。  
> 结论先行：**P2 loss-aware 版本在 train/val 与 test 上改善明显，但 inference OOD 明显退化；不建议上线。当前最稳健的 P2 候选仍是上一轮 `P2_mode_classifier_regressor`，而非 loss-aware 版本。**

---

## 1. 约束与实验纪律

本轮严格遵守：

| 约束 | 执行情况 |
|---|---|
| 参数只能从 train/val 推导 | 是。模式阈值、特征、候选模型、loss-aware 目标均只用 train/val。 |
| test 只作验证 | 是。test 未参与参数选择。 |
| inference / 7 月 OOD 只作最终验证 | 是。未用 inference/7月反向选阈值、特征或模型。 |
| P2 与 P1 分开 | 是。P2 只改已判 ON 点的功率，不改 `state_pred_main`；P1 recall guard 作为独立 variant 保留。 |

---

## 2. P2 loss-aware 新增内容

### 2.1 日级 loss-aware 目标

候选模型选择时，使用 train-only fit + val objective，最终再用 train+val 重训选中候选。

目标函数：

```text
objective =
  1.0   * point_MAE_W
+ 80.0  * overall_SAE
+ 40.0  * mean_daily_SAE
+ 25.0  * p95_daily_SAE
+ 200.0 * SAE_gt_20_day_rate
```

目的：不只优化逐点 MAE，也惩罚日级 SAE 和 `SAE>20%` 日。

### 2.2 候选空间

候选数量：54 个。

搜索维度：

```text
n_estimators: 160 / 240
classifier max_depth / regressor max_depth / min_leaf: 3 组
sample_weight:
  daily_sae_weight: 0 / 2 / 5
  bad_day_weight:   0 / 4 / 10
blend: 1.0 / 0.85 / 0.70
```

注意：虽然候选包含日级 sample weight，但最终 val objective 选中的候选为：

```text
daily_sae_weight = 0
bad_day_weight = 0
blend = 0.85
```

这说明：在 train→val 验证下，显式加权日级坏日并没有被选中；loss-aware 主要体现在候选选择目标，而不是最终 sample_weight。

### 2.3 选中候选

```json
{
  "n_estimators": 160,
  "clf_max_depth": 8,
  "reg_max_depth": 10,
  "clf_min_leaf": 6,
  "reg_min_leaf": 6,
  "daily_sae_weight": 0,
  "bad_day_weight": 0,
  "blend": 0.85
}
```

### 2.4 模式标签

模式标签仍来自 train+val 真 ON 样本三分位：

| mode | 规则 | 样本数 |
|---:|---|---:|
| low | `y_true_W <= 564.29W` | 660 |
| mid | `564.29W < y_true_W <= 747.70W` | 660 |
| high | `y_true_W > 747.70W` | 658 |

---

## 3. 新增模式判别特征

上一版 P2 已有：

```text
p_on_main, y_pred_W_main, y_pred_low/high
hour_sin/hour_cos/dow
逐点 temperature / apparent_temperature / humidity
day_pred_on_n, day_pred_on_mean, day_pred_kwh
day_p_on_mean/q25/q50/q75
temp_mean, rh_mean
```

本轮新增：

### 3.1 日内分段特征

```text
day_pred_on_mean_morning / midday / evening
day_pred_kwh_morning / midday / evening
```

目的：区分上午强、下午弱，或全天低功率/高功率模式。

### 3.2 日级低置信/高置信计数

```text
day_p_on_ge_02 / 05 / 10 / 30 / 45 / 57
```

目的：刻画“整日 p_on 偏低但持续存在”的梅雨/低功率模式。

### 3.3 连续 ON 段特征

```text
seg_len
seg_pos_frac
seg_elapsed_h
seg_remaining_h
seg_pred_mean/std/min/max
seg_p_on_mean/min/q25/q50/q75
seg_start_hour
seg_end_hour
```

目的：区分短启动段、长稳态段、低功率长时 ON 与高功率长时 ON。

### 3.4 局部滚动与相对量纲

```text
p_on_roll4_mean / p_on_roll8_mean
pred_roll4_mean / pred_roll8_mean
pred_to_day_mean
pred_to_seg_mean
pred_interval_width / ratio
pred_low_ratio / pred_high_ratio
```

目的：增强模式分类器对局部功率形态、预测不确定性和相对档位的识别能力。

---

## 4. 整体指标对比

### 4.1 train_val

| variant | F1 | SAE | MAE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9962 | 0.0031 | 11.01 | 316.413 | 315.447 | -0.966 | 1972/9/6/4039 |
| P2 mode classifier/regressor | 0.9962 | 0.0024 | 5.98 | 316.413 | 317.180 | +0.767 | 1972/9/6/4039 |
| P2 loss-aware mode model | 0.9962 | 0.0013 | 5.46 | 316.413 | 316.820 | +0.407 | 1972/9/6/4039 |

train_val 上，loss-aware 版本看起来最好：

```text
MAE_W: 11.01 -> 5.46
SAE:   0.31% -> 0.13%
```

这说明新增特征 + loss-aware 目标对选择集拟合很强。

### 4.2 test

| variant | F1 | SAE | MAE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9885 | 0.0362 | 27.67 | 91.798 | 88.472 | -3.325 | 514/2/10/914 |
| P2 mode classifier/regressor | 0.9885 | 0.0309 | 24.15 | 91.798 | 88.957 | -2.841 | 514/2/10/914 |
| P2 loss-aware mode model | 0.9885 | 0.0379 | 24.82 | 91.798 | 88.321 | -3.477 | 514/2/10/914 |

解读：

- loss-aware 版本 test MAE 仍优于 baseline：27.67 → 24.82W；
- 但 test SAE 反而略差于 baseline：3.62% → 3.79%；
- 上一版普通 P2 mode model 在 test 上更稳：SAE 3.09%、MAE 24.15W。

### 4.3 inference OOD

| variant | F1 | SAE | MAE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9144 | 0.1460 | 108.07 | 224.919 | 192.088 | -32.832 | 1201/23/202/1354 |
| P2 mode classifier/regressor | 0.9144 | 0.1407 | 103.46 | 224.919 | 193.273 | -31.647 | 1201/23/202/1354 |
| P2 loss-aware mode model | 0.9144 | 0.1842 | 114.92 | 224.919 | 183.489 | -41.430 | 1201/23/202/1354 |

**关键结论**：

```text
P2 loss-aware 在 inference OOD 上显著退化。
```

退化幅度：

| 指标 | baseline -> loss-aware | 普通 P2 mode -> loss-aware |
|---|---:|---:|
| MAE_W | 108.07 -> 114.92，+6.85W | 103.46 -> 114.92，+11.46W |
| SAE | 14.60% -> 18.42%，+3.82pp | 14.07% -> 18.42%，+4.35pp |
| kWh_pred | 192.09 -> 183.49，-8.60 kWh | 193.27 -> 183.49，-9.78 kWh |

分类完全不变，因为 P2 与 P1 分开，不改 state。

---

## 5. 日级异常数量

| variant | stage | F1<90% ON日 | SAE>20% ON日 |
|---|---|---:|---:|
| baseline | test | 0 | 2 |
| P2 mode classifier/regressor | test | 0 | 2 |
| P2 loss-aware mode model | test | 0 | 2 |
| baseline | inference | 5 | 14 |
| P2 mode classifier/regressor | inference | 5 | 15 |
| P2 loss-aware mode model | inference | 5 | 17 |

日级结论：

- P2 不改分类，F1<90% 天数不变；
- 普通 P2 mode model 已让 inference `SAE>20%` 天数从 14 增到 15；
- loss-aware 版本进一步增到 17，说明日级 OOD 稳定性更差。

---

## 6. 日级效果明细

### 6.1 改善样本

| date | baseline SAE | loss-aware SAE | ΔSAE | baseline MAE | loss-aware MAE | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 2026-06-09 | 31.7% | 19.3% | -12.4pp | 83.87 | 71.03 | 低功率日高估被压低，有效。 |
| 2026-07-11 | 15.3% | 9.0% | -6.3pp | 66.57 | 58.99 | 高估减轻。 |
| 2026-07-08 | 13.8% | 10.5% | -3.3pp | 68.13 | 51.02 | 高估减轻。 |
| 2026-06-19 | 266.6% | 264.8% | -1.8pp | 295.17 | 293.23 | 低功率高估略有改善，但仍严重失败。 |

### 6.2 恶化样本

| date | baseline SAE | loss-aware SAE | ΔSAE | baseline MAE | loss-aware MAE | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 2026-06-24 | 36.0% | 56.8% | +20.8pp | 114.71 | 180.92 | 已低估日被进一步压低。 |
| 2026-07-01 | 37.7% | 51.5% | +13.8pp | 173.33 | 229.84 | 高功率低估加重。 |
| 2026-06-05 | 30.8% | 42.9% | +12.1pp | 63.28 | 83.59 | partial-day / 低功率日高估加重。 |
| 2026-06-26 | 33.3% | 45.6% | +12.3pp | 131.40 | 179.84 | 高功率低估加重。 |
| 2026-07-03 | 35.4% | 43.2% | +7.8pp | 161.09 | 196.68 | 高功率低估加重。 |
| 2026-07-04 | 41.8% | 48.0% | +6.2pp | 186.07 | 213.58 | 高功率低估加重。 |

### 6.3 仍完全无效的分类漏检日

| date | baseline SAE | loss-aware SAE | 原因 |
|---|---:|---:|---|
| 2026-06-08 | 100.0% | 100.0% | 分类整日漏检，P2 无 ON 点可回归。 |
| 2026-07-06 | 100.0% | 100.0% | 分类整日漏检，P2 无 ON 点可回归。 |

---

## 7. 为什么 loss-aware 在 train/val 好、OOD 差？

### 7.1 train/val 的日级坏样本太少

train/val 中 ON 日 `SAE>20%` 数量极少：

```text
baseline train: 0 天
baseline val:   1 天
```

因此 loss-aware 目标在 train/val 上可以把少数坏日拟合好，但无法覆盖 inference OOD 中的多种模式：

```text
低功率长时 ON 高估：6/19
高功率高湿低估：7/1、7/3、7/4
分类漏检：6/8、7/6
partial-day：6/5、6/24
```

### 7.2 目标函数仍只作用于“已识别 ON 点”

P2 不改 state。对整日漏检：

```text
state_pred_main 全 0 -> y_pred_variant 全 0
```

所以 6/8、7/6 无论 P2 怎么训练，都无法修复。

### 7.3 新增段级特征提高了拟合能力，也提高了 OOD 过拟合风险

增强特征让 train/val MAE 大幅下降：

```text
11.01W -> 5.46W
```

但 inference 高功率日被系统性压低，说明模型把 train/val 中某些段级/天气模式错误外推到了 7 月 OOD。

---

## 8. 与 P1 分开处理的验证

本轮 P2 变体均保持：

```text
TP/FP/FN/TN 与 baseline 完全一致
```

例如 inference：

```text
baseline:              TP=1201 FP=23 FN=202 TN=1354
P2 loss-aware model:    TP=1201 FP=23 FN=202 TN=1354
```

说明：

- P2 只负责已识别 ON 点的功率档位；
- P1 才负责 recall/FN；
- 两者没有混在一起归因。

P1 当前独立效果仍是：

```text
F1 0.9144 -> 0.9172
FN 202 -> 195
SAE 14.60% -> 14.07%
```

---

## 9. 最终判断

### 9.1 本轮 P2 loss-aware 是否成功？

结论：**验证失败，不建议上线。**

虽然它满足用户要求的工程动作：

- 加入了日级 loss-aware 目标；
- 增加了模式判别特征；
- 与 P1 分开处理；
- 遵守 train/val 参数纪律；

但 OOD 结果不合格：

```text
inference SAE: 14.60% -> 18.42%
inference MAE: 108.07W -> 114.92W
SAE>20% ON日: 14 -> 17
```

### 9.2 当前 P2 最优候选

当前三种 P2 中，排序为：

| 方案 | 结论 |
|---|---|
| P2_mode_classifier_regressor | 当前最好：test 与 inference MAE/SAE 均改善，但日级 SAE>20 天数 +1。 |
| P2_daily_scale_ref | 总量 SAE 与 P2 mode 接近，但逐点 MAE 改善小。 |
| P2_lossaware_mode_model | train/val 拟合最好，但 OOD 明显退化，不建议。 |

### 9.3 下一步建议

1. **不要上线 P2_lossaware_mode_model**。
2. 若继续 P2，应回到 `P2_mode_classifier_regressor`，再加“train/val 驱动的 OOD 安全闸”：
   ```text
   若预测日属于 train/val 无覆盖模式，则回退 baseline 或低 blend。
   ```
3. 引入真正可泛化的总线/段级原始特征，而不仅是 baseline 预测派生特征：
   ```text
   load_iden_data73 残差形态
   启动后 N 步总线变化
   ON 段内总线斜率/波动
   早/中/晚总线差分
   ```
4. P2 继续只修功率；6/8、7/6 必须由 P1/补样本/分类模型解决。
5. 继续保持 7 月只做 OOD 验证，不用 7 月反向调参。

---

## 10. 总结

本轮按要求完成了 P2 下一步验证。硬结论是：

> **日级 loss-aware + 增强模式特征在 train/val 与 test 上有效，但在 inference OOD 上失败，表现为更严重的系统性低估。当前不能作为生产修复。P2 的可行方向仍是 mode_classifier + per-mode regressor，但必须增加 OOD 安全闸与原始总线/段级特征。**
