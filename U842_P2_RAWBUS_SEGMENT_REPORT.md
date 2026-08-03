# U842 P2 原始总线/段级特征验证报告

> 用户要求：继续引入真正的原始总线/段级特征进行验证测试，而不仅是 baseline 预测派生特征。  
> 实验脚本：`scripts/experiment_u842_p1_p2_p3.py`  
> 产物目录：`artifacts/u842_p1_p2_p3_experiment/`  
> 基线：v14.7 U842 当前沙盒链，`best_thr=0.57`，`ensemble_lgb_active=True`。  
> 结论先行：**引入 raw bus + baseline predicted-ON 段级特征后，P2 显著优于之前所有 P2 变体：inference SAE 14.60%→7.49%，MAE 108.07W→93.22W，SAE>20% ON 日 14→11。仍不能解决 6/8、7/6 这类分类整日漏检，也会恶化 6/19、6/27 等低功率长时高估日。**

---

## 1. 约束与实验纪律

| 约束 | 执行情况 |
|---|---|
| 参数只从 train/val 推导 | 是。模式阈值、raw bus 列、blend 与模型参数均不使用 test/inference 标签。 |
| test 只作验证 | 是。 |
| inference / 7 月 OOD 只作最终验证 | 是。 |
| 与 P1 分开 | 是。P2 不改 `state_pred_main`，只重估 baseline 已判 ON 点功率。 |
| 不再只用 baseline 派生特征 | 是。新增 raw `load_iden_data*` 15min 值、差分、rolling、日级 raw 统计、预测 ON 段内 raw 统计。 |

---

## 2. Raw bus / segment 特征设计

### 2.1 原始总线列

优先取当前 U842 bundle `feat_cols` 中存在于 raw bus 的前 12 个原始电参量列：

```text
load_iden_data73
load_iden_data1
load_iden_data74
load_iden_data2
load_iden_data79
load_iden_data7
load_iden_data75
load_iden_data5
load_iden_data77
load_iden_data3
load_iden_data76
load_iden_data4
```

其中 `load_iden_data73` 是主功率相关列，其他列来自模型已选择的高相关/高重要原始电参量。

### 2.2 点级 raw bus 特征

对 12 个 raw bus 列取 15min resample 值：

```text
raw_load_iden_data*
```

对前 6 个 raw bus 列额外取：

```text
diff1
diff4
rolling4_mean
rolling4_std
```

### 2.3 日级 raw bus 特征

对前 8 个 raw bus 派生列计算：

```text
day_raw_*_mean
day_raw_*_std
day_on_raw_*_mean     # baseline predicted-ON 点内均值
```

### 2.4 baseline predicted-ON 段级 raw bus 特征

对 baseline 已判 ON 的连续段，计算：

```text
raw_*_seg_mean
raw_*_seg_std
raw_*_seg_min
raw_*_seg_max
raw_*_seg_range
raw_*_to_seg_mean
```

注意：这些段级特征只使用 raw bus 和 baseline 预测状态，不使用分路标签；P2 仍不改变 ON/OFF 分类。

### 2.5 模型结构

仍为：

```text
mode_classifier + per-mode regressor
```

模式阈值仍由 train+val 真 ON 样本三分位定义：

| mode | 规则 | 样本数 |
|---:|---|---:|
| low | `y_true_W <= 564.29W` | 660 |
| mid | `564.29W < y_true_W <= 747.70W` | 660 |
| high | `y_true_W > 747.70W` | 658 |

模型参数：

```text
RandomForestClassifier(n_estimators=240, max_depth=8, min_samples_leaf=6, class_weight=balanced)
RandomForestRegressor(n_estimators=240, max_depth=10, min_samples_leaf=6)
selected_blend = 1.0
```

`selected_blend=1.0` 由 train+val objective 在候选 `[1.0,0.85,0.70,0.50,0.30]` 中选择。

---

## 3. 整体指标对比

### 3.1 train_val

| variant | F1 | SAE | MAE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9962 | 0.0031 | 11.01 | 316.413 | 315.447 | -0.966 | 1972/9/6/4039 |
| P2 mode classifier/regressor | 0.9962 | 0.0024 | 5.98 | 316.413 | 317.180 | +0.767 | 1972/9/6/4039 |
| P2 loss-aware mode model | 0.9962 | 0.0013 | 5.46 | 316.413 | 316.820 | +0.407 | 1972/9/6/4039 |
| **P2 rawbus segment model** | **0.9962** | **0.0028** | **5.01** | **316.413** | **317.301** | **+0.888** | **1972/9/6/4039** |

train_val 上，rawbus 模型的 MAE 低于普通 P2 和 loss-aware P2，但总电量略高估。

### 3.2 test

| variant | F1 | SAE | MAE_W | RMSE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9885 | 0.0362 | 27.67 | 79.36 | 91.798 | 88.472 | -3.325 | 514/2/10/914 |
| P2 mode classifier/regressor | 0.9885 | 0.0309 | 24.15 | 76.19 | 91.798 | 88.957 | -2.841 | 514/2/10/914 |
| P2 loss-aware mode model | 0.9885 | 0.0379 | 24.82 | 76.78 | 91.798 | 88.321 | -3.477 | 514/2/10/914 |
| **P2 rawbus segment model** | **0.9885** | **0.0249** | **21.95** | **71.81** | **91.798** | **89.509** | **-2.289** | **514/2/10/914** |

**test 结论**：rawbus 模型最优。

相对 baseline：

```text
MAE_W: 27.67 -> 21.95  (-20.6%)
SAE:   3.62% -> 2.49% (-1.13pp)
kWh_err: -3.325 -> -2.289 kWh
```

### 3.3 inference OOD

| variant | F1 | SAE | MAE_W | RMSE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.9144 | 0.1460 | 108.07 | 213.41 | 224.919 | 192.088 | -32.832 | 1201/23/202/1354 |
| P2 mode classifier/regressor | 0.9144 | 0.1407 | 103.46 | 211.56 | 224.919 | 193.273 | -31.647 | 1201/23/202/1354 |
| P2 loss-aware mode model | 0.9144 | 0.1842 | 114.92 | 229.30 | 224.919 | 183.489 | -41.430 | 1201/23/202/1354 |
| **P2 rawbus segment model** | **0.9144** | **0.0749** | **93.22** | **202.44** | **224.919** | **208.074** | **-16.845** | **1201/23/202/1354** |

**inference 结论**：rawbus 模型显著优于 baseline 和此前所有 P2 变体。

相对 baseline：

```text
MAE_W:    108.07 -> 93.22   (-14.85W, -13.7%)
SAE:      14.60% -> 7.49%   (-7.11pp)
kWh_pred: 192.09 -> 208.07  (+15.99 kWh)
kWh_err:  -32.83 -> -16.85  低估减半
```

相对普通 P2 mode model：

```text
MAE_W: 103.46 -> 93.22
SAE:   14.07% -> 7.49%
```

---

## 4. 异常日数量

ON 日口径：

| variant | test F1<90% | test SAE>20% | inference F1<90% | inference SAE>20% |
|---|---:|---:|---:|---:|
| baseline | 0 | 2 | 5 | 14 |
| P2 mode classifier/regressor | 0 | 2 | 5 | 15 |
| P2 loss-aware mode model | 0 | 2 | 5 | 17 |
| **P2 rawbus segment model** | **0** | **2** | **5** | **11** |

结论：

- P2 不改分类，所以 F1<90% 天数不变；
- rawbus 模型首次让 inference SAE>20% 天数下降：14 → 11；
- 这说明 raw bus/segment features 提供了 baseline 预测派生特征没有的档位信息。

---

## 5. 日级改善分析

### 5.1 改善最大的日

| date | baseline SAE | rawbus SAE | ΔSAE | baseline MAE | rawbus MAE | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 2026-06-09 | 31.7% | 2.7% | -29.0pp | 83.87 | 53.89 | 低功率日高估基本被修正。 |
| 2026-06-26 | 33.3% | 10.3% | -23.0pp | 131.40 | 55.77 | 高功率低估大幅缓解。 |
| 2026-06-18 | 24.1% | 7.5% | -16.5pp | 105.69 | 33.27 | 高功率低估大幅缓解。 |
| 2026-07-01 | 37.7% | 21.3% | -16.4pp | 173.33 | 105.55 | 高功率低估明显缓解，但仍略高于 20%。 |
| 2026-07-03 | 35.4% | 20.6% | -14.7pp | 161.09 | 94.05 | 高功率低估明显缓解，接近 20%。 |
| 2026-07-04 | 41.8% | 30.7% | -11.2pp | 186.07 | 136.36 | FN 仍在，但功率层明显改善。 |
| 2026-07-15 | 14.8% | 4.6% | -10.2pp | 68.31 | 40.91 | 电量误差显著减小。 |

### 5.2 恶化最大的日

| date | baseline SAE | rawbus SAE | ΔSAE | baseline MAE | rawbus MAE | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 2026-06-27 | 42.0% | 58.8% | +16.8pp | 96.04 | 124.73 | 低/中功率长时 ON 被进一步高估。 |
| 2026-06-19 | 266.6% | 274.3% | +7.7pp | 295.17 | 303.53 | 低功率长时 ON 仍严重高估。 |
| 2026-06-05 | 30.8% | 37.7% | +6.9pp | 63.28 | 74.85 | partial-day/低功率日高估加重。 |
| 2026-07-11 | 15.3% | 21.9% | +6.7pp | 66.57 | 77.23 | 原本未越界，rawbus 上抬后 SAE>20%。 |
| 2026-07-10 | 10.1% | 18.5% | +8.3pp | 55.53 | 70.48 | 高估加重但未超过 20%。 |

### 5.3 仍无效的分类漏检日

| date | baseline SAE | rawbus SAE | 原因 |
|---|---:|---:|---|
| 2026-06-08 | 100.0% | 100.0% | 分类整日漏检，baseline `state_pred=0`，P2 无 ON 点可回归。 |
| 2026-07-06 | 100.0% | 100.0% | 同上。 |

---

## 6. 为什么 raw bus 特征有效？

与之前只用 baseline 预测派生特征相比，raw bus 直接提供了：

1. **总线真实负荷水平**：`load_iden_data73` 等 raw 电参量能反映当天实际负荷水位；
2. **总线动态变化**：diff/rolling 能识别档位变化和稳定段；
3. **段内 raw bus 形态**：seg_mean/std/range 能把低功率长时 ON 与高功率长时 ON 分开；
4. **日级 raw bus 上下文**：day mean/std 与 predicted-ON mean 能提供当天负荷背景。

这解释了为什么 rawbus 模型可以显著缓解 6/18、6/26、7/1、7/3、7/4 的高功率低估。

---

## 7. 与 P1 分开处理

P2 rawbus 仍保持分类不变：

```text
baseline inference:  TP=1201 FP=23 FN=202 TN=1354
P2 rawbus inference: TP=1201 FP=23 FN=202 TN=1354
```

因此：

- P2 rawbus 的收益完全来自功率层；
- 6/8、7/6 的整日漏检仍必须由 P1/补样本/分类模型解决；
- P1 recall guard 与 P2 rawbus 可以组合，但需要单独验证，不能混淆归因。

---

## 8. 是否建议上线？

### 8.1 结论

**P2 rawbus segment model 是目前最强的 P2 候选，但仍建议先灰度/加安全闸，不建议直接全量上线。**

硬收益：

```text
inference SAE: 14.60% -> 7.49%
inference MAE: 108.07W -> 93.22W
SAE>20% ON日: 14 -> 11
```

主要风险：

```text
6/19、6/27 低功率长时 ON 高估加重
7/11 从 SAE 15.3% 恶化到 21.9%
6/8、7/6 分类整日漏检仍无效
```

### 8.2 建议的安全闸

上线前应增加 train/val-only 安全闸：

```text
如果 rawbus model 输出相对 baseline 上抬过大，且该日/段 rawbus 模式在 train+val 覆盖不足，则回退 baseline 或降低 blend。
```

安全闸特征可使用：

```text
rawbus segment mean/range
pred_on_mean
day_pred_kwh
p_on 分布
coverage
```

阈值仍必须只用 train/val 选择。

---

## 9. 最终结论

本轮验证回答了用户问题：**真正的原始总线/段级特征是有效的。**

相比之前 P2：

| 方案 | inference SAE | inference MAE | SAE>20% ON日 | 结论 |
|---|---:|---:|---:|---|
| baseline | 14.60% | 108.07W | 14 | 原始基线 |
| P2 daily scale | 14.07% | 107.90W | 14 | 总量小幅修正，逐点弱 |
| P2 mode classifier/regressor | 14.07% | 103.46W | 15 | 逐点改善，但日级不稳 |
| P2 loss-aware enhanced | 18.42% | 114.92W | 17 | OOD 失败，拒绝上线 |
| **P2 rawbus segment** | **7.49%** | **93.22W** | **11** | **当前最佳 P2 候选** |

最终建议：

> **保留 P2 rawbus segment 方向，下一步做 P2 rawbus + 安全闸 + P1 recall guard 的组合灰度验证；不要再推进纯 baseline 派生特征的 loss-aware 版本。**
