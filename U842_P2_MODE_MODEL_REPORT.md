# U842 P2 重做：mode_classifier + per-mode regressor 验证报告

> 更新说明：本报告为普通 P2 mode model 的验证；后续已进一步验证日级 loss-aware + 增强模式特征（`U842_P2_LOSSAWARE_MODE_REPORT.md`）以及真正 raw bus/segment 特征（`U842_P2_RAWBUS_SEGMENT_REPORT.md`）。当前 P2 最佳候选以 rawbus 报告为准。
>
> 目标：按用户要求重做 P2，不再使用 simple daily scale，改为 `mode_classifier + per-mode regressor`，并重新验证 test / inference 效果。  
> 实验脚本：`scripts/experiment_u842_p1_p2_p3.py`  
> 产物目录：`artifacts/u842_p1_p2_p3_experiment/`  
> 基线：v14.7 U842 当前沙盒链，`best_thr=0.57`，`ensemble_lgb_active=True`。  
> 参数纪律：P2 模式阈值、分类器、回归器均只使用 `train+val`；`test` 与 `inference` 仅作为验证，不参与调参。

---

## 1. P2 新方案定义

旧 P2 是 simple daily scale：按每日 `pred_on_mean` 分档后对整天预测功率乘一个 scale。该方案过弱，之前学到的 3 个 scale 都接近 1，不能真正区分低功率长时 ON 与高功率长时 ON。

本次重做为两级模型：

```text
baseline state_pred_main 不变
        ↓
只对 baseline 已判 ON 的点重估功率
        ↓
mode_classifier 预测功率模式：low / mid / high
        ↓
per-mode regressor 输出该模式下的功率
        ↓
y_pred_variant = per-mode regressor 输出；OFF 点仍为 0
```

该方案只改功率层，不改分类层，因此 F1/Precision/Recall/TP/FP/FN/TN 与 baseline 一致。

---

## 2. 训练标签与模型参数

### 2.1 模式标签

模式标签来自 `train+val` 真实 ON 样本的 `y_true_W` 三分位数：

| mode | 规则 | train+val 样本数 |
|---:|---|---:|
| 0 low | `y_true_W <= 564.29W` | 660 |
| 1 mid | `564.29W < y_true_W <= 747.70W` | 660 |
| 2 high | `y_true_W > 747.70W` | 658 |

没有使用 test / inference 标签定义模式。

### 2.2 mode classifier

```text
RandomForestClassifier(
  n_estimators=200,
  max_depth=6,
  min_samples_leaf=10,
  class_weight='balanced',
  random_state=42,
  n_jobs=1
)
```

### 2.3 per-mode regressor

每个 mode 单独训练：

```text
RandomForestRegressor(
  n_estimators=200,
  max_depth=8,
  min_samples_leaf=8,
  random_state=42,
  n_jobs=1
)
```

### 2.4 特征

使用推理时可获得的 baseline 预测、时间与天气特征：

```text
p_on_main
y_pred_W_main
y_pred_low_W_main
y_pred_high_W_main
hour_sin, hour_cos, dow
temperature_2m, apparent_temperature, relative_humidity_2m
day_pred_on_n, day_pred_on_mean, day_pred_kwh
day_p_on_mean, day_p_on_q25, day_p_on_q50, day_p_on_q75
temp_mean, rh_mean
```

---

## 3. 整体指标对比

### 3.1 train / val / train_val

| variant | stage | F1 | Precision | Recall | SAE | MAE_W | kWh_true | kWh_pred | kWh_err | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | train | 1.0000 | 1.0000 | 1.0000 | 0.0033 | 5.75 | 222.520 | 221.778 | -0.742 | 1404 | 0 | 0 | 3086 |
| P2 mode model | train | 1.0000 | 1.0000 | 1.0000 | 0.0032 | 2.95 | 222.520 | 223.233 | +0.713 | 1404 | 0 | 0 | 3086 |
| baseline | val | 0.9870 | 0.9844 | 0.9895 | 0.0024 | 26.39 | 93.893 | 93.669 | -0.225 | 568 | 9 | 6 | 953 |
| P2 mode model | val | 0.9870 | 0.9844 | 0.9895 | 0.0006 | 14.81 | 93.893 | 93.947 | +0.054 | 568 | 9 | 6 | 953 |
| baseline | train_val | 0.9962 | 0.9955 | 0.9970 | 0.0031 | 11.01 | 316.413 | 315.447 | -0.966 | 1972 | 9 | 6 | 4039 |
| P2 mode model | train_val | 0.9962 | 0.9955 | 0.9970 | 0.0024 | 5.98 | 316.413 | 317.180 | +0.767 | 1972 | 9 | 6 | 4039 |

**解释**：

- 分类完全不变，符合 P2 只修功率层的设计。
- train_val MAE 从 11.01W 降到 5.98W，说明模型在选择集上拟合能力明显强于原 MoE 输出。
- 但 train_val 电量由轻微低估转为轻微高估，存在一定过拟合/上抬倾向。

### 3.2 test 验证集

| variant | F1 | Precision | Recall | SAE | MAE_W | RMSE_W | kWh_true | kWh_pred | kWh_err | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9885 | 0.9961 | 0.9809 | 0.0362 | 27.67 | 79.36 | 91.798 | 88.472 | -3.325 | 514 | 2 | 10 | 914 |
| P2 mode model | 0.9885 | 0.9961 | 0.9809 | 0.0309 | 24.15 | 76.19 | 91.798 | 88.957 | -2.841 | 514 | 2 | 10 | 914 |

**test 结论**：

| 指标 | 改善 |
|---|---:|
| MAE_W | 27.67 → 24.15，下降 3.52W，约 -12.7% |
| SAE | 3.62% → 3.09%，下降 0.53pp |
| kWh_err | -3.325 → -2.841 kWh，低估减轻 0.485 kWh |

在未使用 test 调参的前提下，P2 mode model 对 ID test 有明确正收益。

### 3.3 inference OOD 验证集

| variant | F1 | Precision | Recall | SAE | MAE_W | RMSE_W | kWh_true | kWh_pred | kWh_err | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9144 | 0.9812 | 0.8560 | 0.1460 | 108.07 | 213.41 | 224.919 | 192.088 | -32.832 | 1201 | 23 | 202 | 1354 |
| P2 mode model | 0.9144 | 0.9812 | 0.8560 | 0.1407 | 103.46 | 211.56 | 224.919 | 193.273 | -31.647 | 1201 | 23 | 202 | 1354 |
| old daily scale ref | 0.9144 | 0.9812 | 0.8560 | 0.1407 | 107.90 | 213.24 | 224.919 | 193.269 | -31.651 | 1201 | 23 | 202 | 1354 |

**inference 结论**：

| 指标 | P2 mode model 改善 |
|---|---:|
| MAE_W | 108.07 → 103.46，下降 4.61W，约 -4.3% |
| SAE | 14.60% → 14.07%，下降 0.53pp |
| kWh_pred | 192.09 → 193.27，上升 1.18 kWh |
| kWh_err | -32.83 → -31.65 kWh，低估减轻 1.18 kWh |

与 old simple daily scale 相比：

- 总 SAE 几乎相同；
- P2 mode model 的 MAE 明显更优：103.46W vs 107.90W；
- 说明 mode regressor 改善了逐点功率形态，但日级电量仍受分类 FN 与 OOD 模式缺失限制。

---

## 4. 日级效果分析

### 4.1 改善最大的日

| date | baseline SAE | P2 SAE | ΔSAE | baseline MAE_W | P2 MAE_W | ΔMAE | kWh_true | baseline kWh_pred | P2 kWh_pred | 解释 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-09 | 31.7% | 6.9% | -24.8pp | 83.87 | 45.11 | -38.77 | 2.454 | 3.233 | 2.285 | 低功率日高估被显著压低，是 P2 mode model 最成功样本。 |
| 2026-06-18 | 24.1% | 19.8% | -4.2pp | 105.69 | 87.14 | -18.55 | 10.540 | 8.004 | 8.449 | 高功率低估有所缓解，SAE 由 >20% 降到 <20%。 |
| 2026-07-04 | 41.8% | 38.5% | -3.4pp | 186.07 | 170.97 | -15.10 | 10.671 | 6.205 | 6.567 | 高功率低估略缓解，但分类 FN 仍存在。 |
| 2026-07-03 | 35.4% | 32.6% | -2.7pp | 161.09 | 148.69 | -12.40 | 10.936 | 7.070 | 7.368 | 高功率低估略缓解。 |
| 2026-07-15 | 14.8% | 8.3% | -6.5pp | 68.31 | 69.27 | +0.96 | 10.693 | 9.113 | 9.808 | 电量更接近，但逐点 MAE 略升。 |

### 4.2 恶化最大的日

| date | baseline SAE | P2 SAE | ΔSAE | baseline MAE_W | P2 MAE_W | ΔMAE | kWh_true | baseline kWh_pred | P2 kWh_pred | 解释 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-05 | 30.8% | 44.6% | +13.8pp | 63.28 | 86.47 | +23.18 | 3.183 | 4.162 | 4.603 | partial-day/低功率日被进一步估高。 |
| 2026-06-06 | 15.8% | 21.3% | +5.5pp | 50.30 | 64.23 | +13.92 | 6.103 | 7.066 | 7.401 | 原本未超 20%，P2 上抬后越界。 |
| 2026-07-02 | 16.8% | 21.1% | +4.3pp | 90.25 | 109.37 | +19.12 | 10.891 | 9.064 | 8.594 | P2 误下调，导致低估加重。 |
| 2026-07-11 | 15.3% | 20.4% | +5.1pp | 66.57 | 73.45 | +6.88 | 8.142 | 9.387 | 9.804 | 原本高估，P2 进一步高估并越界。 |
| 2026-06-26 | 33.3% | 35.8% | +2.4pp | 131.40 | 141.05 | +9.65 | 9.460 | 6.307 | 6.075 | 高功率低估被进一步压低。 |

### 4.3 核心失败日仍未解决

| date | baseline SAE | P2 SAE | 原因 |
|---|---:|---:|---|
| 2026-06-08 | 100.0% | 100.0% | 分类整日漏检，P2 只改已判 ON 点，无可作用样本。 |
| 2026-06-21 | 79.8% | 78.8% | 大段 FN，P2 只能微调上午已识别段。 |
| 2026-07-06 | 100.0% | 100.0% | 分类整日漏检，P2 无法修复。 |
| 2026-06-19 | 266.6% | 263.2% | 低功率长时 ON 被高估仍严重，mode classifier 未能充分识别为低档。 |

---

## 5. 异常日数量

ON 日口径：

| variant | stage | F1<90% 天数 | SAE>20% 天数 |
|---|---|---:|---:|
| baseline | test | 0 | 2 |
| P2 mode model | test | 0 | 2 |
| baseline | inference | 5 | 14 |
| P2 mode model | inference | 5 | 15 |

解释：

- P2 不改分类，所以 F1<90% 天数不变。
- P2 mode model 改善了总体 MAE/SAE，但让若干原本接近 20% 阈值的日子越界，例如 6/6、7/2、7/11；同时也修复了 6/18、6/9 等部分高 SAE 日。
- 因此，**整体能量指标改善，但日级 SAE>20% 天数反而 +1**。

---

## 6. 与 old simple daily scale 对比

| 指标 | simple daily scale | mode_classifier + per-mode regressor | 结论 |
|---|---:|---:|---|
| inference SAE | 14.07% | 14.07% | 总量 SAE 近似相同。 |
| inference MAE_W | 107.90 | 103.46 | mode model 明显更好。 |
| test SAE | 3.14% | 3.09% | mode model 略优。 |
| test MAE_W | 27.51 | 24.15 | mode model 明显更好。 |
| inference SAE>20% 天数 | 14 | 15 | mode model 日级稳定性更差。 |

结论：重做后的 P2 相比 simple daily scale，**逐点功率形态更好**，但**日级越界风险更高**。

---

## 7. 是否建议上线

不建议直接生产上线当前 P2 mode model。

理由：

1. test 和 inference 整体指标有改善，说明方向正确；
2. 但 inference SAE>20% 天数从 14 增至 15，说明日级稳定性不足；
3. 对 6/8、7/6 等分类整日漏检无效；
4. 对 6/19 低功率长时 ON 严重高估仅从 266.6% 降到 263.2%，远未解决；
5. 当前特征仍不足以可靠区分“低功率长时 ON”和“高功率长时 ON”。

---

## 8. 下一步优化建议

### 8.1 给 P2 增加日级 loss-aware 选择目标

当前训练目标偏逐点 MAE。下一版应在 train+val 内加入日级约束：

```text
优化目标 = point_MAE + λ * daily_SAE + μ * SAE>20_day_penalty
```

参数 `λ/μ` 只能用 train+val 选。

### 8.2 加入更多模式判别特征

当前 mode classifier 主要依赖 baseline 预测、天气与时段。建议增加：

```text
总线残差形态
ON 段内功率斜率/波动
启动后 N 步均值
日内分段统计：早/中/晚 pred_on_mean
连续 ON 段长度、段内 p_on 分位数
```

目标是区分：

```text
6/19 低功率长时 ON（真实约 203W）
7/1~7/4 高功率长时 ON（真实约 850W）
```

### 8.3 与 P1 分类 guard 联合，但需分开归因

P2 只能修已识别 ON 点功率；对 6/8、7/6 这种 `state_pred=0` 整日漏检无效。必须与 P1 或补样本方案分开处理：

```text
P1/P补样本：解决 FN
P2 mode model：解决已识别 ON 的功率档位
```

### 8.4 不使用 7 月反向调参

7 月仍作为最终 OOD 验证。当前 P2 mode model 的失败日可用于归因，但不能用于反向选择模式阈值、模型超参或日级安全阈值。

---

## 9. 最终结论

P2 重做后的 `mode_classifier + per-mode regressor` 相比 simple daily scale 有实质进步：

```text
test MAE: 27.67W -> 24.15W
inference MAE: 108.07W -> 103.46W
inference SAE: 14.60% -> 14.07%
```

但它还不是可直接上线的修复：

```text
inference SAE>20% 天数: 14 -> 15
6/8、7/6 整日漏检无效
6/19 低功率长时高估仍严重
```

因此，当前 P2 新版结论是：

> **方向正确，优于 simple daily scale；但需要加入日级 loss-aware 目标与更多总线/段级模式特征后，才能作为生产候选。**
