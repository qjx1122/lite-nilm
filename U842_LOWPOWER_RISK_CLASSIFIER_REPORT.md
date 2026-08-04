# U842 低功率长时风险识别器增强特征验证报告

> 用户要求：继续按“6.2 增强特征”建议，训练/验证更强的低功率长时风险识别器。  
> 实验脚本：`scripts/experiment_u842_lowpower_risk_classifier.py`  
> 实验产物：`artifacts/u842_lowpower_risk_classifier/`  
> 当前基线：`current_optimized = P0_P1_P2_P3_lowpower_edge`。  
> 纪律：训练标签只使用 train+val；test/inference 只作验证；oracle 仅用于上限量化。

---

## 1. 本轮增强特征

本轮训练 day-level 低功率长时风险识别器，特征包括：

```text
rawbus-to-current ratio
anchor_delta_kwh
raw73_on_mean / raw73_on_std
raw73_day_mean / raw73_day_std
morning / midday / evening raw73 mean
morning / midday / evening p_on mean
morning / midday / evening predicted power mean and count
current_on_mean / base_on_mean
p_on_mean / std / q25 / q50 / q75
p_on_ge02 / ge05 / ge10 / ge30 / ge45 / ge57
coverage / n_samples / pred_on_n
RH_mean / temp_mean
dist_humid_low_cluster
dist_dry_low_cluster
```

目标是识别：

```text
true_on_h >= 10h 且 true_on_mean < 400W
```

并在识别后应用 train+val 推导的低功率锚：

| 条件 | anchor |
|---|---:|
| RH >= 80 | 207.08W |
| RH < 80 | 308.39W |

---

## 2. 训练标签分布

train+val 正样本只有 3 天：

```text
2026-06-04
2026-06-10
2026-06-11
```

标签分布：

| label | 天数 |
|---:|---:|
| 0 | 47 |
| 1 | 3 |

这是本轮验证的核心限制：正样本太少，且没有覆盖 `2026-06-19` 的强错配形态。

---

## 3. 分类器识别结果

| model | stage | true positive days | pred positive days | precision | recall | predicted_dates |
|---|---|---:|---:|---:|---:|---|
| RF | train | 3 | 3 | 1.000 | 1.000 | 2026-06-04, 2026-06-10, 2026-06-11 |
| RF | val | 0 | 0 | 0 | 0 | - |
| RF | test | 0 | 0 | 0 | 0 | - |
| RF | inference | 5 | 2 | 1.000 | 0.400 | 2026-06-08, 2026-06-09 |
| ExtraTrees | inference | 5 | 2 | 1.000 | 0.400 | 2026-06-08, 2026-06-09 |
| Logistic | inference | 5 | 2 | 1.000 | 0.400 | 2026-06-08, 2026-06-09 |

inference 中真实低功率长时日包括：

```text
2026-06-07
2026-06-08
2026-06-09
2026-06-19
2026-06-27
```

但所有模型都只识别出：

```text
2026-06-08
2026-06-09
```

没有识别出：

```text
2026-06-19
2026-06-27
```

---

## 4. 应用锚点后的整体指标

| variant | inference F1 | inference SAE | MAE_W | kWh_pred | 说明 |
|---|---:|---:|---:|---:|---|
| current_optimized | 0.9794 | 0.735% | 66.58 | 226.57 | 当前优化版本 |
| RF lowpower classifier | 0.9794 | 1.707% | 69.12 | 228.76 | 只触发 6/8、6/9，未触发 6/19/6/27，整体变差 |
| ExtraTrees | 0.9794 | 1.707% | 69.12 | 228.76 | 同 RF |
| Logistic | 0.9794 | 1.707% | 69.12 | 228.76 | 同 RF |
| Oracle 6/19+6/27 | 0.9794 | 3.690% | 56.68 | 216.62 | 只验证锚点上限，不是生产规则 |

解释：

- 训练出的识别器没有命中 6/19/6/27；
- 反而对 6/8、6/9 再次应用锚点，使总体 kWh 过高，整体 SAE 从 0.735% 退化到 1.707%；
- oracle 能降低 6/19/6/27 的 MAE，但整体 SAE 因总量低估上升到 3.69%。

---

## 5. 6/19 与 6/27 日级效果

| variant | date | F1 | SAE | MAE_W | kWh_true | kWh_pred | kWh_err | TP/FP/FN/TN |
|---|---|---:|---:|---:|---:|---:|---:|---|
| current_optimized | 2026-06-19 | 0.990 | 266.6% | 295.17 | 2.602 | 9.540 | +6.938 | 50/0/1/45 |
| RF / Extra / Logistic | 2026-06-19 | 0.990 | 266.6% | 295.17 | 2.602 | 9.540 | +6.938 | 50/0/1/45 |
| oracle | 2026-06-19 | 0.990 | 0.53% | 17.27 | 2.602 | 2.589 | -0.014 | 50/0/1/45 |
| current_optimized | 2026-06-27 | 0.980 | 42.0% | 96.04 | 4.718 | 6.701 | +1.984 | 48/0/2/46 |
| RF / Extra / Logistic | 2026-06-27 | 0.980 | 42.0% | 96.04 | 4.718 | 6.701 | +1.984 | 48/0/2/46 |
| oracle | 2026-06-27 | 0.980 | 21.6% | 87.12 | 4.718 | 3.701 | -1.017 | 48/0/2/46 |

结论：

```text
锚点能修目标日，但识别器没有学会命中目标日。
```

---

## 6. 为什么增强特征仍失败？

### 6.1 正样本不足

train+val 正样本只有 3 天，模型只能学到：

```text
常规低功率长时 ON
```

无法学到：

```text
raw bus 高 + baseline 高 + true 低功率
```

这种强错配模式。

### 6.2 6/19 是 OOD 强错配

6/19 特征：

```text
true_on_mean ≈ 203W
current_on_mean ≈ 763W
raw73_on_mean ≈ 2692
RH ≈ 84
p_on_q50 ≈ 0.585
```

train+val 低功率样本没有类似的“current_on_mean≈763W 但 true≈200W”样本。

### 6.3 建议特征缺少“非空调负荷分解”

raw73 很高，但无法区分：

```text
空调高功率
vs
家庭其他负荷高、空调低功率
```

这就是 6/19 的本质。

---

## 7. 结论

本轮“增强特征”验证结论是否定的：

> **当前增强特征 + train/val 标签不足以训练出能识别 6/19 的低功率长时风险识别器。**

锚点方向仍然正确：

```text
6/19 oracle SAE: 266.6% -> 0.53%
6/27 oracle SAE: 42.0% -> 21.6%
```

但识别器不合格：

```text
6/19 未命中
6/27 未命中
```

因此不能上线当前 classifier。

---

## 8. 下一步建议

需要继续增强的是“可识别非空调背景负荷”的特征或数据：

1. **补样本**：增加 `raw bus 高 + baseline 高 + true 低功率` 的同型训练样本。
2. **非空调基线估计**：引入 household/base-load 分解，判断 raw73 高是否来自非空调负荷。
3. **段内负荷分解特征**：ON 段 rawbus 与 p_on/y_pred 的残差特征，而不是 rawbus 绝对值。
4. **跨日相似度**：与 train+val 低功率长时簇的相似度需要包含“baseline 高估”维度，但不能误伤高功率日。
5. **灰度前规则**：在没有更多样本前，继续保留 current_optimized，不上线低功率长时 classifier。
