# 5用户独立 force-off / consistency guard 生产候选验证报告

> 用户要求：保留生产主配置；将新增 6 月样本转为独立 guard 数据集；不训练主分类器，而训练后处理一致性守卫；守卫使用更多物理信号；按每用户方向验证。  
> 实验脚本：`scripts/experiment_5user_independent_guard.py`  
> 实验产物：`artifacts/five_user_independent_guard/`  
> 主模型配置：仍使用生产主配置 `data/time_filters.json`。  
> 独立 guard 数据来源：`data/time_filters_augmented_forceoff.json` 中相对主训练集新增的 6 月日期，只用于 guard 训练，不进入主 NILM 分类器。

---

## 1. 实验设计

### 1.1 保留生产主配置

本轮先用原始生产配置重新 5 用户训练/推理：

```bash
.venv/bin/python scripts/run_batch_users.py \
  --force-retrain \
  --time-filter-config data/time_filters.json
```

批跑日志：

```text
logs/_batch/batch_run_20260807_114524.log
```

结果：

```text
[OK] 5/5 用户成功
```

### 1.2 独立 guard 数据集

guard 训练集由两部分组成：

```text
1. 生产主模型 train+val 中 predicted-ON 点
2. data/time_filters_augmented_forceoff.json 相对主训练集新增的 6 月 guard-train 日期
```

验证集：

```text
生产 inference 中排除 guard-train 日期后的剩余日期
```

本脚本当前也输出全 inference 参考指标。

### 1.3 物理信号特征

相比上一版仅使用模型输出，本轮增加 raw bus 物理特征：

```text
raw load_iden_data* 15min 值
diff1 / diff4
rolling4 mean/std
day_raw_* mean/std
day_on_raw_* mean
predicted-ON segment raw mean/std/range/to_seg_mean
temperature / apparent_temperature / humidity
hour / dow
p_on / y_pred / prediction interval
```

### 1.4 每用户方向

| 用户 | 方向 |
|---|---|
| U842 | 不以 force-off 为主，当前问题是 recall/power；force-off 仅作安全性验证 |
| U2844 | low-true/high-FP + recall 仍需独立处理 |
| U0778 | OFF false block，但 train+val 负样本极少 |
| U0789 | OFF false block，是重点 force-off 用户 |
| U0800 | OFF false block，但 train+val 无 predicted-ON 负样本 |

---

## 2. 训练出的 guard 参数

### 2.1 point-level guard

| 用户 | 是否启用 | 阈值 | train+val 标签 | 说明 |
|---|---:|---:|---|---|
| U842 | 是 | 0.40 | 0:9, 1:1972 | train+val 可选型，但 inference 无实际收益 |
| U2844 | 是 | 0.40 | 0:15, 1:953 | train+val 可选型，但 inference 误杀 TP |
| U0778 | 否 | - | 0:1, 1:771 | 负样本不足，禁用 |
| U0789 | 是 | 0.45 | 0:8, 1:632 | train+val 可选型，但 inference 只小幅降 FP |
| U0800 | 否 | - | 1:202 | 无负样本，禁用 |

### 2.2 day-level guard

```text
enabled = true
threshold = 0.44
train_val_forced_off_days = 22
train_val_on_day_kills = 0
```

但 inference 上几乎不触发有效 OFF 虚块，因此无收益。

---

## 3. 5 用户 inference 验证结果

| variant | F1 | Precision | Recall | SAE | MAE_W | kWh_true | kWh_pred | kWh_err | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 baseline_current | 0.8495 | 0.7743 | 0.9408 | 10.08% | 145.98 | 1113.10 | 1225.33 | +112.23 | 6004 | 1750 | 378 | 8568 |
| TV_point_guard_physical | 0.8503 | 0.7762 | 0.9401 | 9.62% | 144.84 | 1113.10 | 1220.16 | +107.06 | 6000 | 1730 | 382 | 8588 |
| TV_day_guard_physical | 0.8495 | 0.7743 | 0.9408 | 10.08% | 145.98 | 1113.10 | 1225.33 | +112.23 | 6004 | 1750 | 378 | 8568 |
| TV_combined_physical | 0.8503 | 0.7762 | 0.9401 | 9.62% | 144.84 | 1113.10 | 1220.16 | +107.06 | 6000 | 1730 | 382 | 8588 |

### 3.1 结论

物理信号 point guard 只有极小改善：

```text
FP: 1750 -> 1730   (-20)
FN: 378 -> 382     (+4)
F1: 0.8495 -> 0.8503
SAE: 10.08% -> 9.62%
```

收益远低于 oracle 上限：

```text
oracle P0-P4: FP 1750 -> 46, F1 0.849 -> 0.980
```

说明当前物理特征仍不足以支撑生产级 force-off。

---

## 4. 分用户验证结果

### 4.1 U0789

| variant | F1 | Precision | Recall | SAE | FP | FN | kWh_err |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7992 | 0.6677 | 0.9953 | 26.55% | 742 | 7 | +119.26 |
| physical point guard | 0.8023 | 0.6732 | 0.9927 | 25.40% | 722 | 11 | +114.09 |

结论：

```text
只减少 20 个 FP，却增加 4 个 FN。
无法解决 U0789 的 OFF 虚块主问题。
```

### 4.2 U0800

| variant | F1 | Precision | Recall | SAE | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.7227 | 0.5806 | 0.9568 | 19.57% | 767 | 48 |
| physical guard | 0.7227 | 0.5806 | 0.9568 | 19.57% | 767 | 48 |

U0800 train+val 中没有 predicted-ON 负样本：

```text
train_label_counts = {1: 202}
```

所以 point guard 被禁用。

### 4.3 U0778

U0778 train+val predicted-ON 负样本只有 1 个：

```text
train_label_counts = {0:1, 1:771}
```

因此 guard 被禁用。生产化需要更多 OFF false block 样本。

### 4.4 U2844

| variant | F1 | Precision | Recall | SAE | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.8953 | 0.9290 | 0.8640 | 11.21% | 52 | 107 |
| physical point guard | 0.8953 | 0.9290 | 0.8640 | 11.21% | 52 | 107 |

在独立 guard 脚本中 U2844 当前没有实际改善；U2844 的主要问题仍是 recall / FN，需要 bus_guard 滑窗锚或 recall guard。

---

## 5. 为什么仍不能生产上线？

### 5.1 U0789/U0800 OFF 虚块在物理特征上仍像 ON

即使加入 raw bus 物理特征，U0789/U0800 的 OFF 虚块仍表现为：

```text
p_on 高
pred_on_n 多
y_pred 高
raw bus 段内负荷也高
```

这些特征与真实 ON 很难区分。

### 5.2 负样本不足

关键用户缺少 train/val 负样本：

```text
U0778 负样本 1 个
U0800 负样本 0 个
```

没有足够监督信号训练稳定 guard。

### 5.3 缺少“非空调基线”特征

raw bus 高不能说明目标空调高功率。需要进一步估计：

```text
household / non-AC baseline
```

否则无法判断高 raw bus 是目标设备还是其他负荷。

---

## 6. 生产判断

```text
[FAIL] 当前独立 guard 模型不建议上线。
```

原因：

- 只带来极小 FP 改善；
- 没有解决 U0789/U0800 主问题；
- 部分用户负样本不足；
- 距离 oracle 上限差距巨大。

继续建议上线的只有：

```text
P4 日报监控字段
```

---

## 7. 下一步建议

要继续生产化 force-off，必须补充更强物理信号或样本：

1. **新增独立 guard 标注集**  
   不只是把新增日期放入配置，还要明确：
   ```text
   guard_train / guard_val
   false_on / true_on / no_positive / partial-day 标签
   ```

2. **非空调基线估计**  
   估计 raw bus 中非目标负荷基线，构造：
   ```text
   raw_bus - non_ac_baseline - predicted_ac_power
   ```

3. **目标设备启动/停止响应特征**  
   利用 raw bus 在目标 ON/OFF 边界的响应，而不是只看段内均值。

4. **更多 U0789/U0800 OFF false-block 样本**  
   当前 train/val 负样本不足，无法训练出稳定 guard。

5. **先做灰度监控，不做自动 force-off**  
   可以先输出 guard risk score，而不是直接改变预测。

---

## 8. 最终结论

本轮已经按要求完成：

```text
1. 独立 guard 数据集
2. 物理信号特征
3. 每用户方向验证
4. 5 用户验证测试
```

结果显示：

```text
当前独立 guard 模型仍不满足生产上线条件。
```

所以生产策略保持：

```text
上线 P4 报表监控字段；
force-off 预测逻辑继续作为实验，不进入生产。
```
