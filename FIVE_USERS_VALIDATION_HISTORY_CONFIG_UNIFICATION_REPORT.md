# 5用户验证历史复盘与配置统一/自动生成方案

## 0. 结论摘要

1. “现在验证效果差很多”不是单一代码回归，而是**口径从 oracle 上限验证转到可生产约束验证**后，无法继续使用 inference 标签；同时尝试把 6月异常/OFF 样本直接合入主分类器会导致主模型过度保守。

2. 当前生产主路径 `data/time_filters.json` 的 5用户 pooled inference 仍是 F1≈0.849、Precision≈0.774、Recall≈0.941、SAE≈10.08%；最大问题是 U0789/U0800 的 FP 虚块。

3. Oracle P0-P4 上限能把 F1 提到≈0.980，但使用 inference branch 标签，不可生产；train/val-only 独立 guard 目前只能把 F1 从 0.8495 提到 0.8503，说明可用生产信号不足。

4. 5用户配置差异过大，本质是把 main model / guard / calibration / OOD validation 混在一个 `time_filters.json` 里手工维护。建议统一成分层 schema，并用数据画像自动生成。


## 1. 验证历史关键节点

| stage                             | source                                                    |     F1 |   Precision |   Recall |    SAE |    MAE_W |   TP |   FP |   FN |   TN | 结论               |
|:----------------------------------|:----------------------------------------------------------|-------:|------------:|---------:|-------:|---------:|-----:|-----:|-----:|-----:|:-----------------|
| A. 当前生产主路径 data/time_filters.json | artifacts/summary_metrics_all_users.csv                   | 0.8495 |      0.7743 |   0.9408 | 0.1008 | 145.9758 | 6004 | 1750 |  378 | 8568 | 当前线上可复现基线；FP虚块主导 |
| E. 独立guard+raw bus物理信号            | artifacts/five_user_independent_guard/summary_metrics.csv | 0.8495 |      0.7743 |   0.9408 | 0.1008 | 145.9758 | 6004 | 1750 |  378 | 8568 | 见专项报告            |
| C. 6月样本直接合入主训练器                   | FIVE_USERS_AUGMENTED_TRAIN_CONFIG_VALIDATION_REPORT.md    | 0.5836 |      0.8616 |   0.4413 | 0.6420 | nan      | 2584 |  415 | 3272 | 8031 | 失败：主分类器过度保守      |

## 2. 当前生产主路径各用户 inference 指标

| user   |                    user_id |     F1 |   Precision |   Recall |    SAE |    MAE_W |   kWh_true |   kWh_pred |   kWh_err |   TP |   FP |   FN |   TN |   n_samples |
|:-------|---------------------------:|-------:|------------:|---------:|-------:|---------:|-----------:|-----------:|----------:|-----:|-----:|-----:|-----:|------------:|
| U842   | 800080252842_4206894986488 | 0.9144 |      0.9812 |   0.8560 | 0.1460 | 108.0704 |   224.9195 |   192.0877 |  -32.8317 | 1201 |   23 |  202 | 1354 |        2780 |
| U2844  | 800080252844_4206894986488 | 0.8953 |      0.9290 |   0.8640 | 0.1121 |  71.9227 |   127.3355 |   141.6095 |   14.2739 |  680 |   52 |  107 | 1657 |        2496 |
| U0778  | 800080270778_4200903422131 | 0.9458 |      0.9044 |   0.9912 | 0.0038 | 142.8435 |   247.9440 |   246.9901 |   -0.9539 | 1570 |  166 |   14 | 1706 |        3456 |
| U0789  | 800080270789_4206680982373 | 0.7992 |      0.6677 |   0.9953 | 0.2655 | 326.7073 |   449.1140 |   568.3746 |  119.2606 | 1491 |  742 |    7 | 1600 |        3840 |
| U0800  | 800080270800_4200904302272 | 0.7227 |      0.5806 |   0.9568 | 0.1957 |  50.7795 |    63.7890 |    76.2731 |   12.4841 | 1062 |  767 |   48 | 2251 |        4128 |

## 3. 当前配置差异画像

| user   | target_col   |   train_segments |   train_days_declared |   infer_include_segments |   infer_exclude_days |   split_train_segments |   on_thr_w | v14   | bus_guard   | power_temp_calib   | calib_stats_include            |
|:-------|:-------------|-----------------:|----------------------:|-------------------------:|---------------------:|-----------------------:|-----------:|:------|:------------|:-------------------|:-------------------------------|
| U842   | p1           |                4 |                   336 |                        2 |                    6 |                      4 |    10.0000 | True  | False       | False              |                                |
| U2844  | p2           |                1 |                   330 |                        2 |                    0 |                      0 |    10.0000 | True  | True        | True               |                                |
| U0778  | p2           |                1 |                    19 |                        2 |                    0 |                      1 |    50.0000 | True  | False       | True               | [['2026-05-21', '2026-06-30']] |
| U0789  | p1+p2        |                1 |                    15 |                        2 |                    0 |                      0 |    60.0000 | True  | False       | True               | [['2026-05-21', '2026-06-30']] |
| U0800  | p1           |                1 |                     6 |                        2 |                    0 |                      0 |    50.0000 | True  | False       | True               | [['2026-05-21', '2026-06-30']] |

## 4. 数据/日级问题画像

| user   |   train_daily_rows |   train_on_days |   train_off_days |   train_partial_days |   infer_days |   infer_on_days |   infer_off_days |   infer_F1_lt90_on |   infer_SAE_gt20_on |   infer_off_fp_days |   infer_partial_days |
|:-------|-------------------:|----------------:|-----------------:|---------------------:|-------------:|----------------:|-----------------:|-------------------:|--------------------:|--------------------:|---------------------:|
| U842   |                 80 |              62 |               18 |                    4 |           30 |              29 |                1 |                  5 |                  14 |                   0 |                    3 |
| U2844  |                 34 |              26 |                8 |                    1 |           26 |              18 |                8 |                  4 |                   7 |                   0 |                    0 |
| U0778  |                 19 |              19 |                0 |                    0 |           36 |              34 |                2 |                  1 |                  13 |                   2 |                    0 |
| U0789  |                 15 |              15 |                0 |                    0 |           40 |              29 |               11 |                  2 |                  15 |                  11 |                    0 |
| U0800  |                  6 |               6 |                0 |                    0 |           43 |              28 |               15 |                  3 |                  19 |                  15 |                    0 |

## 5. 为什么现在验证效果差很多

### 5.1 Oracle 上限与生产约束不是同一问题

- Oracle P0/P2/P3 直接使用 inference branch label 判断 OFF/非目标点，等价于“已知答案后 force-off”，因此 F1/Precision 极高。
- 生产中无法读取未来/inference 标签，只能使用 train/val 中可观测的总线、模型输出、天气、coverage 等信号；这些信号目前不足以区分 U0789/U0800 的高置信 OFF 虚块。

### 5.2 直接补充 6月样本进主分类器导致边界重塑

- `time_filters_augmented_forceoff.json` 把 OFF/异常样本合入主 NILM 训练后，U0789 best_thr 从 0.02 跳到 0.39，最终 TP=0/F1=0。
- U0778/U0800 Recall 也大幅下降，说明主分类器被训练成过度保守。
- 这证明新增样本应训练独立 guard，而不是污染主分类器。

### 5.3 训练侧负样本不足且不代表 OOD 虚块

- U0800 train+val predicted-ON 负样本为 0，U0778 只有 1 个，U0789 只有 8 个；不足以训练稳定 force-off。
- U0789/U0800 的 OOD OFF 日模型输出 p_on 很高、pred_on_n 很多、raw bus 也高，与真实 ON 在现有特征空间中相似。

### 5.4 总量 SAE 存在抵消假象

- U0778/U0800 清除 FP 后，ON 日低估暴露，SAE 反而可能变差；因此必须把 force-off 与功率补偿成对设计。


## 6. 统一配置方案

建议把当前单层 `time_filters.json` 拆为统一 schema：

```json
{
  "global_defaults": {"holdout_ood_start": "2026-07-01", "sample_period_min": 15},
  "users": {
    "<uid>": {
      "target": {"target_col": "p1", "on_thr_w": 10},
      "main_model": {"train_include": [], "split_policy": "auto_stratified"},
      "guard": {"train_include": [], "val_include": [], "enabled_candidates": []},
      "calibration": {"stats_include": [], "power_temp_calib": {}},
      "evaluation": {"inference_include": [], "inference_exclude": [], "ood_holdout": []}
    }
  }
}
```

核心原则：
- main_model 训练集只训练主分类/回归；
- guard 数据集只训练后处理一致性守卫；
- calibration 数据集只估计功率/温桶/时段先验；
- evaluation/OOD 永不参与参数选型。


## 7. 自动生成配置参数的设计

自动生成器应分 6 步：

1. **数据发现**：扫描 `data/trains/<uid>` 与 `data/infers/<uid>`，识别 bus/branch 文件、可用 pN 列、raw coverage。
2. **目标画像**：按候选 target_col 计算每日 kWh、ON 样本、ON/OFF 天、低/中/高功率模式、coverage。
3. **阈值生成**：`on_thr_w = max(业务下限, OFF噪声p99 + margin)`，并限制到用户类别默认范围；当前 10/50/60W 应由噪声与目标功率分布自动推导。
4. **main train/split 生成**：选择 OOD 之前且覆盖 ON/OFF/功率模式/天气桶的日期；若 OFF 日不足，标记为 “main不足，不强行补 guard 样本”。
5. **guard 数据集生成**：从 OOD 前或可回流区间选 OFF虚块/低真值高FP/非目标日，单独进入 `guard.train/val`，不进入 main_model。
6. **calibration 生成**：按 train侧覆盖的温度/湿度/功率桶选择 `calib_stats_include` 和 power_temp_calib 参数；7月仅评估。


## 8. 当前五用户自动生成建议

- U842：main 仍用原配置；P0-P3 focused guards 应转成 `guard` 候选，并做 train/val 泛化选型；低功率长时需要额外样本。

- U2844：main 可保持，guard 需要低真值高FP与 recall 两类；bus_guard 滑窗锚应自动生成。

- U0778：main 不应加入 6/24-6/25；这些应进入 guard 数据集；同时 calibration 要补偿高温低估。

- U0789：必须单独建 force-off/电路一致性 guard；main 直接补 OFF 会导致全关断。还需要单/双开 mode。

- U0800：main 样本太少，优先补 OFF 与高温 ON；guard 必须等待负样本足够，否则只能监控不自动 force-off。


## 9. 生产建议

当前可生产：P4 报表监控字段。

当前不可生产：P0/P1/P2/P3 预测逻辑、`time_filters_augmented_forceoff.json`、当前 train/val-only force-off guard。

下一步代码优先级：实现 `generate_auto_time_filters.py` 或 `build_user_data_profile.py`，先自动产出上述分层配置草案与数据充分性诊断，再做真正的 guard 模型训练。
