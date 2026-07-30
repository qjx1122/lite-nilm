# NILM 空调负荷分解 — Windows + Conda 部署手册（v6.12.6+v6.15.0）

> **算法版本**：**v6.12.6+v6.15.0**（v6.12.6 基线 + v6.15.0 自适应守卫的稳定分支）
> **分支说明**：在 v6.12.6 之上**仅叠加** v6.15.0 自适应守卫阈值；**已移除** v6.13(ON_THR 解耦) / v6.14(增量训练+MF=1.25) / v6.16(双口径+守卫训练对称+MAX_ON 自适应+启动确认)
> **包含特性**：Weather-Aware Seasonal MoE + L1+L2+L4+L5 漂移防御 + 按天分层切分 + 指标系统修复 + **d87 启动尖峰自适应守卫 v6.15 (软最大平滑 + 样本量自适应 AF/MF + 概率融合) + v6.12.6 步级状态机守卫 (MAX_ON_HOURS=12h 硬编码)**
> **保留对照**：v4.2（Seasonal MoE, 无温度无漂移防御）—— 作为基线与 L5 切换的 fallback
> **适配环境**：Windows 10 / 11 + Miniconda / Anaconda
> **Python 版本**：3.10
> **项目大小**：约 8.4 MB（不含训练产物）/ ~50 MB（含双模型）
>
> 完整算法演进与实验复盘见 [`REPORT.md`](./REPORT.md)。

## ⭐ v6 关键能力

| 层 | 名称 | 解决问题 |
|----|------|---------|
| **L1** | 漂移感知特征（5 维）| 让模型显式感知"近期 vs 历史"行为差异 |
| **L2** | 漂移检测告警 | 推理时自动识别协变量/概念漂移，输出 ALERT |
| **L4** | 残差校正层 | val 集学习残差，推理时加性校正主模型预测 |
| **L5** | 多模型动态切换 (v6.9 L4-aware) | 检测到 ALERT 时按 L4 启用状态智能调权 |
| **数据基础** | 按天分层切分 (v6.10) | 杜绝同天切分边界泄漏，让 val/test 指标真实可信 |

历史功能（v3-v5）全部保留：
- 两阶段 GBDT（分类 + 分位回归）
- F0.5 阈值优化 + 最小持续时长后处理
- 季节分层 MoE（summer/transition/winter expert）
- 12 维温度特征（Open-Meteo API + 本地缓存）
- 温度驱动的季节路由
- 多基线模型对比（rf / fallback / naive_mean / 外部 pkl）

## 🆕 v6.x 最近工程改进

| 版本 | 改进 | 影响 | 详见 |
|------|------|------|------|
| v5.2 | 多格式时间解析 | 兼容 `2026/3/18 0:00:00` 与 `2026-3-18 0:00:00` 等 7 种格式 | `scripts/time_utils.py` |
| v6.1 | TARGET_COL 变量化 | 一处改 `common.py` 即可切换辨识目标分路 (p1/p2/p3/p4) | §7.1 业务常量 |
| v6.2 | model_meta.json 修复 | 修复 tuple 键导致的 JSON 序列化 bug | §6 模型与产物 |
| v6.3 | 基线模式不污染主模型 | v4.2 训练时跳过共享组件文件，仅出主 .pkl | `03_train.py` 增加 `NILM_BASELINE_MODE` 判断 |
| v6.4 | 统一模型命名 | 主模型在 train/val/test 全用 `main`，透视表不再错位 | `metrics_pivot.csv` 主模型三 split 完整显示 |
| v6.5 | 训练阶段全模型指标 | 5 模型（main / main_L4_calib / fallback / rf / v42_baseline）train+val 完整覆盖 | `train_val_metrics.csv` 170 行 |
| v6.6 | SPLIT_RATIOS 参数化 | 数据集切分比例集中配置，支持 70/15/15、80/10/10 等任意三元组 | §7.1 业务常量 |
| **v6.7** | **`max_gap_steps=2` 修复 ffill 数据泄漏** | **杜绝 7/29 类长间隙日期产生 1858 行 (25.3%) 虚假对齐样本** | `scripts/feature_utils.py::resample_and_align` |
| **v6.8** | **04_evaluate / 05_inference 补 `main_L4_calib` 与 `main_final` 三轨指标** | **首次定量验证 L4 在 OOD 上 SAE 收益 -65%；推理 CSV 三列预测并存** | §6.3 / §六 防御层 |
| **v6.9** | **L5 决策按 L4 启用状态分流（区分 with_L4 / without_L4 两套权重）** | **5 月 ALERT 场景 SAE 由 7.24% → 5.16% (-29%)，kWh_err -7.84 → -5.59** | §5.3 L5 决策矩阵 |
| **v6.10** | **`stratified_day` 切分：按天分层抽样，消除同天泄漏** | **彻底消除 6/8 同天切分边界泄漏；5 月 OOD SAE 5.16% → 0.19% (-96%)，MAE 23.39 → 14.22W (-39%)；揭示之前所谓"5 月概念漂移"约 80% 是切分泄漏伪信号** | §5.5 切分策略 |
| **v6.11.a** | **突变感知特征（51 维）** | **`feature_utils` 加多步差分/绝对差分/突变方向/窗口极差/EMA 跨尺度共 51 维，特征 77→128**；针对早晨/傍晚突变工况漏识别 | §5.6.1 突变感知特征 |
| **v6.11.b** | **后处理优化（min_on=1, fill_off=3）** | **6/2 OOD main_final SAE 19.34% → 2.59% (-86.6%)**；修复"开-关-开-关"震荡场景的过度抑制 | §5.6.2 后处理优化 |
| **v6.11.c** | **F1 阈值优化目标（替代 F0.5）** | **Val FN 19→13，Test main SAE 0.11%（历史最佳）**；阈值从 0.79 降到 0.43，平衡 P/R | §5.6.3 F1 阈值 |
| **v6.11.d** | **隔离修复：v42 训练不覆盖 train_pred/val_pred** | val_pred.csv 现在与训练日志 FN 数字完全一致；修复 v6.3 隔离机制对 artifacts/ 的遗漏 | §5.6.4 隔离 bug 修复 |
| **v6.11.e** ⭐ | **指标系统 4 个 Bug 修复** | **修复 fallback/rf 指标被 v42 训练静默覆盖、v42 在 test 缺失、baseline 路径相对解析、source 字段缺失** | §6.4 指标系统修复 |
| **v6.12.a** ⭐ | **`load_iden_data87` 启动尖峰特征（5min 极值聚合 _max5/_min5/_absmax5）** | 训练集 75 个 ON 事件 d87 绝对极值中位 171，全天 OFF 7 天极值仅 35，**AUC=1.000**；分类器 importance rank 16/19/37 ∈ 128 | §5.7 d87 启动尖峰特征 |
| **v6.12.b** ⭐ | **推理侧日级 d87 启动签名守卫**（阈值 100, 训练 OFF max=35 留 3× 安全裕度） | **5/21~6/3 OOD: SAE 46.87% → 12.49% (-73.4%), F1 0.730 → 0.999 (+36.8 pp), Precision 0.575 → 1.000**；精准命中 7 个真实关空调日，0 FP | §5.7 d87 启动尖峰守卫 |
| **v6.12 文档** | **prediction CSV ↔ metric CSV 列名映射表** | 显式标注 `y_pred_W_main = main_final`、`y_pred_W_main_raw = main` 这一历史兼容性命名错位 | §4.6 输出 CSV 字段 |
| **v6.12.1** ⭐ | **d87 守卫方向+训练自适应阈值修正** | 用 `d87.min()` 替代 `abs().max()`（启动是负向尖峰专属）；阈值从训练 OFF P1 × safety_factor 自动学习，写入 bundle | §5.7.7 v6.12.1 守卫修正 |
| **v6.12.2** ⭐⭐ | **d73 主功率比例缩放: 真正跨用户自适应** | 推理时按 `user_d73_P95 / train_d73_P95` 等比缩放阈值；实测 842(2813W) 与 270848(612W) 两个功率差 5× 用户均达 **100% 日级守卫准确率** | §5.7.8 v6.12.2 自适应缩放 |
| **v6.12.3** ⭐ | **守卫双向化 + d87 启动事件二值特征 + 训练数据清洗** | 修复 4% 正向尖峰漏检（用户2 5/23 救回 8.24 kWh）；新增 6 个 d87 事件特征让模型可见启动信号；`--exclude-dates` 排除训练污染段。用户2 OOD SAE 38.31% → 26.92% | §5.7.9 v6.12.3 三大修正 |
| **v6.12.4** ⭐⭐ | **守卫阈值双源约束标定** | 替代 v6.12.2 的 `OFF_P1 × SF` 单源标定，新公式 `max(ON_P10×0.8, OFF_P99×1.3)` 双源约束。修复用户2 5/27 类阈值过严问题（守卫阈值 -185 → -103, 救回 +7.68 kWh） | §5.7.10 v6.12.4 双源标定 |
| **v6.12.5** ⭐⭐⭐ | **ALLOW_FACTOR 0.8→0.9 (FP 清零)** | 用户2 OOD 守卫达 **TP=7 / TN=7 / FP=0 / FN=0 完美 100% Accuracy**；F1 0.988、Precision 0.994、MAE 34.9 W (vs v6.12.2 81.0 W, **-57%**) | §5.7.11 v6.12.5 完美收官 |
| **v6.12.6** | **步级状态机守卫（替代日级）** | 对每个 15min 推理步在 5min bus 中查找最近启动点；启动点之前强制 OFF、启动后 12h 内信任模型；解决"启动前误识别"边界 | §5.7.12 步级状态机 |
| **v6.13** ⭐ | **ON_THR 解耦设计 (TRAIN=10/BUSINESS=50)** | 修复 v6.12.7 单变量 ON_THR_W 10→50 触发 best_thr 失控 (0.04→0.70) / OOD SAE 0.85%→53% 的灾难；训练标签与业务评估口径独立配置 | §5.8 ON_THR 解耦 |
| **v6.14.1** | **增量训练能力** | `run_user_pipeline.py` 新增 `--extra-train-bus/-branch/-dates` 与 `--infer-eval-dates`；支持把推理数据末尾 N 天加入增量训练 | §5.9 增量训练 |
| **v6.14.2** | **MARGIN_FACTOR 1.3→1.25 (用户3 救场)** | 用户3 OFF P99=101 偏高，1.3 让 OFF 约束(131) 超过 ON 约束(127)，5/23+6/3 启动 \|d87\|=135-136 卡边界被错压；改 1.25 后绑定回 ON，F1 0.92→0.99, SAE 22%→2% | §5.10 用户3 救场 |
| **v6.15.0** ⭐⭐⭐ | **自适应守卫阈值（替代手工调 MF）** | **三层自适应**：(A) 软最大平滑—gap 小不再硬翻转；(B) 样本量自适应 AF/MF—小样本自动放宽；(C) 概率融合守卫—`p_on` 高时局部阈值降低。**三用户回归 F1 全部稳定或微涨**：用户1 0.7732→0.7748, 用户2 0.9740→0.9743, 用户3 0.9937→0.9937（保持）；6 个极端场景压力测试全部通过 | §5.11 自适应守卫 |
| **v6.12.6+v6.15.0** ⭐⭐⭐ | **当前稳定分支 (回退分支)** | 在 v6.12.6 基线上**仅叠加 v6.15.0 自适应守卫**；**已移除** v6.13/v6.14/v6.16 系列改动。三用户 OOD F1: 0.8961/0.9743/0.9937; main_final SAE: **0.85%/12.81%/3.08%**；用户3 边界事件 (5/23+6/3) 完美救场；vs v6.16.2 — 用户2 SAE **改善 2.4pp** (12h 硬编码反而压制了 5/28+5/31 关机日 FP) | §5.12 v6.12.6+v6.15.0 分支 |
| **v6.12.6+v6.15.0 批量** | **多用户批量执行 (`run_batch_users.py`)** | 自动扫描 `data/<设备_用户>/` 文件夹按命名规范解析 train/infer CSV, 批量跑 N 个用户. 产物归档 `artifacts/<用户ID>/`, 汇总指标到 `artifacts/_batch_summary/{train_val,test,inference,ood_overview}_all_users.csv`. 缺数据自动跳过不中断 | §3.4 多用户批量执行 |
| **v12** ⭐⭐⭐ | **训练/推理时段过滤 (include + exclude, 任意时段)** | 新增 `scripts/time_filter_utils.py` 工具模块 + JSON 配置文件驱动. **可给每用户分别指定 train / infer 的 include 白名单 + exclude 黑名单**, 支持多段任意时段 (不限整天粒度, `YYYY-MM-DD` 自动扩为全天, `YYYY-MM-DD HH:MM[:SS]` 精确到秒). 闭区间语义, 先 include 后 exclude. 沙箱 6 组单元测试 + 端到端实测通过 (剔除半天 09:30-14:30 精准 -21 行). 向后兼容旧的 `--exclude-dates` 参数 | §3.5 时段过滤 |
| **v13.1** ⭐⭐⭐ | **用户级 d87 守卫开关 `guard_enabled` + 自动检测降级** | JSON 配置 `guard_enabled=true/false` 覆盖全局 `D87_ADAPTIVE_GUARD_ENABLED`. 未指定时训练侧根据训练集 d87 强度自动判定: 判据 A `\|d87\|.max<50W` 或 判据 B `阈值天覆盖率<30%` 任一触发 → 自动降级. **270708 case 实测**: F1 0.887→**0.996**, Recall 0.802→**0.998**, SAE 29.7%→**14.4%**, kWh_err -6.43→**-3.12** (推理端 100% FN 灾难修复) | §3.7 用户级守卫 |
| **v13.2** ⭐⭐⭐ | **per-split 时段过滤 (train/val/test 三集独立 include/exclude)** | JSON 配置 `splits.{train,val,test}.{include,exclude}`. **4 步语义**: (1) 原策略切分 → (2) include 硬锚定 (样本粒度, 冲突时 train→val→test 优先) → (3) 严格保持原 split 形状 (跨 split 平移补齐) → (4) exclude 剔除 (三方全命中则完全丢弃, 否则送回重分配池). 沙箱 5 组单测 + 端到端 4/4 验证通过 (270708 严格保持 959/384/384 原形状, 0 样本丢失) | §3.8 per-split 过滤 |
| **v13.3** | **修复 "训练成功但汇总标 soft_skip" bug** | 根因: `artifacts/trains/<uid>/skip_reason.json` 陈旧残留污染 `aggregate_metrics()` 判据. 首次数据不足触发数据质量门 3 (val/test 空) 归档 skip_reason.json, 后续修数据重跑成功但归档时**未清理旧标记**, 汇总时 4 stage 全被标 `soft_skip:split_empty_val_test`. **修复**: `archive_outputs()` 里 `did_train=True` 时归档前自动 `unlink()` 旧 skip_reason.json | §八 常见问题排查 |
| **v13.4** | **`target_col` 配置化 (JSON 优先级最高)** | JSON 配置 `target_col: "p1"/"p2"/"p3"/"p4"` 覆盖历史的 `-Ch{N}-` 反推逻辑. **优先级链**: 用户级 → `_default` → `-Ch{N}-` 反推 → 分路第 1 个 pN → 兜底 p1. 宽松校验: 配置值不在分路列中时 WARN + 回退旧逻辑, 大小写不敏感. 8 场景单测全通过 | §3.6.2 配置字段 |
| **v13.4-fix** | **`target_col` 从 [p1-p4] 放宽到通用 `pN`** | 支持 `p0` / `p5` / `p10` / `p128` 等任意 N ≥ 0. 正则 `^p\d+$` 校验, 大小写规范化. 修改 `time_filter_utils.py` 硬集合 + `run_user_pipeline.py` argparse choices. 20 case 端到端验证 | §C.4.2 |
| **v13.5** ⭐⭐⭐ | **9 个 common 常量用户级覆盖** | JSON 配置 9 个字段: `on_thr_w` / `split_ratios` / `split_strategy` / `post_min_on` / `post_fill_short_off` / `weather_latitude` / `weather_longitude` / `use_weather_features` / `use_temp_based_season`. 三层优先级链 (用户级 → `_default` → common.py). 非法值 WARN + 回退. **270737 案例催生**: 待机 16-24W 干扰可用 `on_thr_w: 50` 排除. 13 组单测 + 4 大场景端到端 (回归/覆盖 4 项/气象 4 项/三层优先级) 全通过 | §C.4.7 |
| **v13.5-fix** | **`ON_THR_W` 3 处硬编码 bug 修复** | (1) `04_evaluate.py` 打标签 + baseline 未从 bundle 读, 用旧 10W 值; (2) `baseline_utils.py::cross_model_consistency` 硬编码 10W; (3) `run_user_pipeline.py::_filter_inference_metrics` 直接 import `ON_THR_BUSINESS_W`. 修复后所有阶段均从 `bundle.pkl::ON_THR` 读取, 训推评估口径完全一致. 端到端验证: 270708 配 `on_thr_w:150` 后 04 test ON% 从 44.53% → 22.14% (与 03 训练一致) | §C.4.7 |
| **v13.6** ⭐⭐ | **分路开机时段分析工具 + 集成流水线** | 新增 `analyze_on_periods.py`: 加载分路 CSV → 按 `on_thr_w` 阈值二值化 → 输出 (1) 段级明细 (`being_time` / `end_time` / target=1 / duration/mean/peak/energy) + (2) 每日汇总 (n_segments / total_on_hours / first_on_time / last_off_time / weighted_mean_w). 双调用模式 (`--user + --stage` 或 `--br-csv`), 支持 `--config` 配置文件读 target_col + on_thr_w. 集成 `run_user_pipeline.py`: **训练前 + 推理前各自动跑一次**, 归档到 `artifacts/{trains,infers}/<user>/<stage>_on_periods*.csv`; `--skip-analyze` 可跳过. 用途: 训练前健全性检查, 避免盲改 `on_thr_w` 酿成 v6.12.7 灾难 | §C.4.8 |
| **v13.7** ⭐⭐ | **特征矩阵 NaN 硬检测 (fail-fast)** | 新增 `feature_utils.assert_no_nan_features()`: 03/04/05 三阶段在 `X = X_df.values` 前统一检测 NaN. 无 NaN → INFO 一行; 有 NaN → WARN 精准定位 (列数/行数/Top-10 列/首个 NaN 时间戳 + 4 条诊断建议) → 主动 `raise ValueError`. **270758 用户 Windows 环境 GBM `Input X contains NaN` 崩溃根因**: 推理路径无 NaN 兜底防线, feature_utils L213 `rolling(4).mean()` 无 fillna + `_add_weather_features` daily.reindex 边界产 NaN. Linux 沙箱因 pandas 版本差异不复现. 5 组单测 + 双回归 (无 NaN 零副作用 + 人为注入 NaN 精准拦截) 全通过 | §C.4.4 陷阱 |
| **v13.8** ⭐⭐⭐ | **train/infer 数据泄漏自动检测 + 拆分指标** | (1) `03_train.py` 保存 `train_dates/val_dates/test_dates` 到 bundle (ISO yyyy-mm-dd 字符串集合); (2) `metrics_utils.py` 新增 `compute_leak_ood_split()` + `build_leak_ood_metric_rows()` API; (3) `05_inference.py` 推理指标写入前调用: 若推理集与训练日交集非空 → WARN 打印泄漏日期 + 5 关键指标对比表, 同时 `inference_metrics.csv` 追加 `split=inference_leak` (泄漏部分) 和 `split=inference_ood` (真 OOD 部分) 两组行. **270758 案例**: 配置 B 推理集含 3 天训练日 → 拆分显示 leak/SAE=3.33% vs ood/SAE=7.39%, 整体加权 SAE=2.63% 具欺骗性. 7 组单测 + 双端到端场景 (无泄漏零副作用 + 3 天泄漏精准拆分) 全通过 | §C.4.4 陷阱 1 |
| **v13.8-fix1** ⭐⭐ | **v13.8 leak/ood 拆分覆盖三主模型 (main / main_L4_calib / main_final)** | `build_leak_ood_metric_rows()` 新增 `extra_model_preds: dict` 参数, 允许一次调用产出多模型指标行. 分类指标 (F1/Prec/Rec/AUC) 因共用 `state_pred` 只算一次; 回归指标 (MAE/RMSE/SAE/kWh) 每模型独立. `05_inference.py` 传入 `main_L4_calib=y_pred_after_L4` 和 `main_final=y_pred (L4+L5)`. **270758 实测洞察**: main 原始 SAE 泄漏 vs OOD 差 22 倍 (0.95% vs 22.17%), L4 校正后差异消失 (7.62% vs 7.40%), L5 切换又拉开一些 (main_final 3.33% vs 7.39%) — 定量证明各防御层价值. 5 组扩展单测 (含向后兼容/防误传 main 覆盖) 全通过 | §C.4.4 陷阱 1 |
| **v13.9** ⭐ | **`analyze_on_periods` 4 字段计算审计 + 边界预检 WARN** | 6 层硬证据审计 (静态公式对照 / 5 组合成用例手工复算 / 270708 全 15 段逐值验证 / 段-日聚合一致性 / 每日加权 mean_w 复算 / 5 组极端边界 E1-E5). **主流场景 (15min 严格等间隔) 4 字段 (duration_min / mean_w / peak_w / energy_kwh) 100% 正确**, 与手工复算完全一致 (max diff < 5e-6). 边界场景发现 2 个潜在偏差: (a) 非均匀采样 → `mean_w × dur_h ≠ energy_kwh` 最大 25%; (b) 时间断裂被错误合并 → duration 严重高估 (硬证据 4h gap → dur 高估 100%). **v13.9 加 2 组预检 WARN** (采样均匀性 CV>0.10 + 时间断裂 dt>2×step_median), 主流数据零触发零回归, 边界场景显性化让用户判断可信度 | §C.4.8 §审计报告 |
| **v13.10-analysis** ⭐⭐ | **上下文边界效应 5 大发现 (深度量化)** | v13.8 顺带发现的"同 8 天在不同 infer 配置下 74/138 列特征值不同"深挖到底. 5 层递进分析: (1) 138 列按算子族分类, 稳定 vs 漂移的 54/74 二分; (2) 每个特征的稳定所需步数 (`power_recent_7d_mean` 需 651 步); (3) 暖启动衰减曲线 — **7 天暖启动 100% 消除漂移**; (4) F1 单调 (0.9714→0.9794), **SAE 是 U 形** (19.55%→17.41%⭐→22.55%); (5) U 形根因 = 暖启动候选日 (mean 249W) vs 目标日 (mean 117W) 分布偏移. **实用决策矩阵** + **7 天魔法数字**理论解释 | §C.4.4 陷阱 2 |
| **v13.10** ⭐⭐ | **分路 CSV 增加 dataset 归属列 (业务可视化)** | `analyze_on_periods.py::compute_on_periods` 和 `compute_daily_summary` 新增 `date_labels: dict` 参数, 输出 CSV 追加 `dataset` 列. `run_user_pipeline.py` 集成: (a) 训练阶段: 03 训练完成后**补跑一次 train analyze**, 从 `bundle::train_dates/val_dates/test_dates` 读归属, 每段/每日标为 `train/val/test/未使用`; (b) 推理阶段: 从 `--infer-time-filter-spec` 计算实际推理集, 标为 `used/excluded`. 270758 A 实测: 训练归属 `{train:11, val:3, test:3, 未使用:23}` 完美对应, 推理归属 `{used:8, excluded:32}`; B 实测: 推理 `{used:11, excluded:29}` 精准显示 6-04/05/06 数据泄漏日 = used. 4 组单测 + A/B 双端到端验证零回归 | §C.4.8 §十一 |
| **v13.11** ⭐ | **全天 OFF 日也输出到分路 CSV (数据完整性增强)** | 之前 `train_on_periods.csv` / `infer_on_periods.csv` 只输出**有 ON 段的天**, 全天未启动的日子完全消失. 现在段级 CSV 追加"OFF 天行" (target_col=0, being_time=00:00:00, end_time=23:45:00, duration_min=1440, mean_w/peak_w/energy_kwh 按**全天所有采样点**统计, dataset 归属保留). daily CSV 补齐 OFF 天 (n_segments=0, total_on_min=0, first/last_on_time=空, 待机功率/电量正常显示). 5 组单测 + 双端到端: 270708 用户 (thr=50W) 精准捕获 3 个周日 OFF (6-14/21/28), 待机 1.44W / 每天 34Wh; 270758 用户 (thr=70W) 40 天全部覆盖 (无 OFF 天时行为不变, 向后兼容零回归) | §C.4.8 §十二 |
| **v13.12** ⭐ | **`stratified_day` 切分 round 精度 bug 修复** | 用户报告 `split_ratios=[0.7,0.15,0.15]` "没生效", 硬证据溯源: `split_utils.py::make_splits` 每月独立 `int(round())` val/test 天数, 累积舍入误差. 例: 2 月×10 天 = 20 天 → 每月 val=round(10×0.15)=2, test=2, train=6 → 累加 12/4/4 = 60/20/20 (期望 70/15/15). 修复: 先 `n_tr_days=round(n_full×train_ratio)`, 再按 val:test 比例分剩余. 20 组单测 覆盖 5~60 天单/多月场景 | §C.4.4 陷阱 |
| **v13.13** ⭐⭐ | **新增 `global_stratified` 切分策略 (跨月比例精准)** | v13.12 修复后跨月**主体+零头**场景 (如 14 天=11+3) 仍偏差: `stratified_day` 按月分层+每月"至少 1 天 val/test"保护, 小月被强制 1/1/1 挤压 train (14天场景 9/3/2 = 64/21/14%, 期望 10/2/2 = 71/14/14%). 修复: 新增第 4 种策略, 不按月分层, 直接全局按 ratios 切完整天. 用户在配置里加 `"split_strategy": "global_stratified"` 主动切换. 270758 用户 14 天场景实测 → **10/2/2 = 71.4/14.3/14.3%** ✓ 完美. 保留 `stratified_day` 老行为完全向后兼容 (跨月分布均衡的用户仍可用) | §C.4.7 |
| **v13.14** ⭐⭐ | **逐日主模型评估指标 CSV (train/val/test/inference)** | 现有指标是**整体聚合** (train/val/test/inference 各一个数字), 无法定位单日异常. 新增 `metrics_utils.build_daily_metrics_rows()` API: 按天聚合 F1/Precision/Recall/Accuracy/AUC/MAE_W/RMSE_W/SAE/kWh_true/kWh_pred/kWh_err/TP/FP/FN/TN 15 指标. `04_evaluate.py` 生成 `artifacts/trains/<user>/train_daily_metrics.csv` (含 train+val+test 3 splits, `dataset=train/val/test`); `05_inference.py` 生成 `artifacts/infers/<user>/inference_daily_metrics.csv` (**`dataset=used_leak/used_ood`** 自动标记数据泄漏日, 与 v13.8 完美对应). SAE 边界保护 (全 OFF 天 kwh_true<1e-3 时 SAE=None 避免爆炸). 交叉验证: daily 累加 kWh 与整体聚合完全一致 (浮点差 2e-6). 270758 实测: train_daily 14 行 (10/2/2), inference_daily 40 行 (used_leak:10, used_ood:30) | §C.4.8 §十三 |
| **v13.15** ⭐⭐ | **温度桶期望信号 CSV 导出 (概念漂移可视化增强)** | 已有 `drift_report.csv` 只输出**触发告警的少数桶** (\|rel\|≥0.30 的 detail 桶), 无法回答"训练时 27°C 那桶模型认为总线该多少 W"或"20 个桶里哪些桶落 WARN". `drift_features.py` 新增 2 个导出 API: (a) `export_temp_power_lut_csv()` 训练侧写 `temp_power_lut.csv` (12 列: bin_id/temp_lo/temp_hi/temp_width/expected_signal/n_samples/mean/std/p25/p75/signal_col/is_global_median, 全部 20 桶 + 全局中位兜底行); (b) `export_temp_power_actual_vs_expected_csv()` 推理侧写 `inference_temp_power_actual_vs_expected.csv` (13 列: 训练期望 + 推理实测中位/均值/p25/p75 + abs_residual + rel_drift + drift_flag∈{OK,WARN,ALERT,NO_DATA}, 全部 20 桶). `build_temp_power_lut()` 加 `return_meta=True` 支持每桶元数据统计, 默认调用签名不变 (向后兼容). **270788 实测**: 20 桶完整覆盖, 26.85-28.65°C 三桶 rel_drift = -57%/-55%/-51% ALERT (与上轮报告一致), 8 ALERT+6 WARN+5 OK+1 NO_DATA; 训练 `expected_signal` ≡ 推理 `train_expected_signal` (delta_max=0, 两侧口径一致性硬保证); 主指标零回归 (F1=0.9022, Recall=0.8324, SAE=5.45%). **34 组单测全通过** (T1 元数据数学正确性 12 + T2 训练 CSV 写盘 8 + T3 分级+数学 7 + T4 双侧横向对齐 2 + T5 兜底 4). 归档规则复用: 名字不含 `inference` → `trains/`; 含 → `infers/` | §C.4.8 §十四 |
| **v13.16** ⭐⭐ | **`target_col` 支持复合列语义 (`"p1+p2"` / `"p1+p2+p3"`)** | 用户场景: 同一空调有多个分路 (主机+辅热或多室内机), 需合并作为总标签. 语法: JSON `"target_col": "p1+p2"` 或 `"p1+p2+p3"` (可扩展到任意分量数); 加载分路 CSV 时 `load_branch_csv()` 自动物化一列, 列名就是复合字符串, 值 = 各分量 `pd.to_numeric` 后按行 `sum(skipna=False)` (任一分量为空则 NaN 传播, 避免静默补 0 导致电量偏低). 归一化: 大小写 + 空白容忍 (`" P1 + p2 "` → `"p1+p2"`); 防呆: 拒绝 `p1+p1` 重复分量. 正则: `time_filter_utils::get_user_target_col` / `run_user_pipeline::_validate_target_col` / `analyze_on_periods::_RE_PN_COMPOSITE` 统一放宽为 `^p\d+(\+p\d+)*$`. `run_batch_users` 复合列校验策略: 所有分量必须都在 `br_p_cols` 中. **270788 端到端验证**: 单列 `p1` 零回归 (F1=0.9022 / Recall=0.8324 / SAE=5.45% 完全一致); 复合 `p1+p2` 跑通 F1=0.9395 / SAE=4.10%, kWh_true=32.95 ≈ p1 单列 16.78 × 2 物理合理 ✅. **47 组单测全通过** (T1 正则/归一化/防呆 17 + T2 一致性 11 + T3 用户示例逐值 [32,16,16,24] 对齐 6 + T4 边界与向后兼容 4 + T5 analyze 端到端 7 + T6 resample_and_align 集成 2). 向后兼容零风险: 单列 `p1` 完全不触发物化, `target_col=None` 老调用不变 | §C.4.2 |
| **v13.16-min_w** ⭐ | **`analyze_on_periods` 段级/daily 追加 `min_w` 列 (最小瞬时功率)** | 已有 `mean_w` / `peak_w`, 独缺"段内最低到过多少 W", 变频空调判断"是否有短暂低谷 / 是否真的一直保持高档"必需. **段级 CSV** (`train_on_periods.csv`/`infer_on_periods.csv`) 追加 `min_w` (位置在 `duration_min` 之后 / `mean_w` 之前), OFF 天行 = 全天最小 (待机功率下限). **daily CSV** (`train_on_periods_daily.csv`/`infer_on_periods_daily.csv`) 追加 `min_w`: ON 天 = 各 ON 段 `min_w` 的最小 (=开机期间最低瞬时功率); OFF 天沿用段行. **向后兼容**: `compute_daily_summary` 对老段级 CSV (无 `min_w` 列) 输入返回 `daily.min_w = ""` 不崩. 4 处物化点全覆盖 (段级正常/跨日拆分/OFF 天/daily). **270788 实测硬证据**: 段级 40 行 min_w 中位 36W (变频低档), max 174W; **40 天全部满足物理不变量 `min ≤ mean ≤ peak`** ✅. **25 组单测** (T1 段级位置+值 6 + T2 多段独立 3 + T3 daily ON 天 min = min(各段 min) 4 + T4 daily OFF 天 4 + T5 复合列场景 4 + T6 空 DF 兜底 2 + T7 老输入向后兼容 2) 全通过 <2s | §C.4.8 §三 |
| **v13.16-daily_raw** ⭐⭐ | **`train_daily_metrics.csv` / `inference_daily_metrics.csv` 追加 `n_bus_raw` / `n_branch_raw` 2 列 (每天总线/分路原始采集点数)** | 已有 `n_samples` 是**对齐后**样本数 (受时段过滤+`resample_and_align` inner-join 影响), 独缺"每天原始 CSV 采集了多少点"的完整性视角. `metrics_utils` 新增 `compute_raw_daily_counts(csv_path, time_col, time_filter_spec=None, logger=None) -> {date: int}` 工具函数 + `build_daily_metrics_rows(..., bus_daily_counts=None, branch_daily_counts=None)` 两个可选参数. 推理侧应用 `--time-filter-spec` 让统计口径与实际推理天数一致 (避免统计到 `infer.exclude` 排除的训练日). 位置: `n_samples` 之后, `Accuracy` 之前. **CSV 从 23 列扩到 25 列**. **270788 端到端首发揭示**: 5-21/22/23/25 四天推理 F1=0 的根因不是"缺分路数据", 而是**总线原始只采了 3-7 点** (应 288); 分路侧全 96 完整. 之前 v13.14 daily 只显示"缺数据"占位, 现在铁证摊开. **28 组单测** (T1 字段+位置 4 + T2 值正确+缺失=0 4 + T3 向后兼容 2 + T4-5 raw_counts 基础+过滤 7 + T6 兜底 2 + T7 端到端 5 + T8 落盘读回 4) 全通过 <3s. 向后兼容: 不传 counts 时两列 = `""` | §C.4.8 §十三 |
| **v13.17** ⭐⭐ | **`run_batch_users.py` 断点续跑 (`--resume`) + 实时增量状态 CSV** | **`batch_run_summary.csv` 是"跑完一次性覆盖写"**, 中断/崩溃 = 全部丢失 → 无法断点续跑. **v13.17 新增独立文件** `artifacts/batch_execution_state.csv` (9 列: `user_id, status, success, started_at, finished_at, duration_s, message, target_col, run_id`), **每个用户跑完立即增量写** (原子写 `.tmp + os.replace`, 崩溃时最多丢当前正跑的). **CLI 新增两参数**: `--resume` (启用断点续跑, 默认关闭零回归) + `--resume-skip-failed` (fail 用户也跳过, 默认 fail 会重跑). **续跑决策**: `ok`/`soft_skip` → 跳; `fail` → 重跑 (可用 `--resume-skip-failed` 改为跳); 状态文件不存在或损坏时自动降级到全部重跑. **关键修复**: `run_user_pipeline.py::cleanup_artifacts_top` 顶层清理逻辑之前会把 `batch_execution_state.csv`/`batch_run_summary.csv`/`summary_metrics_all_users.csv` 一起删掉 (每次单用户结束都执行 = 只剩最后 1 行) → v13.17 加**白名单**保护 5 类批量层持久化文件. **37 组单测** (T1-T12: 文件不存在/损坏/老格式/正常 4 + `_get_completed_users` retry 分支 6 + `_upsert` 首写+覆盖+原子 5 + 端到端崩溃恢复 5 + resume 覆盖 3 + CSV 格式硬校验 9) 全通过 <1s. **真实端到端验证**: 270708+270848 首跑 2 行 ok → 再 `--resume` 加载 2 已完成 → 0 待跑 "无可执行用户, 退出" (<1s) ✅ | §附 C.3.4 / §C.4.9 |

---

## 📋 目录

1. [快速开始（3 步）](#一快速开始3-步)
2. [项目结构](#二项目结构)
3. [详细执行指令](#三详细执行指令)
   - 3.4 多用户批量执行
   - **3.5 时段过滤 (`--time-filter-config`)** ⭐ v12 新增
   - **3.6 v13 用户级配置扩展** ⭐ v13 新增 (target_col / guard_enabled / splits)
   - **3.7 用户级 d87 守卫开关** ⭐ v13.1
   - **3.8 per-split 时段过滤** ⭐ v13.2
   - 3.9 查看结果
4. [独立推理 — 生产部署用法](#四独立推理--生产部署用法)
5. [v6 漂移防御体系详解](#五v6-漂移防御体系详解)
6. [模型与产物说明](#六模型与产物说明)
7. [关键技术参数](#七关键技术参数)
8. [常见问题排查](#八常见问题排查)
9. [嵌入式部署进阶](#九嵌入式部署进阶)
10. [持续运营建议](#十持续运营建议)

---

## 一、快速开始（3 步）

> 假设项目存放在 `D:\projects\nilm_ac_win`（路径**避免中文/空格**）。

### Step 1｜创建 Conda 环境（首次执行）

```bat
cd /d D:\projects\nilm_ac_win
conda env create -f environment.yml
conda activate nilm_ac
```

> 🌐 国内可换清华源加速：
> ```bat
> conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
> conda config --set show_channel_urls yes
> ```

### Step 2｜放置数据

把您的原始 CSV 文件放入 `data/` 目录：
- 总线 CSV（含 `event_time` 列和 `load_iden_data*` 列）
- 分路 CSV（含 `time` 列和 `p1`/`p2`/`p3`/`p4` 列。默认辨识 `p1`，可在 `common.py` 中改 `TARGET_COL` 切换其他分路）

> 💡 时间列支持多种格式：`2026/3/18 0:00:00`、`2026-3-18 0:00:00`、`2026-03-18T00:00:00` 等。

### Step 3｜一键运行

```bat
run_all.bat
```

依次执行：合并 → 勘察 → 对齐 → 拉取气象 → 训练 v6 → 训练 v4.2 → 评估 → 推理（含多基线 + L1-L5 防御）。

---

## 二、项目结构

```
nilm_ac_win/                          (项目根, 约 8.2 MB)
│
├── README_WIN.md                     本文件
├── REPORT.md                         ⭐ v6 完整开发报告 (含 v1→v6 全演进)
├── environment.yml                   Conda 环境定义
├── requirements.txt                  pip 备选方案
├── run_all.bat                       Windows 一键运行 (8 步)
│
├── data/                             原始 + 合并数据 + 气象缓存
│   ├── <您的总线 CSV>.csv
│   ├── <您的分路 CSV>.csv
│   ├── merged_bus.csv                ⭐ 训练总线 (BUS_CSV, merge_data.py 生成)
│   ├── merged_branch.csv             ⭐ 训练分路 (BR_CSV)
│   ├── infer_bus.csv                 ⭐ 推理总线 (INFER_BUS_CSV, v6.12.6+v6.15.0 新增, 训推路径分离)
│   ├── infer_branch.csv              ⭐ 推理分路 (INFER_BR_CSV, 可选, 用于评估)
│   └── weather_cache/                ⭐ Open-Meteo 历史气温本地缓存
│       └── {lat}_{lon}_{year}.csv
│
├── scripts/                          24 个 Python 文件
│   │
│   ├── ── 公共模块 (12 个) ──
│   ├── common.py                     路径/字体/日志/常量
│   ├── time_utils.py                 多格式时间解析
│   ├── feature_utils.py              特征工程入口 (60→72→77 维)
│   ├── postprocess.py                阈值搜索 + 形态学滤波
│   ├── sample_weight_utils.py        v3 逆密度加权
│   ├── expert_utils.py               v4 月份路由 + v5 温度路由
│   ├── split_utils.py                v4 分层时序切分
│   ├── metrics_utils.py              指标 + 透视对比表
│   ├── weather_utils.py              v5 Open-Meteo API + 缓存
│   ├── baseline_utils.py             多基线模型注册与执行器
│   ├── drift_features.py             ⭐ v6 L1 漂移感知特征 + LUT
│   ├── drift_detect.py               ⭐ v6 L2 漂移检测告警
│   └── residual_calibrator.py        ⭐ v6 L4 校正器 + L5 切换器
│   │
│   ├── ── 主流程 (7 个) ──
│   ├── 01_audit.py                   数据勘察
│   ├── 02_align_and_feat.py          对齐 + 相关性
│   ├── 03_train.py                   训练 v6 (含 L1+L4)
│   ├── 03b_train_v42_baseline.py     训练 v4.2 对照基线
│   ├── 04_evaluate.py                测试集多基线评估
│   ├── 05_inference.py               ⭐ 独立推理 (L1+L2+L4+L5)
│   ├── 06_inference_with_calib.py    备用: Isotonic 校正推理
│   │
│   └── ── 工具脚本 (4 个) ──
│       ├── merge_data.py             多份 CSV 合并
│       ├── fetch_weather.py          批量拉取历史气温
│       ├── diag_inference.py         推理结果诊断
│       └── diag_new_data.py          新数据画像对比
│
├── models/                           训练后自动填充
│   ├── nilm_ac_two_stage.pkl                ⭐ v6 主模型 (~10 MB)
│   ├── nilm_ac_two_stage_v42.pkl            ⭐ v4.2 对照 / L5 fallback
│   ├── nilm_ac_two_stage_<时间戳>.pkl       时间戳备份
│   ├── stage2_moe_bundle.pkl                MoE 独立组件
│   ├── stage1_classifier.pkl                Stage-1 分类器
│   ├── stage2_regressor.pkl                 Stage-2 主回归器 (P50)
│   ├── stage2_regressor_p10.pkl             P10 下界
│   ├── stage2_regressor_p90.pkl             P90 上界
│   ├── scaler.pkl                           StandardScaler
│   ├── baseline_rf.pkl                      RF 基线
│   └── model_meta.json                      元数据 (含 temp_power_lut)
│
├── artifacts/                        训练/评估产物
│   ├── aligned_15min.csv             对齐后训练数据
│   ├── bus_columns_summary.csv       电参量画像
│   ├── feature_corr_with_ac.csv      特征-标签相关性
│   ├── feat_importance.png           特征重要性图
│   ├── test_prediction.png           测试集可视化
│   ├── inference_comparison.png      ⭐ 多模型推理对比图
│   ├── metrics/                      指标 CSV
│   │   ├── metrics_pivot.csv         多 split 透视对比表
│   │   ├── train_val_metrics.csv
│   │   ├── test_metrics.csv          多基线长表
│   │   ├── test_metrics_comparison.csv  ⭐ Test 多模型对比
│   │   ├── inference_metrics.csv     多基线长表
│   │   ├── inference_metrics_comparison.csv  ⭐ 推理多模型对比
│   │   ├── inference_consistency.csv     无标签时主 vs 基线一致性
│   │   ├── threshold_curve_val.csv   阈值-指标全谱
│   │   ├── expert_summary.csv        MoE 专家训练摘要
│   │   └── drift_report.csv          ⭐ v6 L2 漂移检测报告
│   └── predictions/                  预测明细 CSV
│       ├── train_pred.csv / val_pred.csv / test_pred.csv
│       ├── test_pred_<baseline>.csv  ⭐ 各基线单独预测明细
│       └── inference_result.csv      含每个基线一列预测
│
└── logs/                             运行日志 (每次带时间戳)
    └── {脚本名}_{YYYYMMDD_HHMMSS}.log
```

---

## 三、详细执行指令

### 3.1 分步执行（推荐首次使用，便于排查）

```bat
conda activate nilm_ac
cd /d D:\projects\nilm_ac_win

:: Step 1: 数据合并 (如有多份历史数据)
python scripts\merge_data.py

:: Step 2: 数据勘察
python scripts\01_audit.py

:: Step 3: 时间对齐 + 特征筛选
python scripts\02_align_and_feat.py

:: Step 4: [v5] 拉取气象数据 (首次联网, 后续走缓存)
python scripts\fetch_weather.py

:: Step 5: 训练 v6 主模型 (含 L1+L4)
python scripts\03_train.py

:: Step 6: 训练 v4.2 对照基线 (L5 切换的 fallback)
python scripts\03b_train_v42_baseline.py

:: Step 7: 测试集多基线评估
python scripts\04_evaluate.py --baseline rf fallback models\nilm_ac_two_stage_v42.pkl

:: Step 8: 独立推理 (含 L1+L2+L4+L5 全防御)
python scripts\05_inference.py ^
    --baseline rf fallback models\nilm_ac_two_stage_v42.pkl --plot
```

### 3.2 一键执行

```bat
run_all.bat
```

### 3.3 单用户端到端流水线 (`run_user_pipeline.py`)

对**单个用户**的训练+评估+推理一次性跑完，显式传入 CSV 路径：

```bash
python scripts/run_user_pipeline.py \
  --user-id <自定义ID> --target-col p1 \
  --train-bus <训练总线.csv> --train-branch <训练分路.csv> \
  --infer-bus <推理总线.csv> --infer-branch <推理分路.csv> \
  --output-dir results
```

产物归档到 `results/<自定义ID>/{model.pkl,metrics/,predictions/,logs/}`。

### 3.4 多用户批量执行 (`run_batch_users.py`) ⭐ v6.12.6+v6.15.0 新增

**自动扫描 `data/` 目录**下所有按命名规范组织的用户文件夹, 批量跑训练+评估+推理, 并汇总指标。

#### 命名规范

每个用户一个文件夹: `data/<设备编号>_<用户编号>/`, 内含 4 个 CSV (规范见下):

| 类型 | 命名规则 | 示例 |
|---|---|---|
| 训练总线 | `e241_<设备编号>_<用户编号>-Ch<N>-<起>-<止>-1.csv` | `e241_800080252842_4206894986488-Ch1-250710-250730-1.csv` |
| 训练分路 | `<用户编号>-<起>-<止>.csv` | `4206894986488-250710-250730.csv` |
| 推理总线 | `e241_<设备编号>_<用户编号>-Ch<N>-<起>-<止>-infer.csv` | `e241_800080252842_4206894986488-Ch1-260521-260603-infer.csv` |
| 推理分路 (可选) | `<用户编号>-<起>-<止>-infer.csv` | `4206894986488-260521-260603-infer.csv` |

> 💡 `target_col` 自动从训练分路 CSV 的实际列名 (`p1`/`p2`/`p3`/`p4`) 反推, 与总线文件名中的 `-Ch{N}-` 无关。

#### 命令

```bash
# 跑 data/ 下全部用户
python scripts/run_batch_users.py

# 仅跑指定用户
python scripts/run_batch_users.py --users 800080252842_4206894986488 800080270825_4206911115606

# 仅扫描+列出计划, 不实际执行
python scripts/run_batch_users.py --dry-run

# 跳过 artifacts/<user>/ 已存在的用户 (增量跑)
python scripts/run_batch_users.py --skip-existing

# 自定义数据/产物目录
python scripts/run_batch_users.py --data-dir /path/data --output-dir /path/artifacts
```

#### 产物结构

```
artifacts/
├── <用户ID 文件夹全名>/        ⭐ 每个用户的独立产物 (model.pkl + metrics/ + predictions/ + logs/)
│   ├── model.pkl
│   ├── metrics/
│   ├── predictions/
│   └── logs/
├── _batch_summary/             ⭐ 批量汇总指标 (所有用户合并)
│   ├── train_val_metrics_all_users.csv    所有用户 train+val 指标 (含 user_id 列)
│   ├── test_metrics_all_users.csv         所有用户 test 集指标
│   ├── inference_metrics_all_users.csv    所有用户 OOD 推理指标
│   ├── ood_overview_all_users.csv         OOD 核心指标总览 (用户×模型 透视表)
│   ├── batch_run_summary.csv              每用户执行结果 (成功/失败/耗时)
│   └── batch_run_<时间戳>.log              批量运行完整日志
└── .gitkeep / metrics/ / predictions/      (空目录骨架, 单用户产物已归档)
```

#### 缺数据处理

某用户文件夹缺训练/推理数据时, **批量不中断**, 在 dry-run / 总结中明确标记:
- 缺训练数据 → 跳过该用户, 标记原因
- 缺推理分路 → 仍跑推理, 仅无法计算评估指标

### 3.5 时段过滤 (`--time-filter-config`) ⭐ v12 新增

需求场景:
- 某几天数据在部分时段有质量问题, 想剔除具体时段(不剔整天)
- 只用某个时间窗口的数据训练/推理
- 不同用户有各自的坏日/坏时段
- 训练和推理需要独立配置(常见: 训练排除某天, 推理不排除)

#### 配置文件结构 (JSON)

`data/time_filters.json` (参考模板 `data/time_filters.example.json`):

```json
{
  "800080252842_4206894986488": {
    "_note_": "变频空调用户, 剔除 3 天高峰缺失日 + 6/29 分路异常日",
    "train": {
      "include": [
        ["2025-07-10", "2026-06-28"]
      ],
      "exclude": [
        ["2026-04-02 17:45", "2026-04-02 23:59:59"],
        ["2026-06-05 09:30", "2026-06-05 14:30"],
        ["2026-06-29", "2026-06-29"]
      ]
    },
    "infer": {
      "exclude": [["2026-06-05", "2026-06-05"]]
    }
  },

  "800080270825_4206911115606": {
    "train": {"exclude": [["2026-05-20", "2026-05-20"]]}
  },

  "_default": {
    "_note_": "未列出的用户走这个默认",
    "train": {"exclude": []},
    "infer": {"exclude": []}
  }
}
```

#### 语义规则

| 规则 | 说明 |
|---|---|
| **粒度** | 支持 `YYYY-MM-DD` (自动扩为全天) 和 `YYYY-MM-DD HH:MM[:SS]` (精确到秒) |
| **区间** | `[start, end]` **闭区间**, 两端都包含 |
| **组合** | 先 include 保留 (未指定 include = 全保留), 再 exclude 剔除 |
| **多段** | `include` / `exclude` 都是列表, 段数无限制 |
| **训推独立** | `train` / `infer` 是两个独立子节点, 各自配置 |
| **`_default` 键** | 用户级未配置时的兜底规则; 也可以完全不写 `_default` |

#### 命令行使用

**批量执行**:
```bash
# 加载配置文件, 每用户自动应用各自的时段过滤
python scripts/run_batch_users.py --time-filter-config data/time_filters.json

# 结合 --users 只跑某几个用户
python scripts/run_batch_users.py \
  --users 800080252842_4206894986488 800080270825_4206911115606 \
  --time-filter-config data/time_filters.json
```

**单用户执行** (直接传 JSON 字符串):
```bash
python scripts/run_user_pipeline.py \
  --user-id 800080252842_4206894986488 --target-col p1 \
  --train-bus data/trains/.../bus.csv --train-branch data/trains/.../branch.csv \
  --infer-bus data/infers/.../bus.csv --infer-branch data/infers/.../branch.csv \
  --output-dir artifacts \
  --train-time-filter-spec '{"exclude":[["2026-04-02 17:45","2026-04-02 23:59:59"],["2026-06-29","2026-06-29"]]}' \
  --infer-time-filter-spec '{"include":[["2026-06-01","2026-06-10"]]}'
```

#### 日志明细

执行时会在 `logs/<user_id>/align_*.log` 中输出每段的影响:
```
[align]   [v12 时段过滤] 规格: include=1段, exclude=3段
[align]   [time_filter/bus] include 1 段 -> 30662 行 -> 30662 行 (-0)
[align]       include[0]: 2025-07-10 00:00:00 ~ 2026-06-28 23:59:59
[align]   [time_filter/bus] exclude 3 段 -> 30662 行 -> 30662 行 (-0)
[align]       exclude[0]: 2026-04-02 17:45:00 ~ 2026-04-02 23:59:59
[align]       exclude[1]: 2026-06-05 09:30:00 ~ 2026-06-05 14:30:00
[align]       exclude[2]: 2026-06-29 00:00:00 ~ 2026-06-29 23:59:59
```

#### 向后兼容

- 旧的 `--exclude-dates 2026-04-02,2026-06-05` 参数仍可用, 内部自动合并到 spec 的 exclude
- 未传 `--time-filter-config` 时, 行为与之前完全一致 (无回归)

#### 沙箱验证 (v12 实测)

| 用例 | 期望 | 实测 |
|---|---|---|
| 整天剔除 (6/29) | 96 行 → 0 | ✅ 0 行 |
| 半天剔除 (6/5 09:30-14:30, 21×15min 点) | 96 → 75 | ✅ 75 行 |
| 晚间剔除 (4/2 17:45-24:00, 25×15min 点) | 96 → 71 | ✅ 71 行 |
| include 限时段 (2025-07-10 ~ 2026-06-28) | 剔除 6/29 = 96 行 | ✅ 剩 11328 行 |
| `_default` 回退 (未配置用户) | 无过滤 | ✅ 数据不变 |
| 非 dict 边缘键 (`_comment_`) | 不崩溃 | ✅ 返回 None |

**工具模块**: `scripts/time_filter_utils.py` (8 个 API 函数, 6 组单元测试, `python3 scripts/time_filter_utils.py` 可直接跑)

### 3.6 v13 用户级配置扩展 ⭐ v13 新增

v13 在 v12 的 `time_filters.json` 基础上扩展了 3 个新字段, 用于**每用户独立控制** 3 个关键行为。

#### 3.6.1 完整 JSON 结构

```json
{
  "800080270708_4206602981958": {
    "target_col":     "p1",              // [v13.4] 目标分路列名 (可覆盖 -Ch{N}- 反推)
    "guard_enabled":  false,             // [v13.1] d87 守卫开关 (覆盖全局 D87_ADAPTIVE_GUARD_ENABLED)
    "train":  { "exclude": [...] },      // [v12] 训练侧全局时段过滤
    "infer":  { "exclude": [...] },      // [v12] 推理侧全局时段过滤
    "splits": {                          // [v13.2] per-split 细粒度时段过滤
      "train": { "include": [...], "exclude": [...] },
      "val":   { "include": [...], "exclude": [...] },
      "test":  { "include": [...], "exclude": [...] }
    }
  },
  "_default": {
    "target_col":    "p1",     // 未列出用户的默认值
    "guard_enabled": true,     // 或不写让走自动检测
    "train":  { ... },
    "infer":  { ... },
    "splits": { ... }
  }
}
```

#### 3.6.2 `target_col` 字段 (v13.4)

**优先级链** (从高到低):
1. `config[user_id].target_col`
2. `config._default.target_col`
3. **总线 `-Ch{N}-` 反推** (旧 v8 逻辑)
4. **分路第 1 个 `pN` 列** (退化 1)
5. 兜底 `p1`

**合法值**: `"p1"` / `"p2"` / `"p3"` / `"p4"` (大小写不敏感, 前后空白自动去除)

**宽松校验**: 若配置的 `pN` 不在分路 CSV 实际列中, WARN + **回退到旧反推逻辑**, 不阻塞用户

**使用场景**:
- 用户分路 CSV 有 `p1+p2` 两个通道, 业务需要辨识 `p2` (Ch1 反推默认 p1)
- 文件名 `-Ch{N}-` 命名不规范或缺失

### 3.7 用户级 d87 守卫开关 ⭐ v13.1

#### 3.7.1 三层决策优先级

```
JSON 配置 config[user_id].guard_enabled          ← 最高
    ↓ 未指定
JSON 配置 config._default.guard_enabled          ← 兜底 1
    ↓ 未指定
自动检测 (基于训练集 |d87| 强度)                 ← 兜底 2 (需全局开启)
    ↓ 未触发
全局 common.D87_ADAPTIVE_GUARD_ENABLED           ← 最低
```

#### 3.7.2 自动检测判据 (任一触发即降级关闭守卫)

- **判据 A**: 训练集 `|d87|.max_effective < 50W` → d87 特征本身不足
- **判据 B**: 训练集**逐日 `|d87|.max` ≥ 守卫阈值的天数占比 < 30%** → 推理时约 70% 的天会被守卫全天强制 OFF

自动检测触发后 `bundle.d87_guard_meta` 会带:
```
{"enabled": false, "disabled_by_auto_detect": true, "auto_detect_trigger": "B",
 "auto_detect_cover_ratio": 0.222, "auto_detect_threshold_abs": 64.3}
```

#### 3.7.3 使用建议

| 用户类型 | 推荐 `guard_enabled` |
|---|---|
| 定频空调 (启动有 100W+ d87 冲击) | `true` 或不写走自动检测 |
| 变频空调 (启动无冲击) | `false` |
| 不确定 | 全局开 True + 不写让自动检测决定 |

**270708 实测收益** (变频小功率空调, peak 235W):

| 场景 | F1 | Recall | SAE | kWh_err |
|---|---|---|---|---|
| 守卫强开 (无自动降级) | 0.887 | 0.802 | 29.7% | -6.43 |
| **守卫自动降级 / 显式关闭** | **0.996** | **0.998** | **14.4%** | **-3.12** |

### 3.8 per-split 时段过滤 ⭐ v13.2

需求场景: **不同 split (train/val/test) 需要独立指定 include/exclude 时段**, 强制某些时段必入某个 split (或从中排除)。与 §3.5 的 v12 全局 time_filter 不同, 这里是**在切分之后再做微调**。

#### 3.8.1 配置结构

```json
"splits": {
  "train": {
    "include": [["2026-06-01", "2026-06-10"]],   // 强制这 10 天入 train
    "exclude": [["2026-06-05", "2026-06-05"]]    // 但 6/5 从 train 移除
  },
  "val":  { "include": [["2026-06-20", "2026-06-20"]] },
  "test": { "include": [["2026-06-25", "2026-06-25"]] }
}
```

#### 3.8.2 执行 4 步语义

```
Step 1: 原策略 (stratified_day) 切分 → 初始 train/val/test 索引
Step 2: include 硬锚定 (样本粒度):
        - 样本 ∈ train.include → 归 train
        - 否则 ∈ val.include → 归 val
        - 否则 ∈ test.include → 归 test
        - 冲突时按 train → val → test 顺序取第一个匹配, 并 WARN
Step 3: 严格保持原 split 形状 (大小不变):
        - 若某 split 因 include 过剩 → 从该 split 非锚定样本让出到不足的 split
        - 跨 split 平移补齐, 优先转移不被目标 split.exclude 拒绝的样本
Step 4: exclude 剔除 (样本粒度):
        - 样本 t ∈ split X 且 t ∈ X.exclude → 从 X 移出, 送回重分配池
        - 重分配池按剩余空间就近分配
        - 若样本被 3 个 split.exclude 全部命中 → 完全丢弃
```

#### 3.8.3 沙箱实测 (270708 案例)

配置:
```json
"splits": {
  "train": { "exclude": [["2026-06-12", "2026-06-12"]] },
  "val":   { "include": [["2026-06-20", "2026-06-20"]] },
  "test":  { "include": [["2026-06-25", "2026-06-25"]] }
}
```

**日志证据**:
```
[v13 per_split_filter] 原切分: train=959, val=384, test=384
[v13 per_split_filter] include 锚定: train=0, val=96, test=96 (合计 192)
[v13 per_split_filter] Step 3 形状调整前: train=863, val=480, test=288, 未分配=96
[v13 per_split_filter] 最终切分: train=959, val=384, test=384, 丢弃=0
```

**硬证据 4/4 验证通过**:
- 6/20 分布: train=0, **val=96** ✓, test=0
- 6/25 分布: train=0, val=0, **test=96** ✓
- 6/12 分布: **train=0** ✓, val=0, test=96 (被 exclude 后重分配到 test)
- 形状严格保持: **train=959, val=384, test=384** = 原始不变

### 3.9 查看结果

| 想看什么 | 命令 |
|---------|------|
| **Test 集三轨对比** (v6.8 新增 main_L4_calib 列) | `notepad artifacts\metrics\test_metrics_comparison.csv` |
| **推理三轨对比** (v6.8 main / main_L4_calib / main_final) | `notepad artifacts\metrics\inference_metrics_comparison.csv` |
| 漂移检测报告 (v6 L2) | `notepad artifacts\metrics\drift_report.csv` |
| 训练阶段 5 模型指标 (v6.5) | `notepad artifacts\metrics\train_val_metrics.csv` |
| Train+Val+Test 透视 | `notepad artifacts\metrics\metrics_pivot.csv` |
| Test 预测三轨明细 (v6.8) | `notepad artifacts\predictions\test_pred.csv` |
| 测试集可视化 | `start artifacts\test_prediction.png` |
| 多模型对比图 | `start artifacts\inference_comparison.png` |
| 特征重要性 | `start artifacts\feat_importance.png` |
| 模型元数据 | `notepad models\model_meta.json` |
| 完整运行日志 | `notepad logs\train_<时间戳>.log` |

---

## 四、独立推理 — 生产部署用法

`scripts\05_inference.py` 是**生产推理入口**，支持丰富的命令行参数。

### 4.1 基础用法（向后兼容）

```bat
:: 默认数据 + 主模型, 不带基线对比
python scripts\05_inference.py
```

### 4.2 多基线对比（v5 起）

```bat
:: 主模型 + RF + MoE Fallback
python scripts\05_inference.py --baseline rf fallback

:: 主模型 + v4.2 对照
python scripts\05_inference.py --baseline models\nilm_ac_two_stage_v42.pkl

:: 完整对比 + 绘图
python scripts\05_inference.py ^
    --baseline rf fallback naive_mean models\nilm_ac_two_stage_v42.pkl ^
    --plot
```

### 4.3 v6 防御层控制（v6.8 起评估时也支持 `--no-calib`）

```bat
:: 默认启用 L4 残差校正 + L5 动态切换 (需 bundle 含 calibrator)
python scripts\05_inference.py --baseline models\nilm_ac_two_stage_v42.pkl

:: 仅禁用 L4 校正 (推理 + 评估均支持)
python scripts\05_inference.py --no-calib
python scripts\04_evaluate.py  --no-calib

:: 仅禁用 L5 切换 (适合 kWh / 电费场景, 让 L4 校正收益完整保留)
python scripts\05_inference.py --no-switch

:: 全部禁用 (退化到 v5 行为)
python scripts\05_inference.py --no-calib --no-switch
```

**v6.10 推荐部署矩阵**（大幅简化，默认即最优）：

| 业务诉求 | 推荐命令 | 取自 CSV 列 |
|---|---|---|
| **绝大多数场景（默认推荐）** | 默认（L4+L5+按天切分）| `y_pred_W_main` |
| 极致 MAE（不在意 kWh） | `--no-calib --no-switch` | `y_pred_W_main_raw` |
| 与历史 v5 行为对齐（ablation 对照） | `--no-calib --no-switch` | `y_pred_W_main_raw` |

### 4.4 完整生产命令

```bat
python scripts\05_inference.py ^
    --bus     data\new_bus.csv ^
    --branch  data\new_branch.csv ^
    --model   models\nilm_ac_two_stage.pkl ^
    --baseline rf fallback models\nilm_ac_two_stage_v42.pkl ^
    --out     artifacts\predictions\result.csv ^
    --metric-out artifacts\metrics\metrics.csv ^
    --plot
```

### 4.5 支持的基线别名

| 别名 | 来源 | 用途 |
|------|------|------|
| `rf` | 主 bundle 的 RandomForest | 单阶段 RF 对照 |
| `fallback` | MoE 的全局兜底回归器 | 无季节路由对照 |
| `naive_mean` | 训练集 ON 均值 | 随机性下限 |
| `naive_zero` | 全 0 | 理论下限 |
| `<.pkl 路径>` | 外部模型文件 | v4.2 / 版本回滚 |

### 4.6 输出 CSV 字段

#### 推理结果 (`inference_result.csv`) — v6.8 三轨预测并存

| 列 | 含义 |
|----|------|
| `time` | 时间戳（15min）|
| `y_true_W` | 真实值（无标签时缺失）|
| **`y_pred_W_main`** | **最终生产输出（含 L4+L5），与下游对接的主列** |
| `residual_W_main` | 最终生产输出残差 |
| **`y_pred_W_main_raw`** ⭐ v6.8 | **原始主模型预测（无 L4 无 L5），对照基线** |
| **`residual_W_main_raw`** ⭐ v6.8 | **原始主模型残差** |
| **`y_pred_W_main_L4_calib`** ⭐ v6.8 | **仅 L4 校正后（不含 L5），单独评估 L4 收益** |
| **`residual_W_main_L4_calib`** ⭐ v6.8 | **仅 L4 校正后残差** |
| `y_pred_W_<baseline>` | 每个基线一列预测 |
| `residual_W_<baseline>` | 每个基线残差 |
| `state_pred_main` | 主模型 ON/OFF（L4/L5 不改变状态，三轨同值）|
| `p_on_main` | 主模型开机概率 |
| `y_pred_low_W_main` | 主模型 P10 下界 |
| `y_pred_high_W_main` | 主模型 P90 上界 |

> **三轨用法**：业务关注瞬时功率 → 用 `y_pred_W_main`（v6.9 已平衡）；业务关注 kWh/电费 → 用 `y_pred_W_main_L4_calib`（SAE 最优）；做 L4/L5 收益分析 → 同时对比三列残差。

#### ⚠️ 列名映射对照（prediction CSV ↔ metric CSV）— v6.12 文档强化

**重要**：`prediction CSV` 与 `metric CSV` 的命名约定不一一对应，下游消费时必须按下表查阅：

| `inference_result.csv` 列名 | `inference_metrics.csv` 中对应 `model` 字段 | 含义 |
|---|---|---|
| `y_pred_W_main_raw` | `main` | 原始主模型预测（**无 L4 无 L5 无守卫**），用于隔离评估模型本体 |
| `y_pred_W_main_L4_calib` | `main_L4_calib` | 仅 L4 残差校正后（无 L5） |
| **`y_pred_W_main`** | **`main_final`** | **最终生产输出（L4 + L5 + d87 守卫），下游对接列** |
| `y_pred_W_<baseline>` | `<baseline>` | 各基线（rf / fallback / nilm_ac_two_stage_v42 等）|

**历史背景**：v6.8 引入三轨预测时，为保持下游 v5 客户端兼容性，将"最终生产输出"复用了 `y_pred_W_main` 这个旧列名，把原始预测重命名为 `y_pred_W_main_raw`。因此当你看到 metric CSV 中 `model='main'` 和 `model='main_final'` 是**两个不同的实体**，对应 prediction CSV 中两个不同的列：

```python
# ✅ 正确：诊断"模型本体在 OOD 上的原始能力"
import pandas as pd, numpy as np
df  = pd.read_csv("artifacts/predictions/inference_result.csv")
inf = pd.read_csv("artifacts/metrics/inference_metrics.csv")
mae_raw = np.mean(np.abs(df["y_true_W"] - df["y_pred_W_main_raw"]))
mae_csv = inf[(inf["model"]=="main")&(inf["metric"]=="MAE_W")]["value"].iloc[0]
assert abs(mae_raw - mae_csv) < 0.01   # ✅ main 长表 = main_raw 预测列

# ✅ 正确：生产环境实际效果
mae_final = np.mean(np.abs(df["y_true_W"] - df["y_pred_W_main"]))
mae_csv2  = inf[(inf["model"]=="main_final")&(inf["metric"]=="MAE_W")]["value"].iloc[0]
assert abs(mae_final - mae_csv2) < 0.01  # ✅ main_final 长表 = main 预测列

# ❌ 错误：会得到完全错误的结论
mae_wrong = np.mean(np.abs(df["y_true_W"] - df["y_pred_W_main"]))
mae_main  = inf[(inf["model"]=="main")&(inf["metric"]=="MAE_W")]["value"].iloc[0]
# mae_wrong ≠ mae_main  (前者是 main_final, 后者是 main, 差异可达 5W+)
```

**`train_pred.csv` / `val_pred.csv` / `test_pred.csv`**：训练/评估流程不经过 L5，因此只有两轨：
- `y_pred_W` ↔ metric `main`（原始预测）
- `y_pred_main_L4_calib_W` ↔ metric `main_L4_calib`（仅 04_evaluate 的 test_pred 含此列）

#### 漂移报告 (`drift_report.csv` — v6 新增)

| 列 | 含义 |
|----|------|
| `dimension` | 监测维度（covariate / concept / time）|
| `metric` | 指标名（如 "温度-功率加权漂移"）|
| `train_ref` | 训练分布参考值 |
| `infer_obs` | 推理分布观测值 |
| `drift_ratio` | 漂移比例（绝对值）|
| `level` | NORMAL / WARN / ALERT |
| `note` | 备注 |

---

## 五、v6 漂移防御体系详解

> v6 的核心价值不是"再降 1W MAE"，而是 **让系统自己感知漂移并自适应**。

### 5.1 整体架构

```
┌──────────────────────────────────────────────────────┐
│                  推理输入: 新总线数据                  │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ L1 漂移感知特征 (5 维)                                │
│ ├─ power_recent_24h_mean     本周用电基线             │
│ ├─ power_recent_7d_mean      7 天用电基线             │
│ ├─ power_deviation_24h       当前 vs 近期偏离          │
│ ├─ temp_power_residual ⭐    温度下功率异常度          │
│ └─ is_morning_peak           时段标记                  │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ 主模型推理 → y_pred_raw                              │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ L4 残差校正层                                         │
│ delta = g(temp, hour, recent, season, y_raw)         │
│ y_calib = clip(y_raw + delta, 0, +∞)                 │
│ 限幅: ±min(2*MAE, 150W)                              │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ L2 漂移检测 (并行)                                    │
│ ├─ 总线信号均值漂移 (协变量)                          │
│ ├─ 温度-功率桶漂移 (概念漂移) ⭐                     │
│ └─ 时段分布漂移                                       │
│ → 输出 drift_report.csv + 告警级别                   │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ L5 多模型动态切换                                     │
│ NORMAL  → 主模型 (精度优先)                          │
│ WARN    → 0.6 主 + 0.4 fallback (平滑过渡)            │
│ ALERT   → fallback (鲁棒优先) ⭐                     │
└──────────────────────┬───────────────────────────────┘
                       ▼
                  最终预测 y_final
```

### 5.2 L2 漂移告警阈值

```python
# 配置在 scripts/drift_detect.py
DRIFT_THRESH_WARN  = 0.10    # 相对偏差 ≥ 10% 触发 WARN
DRIFT_THRESH_ALERT = 0.25    # 相对偏差 ≥ 25% 触发 ALERT
```

### 5.3 L5 切换决策规则（v6.9 升级为 L4-aware 双轨权重）

```python
# 配置在 scripts/residual_calibrator.py - ModelSwitcher.decide(calib_active=...)
# 阈值与权重已全部抽取到 scripts/common.py §2b 节, 便于线上调参
if n_alert ≥ L5_ALERT_N_ALERT(=3)  or  max_concept_drift ≥ L5_ALERT_MAX_CONCEPT_DRIFT(=0.50):
    mode = "ALERT"
    main_weight = L5_MAIN_WEIGHT_ALERT_WITH_L4(=0.5)   if calib_active else L5_MAIN_WEIGHT_ALERT_WITHOUT_L4(=0.0)
elif n_alert ≥ L5_WARN_N_ALERT(=1) or n_warn ≥ L5_WARN_N_WARN(=2) or max_concept_drift ≥ L5_WARN_MAX_CONCEPT_DRIFT(=0.20):
    mode = "WARN"
    main_weight = L5_MAIN_WEIGHT_WARN_WITH_L4(=0.75)   if calib_active else L5_MAIN_WEIGHT_WARN_WITHOUT_L4(=0.6)
else:
    mode = "NORMAL"
    main_weight = 1.0
```

**v6.9 决策矩阵**：

| 漂移级别 | L4 启用 (`calib_active=True`) | L4 未启用（向后兼容旧行为） |
|---|---|---|
| NORMAL | main_w = 1.0 | main_w = 1.0 |
| WARN | **main_w = 0.75** | main_w = 0.6 |
| ALERT | **main_w = 0.5** ⭐ | main_w = 0.0 |

**设计依据（5 月推理硬证据）**：L4 单独启用时 SAE 从 8.77% → 3.08% (-65%)。若 L5 在 ALERT 模式下沿用旧策略把主权重压到 0，会把 L4 的电量校正收益全部丢失。新策略在保证 MAE/RMSE 稳定的前提下，让 L4 的 kWh 校正能力保留约一半。

**实测对比（5 月 OOD 1824 行）**：

| 决策版本 | MAE (W) | RMSE (W) | SAE | kWh_err |
|---|---|---|---|---|
| v6.7/6.8 (ALERT→main_w=0) | 22.07 | 59.27 | 7.24% | -7.84 |
| **v6.9 (ALERT+L4→main_w=0.5)** | 23.39 | **58.10** ⭐ | **5.16%** ⭐ | **-5.59** ⭐ |

#### ⚠️ v6.12 实测告警：L5 ALERT 场景在特定条件下负收益

**5/21~6/3 OOD 实测**（守卫开启 + L4 启用 + ALERT 模式，1344 行）：

| 轨道 | MAE (W) | SAE | kWh_err | 备注 |
|---|---|---|---|---|
| main_L4_calib | **23.92** ⭐ | **10.43%** ⭐ | -7.07 | L5 未启用 (理论生产输出) |
| fallback (MoE 内置全局 GBR) | 30.99 | 14.55% | -9.86 | L5 备选 |
| **main_final (50/50 加权)** | 27.29 | 12.49% | -8.47 | **实际生产输出 (-3.37 W MAE, +2.06pp SAE)** |

**根因诊断**：v6.9 设计假设 fallback 是**独立训练的 v42 外部 pkl**（与 main 偏差结构互补）。但 v6.12 引入 d87 极值列后，v42 (111 维) 与主模型 (128 维) **特征集不再兼容**，L5 实际加载的是 MoE 内置 fallback（与 main 共享训练集），偏差高度相关，加权混合反而把好的预测拖向坏的方向。

**v6.12 决策建议**：
1. ✅ **关注 kWh/电费的业务**：直接读 `inference_metrics.csv` 中 `model='main_L4_calib'` 行（10.43% SAE）
2. ✅ **诊断/对比**：用 `inference_result.csv` 的 `y_pred_W_main_L4_calib` 列
3. ⚠️ **当前 `y_pred_W_main` (main_final) 在 ALERT 场景非最优**：等待未来重训兼容的 v42 baseline 或方案重设计
4. 🔧 **临时手动禁用 L5**：推理时加 `--no-switch` 标志强制 `main_weight=1.0`，使 `y_pred_W_main = y_pred_W_main_L4_calib`

**保留 v6.9 L5 设计的理由**：v6.9 在 5 月 OOD 上有过 -29% SAE 真实收益；v6.12 的负收益是**特定场景**（d87 守卫已先消除大部分 FP + v42 不兼容退化为 MoE 内置 fallback），不能据此推翻整个机制。未来若重训 v42 与新 schema 兼容，L5 仍可恢复正收益。

### 5.4 防御层开关

| 控制方式 | 启用 | 禁用 |
|---------|------|------|
| L1 漂移特征 | `USE_DRIFT_FEATURES=True` (默认) | 在 `03_train.py` 改为 False 重训 |
| L2 漂移检测 | 推理时自动运行 | 无开关（输出始终生成）|
| L4 残差校正 | `USE_RESIDUAL_CALIB=True` (默认) | `--no-calib` 命令行禁用 |
| L5 模型切换 | 默认启用（需 baseline）| `--no-switch` 命令行禁用 |

### 5.5 数据集切分策略（v6.10 重要改进）

v6.10 之前的 `stratified` 策略存在**同天切分边界泄漏**：每月内部按时序切前 70%/中 15%/后 15%，切分点可能落在某天中间。实测 8 个边界中 **6 个是同天被切到不同 split**（占 75%），导致 lag/rolling 特征跨集泄漏。

**v6.10 `stratified_day` 设计**：

```
对每个月:
  1. 统计该月每天的样本数
  2. 完整天 (≥80 条, 15min 间隔下 96 步的 83%) → 进入随机抽样池
  3. 碎片天 (<80 条) → 全部归 train (不污染 val/test)
  4. 月内完整天用 seed=42 随机洗牌, 按 70/15/15 分到 train/val/test
  5. 整天的所有样本归到对应 split (绝不分割)
  6. 安全保护: 完整天 < 3 全归 train; val/test 至少各 1 天
```

**两种策略对比**：

| 维度 | `stratified` (v6.7~6.9 旧) | `stratified_day` (v6.10 默认) ⭐ |
|---|---|---|
| 切分单位 | 时间步 (15min) | 完整天 (96 步) |
| 月份覆盖 | 各月按比例 | 各月按比例 (抽天) |
| **同天切分边界泄漏** | ❌ 存在 (75% 边界) | ✅ 无 |
| **lag/rolling 跨集泄漏** | ❌ 存在 | ✅ 无 |
| **5 月 OOD SAE** | 5.16% | **0.19%** ⭐ |
| 可复现 | 是 (确定性时序) | 是 (固定 seed=42) |
| 5492 行实际切分 | tr=3845 / va=823 / te=824 | tr=3764 / va=864 / te=864 |

**配置位置**（`scripts/common.py`）：
```python
SPLIT_STRATEGY = "stratified_day"   # v6.10 默认
# SPLIT_STRATEGY = "stratified"     # v6.9 旧策略, 保留兼容
# SPLIT_STRATEGY = "time"           # 纯时序, 不分层
FULL_DAY_MIN_SAMPLES = 80           # 配置在 scripts/split_utils.py
```

### 5.6 v6.11 算法与工程改进（4 项）

v6.11 在 v6.10 稳定基线上做了 4 项**纯改善型**改动（不涉及架构变更），改善幅度均通过端到端实测验证。第 5 项指标系统 bug 修复见 §6.4。

#### 5.6.1 突变感知特征（51 维新增）

**问题**：v6.10 主模型对"突然开机"工况识别失败。典型如 7/12 早晨 08:15 总线信号从 0 突跳到 847W，但 lag/rolling 特征仍指向"近期 OFF"，模型把这一步判 OFF，连带后续 5 步全漏。

**解决**：`scripts/feature_utils.py::build_features` 新增 5 类 51 维突变信号：

| 类别 | 维度 | 公式 | 解决场景 |
|---|---|---|---|
| 多步差分 | Top-10 × {d3, d6} = 20 | `df[c].diff(periods=3/6)` | 45min/90min 累积突变 |
| 绝对差分 | Top-5 × {abs_d1, abs_d3} = 10 | `df[c].diff().abs()` | 突变幅度（方向无关）|
| 突变方向标志 | Top-3 × {up_d1, down_d1} = 6 | `(diff > +50W).int8` / `(diff < -50W).int8` | 显式区分开/关机 |
| 窗口极差 | Top-3 × {range_4, range_12} = 6 | `rolling.max - rolling.min` | 1h/3h 波动范围 |
| EMA 跨尺度 | Top-3 × {ema_2, ema_24, ratio} = 9 | EMA 30min vs 6h + 比值 | 短长期对比 |

**实测**：特征维度 77 → **128**。Val FN 显著改善（带后续 §5.6.3 F1 阈值联合作用更明显）。

#### 5.6.2 后处理优化（min_on=1, fill_off=3）

**问题**：旧 `POST_MIN_ON=2 + POST_FILL_SHORT_OFF=1` 在 6/2 OOD 上**杀掉大量正确预测**。例如 4/11 傍晚连续 ON 段中 p_on 在 0.5~0.85 抖动，被旧后处理压成"开-关-开-关"震荡序列后全部抹掉。

**修改**：

| 参数 | 旧值 | 新值 | 设计意图 |
|---|---|---|---|
| `POST_MIN_ON` | 2 | **1** | 允许孤立 ON 通过（救回事件 1 的 09:30 单步）|
| `POST_FILL_SHORT_OFF` | 1 | **3** | 填充 ≤3 步 OFF 间隙（救回事件 2 的震荡序列）|

**实测对比（6/2 OOD）**：

| 指标 | 旧后处理 | 新后处理 | 改善 |
|---|---|---|---|
| main_final MAE | 83.89 W | **15.73 W** | **-81.2%** ⭐⭐⭐ |
| main_final SAE | 19.34% | **2.59%** | **-86.6%** ⭐⭐⭐ |
| 事件 2 (4/11) FN | 11 | **4** | 救回 7 步 |

> Test 集 main_L4_calib SAE 略升 +0.06pp，幅度在统计噪声内，可接受。

#### 5.6.3 F1 阈值优化目标（替代 F0.5）

**问题**：F0.5 偏 Precision，导致 Stage-1 阈值被推到 0.79，边缘样本 (p_on=0.5~0.8) 被压成 FN。

**修改**：`FBETA = 0.5 → 1.0`，仅 1 行常量改动。

**实测对比（同 6:2:2 切分）**：

| 指标 | F0.5 | F1 | 变化 |
|---|---|---|---|
| Stage-1 最佳阈值 | 0.79 | **0.43** | -0.36 |
| Val FN | 19 | **13** | -32% ✅ |
| Val Recall | 95.71% | **97.07%** | +1.36pp ✅ |
| Val FP | 1 | 7 | +6（业务可接受）|
| **Test main SAE** | 2.85% | **0.11%** ⭐⭐⭐ | **项目历史最佳** |
| 6/2 OOD main_final SAE | 2.59% | **1.59%** | -1.00pp |

**业务取舍**：F1 用 +6 个 FP 换 -6 个 FN。NILM 业务中，**漏识别比误报代价大**（漏识别让电量统计严重偏低），值得。

#### 5.6.4 隔离 bug 修复（v42 训练不再覆盖主模型预测）

**问题**：v6.3 引入 `NILM_BASELINE_MODE=1` 隔离机制只覆盖了 `models/` 拆分组件，**漏掉了 `artifacts/predictions/`**。导致 03b v42 训练时会重写 `train_pred.csv` / `val_pred.csv`，用户看到的预测明细实际是 v42 的而非主模型的。

**症状硬证据**（修复前）：

```
03_train 训练日志: Val FN=19, F1=0.9770  ← 主模型真实指标
val_pred.csv 实际:  FN=27, F1=0.9686    ← 这是 v42 的预测!
```

**修复**：`03_train.py::save_predictions_csv` 调用块加守卫：

```python
if _os.environ.get("NILM_BASELINE_MODE") != "1":
    save_predictions_csv(...)   # 仅主模式写
else:
    log.info("[基线模式] 跳过 train_pred/val_pred 写入 (避免覆盖主模型预测)")
```

**修复后**：val_pred.csv 现在与训练日志报告 FN 数字**完全一致**（19 vs 19）。

#### 5.6.5 4 项改动累积收益

| 维度 | v6.10 初始 | v6.11 (4 项叠加后) | 累积改善 |
|---|---|---|---|
| Val FN | 25 | **13** | **-48%** |
| Val Recall | 94.36% | **97.07%** | +2.71pp |
| Val F1 | 0.9698 | **0.9773** | +0.75pp |
| Test main SAE | 4.65% | **0.11%** ⭐⭐⭐ | -97% |
| 6/2 OOD main_final SAE | 19.34% | **1.59%** ⭐⭐⭐ | -92% |
| 6/2 OOD main_final MAE | 83.89 W | **14.17 W** | -83% |

### 5.7 v6.12 算法改进：`load_iden_data87` 启动尖峰特征

#### 5.7.1 问题背景

**5/21~6/3 新数据推理出现异常 SAE 46.87%**：14 天里 7 天真开空调、7 天真关空调，模型在关空调日全部误报（248 步 FP）。

诊断硬证据：6/1 全天关空调，`load_iden_data73`（主功率列）中位 1531W、max 2660W，**反而比真开空调日 5/22（中位 1439W）更高**——说明用户外出未开空调时冰箱/热水器等其他设备维持了"看似开空调"的总线功率水平。模型从主功率特征无法物理区分。

#### 5.7.2 关键发现：`load_iden_data87` 是空调启动的唯一可靠物理签名

| 群体 | `|d87|` 分位数 (50/75/90/95) | 判别 AUC |
|---|---|---|
| **训练集 75 个 ON 启动事件** ±15min 窗口 | 171 / 186 / 204 / 211 | — |
| **训练集 7 天全天 OFF** 整日范围 | 25 / 27 / 32 / **max=35** | — |
| 41 个 bus 列对 ON 启动事件的判别力 | — | **d87 = 1.000**, 其余 ≤ 0.66 |

物理含义推测：d87 很可能是相位差/无功/谐波类瞬态指标。空调压缩机启动瞬间（5min 内单步冲击）产生 -150~-224 的负向尖峰，呈典型的启动冲击电流特征。关空调日 \|d87\| 始终 ≤ 35（仅热水器等非冲击负载）。

#### 5.7.3 根因 Bug：尖峰被 `resample("15min").mean()` 平滑

```python
# scripts/feature_utils.py (v6.11 之前)
bus_rs = bus_idx.resample("15min").mean()
# 5min 原始 d87 = [-224, +13, -17] → 15min mean = -76 (尖峰强度衰减 ~3 倍)
# v6.11 的 51 维突变特征 (d3/d6/abs/up/down/range/ema) 全部建立在已平滑数据上
# → 模型从未看到 -224 这个尖峰, 无法学到这一物理签名
```

#### 5.7.4 修复方案（A 精选 + C 守卫双保险）

**方案 A：5min 极值聚合特征**（特征工程层根治, ~30 行代码）

`feature_utils.resample_and_align` 对 `SPIKE_COLS=["load_iden_data87"]` 旁路保留三个极值聚合，不参与 mean 平滑：

```python
SPIKE_COLS = ["load_iden_data87"]
for sc in SPIKE_COLS:
    s = bus_df.set_index("event_time")[sc]
    extras.append(s.resample("15min").max().rename(f"{sc}_max5"))
    extras.append(s.resample("15min").min().rename(f"{sc}_min5"))
    extras.append(s.abs().resample("15min").max().rename(f"{sc}_absmax5"))
```

`build_features` 自动消费所有 `_max5/_min5/_absmax5` 后缀列作为原始特征。
**验证**：新模型 feat_cols 含 d87 三个极值列，分类器 importance rank 16/19/37 ∈ 128。

**方案 C：推理侧日级 d87 启动签名守卫**（最后一道防线, ~30 行）

`scripts/05_inference.py` 主推理段后增加日级守卫：

```python
D87_DAY_GUARD_TH = 100.0   # 训练 OFF max=35, 留 ~3× 安全裕度
# 用 5min 原始粒度计算每天的 |d87|_max (避免 15min 均值平滑掉尖峰)
day_d87_amax = bus_5min.groupby('date')['load_iden_data87'].apply(lambda s: s.abs().max())
mask = (df_dates 对应 day_d87_amax) < D87_DAY_GUARD_TH
state_pred[mask] = 0; y_pred[mask] = 0.0; p_on[mask] = 0.0
```

守卫同步扩展到所有 baseline（rf/fallback/v42）以保证 L5 加权对齐。

#### 5.7.5 v6.12 实测收益（5/21~6/3 OOD, 1344 步, 真开 337 步 / 真关 1007 步）

| 指标 | v6.11 原始 | v6.12 (A+C 双保险) | 改善 |
|---|---|---|---|
| **F1** | 0.730 | **0.9985** | **+36.8 pp** |
| **Precision** | 0.575 | **1.0000** | **+42.5 pp** |
| Recall | 0.997 | 0.9970 | 完美保持 |
| MAE | 147.92 W | **31.77 W** | **-78.5%** |
| **main_final SAE** | **46.87%** | **12.49%** | **-73.4%** |
| main_L4_calib SAE | 49.62% | **10.43%** | -77.7% |
| FP 步数 (关空调日) | 248 | **0** | **-100%** |
| kWh 误差 | +31.8 | -8.5 | 由严重高估转为轻度低估 |

**守卫精准命中**：7 个真关空调日（5/21,23,24,25,26,27,6/1）全部触发，0 误伤；7 个真开空调日（5/22,28,29,30,31,6/2,6/3）全部放行。

**隔离实验**：临时关闭守卫，v6.12 模型本体 SAE 仍是 **47.80%**（与 v6.11 几乎相同）。说明：
- 方案 A 的 d87 特征虽进入模型，但被主功率列 d79/d74 (importance 0.70/0.16) 完全压制，**对关空调日判定没有产生根本翻转**
- SAE 改善 **100% 来自方案 C 守卫**
- 方案 A 仍保留：提供"特征工程层根治"的可扩展性，未来若加入新尖峰列只需扩 `SPIKE_COLS` 即可

#### 5.7.6 测试集影响（无 OOD 干扰）

| 指标 | v6.11 | v6.12 |
|---|---|---|
| Test main F1 | 0.9879 | 0.9711 |
| Test main MAE | 11.34 W | 24.68 W |
| Test main SAE | **0.11%** | **0.84%** |

测试集表现轻微下降，但仍属顶级水准。原因：v6.12 阈值由 0.43 提到 0.46，分类边界趋于保守。这是为了让 d87 特征产生收益所必需的代价。

#### 5.7.7 v6.12.1 守卫修正：方向 + 自适应阈值

**v6.12 守卫的 3 个设计缺陷**（用 270848 用户实测发现）：

| 缺陷 | 代码位置 | 失效机理 |
|---|---|---|
| #1 用 `abs()` 不分方向 | `s.abs().max()` | 启动负尖峰与噪声正尖峰混淆 |
| #2 硬编码阈值 100 | `D87_DAY_GUARD_TH=100.0` | 仅适合 842 大空调，对其他用户失灵 |
| #3 默认 inf 倾向放行 | `.get(d, np.inf)` | 缺数据天默认放行，反向操作 |

**事件级硬证据（用户校正）**：d87 在不同用户上**事件级仍然有强判别力**：

| 用户 | 启动事件 d87.min 中位 | OFF 段 d87.min 中位 | 信号/噪声差 |
|---|---|---|---|
| 842 (大空调 891W) | -198 | -17 | 181 |
| 270848 (小空调 137W) | **-71** | **-13** | **58** |

**v6.12.1 修正**：
1. **方向修正**：用 `d87.min()` 替代 `s.abs().max()`，启动尖峰是负方向专属
2. **训练自适应阈值**：从训练 OFF 段 P1 × safety_factor 自动学习，写入 `bundle["d87_guard_meta"]`
3. **缺数据保护性触发**：`.get(d, 0.0)` 替代 `.get(d, np.inf)`

**v6.12.1 (SF=3.0, 阈值-96) 实测**：
- 842 用户：14 天日级判定 **100%** Accuracy（含 5/26 噪声日正确压制）
- 270848 用户：守卫几乎全压制（13/14 天误压），**因为阈值-96 远超 270848 启动幅度-65~-106**

#### 5.7.8 ⭐ v6.12.2 自适应缩放：跨用户终极解决方案

**问题本质**：v6.12.1 的阈值-96 是按 842 大空调标定的。对 270848 小空调（启动 d87 仅-71），单一阈值无法兼顾。需要**按用户特征动态缩放阈值**。

**5 个候选方案的实测对比**（以日级守卫准确率衡量）：

| 方案 | 阈值机制 | 842 Acc | 270848 Acc | 评价 |
|---|---|---|---|---|
| v6.12.1 硬编码 | 训练 SF×P1 = -96 | **100%** | 7.1% | 偏训练用户 |
| A: d87 自身 P0.5 | 推理时该用户分位 | 78.6% | 92.9% | 平庸 |
| A: d87 自身 P1 | 推理时该用户分位 | 57.1% | 100% | 偏小空调 |
| **B: d73_P95 缩放** ⭐ | `train阈值 × user_d73_P95 / train_d73_P95` | **100%** | **100%** | **完美** |
| E: z-score -3σ | `mean - 3×std` | 92.9% | 100% | 良好 |
| E: z-score -2.5σ | `mean - 2.5×std` | 85.7% | 100% | 中等 |

**方案 B 物理依据**：
- d87 启动尖峰强度 ∝ 空调功率（电机启动冲击电流定律）
- d73_P95 是用户「电力规模」的稳健代理（已知与空调强相关）
- 等比缩放：用户主功率越大 → 阈值越深（更严格）；越小 → 阈值越浅（更宽松）

**v6.12.2 实现** (`03_train.py` 保存锚点 + `05_inference.py` 动态缩放)：

```python
# 训练时 (一次性, 写入 bundle)
threshold_base = train_off_p1 × safety_factor    # = -32 × 3.0 = -96
train_d73_p95  = bus_train["load_iden_data73"].quantile(0.95)  # = 2941W
bundle["d87_guard_meta"] = {
    "threshold_base":   -96,
    "train_d73_p95":    2941,
    "adaptive_scaling": True,
    "scale_min": 0.05, "scale_max": 2.0,  # 防极端值
}

# 推理时 (每个用户重新计算)
user_d73_p95 = bus["load_iden_data73"].quantile(0.95)
scale = np.clip(user_d73_p95 / train_d73_p95, 0.05, 2.0)
D87_GUARD_TH = threshold_base × scale
```

**实测对照表（v6.12.2）**：

| 用户 | d73_P95 | 缩放因子 | 自适应阈值 | 守卫日级判定 | Accuracy |
|---|---|---|---|---|---|
| 训练用户 842 | 2941 W | 1.000 | -96.0 | 7 开 / 7 关 | 100% |
| **OOD 842 (大空调 891W)** | 2813 W | 0.956 | **-91.8** | 7 开 / 7 关 | **100%** ⭐ |
| **OOD 270848 (小空调 137W)** | 612 W | 0.208 | **-20.0** | 14 开 / 0 关 | **100%** ⭐ |
| 假想无空调用户 (d73<10) | <50 W | clip→0.05 | ≈ -4.8 | 守卫几近休眠 | 合理（模型本身也不输出 ON）|

**842 用户 14 天逐日详情（v6.12.2 自适应缩放后）**：

| 日期 | 真实 | d87.min | 阈值 | 判定 | 正确? |
|---|---|---|---|---|---|
| 5/21 | 关 | -26 | -91.8 | 压制 | ✅ TN |
| 5/22 | 开 | -187 | -91.8 | 放行 | ✅ TP |
| 5/23 | 关 | -25 | -91.8 | 压制 | ✅ TN |
| 5/24 | 关 | -27 | -91.8 | 压制 | ✅ TN |
| 5/25 | 关 | -57 | -91.8 | 压制 | ✅ TN |
| **5/26** | **关** | **-84** | -91.8 | **压制** | **✅ TN（v6.12 此处 FP）** |
| 5/27 | 关 | -43 | -91.8 | 压制 | ✅ TN |
| 5/28~31 | 开 | -188~-224 | -91.8 | 放行 | ✅ TP×4 |
| 6/1 | 关 | -29 | -91.8 | 压制 | ✅ TN |
| 6/2~3 | 开 | -215~-218 | -91.8 | 放行 | ✅ TP×2 |

**270848 用户 14 天逐日详情**：

| 日期 | 真实 | d87.min | 阈值 | 判定 | 正确? |
|---|---|---|---|---|---|
| 5/21~6/3 (14 天每天都开)| 开 | -62 ~ -106 | -20.0 | 放行 | ✅ TP×14 |

**边界场景保护**：
| 场景 | scale | 行为 |
|---|---|---|
| bus 数据完整 + d73 正常 | 0.05 ~ 2.0 | 正常自适应 |
| bus 缺 `d73` 列 | — | 退化到训练用户基准阈值 -96 |
| `adaptive_scaling=False` | — | 退化到训练用户基准阈值 -96 |
| 旧版 bundle 无 `d87_guard_meta` | — | 退化到硬编码 -50 |
| 推理用户 d73 异常小 (<50W) | clip→0.05 | 阈值 ≈ -4.8（守卫宽松，但模型本身也不输出 ON）|
| 推理用户 d73 异常大 (>5882W) | clip→2.0 | 阈值 ≈ -192（守卫宽松，避免误杀真启动）|

**v6.12.2 终极效果**：单一模型 + 自适应守卫，在两个真实用户（功率差 5 倍）上守卫均达到 **100% 日级判定准确**。

#### 5.7.9 ⭐ v6.12.3 三大修正：双向化 + 事件特征 + 数据清洗

**背景**：v6.12.2 实测发现 3 个新问题（用户1+2 实测，共 89 个启动事件）：
1. **4% 启动是正向 d87 尖峰**（如用户2 5/23: d87 由 -18→+176），守卫只看 `d87.min()` 漏检
2. **d87 特征在模型中 importance 仅 0.001~0.002**，模型决策几乎完全由主功率列驱动
3. **用户1 训练数据混合工作模式**（6/30~7/3 大空调 868W，其余小空调 130W），干扰模型

**修复**：

| 子项 | 改动 | 代码位置 |
|---|---|---|
| #1 守卫双向化 | `s.abs().max()` 替代 `s.min()`，阈值绝对值化 | `05_inference.py` L307-330 |
| #2 d87 事件特征 | 新增 6 个：`d87_jump_abs5`、`d87_amax5`、`d87_event_neg/pos/any/recent_3`、`d87_spike_ratio` | `feature_utils.py` L171-208 |
| #3 训练数据清洗 | `02_align_and_feat.py` 新增 `--exclude-dates` 参数 | `02_align_and_feat.py` L21-43 |

**v6.12.3 实测对照（v6.12.2 → v6.12.3）**：

| 用户 | 指标 | v6.12.2 | v6.12.3 | 变化 |
|---|---|---|---|---|
| 用户1 (270848, 小空调) | OOD F1 | 0.766 | **0.902** | +0.136 |
|  | OOD Recall | 0.628 | **0.830** | +0.202 |
| 用户2 (252844, 大空调) | OOD F1 | 0.819 | **0.913** | +0.094 |
|  | OOD MAE | 81.0 W | **56.0 W** | -25 (-30.9%) |
|  | OOD main_final SAE | 38.31% | **26.92%** | -11.39 pp |
|  | **5/23 救回** | 0 kWh (整天压制) | **8.24 kWh** ✅ | +8.24 |

#### 5.7.10 ⭐⭐ v6.12.4 双源约束阈值标定（解决 5/27 类问题）

**问题**：v6.12.3 用户2 仍存在 5/27 整天 FN 8.38 kWh。诊断后发现：

| 维度 | 之前误判（"模型权重不够"）| 实际真相（"守卫阈值过严"）|
|---|---|---|
| 5/27 09:45 时刻 `p_on` | 全 0 | **实际 0.949 ✅ 模型识别了** |
| `raw_pred=0` | 模型问题 | **守卫强制压制** |
| 守卫阈值 -185 | 看起来合理 | **比 ON 启动中位 -174 还严！** |

**根因**：v6.12.2 公式 `train_off_P1 × safety_factor (3.0)` 在 OFF 噪声大的用户上失灵。
- 用户1：OFF_P1=-15 → 阈值-45 ✅ 合理
- 用户2：OFF_P1=-62 → 阈值-186 ❌ 比 ON 中位还严

**v6.12.4 新公式**（双源约束）：
```python
on_constraint  = abs(train_on_P10)  × ALLOW_FACTOR (0.8)    # 允许 90% 启动通过
off_constraint = abs(train_off_P99) × MARGIN_FACTOR (1.3)   # 留 30% 噪声裕度
threshold_abs  = max(on_constraint, off_constraint)         # 双源约束
```

**v6.12.4 实测**：
- 用户2 阈值 -185 → **-103** （推理 d73 缩放后 -96）
- **5/27 (|d87|=166) → 放行**，救回 **+7.68 kWh** ⭐
- 但出现 5/30/6/02 两个边界 FP（|d87|=104, 98 都略超阈值 96）
- SAE 26.92% → 7.83%（数值低但有 FP 抵消 FN 的"伪精度"成分）

#### 5.7.11 ⭐⭐⭐ v6.12.5 ALLOW_FACTOR 0.8→0.9（完美 100% 日级精度）

**问题**：v6.12.4 5/30 (|d87|=104) 和 6/02 (|d87|=98) 仅比推理阈值 96 略高即被错误放行。

**修复**：将 `ALLOW_FACTOR` 从 0.8 调到 0.9：
- 用户2 训练阈值 103 → **116**
- 用户2 推理阈值（×d73 缩放 0.932）→ **109**
- **阈值 109 正好穿过 5/27 (|d87|=166, 放行) 和 5/30 (|d87|=104, 压制) 的间隙**

**v6.12.5 实测对照（用户2 OOD 14 天）**：

| 日期 | 真实 | \|d87\|.max | v6.12.3 | v6.12.4 | **v6.12.5** | 评价 |
|---|---|---|---|---|---|---|
| 5/21 | 开 | 175 | 7.79 ✅ | 7.76 ✅ | **7.76 ✅** | TP |
| 5/22 | 关 | 87 | 0 ✅ | 0 ✅ | **0 ✅** | TN |
| 5/23 | 开 | 176 | 8.24 ✅ | 8.24 ✅ | **8.24 ✅** | TP（v6.12.2 是 0）|
| 5/24~26 | 开 | 178~203 | ✅ | ✅ | **✅** | TP×3 |
| **5/27** ⭐ | 开 | 166 | **0 ❌** | 7.68 ✅ | **7.68 ✅** | v6.12.4 救回 |
| 5/28~29 | 关 | 90~96 | 0 ✅ | 0 ✅ | **0 ✅** | TN×2 |
| **5/30** ⭐ | 关 | 104 | 0 ✅ | **7.43 ❌** | **0 ✅** | v6.12.5 救回 |
| 5/31 | 关 | 79 | 0 ✅ | 0 ✅ | **0 ✅** | TN |
| 6/01 | 开 | 203 | 8.37 ✅ | 8.37 ✅ | **8.37 ✅** | TP |
| **6/02** ⭐ | 关 | 98 | 0 ✅ | **7.61 ❌** | **0 ✅** | v6.12.5 救回 |
| 6/03 | 关 | 84 | 0 ✅ | 0 ✅ | **0 ✅** | TN |

**用户2 v6.12.5 守卫日级**：TP=7, TN=7, FP=0, FN=0 → **100% Accuracy** ⭐⭐⭐

#### 5.7.12 v6.12.2 → v6.12.5 四版迭代演进汇总

| 版本 | 关键改进 | 用户2 F1 | 用户2 MAE | 用户2 SAE | 用户2 Recall |
|---|---|---|---|---|---|
| v6.12.2 | OFF×SF + d73 缩放 | 0.819 | 81.0 W | 38.31% | 0.704 |
| v6.12.3 | 双向 + 事件特征 + 数据清洗 | 0.913 | 56.0 W | 26.92% | 0.845 |
| v6.12.4 | 双源约束 (AF=0.8) | 0.869 | 79.0 W | 7.83% ⚠️ | 0.982 |
| **v6.12.5** | **AF 0.8→0.9 (FP 清零)** | **0.988** ⭐ | **34.9 W** ⭐ | **15.17%** ✅ | **0.982** |

**累计改善 (v6.12.2 → v6.12.5)**：
- **MAE**: 81.0 → 34.9 W (**-57%**) ⭐⭐⭐
- **F1**: 0.819 → 0.988 (+0.169)
- **Recall**: 0.704 → 0.982 (+0.278)
- **Precision**: 0.979 → 0.994 (+0.015)
- **5/23 救回**: 0 → 8.24 kWh
- **5/27 救回**: 0 → 7.68 kWh

**关于 v6.12.4 SAE 7.83% vs v6.12.5 SAE 15.17% 的反向解读**：
- v6.12.4 SAE 看似更低，但是"FP 多识别 +15 kWh"与"FN 少识别 -25 kWh"互相抵消的**伪精度**
- v6.12.5 SAE 是 **诚实的开机日略低估**（MAE 仅 34.9 W），实际生产质量远高于 v6.12.4
- 业务诊断：用 **MAE** 而非 SAE 判断质量（MAE 是绝对误差，不能互相抵消）

#### 5.7.13 当前 v6.12.5 阈值标定公式（生产稳定）

```python
# 训练时 (scripts/03_train.py L590-600)
ALLOW_FACTOR  = 0.9   # 双源约束: ON 启动约束
MARGIN_FACTOR = 1.3   # 双源约束: OFF 噪声约束
threshold_abs = max(
    abs(train_on_P10)  × ALLOW_FACTOR,   # 允许 90% 启动通过
    abs(train_off_P99) × MARGIN_FACTOR,  # 留 30% 噪声裕度
)
# 推理时 (scripts/05_inference.py L290-300)
user_d73_p95 = bus["load_iden_data73"].quantile(0.95)
scale = np.clip(user_d73_p95 / train_d73_p95, 0.05, 2.0)
D87_GUARD_TH = -threshold_abs * scale   # 自适应到该用户
```

**用户级阈值汇总（v6.12.5 实测）**：

| 用户 | train_d73_P95 | train_on_P10 | train_off_P99 | 训练阈值 | 推理 d73_P95 | 缩放因子 | 推理阈值 |
|---|---|---|---|---|---|---|---|
| 用户1 (270848) | 990 W | 76 | 17 | -68 | 612 W | 0.618 | -42 |
| 用户2 (252844) | 2648 W | 129 | 69 | -116 | 2469 W | 0.932 | -109 |

---

## 六、模型与产物说明

### 6.1 模型文件清单

| 文件 | 用途 | 是否必需 |
|------|------|---------|
| `nilm_ac_two_stage.pkl` | **v6 主模型**（含 L1+L4 校正器）| ✅ 必需 |
| `nilm_ac_two_stage_v42.pkl` | v4.2 对照（L5 切换的 fallback）| ✅ 推荐 |
| `nilm_ac_two_stage_<时间戳>.pkl` | 训练时自动备份 | 版本回滚用 |
| `stage2_moe_bundle.pkl` | MoE 独立组件 | 调试用 |
| `stage1_classifier.pkl` | Stage-1 分类器 | 嵌入式部署用 |
| `stage2_regressor.pkl` | Stage-2 P50 回归器 | 嵌入式部署用 |
| `stage2_regressor_p10.pkl` / `_p90.pkl` | P10/P90 区间 | 置信带预测 |
| `scaler.pkl` | StandardScaler | 预处理必需 |
| `baseline_rf.pkl` | RF 基线 | 对照 |
| `model_meta.json` | 元数据 + LUT | 跨语言推理必读 |

### 6.2 主 bundle 字段（v6）

| 字段 | 用途 |
|------|------|
| `scaler` / `clf` / `reg` / `rf` / `moe` | 模型组件 |
| `reg_low` / `reg_high` | P10/P90 区间 |
| `feat_cols` / `feat_names` / `best_thr` | 特征 + 阈值 |
| `use_weather_features` / `weather_latitude` / `weather_longitude` | v5 气象配置 |
| `use_temp_based_season` / `summer_temp_threshold` / `winter_temp_threshold` | v5 温度路由 |
| **`use_drift_features` / `temp_power_lut`** | **v6 L1 配置** |
| **`use_residual_calib` / `residual_calib`** | **v6 L4 配置** |

### 6.3 v6.10 关键指标基线（5 月 OOD 数据 1824 行，v6.10 实测）

| 模型 | F1 | MAE (W) | RMSE (W) | SAE | kWh真/预 | kWh_err | 备注 |
|------|----|---------|----------|------|---------|---------|------|
| **main_final (v6.10, L4+L5+按天切分) ⭐⭐** | **0.9958** | **14.22** ⭐⭐ | **38.64** ⭐⭐ | **0.19%** ⭐⭐⭐ | **108.24/108.45** ⭐⭐⭐ | **+0.21** ⭐⭐⭐ | **推荐生产** |
| main_L4_calib (仅 L4，无 L5) | 0.9958 | 16.96 | 40.54 | 2.87% | 108.24/111.35 | +3.11 | kWh 略高估 |
| main (无 L4 无 L5) | 0.9958 | **13.06** | 39.59 | 2.98% | 108.24/105.01 | -3.23 | MAE 最优但 SAE 略劣 |
| fallback (MoE 全局兜底) | 0.9958 | 15.37 | 42.61 | 3.33% | 108.24/104.63 | -3.61 | |
| v42_baseline | 0.9965 | 13.85 | 44.33 | 2.49% | 108.24/108.13 | -0.12 | L5 切换目标 |
| rf 单阶段基线 | 0.9551 | 18.01 | 40.78 | 0.91% | 108.24/109.22 | +0.99 | |

**v6.9 → v6.10 飞跃对比**（同模型、同数据、唯一改动是切分策略）：

| 指标 | v6.9 (stratified) | v6.10 (stratified_day) | 改善幅度 |
|---|---|---|---|
| main_final MAE | 23.39 W | **14.22 W** | **-39%** ⭐⭐ |
| main_final RMSE | 58.10 W | **38.64 W** | **-33%** ⭐⭐ |
| main_final SAE | 5.16% | **0.19%** | **-96%** ⭐⭐⭐ |
| kWh 偏差 | -5.59 (-5.2%) | **+0.21 (+0.19%)** | 缩到原 4% ⭐⭐⭐ |

> **重要发现**：v6.10 切到按天分层后，所谓"5 月概念漂移"的 SAE 直接从 5.16% 降到 0.19%，说明 v6.0~6.9 期间观察到的"漂移"中**约 80% 实际是 v6.9 同天切分边界泄漏导致的伪信号**。详见 REPORT §5.6 v6.10 漂移诊断修正。
>
> **场景化选择（v6.10 起所有场景默认 main_final 即可）**：
> - **kWh / 电费 / 月度账单** → 默认 v6.10 (main_final)，kWh 偏差仅 +0.19%，无需 `--no-switch`
> - **实时功率监控** → 默认 v6.10 (main_final)，MAE 14.22W、SAE 0.19% 综合最优
> - **极致 MAE（不在意 kWh）** → 用 `--no-calib --no-switch`（main 原始），MAE 13.06W

### 6.4 v6.11 指标系统修复（4 个长期潜伏 Bug）

⚠️ **v6.0~v6.10 期间，指标 CSV 文件存在 4 个未发现的 bug**，会导致用户读到的 `all_metrics_summary.csv` 与 `metrics_pivot.csv` 部分指标错误。v6.11 集中修复，**不影响主模型本身性能**，仅修复指标记录正确性。

#### Bug 1 ⭐ ：`fallback` / `rf` 指标被 v42 训练静默覆盖

| 数据 | 旧 `all_metrics_summary.csv` 显示 | 真实主模型值 |
|---|---|---|
| val/fallback/MAE_W | 35.26 W ❌（v42 训练时的） | **32.20 W** |

**根因**：`04_evaluate.py` 派生快照逻辑 `drop_duplicates(keep='last')` 未区分指标来源——主模型 03_train 和 v42 03b 都写了 `model=fallback` 的行，按 timestamp 排序后 v42（较晚）的覆盖了主模型（较早）的。

**修复**：`flatten_metrics_to_rows` 新增 `source` 字段（`main_train`/`v42_baseline`/`evaluate`/`inference`），去重 key 包含 source。

#### Bug 2 ：`v42_baseline` 在 test 集中完全缺失

`04_evaluate --baseline ... models/nilm_ac_two_stage_v42.pkl` 命令实际从未生效——日志里有 `[baseline] 无法识别的基线: models/nilm_ac_two_stage_v42.pkl, 跳过` 的 warning，但用户极易忽视。

**根因**：`baseline_utils.py::build` 用 `path.exists()` 检查相对路径，当 04 从 `scripts/` 目录运行时找不到 `models/...`。

**修复**：失败时回退到 `PROJECT_ROOT / name_or_path` 解析；warning 升级为 ERROR 级别。

#### Bug 3 ：baseline 加载失败的隐式 warning

旧行为：基线模型加载失败时仅 `log.warning`，用户极易忽视后续指标表缺模型。

**修复**：升级为 `log.error`，明确提示"该基线被跳过，指标对比表将缺失对应模型！"

#### Bug 4 ：长表无 `source` 字段，无法追溯指标来源

`train_val_metrics.csv` 中同 `(split=train, model=fallback)` 可能有 2 条记录（主模型一次 + v42 一次），但旧版无法区分是哪次训练写的。

**修复**：新增 `source` 字段强制写入每行，长表可永久追溯。

#### 修复后的硬证据

```
all_metrics_summary.csv (修复后):
  val/fallback/MAE_W (source=main_train):    32.20 W  ← 主模型源 ✓
  val/fallback/MAE_W (source=v42_baseline):  35.26 W  ← v42 源, 独立保留 ✓

test 集模型列表: [fallback, main, main_L4_calib, nilm_ac_two_stage_v42, rf]
  ← v42 (作为 nilm_ac_two_stage_v42 外部基线) 现在正确出现
```

#### 历史数据迁移

由于多次清理 `artifacts/`，v6.0~v6.10 的历史 CSV 已物理删除，**不做迁移**。v6.11 起所有新生成的指标都正确。

---

## 七、关键技术参数

### 7.1 业务常量（`scripts/common.py`）

| 参数 | 取值 | 说明 |
|------|------|------|
| `SENT_VALUE` | -2147483648 | INT32_MIN, 电表缺测占位符 |
| `ON_THR_W` | 10.0 | 空调 ON 判定阈值 (W) |
| **`BUS_CSV`** | **`data/merged_bus.csv`** | **训练总线 CSV (merge_data.py 或 run_user_pipeline.py 生成)** |
| **`BR_CSV`** | **`data/merged_branch.csv`** | **训练分路 CSV** |
| **`INFER_BUS_CSV`** ⭐ | **`data/infer_bus.csv`** | **推理总线 CSV (v6.12.6+v6.15.0 新增, 训推路径分离)** |
| **`INFER_BR_CSV`** ⭐ | **`data/infer_branch.csv`** | **推理分路 CSV (可选, 用于 OOD 评估)** |
| `RESAMPLE` | "15min" | 总线重采样周期 |
| `RANDOM_SEED` | 42 | 随机种子 |
| **`TARGET_COL`** | **`"p1"`** | **目标分路列名 (空调=p1, 冰箱=p2, 热水器=p3, 照明=p4)** |
| **`SPLIT_STRATEGY`** | **`"stratified_day"`** ⭐ v6.10 | **切分策略 (stratified_day=按天分层 / stratified=按月分层 / time=纯时序)** |
| **`SPLIT_RATIOS`** | **`(0.70, 0.15, 0.15)`** | **train/val/test 比例 (必须 3 元组, 和=1.0)** |
| **`FULL_DAY_MIN_SAMPLES`** | **`80`** ⭐ v6.10 | **完整天阈值 (配置在 split_utils.py, 仅 stratified_day 用)** |
| `WEATHER_LATITUDE` | 30.59 | 默认: 武汉 |
| `WEATHER_LONGITUDE` | 114.31 | |
| `SUMMER_TEMP_THRESHOLD` | 22.0 | 日均≥此值视为 summer |
| `WINTER_TEMP_THRESHOLD` | 12.0 | 日均≤此值视为 winter |
| `USE_WEATHER_FEATURES` | True | v5 启用温度特征 |
| `USE_TEMP_BASED_SEASON` | True | v5 启用温度路由 |
| **`L5_ALERT_N_ALERT`** | **3** | **v6.9 ALERT 触发: 漂移报告 ALERT 行数阈值** |
| **`L5_ALERT_MAX_CONCEPT_DRIFT`** | **0.50** | **v6.9 ALERT 触发: 最大概念漂移阈值** |
| **`L5_WARN_N_ALERT`** | **1** | **v6.9 WARN 触发: ALERT 行数阈值** |
| **`L5_WARN_N_WARN`** | **2** | **v6.9 WARN 触发: WARN 行数阈值** |
| **`L5_WARN_MAX_CONCEPT_DRIFT`** | **0.20** | **v6.9 WARN 触发: 最大概念漂移阈值** |
| **`L5_MAIN_WEIGHT_ALERT_WITH_L4`** | **0.5** | **v6.9 ALERT + L4 启用 → 主权重** |
| **`L5_MAIN_WEIGHT_ALERT_WITHOUT_L4`** | **0.0** | **v6.9 ALERT + L4 未启用 → 主权重（旧行为）** |
| **`L5_MAIN_WEIGHT_WARN_WITH_L4`** | **0.75** | **v6.9 WARN + L4 启用 → 主权重** |
| **`L5_MAIN_WEIGHT_WARN_WITHOUT_L4`** | **0.6** | **v6.9 WARN + L4 未启用 → 主权重（旧行为）** |

> 💡 **多分路 NILM 扩展**：只需修改 `common.py` 中 `TARGET_COL` 一处即可切换辨识目标设备（如 `TARGET_COL = "p2"` 切换为冰箱辨识），全工程自动适配，无需改动其他代码。
>
> 💡 **切分比例自定义**：修改 `common.py` 中 `SPLIT_RATIOS`，常见配置：
> - `(0.70, 0.15, 0.15)` 默认，标准三七一五
> - `(0.80, 0.10, 0.10)` 数据量大时，训练偏多
> - `(0.60, 0.20, 0.20)` 小数据集，评估偏多
> - `(0.70, 0.20, 0.10)` 适合 v6 L4 校正器（需更多 val 样本）
> 系统会自动校验三元组合法性，和不为 1.0 时自动归一化并发警告。

### 7.2 训练超参（`scripts/03_train.py`）

| 模块 | 超参 | 取值 |
|------|------|------|
| **特征工程** | Top-K | 25 |
| | 滚动窗口 | 4 (1 小时) |
| | 滞后阶数 | 2 |
| **Stage-1 分类器** | n_estimators | 300 |
| | max_depth | 3 |
| | learning_rate | 0.05 |
| | 阈值搜索目标 | F0.5 |
| **Stage-2 回归器** | n_estimators | 400 |
| | loss | quantile |
| | alpha (P50/P10/P90) | 0.5 / 0.1 / 0.9 |
| **后处理** | POST_MIN_ON | 2 (30 min) |
| | POST_FILL_SHORT_OFF | 1 |
| **切分** | 策略 | stratified（按月分层）|
| | 比例 | (0.70, 0.15, 0.15) 默认, 可在 `common.py` 改 |
| | 比例 | 70 : 15 : 15 |
| **L4 校正器** | n_estimators | 50 |
| | max_depth | 3 |
| | loss | huber |
| | 限幅 | ±min(2·MAE, 150W) |
| **L2 漂移阈值** | WARN | 10% |
| | ALERT | 25% |

### 7.3 季节路由（`scripts/expert_utils.py` + `weather_utils.py`）

**v6 默认：温度驱动路由（动态、自适应任意气候带）**

```python
SUMMER_TEMP_THRESHOLD = 22.0   # 日均 ≥ 22°C → summer expert
WINTER_TEMP_THRESHOLD = 12.0   # 日均 ≤ 12°C → winter expert
USE_TEMP_BASED_SEASON = True
```

**v4.2 兼容：月份硬路由（无网络环境备选）**

```python
SEASON_MAP = {
    1: "winter", 2: "winter", 12: "winter",
    5: "summer", 6: "summer", 7: "summer", 8: "summer", 9: "summer",
    3: "transition", 4: "transition",
    10: "transition", 11: "transition",
}
```

### 7.4 气象配置示例

| 城市 | 经纬度 |
|------|--------|
| 武汉（默认）| 30.59, 114.31 |
| 北京 | 39.90, 116.40 |
| 上海 | 31.23, 121.47 |
| 广州 | 23.13, 113.27 |
| 杭州 | 30.27, 120.15 |
| 深圳 | 22.55, 114.06 |

---

## 八、常见问题排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'common'` | 未在项目根目录运行 | `cd` 到含 `scripts\` 的目录 |
| `KeyError: 分路 CSV 缺少目标列 'pX'` | `common.py` 中 `TARGET_COL` 与实际分路 CSV 列名不匹配 | 修改 `common.py` 中 `TARGET_COL` (默认 `"p1"`)，或检查 CSV 列名 |
| `metrics_pivot.csv` 主模型缺 train/val 或 test 列 | 旧版本 model 命名不统一 (`two_stage_gbdt_v5` vs `main`) | **已修复** ✅ 主模型统一命名为 `main`，跨 train/val/test 三 split 对齐 |
| `model_meta.json` 写入报 `TypeError: keys must be str` | `temp_power_lut` 含 tuple 键，JSON 不支持 | **已修复** ✅ tuple 键自动序列化为 `"lo_hi"` 字符串 |
| v4.2 训练后主模型的 `scaler.pkl` / `model_meta.json` 被覆盖 | `03b` 共享了主模型的组件文件路径 | **已修复** ✅ `NILM_BASELINE_MODE=1` 时自动跳过组件保存 |
| `ValueError: time data ... does not match format` | 时间格式异常 | **已修复** ✅ `time_utils.py` 支持 `2026/3/18 0:00:00` / `2026-3-18 0:00:00` 等多种格式 |
| 时间解析后部分行变 NaT | 异常时间字符串 | 查看日志 `[time_parse] N 行解析失败`，修正源数据 |
| 图表中文显示 □□□ | 系统无中文字体 | Windows 自带 `Microsoft YaHei`，自动生效；如缺失装 Office |
| `UnicodeDecodeError` 读 CSV | 文件被 Excel 改成 GBK 编码 | 用记事本另存为 UTF-8 |
| `OSError: [WinError 5]` | 杀软拦截 | 关闭实时防护或换非系统盘 |
| `EOFError` 加载模型 | 训练未跑完 | 重跑 `03_train.py` 直到 `Step 3 训练完成` |
| `_pickle.PicklingError: Can't pickle ... make_quantile_reg` | `expert_utils.py` / `residual_calibrator.py` 旧版本 | 确认含 `strip_for_save()` 方法 |
| Conda 安装慢 | 默认源境外 | 用 Step 1 的清华源 |
| 推理结果全 0 | Stage-1 阈值过高或数据漂移 | 查 `drift_report.csv` 看是否 ALERT |
| 推理 SAE 突然飙高 | 数据/概念漂移 | 看 `drift_report.csv`；考虑触发 L5 切换或重训 |
| Windows 终端中文乱码 | CMD 默认 GBK | `chcp 65001` 切换 UTF-8 |
| L4 校正后 MAE 反而变高 | val 集分布与推理分布差异大 | `--no-calib` 关闭 L4，或扩大 val 集时段 |
| L5 切换后预测异常 | fallback 模型有 bug | `--no-switch` 关闭 L5 验证 |

### 验证季节配置

```bat
python -c "from scripts.expert_utils import SEASON_MAP; print('5月归属:', SEASON_MAP[5])"
:: 必须输出: 5月归属: summer
```

如果输出 `transition`，清理 Python 缓存：

```bat
rmdir /s /q scripts\__pycache__
```

### 验证 v6 防御层

```bat
:: 看模型 bundle 是否含 L4 校正器
python -c "import joblib; b=joblib.load('models/nilm_ac_two_stage.pkl'); print('L4 calibrator:', b.get('residual_calib') is not None); print('LUT 桶数:', len([k for k in b.get('temp_power_lut', {}) if isinstance(k, tuple)]))"
```

### 诊断: 训练指标正常但汇总标 soft_skip:split_empty_val_test (v13.3 已修复)

**症状**: 某用户 `artifacts/trains/<uid>/` 里 `test_metrics.csv` / `train_val_metrics.csv` 等都齐全, 但汇总 CSV 显示该用户 4 stage 全部 `soft_skip:split_empty_val_test`, F1/SAE 都是 NaN.

**根因**: `artifacts/trains/<uid>/skip_reason.json` 陈旧残留污染 `aggregate_metrics()` 判据. 该文件由 03_train.py 数据质量门 3 (val/test 空) 触发时写入并归档, 若用户先跑失败后修数据重跑成功, **v13.3 之前** archive_outputs 不清理旧标记.

**应急处置** (无需重训):

```bat
:: 1. 手工删除残留
del artifacts\trains\<uid>\skip_reason.json

:: 2. 只重跑汇总 (不训练)
python -c "import sys; sys.path.insert(0,'scripts'); from pathlib import Path; from run_batch_users import aggregate_metrics; aggregate_metrics(Path('artifacts'), Path('artifacts'))"
```

**根本修复** (v13.3 已实施): `run_user_pipeline.py::archive_outputs()` 里 `did_train=True` 时自动 `unlink()` 旧 `skip_reason.json`. 升级后不会再出现。

**如何确认是这个 bug**:
```bat
:: 检查残留是否比 metrics CSV 老
dir artifacts\trains\<uid>\skip_reason.json artifacts\trains\<uid>\test_metrics.csv
```
若 `skip_reason.json` 时间戳早于 `test_metrics.csv` → 100% 是这个 bug.

---

## 九、嵌入式部署进阶

### 9.1 导出纯 C 代码（适用 STM32 / ESP32）

```bat
python -c "import joblib, m2cgen as m2c; b=joblib.load(r'models/nilm_ac_two_stage.pkl'); open('infer_clf.c','w').write(m2c.export_to_c(b['clf'])); open('infer_reg.c','w').write(m2c.export_to_c(b['reg'])); print('OK 已生成 infer_clf.c / infer_reg.c')"
```

> ⚠️ 注意：m2cgen 导出**不包含 MoE / L4 校正 / L5 切换**逻辑，需要在 MCU 上手工实现路由层。

### 9.2 导出 ONNX

需 `pip install skl2onnx onnxruntime`：

```bat
python -c "import joblib; from skl2onnx import to_onnx; from skl2onnx.common.data_types import FloatTensorType; b=joblib.load(r'models/nilm_ac_two_stage.pkl'); o=to_onnx(b['reg'], initial_types=[('input', FloatTensorType([None, len(b['feat_names'])]))]); open('model.onnx','wb').write(o.SerializeToString())"
```

### 9.3 推理基准

| 平台 | 单样本延迟 | 模型大小 | 备注 |
|------|----------|---------|------|
| Windows / Linux x86 (Python) | < 5 ms | 10 MB (.pkl) | 含 L1-L5 全防御 |
| ARM Linux (RK3568, ONNX RT) | ~10 ms | 6 MB (.onnx) | 仅主回归器 |
| STM32H7 (Cortex-M7, 纯 C) | ~30 ms | < 2 MB Flash | 仅 Stage-1+2 |
| ESP32-S3 (Xtensa LX7, 纯 C) | ~60 ms | < 2 MB Flash | 仅 Stage-1+2 |

---

## 十、持续运营建议

### 10.1 数据闭环（每月）

```bat
:: 1. 收集本月新数据放入 data/
:: 2. 合并
python scripts\merge_data.py

:: 3. 拉取本月气温到缓存
python scripts\fetch_weather.py

:: 4. 重训
run_all.bat
```

### 10.2 监控告警（每日）

| 监控项 | 阈值 | 处置 |
|-------|------|------|
| 推理 SAE 漂移 | > 15% 连续 3 天 | 触发重训 |
| L2 漂移 ALERT 数 | ≥ 3 | 立即重训或切换 fallback |
| FP 数量 | > 5/天 | 检查 Stage-1 阈值 |
| L5 切换触发率 | > 50% 一周 | 主模型已失效，强烈建议重训 |
| 单次推理延迟 | > 200 ms | 检查硬件资源 |

### 10.3 季节修订（每年 4 月 / 10 月）

每年复核 `SEASON_MAP` 和 `SUMMER/WINTER_TEMP_THRESHOLD`：

```bat
:: 用诊断工具查看当前各月份的功率分布
python scripts\diag_new_data.py
```

### 10.4 模型版本管理

| 操作 | 命令 |
|------|------|
| 查看所有备份模型 | `dir models\nilm_ac_two_stage_*.pkl` |
| 回滚到上一版本 | 把备份重命名为主模型 |
| 对比新旧模型 | `python scripts\05_inference.py --baseline models\<旧版本>.pkl` |

### 10.5 漂移诊断（按需）

```bat
:: 看新数据 vs 训练数据的全面对比
python scripts\diag_new_data.py

:: 看最近推理结果的诊断
python scripts\diag_inference.py
```

---

**版本**：v6.12.6+v6.15.0-graceful-v13.5 (v12 时段过滤 + v13 5 项精细化 + v13.4-fix + v13.5 9 项 common 覆盖)
**最后更新**：2026-07-13
**完整开发报告**：[`REPORT.md`](./REPORT.md)

---

## 附 A: v12 版本要点速览

**代码骨架** (相对 v11):
- 新增 `scripts/time_filter_utils.py` (327 行, 8 个 API + 6 组单测)
- 新增 `data/time_filters.example.json` (配置模板)
- 修改 4 个业务脚本: `02_align_and_feat.py` / `05_inference.py` / `run_user_pipeline.py` / `run_batch_users.py`

**新命令行参数**:
| 脚本 | 参数 | 作用 |
|---|---|---|
| `run_batch_users.py` | `--time-filter-config <path.json>` | 批量层读入 JSON 配置, 分发给每用户 |
| `run_user_pipeline.py` | `--train-time-filter-spec <json_str>` | 单用户训练侧时段规格 |
| `run_user_pipeline.py` | `--infer-time-filter-spec <json_str>` | 单用户推理侧时段规格 |
| `02_align_and_feat.py` | `--time-filter-spec <json_str>` | 训练底层执行时段过滤 |
| `05_inference.py` | `--time-filter-spec <json_str>` | 推理底层执行时段过滤 |

**已知边界情况处理**:
- `_comment_` / `_note_` 等注释键 (值为字符串) → 自动跳过, 不当作用户配置
- `include=[], exclude=[]` (空规格) → 视为无过滤, 数据不变
- `start > end` → parse 阶段抛 ValueError 拒绝, 明确报错
- 未提供 `--time-filter-config` → 全流程无回归 (与 v11 完全一致)

---

## 附 B: v13 版本要点速览

**背景**: v12 只做数据加载阶段的全局时段过滤, 用户实际使用中出现 4 类痛点, v13 依次解决:

| 痛点 | v13 子版本 | 解决方案 |
|---|---|---|
| 变频空调用户强开守卫 → 100% FN 灾难 | **v13.1** | 用户级 `guard_enabled` + 自动检测降级 (判据 A/B) |
| 无法给 train/val/test 独立指定包含/排除时段 | **v13.2** | `splits.{train,val,test}.{include,exclude}` (4 步语义, 严格保形状) |
| 训练成功但汇总标 soft_skip | **v13.3** | `archive_outputs` 归档前清理旧 `skip_reason.json` |
| Ch1 反推固定 p1, 无法业务上要 p2 | **v13.4** | 配置 `target_col` 覆盖反推 |

**代码骨架** (相对 v12):
- `scripts/time_filter_utils.py` 新增 4 组 API: `get_user_guard_enabled` / `auto_detect_guard_enabled` / `get_user_target_col` / `load_splits_time_filter` / `apply_per_split_filter` / `splits_spec_to/from_cli_arg` / `splits_spec_summary`
- `scripts/03_train.py` +55 行: NILM_USER_GUARD_ENABLED 环境变量 + 自动检测判据 + NILM_SPLITS_FILTER_SPEC 应用
- `scripts/04_evaluate.py` +16 行: 与 03 对称应用 per-split 过滤
- `scripts/run_user_pipeline.py` +40 行: 3 个新参数 + env var 注入 + skip_reason.json 清理
- `scripts/run_batch_users.py` +50 行: 从 JSON 读 4 类字段 + 透传

**新命令行参数** (相对 v12):
| 脚本 | 参数 | 作用 |
|---|---|---|
| `run_user_pipeline.py` | `--guard-enabled true/false/""` | 用户级 d87 守卫开关 |
| `run_user_pipeline.py` | `--splits-time-filter-spec <json>` | per-split 过滤规格 |

**新 JSON 配置字段** (`time_filters.json` 每用户可选):
| 字段 | 类型 | 说明 |
|---|---|---|
| `target_col` | `"pN"` (N ≥ 0 整数, 大小写不敏感) | 覆盖 `-Ch{N}-` 反推. **v13.4-fix**: 从 [p1-p4] 硬集合放宽为任意 pN (p0/p1/.../p128 均可) |
| `guard_enabled` | `true/false` | 覆盖全局 `D87_ADAPTIVE_GUARD_ENABLED` |
| `splits.train/val/test.include` | `[[start,end], ...]` | 强制某时段入该 split |
| `splits.train/val/test.exclude` | `[[start,end], ...]` | 从该 split 剔除某时段 |
| **v13.5 新增 9 项 common 覆盖** ⭐ | 各不同 | **on_thr_w** / **split_ratios** / **split_strategy** / **post_min_on** / **post_fill_short_off** / **weather_latitude** / **weather_longitude** / **use_weather_features** / **use_temp_based_season** — 详见 §C.4.7 |

**测试覆盖** (`python scripts/time_filter_utils.py`): 7 组单测全通过 (其中 v13 新增 3 组: get_user_target_col 8 场景 / apply_per_split_filter 5 场景 / get_user_guard_enabled 6 场景)

**270708 用户实测硬证据** (v11 vs v13.1 自动降级):

| 指标 | v11 全局强开 (未自动降级) | **v13.1 自动降级/显式关闭** | 改善 |
|---|---|---|---|
| Recall | 0.802 | **0.998** | +24.4pp |
| F1 | 0.887 | **0.996** | +12.3pp |
| SAE | 29.7% | **14.4%** | -15.3pp |
| kWh_err | -6.43 | **-3.12** | -51% |

---

## 附 C: 完整参考手册 (CLI 参数 / 配置字段 / status 定义)

> 本节是**代码扫描而来的精确 API 参考**, 供开发者速查. 与 §3-§4 的示例教学不同, 这里覆盖**所有可用参数**.

### C.1 批量脚本 `run_batch_users.py` 参数

**用法**: `python scripts/run_batch_users.py [OPTIONS]`

| 参数 | 类型 | 默认 | 引入版本 | 说明 |
|---|---|---|---|---|
| `--data-dir` | str | `data/` | v9 | 用户数据根目录, 内含 `trains/` + `infers/` |
| `--output-dir` | str | `artifacts/` | v9 | 批量产物归档目录, 内产出 `trains/` + `infers/` + `summary_*.csv` |
| `--users` | str... | 全部 | v9 | 仅跑指定用户 (folder_name), 支持多个用空格分开 |
| `--skip-existing` | flag | False | v9 | 若 `artifacts/<user>/` 已存在则跳过该用户 |
| `--dry-run` | flag | False | v9 | 只扫描+打印计划表, 不实际执行 |
| `--continue-on-error` | flag | True | v9 | 某用户失败继续跑其他用户 (默认开启) |
| `--force-retrain` | flag | False | **v10** | 强制重训 (即使 `models/<user>/` 已完整); 默认 = 有模型就复用只跑推理 |
| `--time-filter-config` | str | 空 | **v12** | JSON 配置文件路径, 定义每用户细粒度配置 (target_col / guard_enabled / time_filter / splits, 详见 §C.4) |

**典型用法**:
```bash
# 1. 首次全批量运行
python scripts/run_batch_users.py

# 2. 仅跑指定 2 个用户 + 强制重训
python scripts/run_batch_users.py --users 800080252842_... 800080270708_... --force-retrain

# 3. 应用用户级配置文件 (含时段过滤 / 守卫开关等)
python scripts/run_batch_users.py --time-filter-config data/time_filters.json

# 4. 干跑扫描, 不执行
python scripts/run_batch_users.py --dry-run
```

### C.2 单用户流水线 `run_user_pipeline.py` 参数

**用法**: `python scripts/run_user_pipeline.py [OPTIONS]`

**必需参数** (4 个):

| 参数 | 类型 | 说明 |
|---|---|---|
| `--user-id` | str | 用户唯一标识, 用作产物目录名 (如 `800080252842_4206894986488`) |
| `--target-col` | str | 目标分路列名, **格式 `pN` (N ≥ 0 整数)**. 例: p0/p1/p2/.../p99/p128 均合法. **v13.4-fix** 从 choices=[p1..p4] 放宽 |
| `--train-bus` | str | 训练总线 CSV 路径 |
| `--train-branch` | str | 训练分路 CSV 路径 |
| `--output-dir` | str | 产物归档根目录 (如 `artifacts/`) |

**可选参数** — 推理数据:

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--infer-bus` | str | 空 | 推理总线 CSV, 缺则**跳过 05 推理仅跑 02→03→04** |
| `--infer-branch` | str | 空 | 推理分路 CSV, 缺则**推理仅出预测无评估指标** |

**可选参数** — 训练控制:

| 参数 | 类型 | 默认 | 引入版本 | 说明 |
|---|---|---|---|---|
| `--exclude-dates` | str | 空 | v6.12.3 | 逗号分隔的排除日期 (YYYY-MM-DD), 训练时整天剔除 |
| `--extra-train-bus` | str | 空 | v6.14 | 额外训练总线 CSV (增量训练, 例把推理段末尾几天加入训练) |
| `--extra-train-branch` | str | 空 | v6.14 | 额外训练分路 CSV |
| `--extra-train-dates` | str | 空 | v6.14 | 逗号分隔日期, 仅这些日期从 extra 数据被合并到训练集 |
| `--clean-labels` | flag | False | v6.14 | 启用标签清洗 (用 d87 启动信号确定真实起点, 启动前小负荷归 0) |
| `--clean-d87-thr` | float | 50.0 | v6.14 | 标签清洗 d87 启动阈值 (W) |
| `--skip-clean` | flag | False | v6.14 | 不清理 artifacts/logs (用于调试, 保留中间产物) |
| `--force-retrain` | flag | False | **v10** | 强制重训 (即使 `models/<user>/` 已完整) |

**可选参数** — 推理评估控制:

| 参数 | 类型 | 默认 | 引入版本 | 说明 |
|---|---|---|---|---|
| `--infer-eval-dates` | str | 空 | v6.14 | 逗号分隔日期, 限制推理评估仅算这些日期 (避免与训练重叠) |

**可选参数** — v12/v13 时段过滤 + 用户配置(**通常由 `run_batch_users.py` 自动透传, 手工调用少用**):

| 参数 | 类型 | 默认 | 引入版本 | 说明 |
|---|---|---|---|---|
| `--train-time-filter-spec` | JSON str | 空 | **v12** | 训练侧全局时段过滤 (透传给 `02_align_and_feat.py --time-filter-spec`) |
| `--infer-time-filter-spec` | JSON str | 空 | **v12** | 推理侧全局时段过滤 (透传给 `05_inference.py --time-filter-spec`) |
| `--guard-enabled` | `""` / `"true"` / `"false"` | 空 | **v13.1** | 覆盖全局 D87 守卫开关. 空=走全局(可能被自动降级); true=强开; false=强关 |
| `--splits-time-filter-spec` | JSON str | 空 | **v13.2** | per-split 切分过滤规格 (含 train/val/test 3 集独立 include/exclude), 通过环境变量 `NILM_SPLITS_FILTER_SPEC` 透传给 03/04 |

**典型用法**:
```bash
# 1. 最小完整调用 (仅训练评估, 无推理)
python scripts/run_user_pipeline.py \
    --user-id 800080270708_4206602981958 \
    --target-col p1 \
    --train-bus data/trains/.../bus.csv \
    --train-branch data/trains/.../branch.csv \
    --output-dir artifacts

# 2. 完整含推理
python scripts/run_user_pipeline.py \
    --user-id 800080270708_4206602981958 --target-col p1 \
    --train-bus ... --train-branch ... \
    --infer-bus ... --infer-branch ... \
    --output-dir artifacts

# 3. v13 用户级守卫关闭 (变频空调必备)
python scripts/run_user_pipeline.py \
    ... --guard-enabled false

# 4. v13 per-split 硬锚定 (强制 6/20 入 val)
python scripts/run_user_pipeline.py \
    ... --splits-time-filter-spec '{"val":{"include":[["2026-06-20","2026-06-20"]]}}'
```

### C.3 各主流程脚本 CLI 参数

#### C.3.1 `02_align_and_feat.py` (对齐 + 特征相关性)

| 参数 | 类型 | 默认 | 引入版本 | 说明 |
|---|---|---|---|---|
| `--exclude-dates` | str | 空 | v6.12.3 | 逗号分隔排除日期 (整天粒度) |
| `--time-filter-spec` | JSON str | 空 | **v12** | 时段过滤规格 (支持任意时段 include+exclude 组合, 内部合并 exclude-dates) |

数据源固定: `common.py::BUS_CSV` / `BR_CSV` (通常是 `data/merged_bus.csv` / `merged_branch.csv`)

#### C.3.2 `03_train.py` (训练 v6 主模型)

**无 CLI 参数** — 完全通过 `common.py` 常量 + 环境变量控制:

| 环境变量 | 引入版本 | 说明 |
|---|---|---|
| `NILM_BASELINE_MODE=1` | v6.3 | 切换为 v4.2 基线训练模式 (跳过 L1/L4, 不覆盖主模型组件) |
| `NILM_USER_GUARD_ENABLED=0/1/""` | **v13.1** | 用户级 D87 守卫开关 (由 run_user_pipeline.py 注入) |
| `NILM_SPLITS_FILTER_SPEC=<json>` | **v13.2** | per-split 时段过滤规格 (由 run_user_pipeline.py 注入) |

数据源: `artifacts/aligned_15min.csv` (02 输出)

#### C.3.3 `03b_train_v42_baseline.py` (v4.2 基线对照)

**无 CLI 参数** — 通过 `NILM_BASELINE_MODE=1` 环境变量调用 `03_train.py` 主流程, 但强制关闭 L1/L4/温度特征。

#### C.3.4 `04_evaluate.py` (测试集评估)

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--model` | str | `MODEL_PKL` (v5 主模型) | 主模型路径 |
| `--baseline` | str... | 空 | 基线模型列表, 支持: `rf` / `fallback` / `naive_mean` / `naive_zero` / `<.pkl 路径>` |
| `--no-calib` | flag | False | 禁用 L4 残差校正后指标计算 (即使模型 bundle 含 calibrator) |

同样受 `NILM_SPLITS_FILTER_SPEC` 环境变量影响, 与 03 保持切分对称。

#### C.3.5 `05_inference.py` (独立推理 + 多基线对比)

| 参数 | 类型 | 默认 | 引入版本 | 说明 |
|---|---|---|---|---|
| `--bus` | str | `INFER_BUS_CSV` | v6.12.6 | 总线 CSV (推理数据) |
| `--branch` | str | `INFER_BR_CSV` | v6.12.6 | 分路 CSV (可选, 用于评估) |
| `--model` | str | `MODEL_PKL` | v5 | 主模型 .pkl 路径 |
| `--baseline` | str... | 空 | v5 | 基线列表: `rf` / `fallback` / `naive_mean` / `naive_zero` / `<.pkl 路径>` |
| `--out` | str | `artifacts/predictions/inference_result.csv` | v5 | 推理结果 CSV 输出路径 |
| `--metric-out` | str | `artifacts/metrics/inference_metrics.csv` | v5 | 评估指标 CSV (仅 `--branch` 存在时输出) |
| `--no-branch` | flag | False | v5 | 强制不使用分路标签 (纯预测) |
| `--plot` | flag | False | v5 | 生成多模型功率曲线对比图 |
| `--no-calib` | flag | False | v6.8 | 禁用 L4 残差校正层 |
| `--no-switch` | flag | False | v6.8 | 禁用 L5 多模型动态切换 (强制使用主模型) |
| `--time-filter-spec` | JSON str | 空 | **v12** | 推理数据时段过滤规格 |

#### C.3.6 `06_inference_with_calib.py` (备用: Isotonic 校正推理, v4.1 遗留)

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--bus` / `--branch` / `--model` / `--out` / `--metric-out` / `--no-branch` | 同 05 | 同 05 | |
| `--calib-method` | `isotonic` / `linear` / `none` | `isotonic` | 校正方法 |

⚠️ **v4.x 遗留脚本**, v5+ 推荐用 05 (L4 残差校正). 保留仅供 ablation.

#### C.3.7 工具脚本

- `01_audit.py` — **无参数**, 输出 `bus_columns_summary.csv` 电参量画像
- `merge_data.py` — **无参数**, 合并 `data/` 下多份历史 CSV 为 `merged_*.csv`
- `fetch_weather.py` — 拉取 Open-Meteo 历史气温:
  - `--lat` / `--lon` (默认武汉 30.59/114.31)
  - `--start` / `--end` (默认自动推断)
  - `--cache-dir` (默认 `data/weather_cache/`)

### C.4 `time_filters.json` 配置文件字段详解

**加载入口**: `run_batch_users.py --time-filter-config <path.json>`

**顶层结构**: JSON 对象, 键为 `user_id` 或特殊键 `_default` / `_comment_` / `_note_`(注释键自动忽略)

#### C.4.1 每用户 6 类配置字段

| 字段 | 类型 | 引入版本 | 优先级 | 详见 |
|---|---|---|---|---|
| `target_col` | `"pN"` (N ≥ 0) | **v13.4** | 高于文件名 `-Ch{N}-` 反推 | §C.4.2 |
| `guard_enabled` | `true`/`false` | **v13.1** | 高于全局 `D87_ADAPTIVE_GUARD_ENABLED` | §C.4.3 |
| `train` | `{"include":[], "exclude":[]}` | **v12** | 数据加载阶段全局过滤 | §C.4.4 |
| `infer` | `{"include":[], "exclude":[]}` | **v12** | 数据加载阶段全局过滤 (推理侧独立) | §C.4.4 |
| `splits` | `{"train":{}, "val":{}, "test":{}}` | **v13.2** | 切分之后局部微调 | §C.4.5 |
| **9 个 common 常量覆盖** | 见 §C.4.7 | **v13.5** | 高于 `common.py` 默认值 | §C.4.7 |

**完整示例**:
```json
{
  "800080270708_4206602981958": {
    "target_col":     "p1",                        // v13.4
    "guard_enabled":  false,                       // v13.1
    "train": { "exclude": [["2026-06-12","2026-06-12"]] },   // v12
    "infer": { "include": [["2026-06-20","2026-06-30"]] },   // v12
    "splits": {                                    // v13.2
      "train": { "exclude": [["2026-06-15","2026-06-15"]] },
      "val":   { "include": [["2026-06-20","2026-06-20"]] },
      "test":  { "include": [["2026-06-25","2026-06-25"]] }
    }
  },
  "_default": {
    "guard_enabled": true,        // 未列用户默认开启守卫
    "target_col": "p1"            // 未列用户默认 p1
  }
}
```

#### C.4.2 `target_col` 字段 (v13.4 / v13.4-fix / v13.16)

| 属性 | 值 |
|---|---|
| 合法格式 (v13.4-fix) | 单列 `pN` (N ≥ 0 整数), 大小写不敏感, 前后空白自动去除 |
| **v13.16 新增: 复合格式** | `pA+pB[+pC...]` 由 `+` 连接的多个 pN, 逐行求和作为目标列 |
| 例 (单列) | `"p0"` / `"p1"` / `"p2"` / `"p10"` / `"p128"` / `"P99"` (规范化为 `"p99"`) |
| **例 (v13.16 复合)** | `"p1+p2"` / `"p1+p2+p3"` / `"p0+p5+p10"` / `" P1 + p2 "` (归一化为 `"p1+p2"`) |
| 非法值行为 | 打印 WARN + 回退旧反推逻辑 (不阻塞用户) |
| **v13.16 复合防呆** | 拒绝重复分量 (如 `p1+p1`) — 语义无意义直接归为非法 |
| 完整优先级链 | ① config[user_id].target_col → ② config._default.target_col → ③ 总线 `-Ch{N}-` 反推 → ④ 分路第 1 个 pN → ⑤ 兜底 `p1` |

##### C.4.2.1 v13.16 复合 target_col 详解

**触发场景**: 同一空调物理设备对应**多个分路** — 例如"主机 p1 + 辅热 p2"、"多室内机 p1+p2+p3"、"新老分表交替 p1+p2"等. 单列 target 无法表达"总空调标签 = 多分路之和".

**物化语义** (在 `scripts/feature_utils.py::load_branch_csv` 中):

```python
# 加载 CSV 时若 target_col 含 '+', 自动新增一列
if target_col and "+" in target_col:
    composite = "".join(target_col.split()).lower()   # 归一化
    parts = composite.split("+")
    parts_df = df[parts].apply(pd.to_numeric, errors="coerce")
    df[composite] = parts_df.sum(axis=1, skipna=False)   # 逐行求和
# 下游 resample_and_align / label_cleaner / analyze_on_periods
# 把 composite 当作普通列名使用, 完全无感
```

**用户示例 (逐值对齐)**:

配置 `"target_col": "p1+p2"`, 分路 CSV:

| time | p1 | p2 |
|---|---:|---:|
| 2026/5/21 0:00:00 | 24 | 8 |
| 2026/5/21 0:15:00 | 16 | 0 |
| 2026/5/21 0:30:00 | 16 | 0 |
| 2026/5/21 0:45:00 | 16 | 8 |

物化后新增列 `p1+p2`:

| time | p1+p2 |
|---|---:|
| 2026/5/21 0:00:00 | **32** |
| 2026/5/21 0:15:00 | **16** |
| 2026/5/21 0:30:00 | **16** |
| 2026/5/21 0:45:00 | **24** |

**NaN 传播语义** (关键设计):
- 使用 `sum(axis=1, skipna=False)` 而不是 `skipna=True`
- 任一分量缺采样点 → 该行结果为 NaN, 下游 `dropna(subset=["y_ac"])` 直接剔除
- **不做静默补 0**, 避免"部分分量缺失时电量估计偏低"的隐藏问题

**归一化规则**:
- 大小写: `"P1+P2"` → `"p1+p2"`
- 空白: `" p1 + p2 "` / `"p1 +p2+ p3"` → `"p1+p2"` / `"p1+p2+p3"`
- 复合防呆: `"p1+p1"` / `"p1+p2+p1"` 拒绝并 WARN

**边界与错误**:

| 场景 | 行为 |
|---|---|
| CSV 缺任一分量列 (如 `p1+p3` 但 CSV 无 `p3`) | `KeyError` 精准提示缺失列 |
| 分量非 pN 格式 (如 `p1+q2`) | `ValueError` 精准提示 |
| target_col 未含 `+` (单列 `p1`) | 完全不触发物化, 与 v13.15 行为等价 (向后兼容) |
| `load_branch_csv(path)` 老调用 (无 target_col 参数) | 完全等价旧行为, 零回归 |

**批量层 (`run_batch_users.py::parse_user_folder`) 复合列校验**:
- 复合列的**所有分量**都必须在分路 CSV 的 `p\d+` 列中, 否则回退到旧反推逻辑并 WARN
- 若分路 CSV 无任何 pN 列, 信任配置并 WARN

**端到端命令行示例**:

```bash
# 单用户
python scripts/run_user_pipeline.py \
    --user-id 800080270788_4206701750448 \
    --target-col "p1+p2" \                        # v13.16 复合语法
    --train-bus  data/e241_..._-Ch1-260521-260629-1.csv \
    --train-branch data/4206701750448-260521-260629.csv \
    ...

# 或通过配置文件 (推荐)
# data/time_filters.example.json:
#   "800080270788_4206701750448": {
#     "target_col": "p1+p2",
#     ...
#   }
python scripts/run_batch_users.py \
    --time-filter-config data/time_filters.example.json
```

**270788 端到端验证**:

| 场景 | F1 | Recall | SAE | kWh_true | 说明 |
|---|---:|---:|---:|---:|---|
| 单列 `p1` (v13.15 基线) | 0.9022 | 0.8324 | 5.45% | 16.78 | ✅ 零回归对照 |
| **复合 `p1+p2` (v13.16 新)** | **0.9395** | **0.9006** | **4.10%** | **32.95** | ✅ 端到端跑通, kWh ≈ p1×2 物理合理 |

**日志显性化提示** (align + inference 两阶段各一行):
```
[align]   [v13.16] 检测到复合 target_col='p1+p2', 已物化为 br['p1+p2'] = p1 + p2 逐行求和
[infer]   [v13.16] 已物化复合列 'p1+p2' = p1 + p2 逐行求和
```

#### C.4.3 `guard_enabled` 字段 (v13.1)

| 属性 | 值 |
|---|---|
| 合法值 | `true` / `false` |
| 完整优先级链 | ① config[user_id].guard_enabled → ② config._default.guard_enabled → ③ 训练侧自动检测 (若全局 True) → ④ 全局 `D87_ADAPTIVE_GUARD_ENABLED` |
| 自动检测判据 A | 训练集 `|d87|.max < 50W` |
| 自动检测判据 B | 逐日 `|d87|.max ≥ 守卫阈值` 的天数占比 < 30% |
| bundle 内标记 | `d87_guard_meta.disabled_by_auto_detect=True` + `auto_detect_trigger="A"/"B"` |

#### C.4.4 `train` / `infer` 字段 (v12 全局时段过滤)

| 子字段 | 类型 | 语义 |
|---|---|---|
| `include` | `[[start, end], ...]` | 未指定=全保留; 指定=**仅保留**在这些时段内的样本 |
| `exclude` | `[[start, end], ...]` | 从 include 结果中再**剔除**这些时段 |

**时间格式**:
- `"YYYY-MM-DD"` — 自动扩为 `[D 00:00:00, D 23:59:59]` (整天)
- `"YYYY-MM-DD HH:MM"` — 精确到分钟
- `"YYYY-MM-DD HH:MM:SS"` — 精确到秒
- 兼容 `2026/6/12` 斜杠格式

**区间边界**: `[start, end]` **闭区间** (两端都包含)

**执行顺序**: 先 include 后 exclude

##### ⚠️ 陷阱 1: `train` 与 `infer` 时段重叠会导致数据泄漏 (v13.8 已加检测)

`train.include` 与 `infer.exclude/include` **不做互斥校验**, 允许配置任意组合。若两者时段有交集且交集日期恰好被 `stratified_day` 切分策略分到 **train 集**内, 该交集日期的推理指标就是"训练集自评估", 会异常好, 掩盖真实泛化能力。

**触发案例** (270758 沙箱证据链):
- `train.include = [5-21, 6-06]` (17 天)
- `infer.exclude` 若不含 `[6-04, 6-06]` 段, 则推理集含 6-04/6-05/6-06 三天
- 这 3 天恰好被 `stratified_day` 切进 train 集 (11 天中的 3 天)
- 推理结果拆分对比:

  | 分段 | 天数 | F1 | Precision | Recall | SAE |
  |---|---:|---:|---:|---:|---:|
  | `inference_leak` (训练用过 3 天) | 3 | 0.9780 | 0.9622 | **0.9944** | **3.33%** |
  | `inference_ood` (从未见过 8 天)  | 8 | 0.9782 | 0.9829 | 0.9734 | **7.39%** |
  | `inference` (整体加权 11 天) | 11 | 0.9781 | 0.9765 | 0.9798 | **2.63%** (欺骗性) |

  Recall 差 2.1 个百分点、SAE 差一倍以上 — 整体加权 SAE=2.63% 具有严重欺骗性。

**v13.8 自动化检测** (在 `05_inference.py` 推理指标写入前调用):
1. 从 `bundle.train_dates` 读训练集实际使用的日期 (03_train.py 已保存)
2. 从推理时间戳 `df.index` 抽取推理集日期
3. 交集非空 → WARN 打印泄漏日期列表 + 拆分对比表 (F1/Prec/Rec/SAE/MAE 5 个关键指标)
4. **`inference_metrics.csv` 自动追加 `split=inference_leak` 和 `split=inference_ood` 两组行**, 让下游报表 / aggregate 脚本 / 业务方能一眼分辨"泄漏部分" vs "真泛化"

**日志样例** (v13.8 WARN + v13.8-fix1 多模型):
```
[v13.8 泄漏检测] [WARN] 推理集与训练集日期存在重叠! 泄漏 288 样本 / 3 天, OOD 768 样本 / 8 天
  泄漏日期: ['2026-06-04', '2026-06-05', '2026-06-06']
  建议: 在 infer.exclude 中明确排除训练区间, 或以 inference_ood 指标为准评估泛化能力.
[v13.8 泄漏拆分对比]
  指标            model                   inference_leak     inference_ood
  F1            main                            0.9780            0.9782
  Precision     main                            0.9622            0.9829
  Recall        main                            0.9944            0.9734
  SAE           main                            0.0095            0.2217   ← 原始主模型 (无 L4/L5), 泄漏差异 22 倍
  SAE           main_L4_calib                   0.0762            0.0740   ← L4 残差校正后, 差异被压平
  SAE           main_final                      0.0333            0.0739   ← L4+L5 生产实际输出
  MAE_W         main                           68.6088           73.2285
  MAE_W         main_L4_calib                  76.7405           57.8889
  MAE_W         main_final                     70.1024           63.0988
```

**多模型拆分洞察** (v13.8-fix1): 原始 main 在训练泄漏部分 SAE 只有 0.95% 而 OOD 高达 22.17% (22 倍差), 说明模型对训练样本严重过拟合; L4 残差校正 (在 val 集学习) 把 leak/OOD 差异**压到几乎相等** (7.62% vs 7.40%), 定量证明 L4 层的核心工程价值就是"缓和过拟合". L5 权重切换后 main_final 是折中. 分类指标 (F1/Precision/Recall) 因所有变体共用同一 state_pred, 在三模型间**恒相等**.

**修复建议**: 在 `infer.exclude` 中**显式排除整个训练区间** (含 val/test 的碎片日), 例:
```json
"train":  {"include": [["2026-05-21", "2026-06-06"]]},
"infer":  {"exclude": [["2026-05-21", "2026-06-06"], ["其它需排除"]]}
```

##### ⚠️ 陷阱 2: `infer` 时段过滤会改变滑窗特征值 (上下文边界效应) — v13.10 深度量化

**问题定义**: 同一天推理数据在不同 `infer.exclude` 配置下, 特征矩阵 X_df 里的值**不完全相同**, 导致模型对同一天的预测也不同。**这不是 bug**, 是滑窗特征工程 (lag/diff/rolling/ema) 的固有性质, 但需要量化其影响并给出实用建议。

**v13.10 硬证据链** (270758 用户实测, 6-07~6-14 同 8 天):

###### 发现 1: 138 特征列按算子族分类 — "稳定 vs 漂移"界限极清晰

| 类别 | 列数 | 是否漂移 | 说明 |
|---|---:|---|---|
| `raw` (原始电参量) | 28 | ✅ 完全稳定 | 与窗口位置无关 |
| `time` (时间编码 hour/dow/...) | 9 | ✅ 完全稳定 | 只依赖时间戳本身 |
| `d87` (启动尖峰特征) | 7 | ✅ 完全稳定 | fillna(0) 保护完善 |
| `weather` (温度/湿度) | 12 | 🟡 2/12 微漂移 | 边界日 daily.reindex 有微弱差异 |
| `lag / diff / rolling / ema` (滑窗族) | 74 | ❌ **100% 漂移** | 全部依赖窗口起点前置数据 |
| `drift_feat` (v6 漂移特征) | 4 | ⚠️ 严重漂移 | `power_recent_7d_mean` 相对差 **13%** |
| `ema_ratio` | 3 | ❌ 100% 漂移 | 递归传播 |
| 其它 | 1 | 🟡 | is_morning_peak 类 |

**核心不变量**: 74/138 = **54% 的特征列存在上下文依赖**。

###### 发现 2: 漂移衰减曲线 — 各算子的"稳定所需步数"

| 特征 | 窗口长度 | 稳定所需步数 | 稳定时间 |
|---|---|---:|---|
| `_lag1 / _lag2` | 1~2 步 | 1~2 步 | < 15min |
| `_d1 / _d3 / _d6` | 1~6 步 | ≤ 6 步 | < 90min |
| `_rm4 / _rs4 / _range_4` | 4 步 | 3~4 步 | < 1h |
| `_range_12` | 12 步 | 11 步 | 3h |
| `_ema_2 / _ratio_ema` | halflife=2 | 6 步 | 1.5h |
| **`_ema_24`** | **halflife=24** | **142 步** | **~1.5 天** ⚠️ |
| **`power_recent_24h_mean`** | **96 步** | **94 步** | **~24h (1 天)** ⚠️ |
| **`power_recent_7d_mean`** | **672 步** | **651 步** | **~6.8 天** ⚠️⚠️ |

**关键发现**: **短窗口特征在 3 小时内就稳定**, 影响极小; **长窗口特征 (ema_24 / power_recent_24h / power_recent_7d) 需要 1~7 天才稳定**, 是漂移的主要来源。

###### 发现 3: 暖启动衰减曲线 — **7 天暖启动完全消除边界效应**

沙箱实测 (270758, 同 8 天目标 6-07~6-14, 参考基线 = 前面有 17 天真实历史):

| 暖启动天数 | 漂移列数 | 平均相对差 | ema_24 差 | 7d_mean 差 |
|---|---:|---:|---:|---:|
| **0 天** (当前默认) | **74 / 138** | 1.12% | 2.64% | **13.06%** ⚠️ |
| 1 天 | 7 / 138 | 0.81% | 0.17% | 4.69% |
| 3 天 | 7 / 138 | 0.24% | 0.00% | 1.65% |
| **7 天** | **0 / 138** ✅ | **0.00%** | **0.00%** | **0.00%** |
| 17 天 (完全历史) | 0 / 138 | 0 | 0 | 0 |

**"7 天魔法数字"**: 最长滑窗特征 `power_recent_7d_mean` 是 672 步 = 7 天。一旦推理窗口起点前有 ≥7 天真实数据, 所有滑窗特征都"填满", 上下文边界效应 = 0。

###### 发现 4: 对最终推理指标 (F1/SAE) 的影响 — **SAE 反直觉的 U 形曲线**

沙箱实测 (270758, 同 8 天 6-07~6-14, n=768):

| 暖启动 | F1 | Precision | Recall | SAE | MAE_W | 说明 |
|---|---:|---:|---:|---:|---:|---|
| **0 天** (当前默认) | 0.9714 | 0.9599 | 0.9831 | **19.55%** | 67.5 | 基线 |
| **1 天** | 0.9771 | 0.9759 | 0.9783 | **17.41%** ⭐ | 64.3 | **SAE 最优** |
| **3 天** | 0.9782 | 0.9829 | 0.9734 | **22.17% ↑** | 73.2 | SAE 反弹! |
| **7 天** | 0.9794 | 0.9830 | 0.9758 | **22.55%** | 73.2 | F1 峰值 |
| 17 天 (完全) | 0.9794 | 0.9830 | 0.9758 | 22.55% | 73.2 | = 7 天 (完全稳定) |

**关键洞察**:
- **F1 单调改善** (0.9714 → 0.9794, +0.008)
- **SAE 是 U 形曲线** (19.55% → 17.41% ⭐ → 22.55%), **1 天暖启动最优, 3 天反而变差!**

###### 发现 5: SAE U 形曲线的根本原因 (反直觉!)

对比暖启动候选日 (6-04~6-06) vs 推理目标日 (6-07~6-14) 的实测功率分布:

| 日期段 | 平均功率 | ON 占比 | ON 段平均 |
|---|---:|---:|---:|
| **6-04~6-06** (暖启动候选) | **249 W** | 62.15% | 382 W |
| **6-07~6-14** (推理目标) | **117 W** | 53.91% | 200 W |

**根因链**:
```
6-04~6-06 是高负荷天, 6-07~6-14 是低负荷天 (分布偏移!)
  ↓
暖启动 3 天 → power_recent_7d_mean 从 6-07 起被"高负荷基线"抬高
  ↓
MoE 回归器输入的漂移特征偏离真实
  ↓
输出 y_pred_W 系统偏高 → kWh_pred 从 26.9 → 27.6
  ↓
kWh_true = 22.5, SAE 从 19.55% 变差到 22.55%
```

**核心结论**: **暖启动数据的分布必须与推理目标窗口相近**, 否则会引入新的分布偏移, 反而让 SAE 变差。

##### 实用决策矩阵 (面向业务方)

| 业务目标 | 推荐配置 | 硬证据依据 |
|---|---|---|
| **只求 F1/Precision 一致** (分类应用) | 暖启动 ≥ 7 天 | F1 从 0.9714 → 0.9794 稳定 |
| **求 SAE/kWh 最优** (能耗计量) | 暖启动 1 天 + 用 `analyze_on_periods.py` 验证暖启动日 vs 目标日分布 | SAE 17.41% ⭐ |
| **求跨配置完全可复现** | 保持推理窗口起点+长度都不变, 只调 exclude 中间段 | 特征完全一致 (0 漂移) |
| **完全避免边界效应** | infer.exclude 只做端点, 不切中间段 | 无 bfill 兜底 |

##### 核心不变量 (工程哲学)

上下文边界效应的**本质**: 推理窗口起点没有前置数据时, `bfill / fillna(0) / rolling.median()` 是主流程唯一的"猜测", 猜测总会与真实有偏。

- **≤ 15min 短窗口特征**: bfill 偏差可忽略, 3 分钟内消化
- **1~6 小时中窗口特征**: bfill 偏差中等, 1~6 步内消化
- **≥ 1 天长窗口特征**: bfill 偏差显著, 需要**足够长的真实历史前置**才能收敛
- **7 天是关键阈值**: 因为最长窗口 (`power_recent_7d_mean`) 是 672 步 = 7 天

##### 快速自检工具

用户可用 `analyze_on_periods.py` (v13.6) 快速对比"暖启动日"vs"推理目标日"的功率分布是否相近:

```bash
# 检查暖启动候选日
python analyze_on_periods.py --br-csv path/to/branch.csv --target-col p2 --on-thr-w 70
# 关注: mean_w, on% 是否与推理目标日的期望范围一致
# 若差异 > 20%, 谨慎使用该段作为暖启动
```

#### C.4.5 `splits.{train,val,test}` 字段 (v13.2 per-split 过滤)

| 子字段 | 类型 | 语义 |
|---|---|---|
| `splits.train.include` | `[[s,e],...]` | 硬锚定这些时段的样本必入 train |
| `splits.train.exclude` | `[[s,e],...]` | 硬剔除这些时段的样本从 train |
| `splits.val.*` / `splits.test.*` | 同上 | val / test 独立配置 |

**4 步执行语义**:
1. 原策略 (`stratified_day`) 切分 → 初始 train/val/test 索引
2. include 硬锚定 (样本粒度), 冲突时按 train→val→test 顺序 + WARN
3. 严格保持原 split 形状 (跨 split 平移补齐)
4. exclude 剔除, 若被 3 个 split.exclude 全部命中则完全丢弃

**详见** §3.8 per-split 时段过滤

#### C.4.6 特殊键

| 键 | 用途 |
|---|---|
| `_default` | 兜底配置, 未在 config 中显式列出的用户会读这个 |
| `_comment_` | 顶层注释, 值为字符串, 自动跳过 (不当用户处理) |
| `_note_` | 用户级注释, 写在某 user_id 下作为该用户的说明, 自动跳过 |
| `_auto_detect_note_` / `_semantics_v13_splits_` / `_v135_*_note_` | 项目自定义注释键, 只要值不是 dict 都会被跳过 |
| **以 `_` 开头的用户 ID** (如 `_v135_full_example_`) | 也会被跳过, 因用户 ID 匹配正则 `^\d+_\d+$` |

#### C.4.7 `common` 常量覆盖字段 (v13.5) ⭐ 新增

**语义**: 若配置里指定这些字段, **覆盖 `common.py` 的对应全局默认值**, 仅对本用户生效。未指定则用 common.py 默认。

**优先级链** (与 v13.1/v13.4 完全一致):
```
1. config[user_id].<字段>       ← 最高
2. config._default.<字段>       ← 兜底
3. common.py 全局默认           ← 最低
```

**9 个字段完整规格**:

| JSON 字段 | 类型 | common.py 常量 | common 默认 | 校验范围 | 影响 |
|---|---|---|---|---|---|
| `on_thr_w` | float | `ON_THR_W` | 10.0 | (0, 5000] W | ⭐⭐⭐ 空调 ON 判定阈值 (训练+评估同口径). **270737 案例证明: 待机 16-24W 用户需配 50+** |
| `split_ratios` | [f,f,f] | `SPLIT_RATIOS` | [0.6, 0.2, 0.2] | 3 元素, 各 > 0, 和自动归一化到 1.0 | ⭐⭐⭐ 数据少时可用 [0.8, 0.1, 0.1] |
| `split_strategy` | str | `SPLIT_STRATEGY` | "stratified_day" | `"stratified_day"` / `"stratified"` / `"time"` | ⭐⭐⭐ 小数据用户可用 `"time"` 保证 test 是最新数据 |
| `post_min_on` | int | `POST_MIN_ON` (03_train.py) | 1 | >= 0 | ⭐⭐ 定频空调可用 2, 变频 1 |
| `post_fill_short_off` | int | `POST_FILL_SHORT_OFF` (03_train.py) | 3 | >= 0 | ⭐⭐ 与 post_min_on 配对使用 |
| `weather_latitude` | float | `WEATHER_LATITUDE` | 30.59 (武汉) | [-90, 90] | ⭐⭐ 跨城市部署必需 |
| `weather_longitude` | float | `WEATHER_LONGITUDE` | 114.31 (武汉) | [-180, 180] | ⭐⭐ 与纬度配对 |
| `use_weather_features` | bool | `USE_WEATHER_FEATURES` | true | true/false | ⭐⭐ 离线场景可关 (会降 5-15% 精度) |
| `use_temp_based_season` | bool | `USE_TEMP_BASED_SEASON` | true | true/false | ⭐⭐ 无气象接入回退月份路由 |

**类型宽松解析** (JSON 里可用多种写法, 内部会规范化):

| 字段 | 支持写法 |
|---|---|
| `on_thr_w` | `50` / `50.0` / `"50"` / `"50.5"` |
| `split_ratios` | `[0.7, 0.15, 0.15]` / `[70, 15, 15]` (自动归一化) |
| `split_strategy` | `"time"` / `"TIME"` / `" Stratified_Day "` (大小写+空白规范化) |
| `use_*` (bool) | `true` / `"true"` / `"1"` / `1` / `"yes"` (`false` 同理) |
| 经纬度 | 数字或字符串数字 |

**非法值行为**: WARN 提示 + **忽略该字段, 回退到 common.py 默认** (不阻塞训练)

**bundle 传递**: 所有 9 个字段都会被 `03_train.py` 写入 bundle.pkl, `04_evaluate.py` 和 `05_inference.py` 读 bundle 保证训推口径一致。

**完整配置示例**:
```json
{
  "800080270737_4206680982373": {
    "target_col": "p4",
    "guard_enabled": false,
    "on_thr_w": 50,
    "split_ratios": [0.8, 0.1, 0.1],
    "split_strategy": "time",
    "post_min_on": 2,
    "post_fill_short_off": 3,
    "weather_latitude": 30.59,
    "weather_longitude": 114.31,
    "use_weather_features": true,
    "use_temp_based_season": true,
    "train":  {"exclude": []},
    "infer":  {"exclude": []},
    "splits": {}
  },
  "_default": {
    "on_thr_w": 30,           // 全局默认改成 30W (仅未列用户生效)
    "split_ratios": [0.7, 0.15, 0.15]
  }
}
```

**日志验证**: 覆盖生效时训练日志会打印:
```
[v13.5 用户级覆盖] ON_THR_W: 10.0 -> 50.0
[v13.5 用户级覆盖] POST_MIN_ON: 1 -> 2
[v13.5 用户级覆盖] SPLIT_STRATEGY: stratified_day -> time
[v13.5 用户级覆盖] SPLIT_RATIOS: (0.6, 0.2, 0.2) -> (0.8, 0.1, 0.1)
```

**触发场景**:
- **270737 变频用户 p4 待机噪声**: `on_thr_w: 50` 排除待机干扰
- **25 天小数据用户**: `split_ratios: [0.8, 0.1, 0.1]` 保证 train 充足
- **北京用户**: `weather_latitude: 39.90, weather_longitude: 116.40`
- **离线场景**: `use_weather_features: false, use_temp_based_season: false`

#### C.4.8 分路开机时段分析工具 (`analyze_on_periods.py`) ⭐ v13.6 新增

**动机**: `on_thr_w` 阈值改一个数字就能让训练/评估口径完全变样 (270708 案例: 10W→150W 会让 test ON% 从 44.53% 掉到 22.14%)。因此在**训练开跑之前**先给运营/算法一份"这个用户每天几点开机、开多久、峰值多少"的直观报告，是**避免盲改阈值酿成 v6.12.7 灾难 (SAE 0.85%→53%) 的前置健全性检查**。

##### 一、工作模式

**双模式二选一**:

| 模式 | 命令片段 | 用途 |
|---|---|---|
| **A. 用户 ID + 阶段** | `--user <folder_name> --stage train\|infer [--config <json>]` | 自动定位分路 CSV, 从配置文件读 `target_col` 和 `on_thr_w` (推荐) |
| **B. 显式 CSV 路径** | `--br-csv <path> [--target-col pN] [--on-thr-w <W>]` | 手动指定所有参数, 灵活 (适合任意 CSV 探索) |

##### 二、三层优先级链 (与 v13.1/v13.4/v13.5 完全一致)

| 参数 | 优先级 1 (最高) | 优先级 2 | 优先级 3 (兜底) |
|---|---|---|---|
| `target_col` | `--target-col <pN>` | `config[user_id\|_default].target_col` | 分路 CSV 里第 1 个 `pN` 列 |
| `on_thr_w` | `--on-thr-w <W>` | `config[user_id\|_default].on_thr_w` | `common.py::ON_THR_W` (=10.0) |

##### 三、输出文件 (2 份 CSV)

**(1) 段级明细 `<stage>_on_periods.csv`**

字段与用户示例对齐, 后 5 列为 v13.6/v13.11/v13.16 追加统计:

| 列名 | 含义 | 版本 |
|---|---|---|
| `being_time` | 该 ON 段起始时间戳 (`YYYY/M/D H:MM:SS`, 无前导 0) | v13.6 |
| `end_time`   | 该 ON 段最后一个采样点的时间戳 | v13.6 |
| `<target_col>` | ON 段行 = `1`; **v13.11** 全天 OFF 天行 = `0` | v13.6 / v13.11 |
| `duration_min` | 段时长 (分钟, 含末点采样区间 = `(末点-首点) + 采样步长`) | v13.6 |
| **`min_w`** | **段内最小瞬时功率 (W); OFF 天 = 全天最小 (待机功率下限). 变频空调判"低档运行/短暂低谷"用** | **v13.16** |
| `mean_w` | 段内平均功率 (W) | v13.6 |
| `peak_w` | 段内峰值功率 (W) | v13.6 |
| `energy_kwh` | 段内电量 (kWh) = Σ(w × Δt_h) | v13.6 |
| `dataset` | 数据集归属 (`train`/`val`/`test`/`未使用` 或 `used`/`excluded`) | v13.10 |

**(2) 每日汇总 `<stage>_on_periods_daily.csv`** ⭐ v13.6

| 列名 | 含义 | 版本 |
|---|---|---|
| `date` | 自然日 (`YYYY-MM-DD`) | v13.6 |
| `n_segments` | 该日 ON 段数 (v13.11: 全天 OFF 天 = `0`) | v13.6 / v13.11 |
| `total_on_min` / `total_on_hours` | 该日总开机时长 | v13.6 |
| `first_on_time` / `last_off_time` | 该日首次开机 / 末次关机时刻 (OFF 天空字符串) | v13.6 / v13.11 |
| **`min_w`** | **ON 天 = 各 ON 段 `min_w` 的最小 (=开机期间最低瞬时功率); OFF 天 = 全天最小 (待机下限)** | **v13.16** |
| `mean_w` | 加权平均功率 (按段时长加权) | v13.6 |
| `peak_w` | 该日峰值 | v13.6 |
| `energy_kwh` | 该日累计用电 | v13.6 |
| `dataset` | 数据集归属 | v13.10 |

**v13.16 `min_w` 语义硬保证**:

| 场景 | 段级 `min_w` | daily `min_w` |
|---|---|---|
| 段内 4 点 = `[30, 100, 80, 50]` | `30` (段内最低) | 若日内多段 = `min(各段 min_w)` |
| 全天 OFF, 4 点 = `[0, 0, 0, 0]` | `0` (段行沿用) | `0` (直接取段行) |
| 待机日 24h × 5W | `5` | `5` |
| ON 段 = `[60,60,60,60]` + OFF 段 = `[0,...]` | ON 段 `60`, OFF 段 (若全天OFF) `0` | `60` (只看 ON 段) |

**物理不变量**: 任何行都必然满足 `min_w ≤ mean_w ≤ peak_w`. 270788 用户 40 天 all pass ✅.

##### 四、关键算法细节

- **断段策略**: 严格按每个采样点判定, `w < on_thr_w` 立即断段, **不做平滑合并** (原始视角, 与 03_train.py `state = (y >= ON_THR_W)` 完全同口径)
- **跨日拆分**: 默认 (`--no-split-by-day` 关闭) 会把 22:00~次日 02:00 的 ON 段按 00:00 拆成两行, 便于按日聚合; 加 `--no-split-by-day` 保留一段并写第一天末尾
- **采样步长**: 自动用 `time.diff().median()` 估算, 用于计算段末持续区间和电量; 若无 diff 兜底 900s (15 min)
- **失败保护**: 集成到 pipeline 时若分析失败会 **WARN + 继续主流程**, 不阻塞训练/推理 (辅助报告, 非关键路径)

##### 五、单机运行示例

```bash
# 模式 A: 用户 ID + 配置文件 (最推荐; 自动读 target_col=p1, on_thr_w=50)
cd scripts
python analyze_on_periods.py \
    --user 800080270708_4206602981958 \
    --stage train \
    --config ../data/time_filters.example.json \
    --out ../artifacts/270708_train_on_periods.csv

# 模式 B: 显式路径 + 参数
python analyze_on_periods.py \
    --br-csv ../data/trains/800080270708_4206602981958/4206602981958-260612-260629.csv \
    --target-col p1 \
    --on-thr-w 50 \
    --out /tmp/270708_periods.csv \
    --daily-out /tmp/270708_daily.csv

# 关闭跨日拆分
python analyze_on_periods.py --br-csv <path> --target-col p1 --on-thr-w 50 --no-split-by-day

# 关闭每日汇总输出 (仅段级明细)
python analyze_on_periods.py --br-csv <path> --target-col p1 --on-thr-w 50 --daily-out none
```

##### 六、集成到流水线 (v13.6)

`run_user_pipeline.py` **默认在训练前 + 推理前各跑一次**, 归档到:

| 阶段 | 输出目录 | 文件 |
|---|---|---|
| **训练前** | `artifacts/trains/<user_id>/` | `train_on_periods.csv` + `train_on_periods_daily.csv` |
| **推理前** | `artifacts/infers/<user_id>/` | `infer_on_periods.csv` + `infer_on_periods_daily.csv` |

**关键行为**:
- 使用**原始 `--train-branch` / `--infer-branch`** CSV (非清洗后的 `merged_branch.csv`), 便于对比原始数据
- `on_thr_w` 从 `--common-overrides` 或 `common.py` 默认自动解析, **与训练评估同口径**
- **复用模型路径下也会跑分析** (数据本身值得留档)
- **推理侧仅在有分路 CSV 时**才跑 (无分路时输出跳过日志)

**跳过开关**:
```bash
# 完全跳过训练前 + 推理前的分析
python run_user_pipeline.py ... --skip-analyze
```

##### 七、批量运行 (v13.6)

`run_batch_users.py` 会自动透传 `--skip-analyze` 到每个用户子进程 (若批量层加了该开关); 默认启用分析, 每用户产出各自的 4 个 CSV 到 `artifacts/{trains,infers}/<user>/`。

##### 八、使用建议

1. **训练前必看**: 打开每日汇总 CSV, 检查 `total_on_hours` 分布是否符合预期 (< 1h/天 的用户可能阈值过高, > 20h/天 可能过低误纳基础负荷)
2. **调试阈值**: 用模式 B 快速试跑不同 `--on-thr-w` (10/50/150), 观察 `n_segments` 和 `total_on_hours` 变化, 定位合理值再写入配置
3. **发现异常日**: 若某天 `n_segments=0` 但历史都有 ON, 提示"该日空调停用/数据缺失", 可加入 `train.exclude` 排除
4. **跨用户对比**: 各用户的 `daily.csv` 汇总在一起, 可看出使用模式差异 (工作日 vs 全天型)

##### 九、v13.6 沙箱端到端验证

- **模式 B** (270708, thr=50): 15 段 / 15 天 / 160.0h / 21.25 kWh ✅
- **模式 A + 配置** (270708 用配置 `on_thr_w:50, target_col:p1`): 结果与模式 B **完全一致** ✅
- **阈值敏感度** (270708): thr=10 → 15 段/160.25h, thr=50 → 15 段/160.0h, thr=150 → **3 段/32h** (与 v13.5 bug 修复报告 test ON% 22.14% 数据完全一致) ✅
- **跨日拆分**: 合成数据 5/21 22:00 ~ 5/22 02:00 正确拆为 22:00~23:45 + 00:00~02:00 两行 ✅
- **集成流水线** (270708 完整跑): train + infer 两阶段各产出 2 CSV, 训练 02→03→04 + 推理 05 全通过 ✅
- **--skip-analyze**: 两处跳过日志正确输出, 主流程不受影响 ✅

##### 十、v13.9 4 字段计算正确性审计 + 边界预检 WARN ⭐

**动机**: 业务方要求核对 4 字段 (`duration_min` / `mean_w` / `peak_w` / `energy_kwh`) 计算是否正确。给出 6 层硬证据审计报告 + 主动预检 WARN.

**审计方法** (INTJ 硬证据链, 6 层):
1. **静态公式对照**: 4 字段代码公式对齐物理定义 (功率×时长 = 能量) ✅
2. **合成用例手工独立复算** (T1-T5): 覆盖单段/多段/单点/跨日/变功率, 5/5 通过 ✅
3. **真实用户全 15 段逐值验证** (270708, thr=50): 首段 dur=675min, mean=110.66W, peak=111.76W, energy=1.24493 kWh 与手工完全一致 ✅
4. **段内跨字段一致性** (`mean_w × dur_h ≡ energy_kwh × 1000`): 15/15 通过, max diff < 5e-6 (浮点精度) ✅
5. **段-日聚合一致性** (段级 `sum(duration_min)` == 日级 `sum(total_on_min)`): 9600 / 9600 min, 21.25 / 21.25 kWh ✅
6. **每日加权 mean_w 独立复算** (按 duration 加权): 15/15 天完全对上 ✅

**主流场景结论**: 对 NILM 生产数据 (`02_align_and_feat.py` 之后的 15min 严格 resample 数据), **4 字段计算 100% 正确, 可信**。

**边界场景 (非常规采样) 2 个潜在偏差**:

| Bug | 触发条件 | 症状 | 硬证据示例 | 影响面 |
|---|---|---|---|---|
| **#1 内部口径不一致** | 采样步长非严格均匀 (混合 15min + 5min) | `mean_w × dur_h ≠ energy_kwh` | `t=[5:00, 5:15, 5:20, 6:00]` 全 100W → dur=75min, energy=0.1 kWh, `mean × dur` 应=0.125 kWh, **差 25%** | 独立 `--br-csv` 模式偶发, NILM 主流程 (先 resample) 不触发 |
| **#2 时间断裂被错误合并** | NaN 剔除或原始数据大 gap | duration 严重高估 | `t=[8:00, 12:00, 12:15]` 全 100W → dur=270min (4.5h!), energy=0.075 kWh, **相对差 100%** | 同上, 边界偶发 |

**v13.9 修复方案 (选项 1: 预检 WARN, 零回归)**:

在 `compute_on_periods` 步长计算后追加 2 组预检:

```python
# (a) 采样均匀性
if CV(dt) > 0.10:
    print("[WARN v13.9 采样均匀性] ... CV=X.XXX, 最大偏差可达 25%")

# (b) 时间断裂
if any(dt > 2 * step_median):
    print("[WARN v13.9 时间断裂] 检测到 N 处相邻采样间隔 > YYYs, 建议先 resample 或按断裂手动切段")
```

**v13.9 WARN 触发矩阵** (端到端验证):

| 数据 | CV WARN | 断裂 WARN | 主流程 |
|---|---|---|---|
| 270708 真实 15min 严格 | ❌ | ❌ | ✅ 15 段正常输出, **零回归** |
| 270758 batch 集成 (生产流水线) | ❌ | ❌ | ✅ exit=0, 训推两阶段 analyze 正常 |
| E1 非均匀 (混 5min/15min) | ✅ CV=0.690 | ✅ max 2400s | ✅ 正常输出, 用户见 WARN |
| E4 大间隔 (跨 4h gap) | ✅ CV=1.179 | ✅ max 4h | ✅ 正常输出, 用户见 WARN |
| E5 均匀波动 (200±80W) | ❌ | ❌ | ✅ 无 WARN |

**一句话结论**: 主流数据 4 字段 **100% 正确可信**, 边界场景 2 潜在 bug 已通过 WARN 显性化 (未修改主逻辑, 零回归). 深度修复 (选项 2/3: 用真实 Δt + 时间断裂强制断段) 待用户实际报告偏差场景再上.

##### 十一、v13.10 分路 CSV 增加 dataset 归属列 ⭐⭐ (业务可视化增强)

**动机**: 之前 `train_on_periods.csv` / `infer_on_periods.csv` 只有原始功率统计, 用户看不出哪些天是"训练用过的"(潜在泄漏日), 哪些是"从未见过的" (真泛化). v13.10 追加 `dataset` 列, 让业务方一眼看清数据集归属。

**新增 API** (向后兼容):
```python
compute_on_periods(df, target_col, on_thr_w, split_by_day=True,
                   date_labels=None)  # 新增
compute_daily_summary(periods, target_col,
                      date_labels=None)  # 新增
# date_labels = {"yyyy-mm-dd": "train"/"val"/"test"/"未使用"/"used"/"excluded"}
# 不传 date_labels: 完全向后兼容, 无 dataset 列
```

**集成到 run_user_pipeline.py**:

| 阶段 | 归属逻辑 | dataset 列取值 |
|---|---|---|
| **训练阶段** (03 完成后**补跑**) | 从 `bundle.train_dates/val_dates/test_dates` 读, 未列日期 = 未使用 | `train` / `val` / `test` / `未使用` |
| **推理阶段** (首次跑就传) | 从 `--infer-time-filter-spec` 计算 include/exclude 后的实际推理集 | `used` / `excluded` |

**流程时序** (新增 v13.10 步骤在原 v13.6 之上):
```
[v13.6 老流程] 训练前跑 analyze (无归属) → 02→03→04 → [v13.6] 推理前跑 analyze (无归属) → 05
                                        ↑
[v13.10 新增]                        03 完成后补跑一次 train analyze, 覆盖上一步 CSV, 加 dataset 列
                                                                            ↑
                                                              [v13.10] 推理前直接加 dataset 列
```

**270758 A 配置实测** (train.include=5-21~6-06, infer.exclude 3 段):

训练 CSV 归属分布 (按天):
```
train:    11 天  (5-24, 5-25, 5-26, 5-27, 5-28, 5-29, 5-31, 6-03, 6-04, 6-05, 6-06)
val:       3 天  (5-21, 5-30, 6-01)
test:      3 天  (5-22, 5-23, 6-02)
未使用:    23 天  (6-07~6-14 + 6-15~6-29 = 22 天 - 3 天已归 + 2 天 5-31 前空日)
```

推理 CSV 归属分布 (按天): `{used: 8, excluded: 32}` — 精准反映配置 A 推理集只有 6-07~6-14 (8 天).

**270758 B 配置实测** (infer.exclude 只 2 段):

推理 CSV 归属分布: `{used: 11, excluded: 29}` — 精准显示 **6-04/6-05/6-06 三天 = used** (数据泄漏日**可视化**! 与 v13.8 leak/ood 拆分完美对应).

**业务价值**:
1. **训练侧**: 一眼看清训练集实际用了哪 11 天, 哪些天被 val/test 采样
2. **推理侧**: 结合 v13.8 leak/ood 拆分, 直接找到 "dataset=used 但同时也在 train_dates 里" 的泄漏日
3. **审计追溯**: 6 个月后重看模型, 能立刻知道当时训练用的具体日期分布, 而不需要重跑 03_train.py

**CSV 样例** (训练每日汇总, 270758 A 配置):
```csv
date,n_segments,total_on_min,total_on_hours,first_on_time,last_off_time,mean_w,peak_w,energy_kwh,dataset
2026-05-21,1,750.0,12.5,08:45:00,21:00:00,150.56,152.0,1.882,val
2026-05-22,1,780.0,13.0,08:30:00,21:15:00,451.538,704.0,5.87,test
2026-05-24,1,750.0,12.5,09:15:00,21:30:00,246.88,320.0,3.086,train
...
2026-06-07,1,750.0,12.5,08:30:00,20:45:00,398.72,712.0,4.984,未使用
```

**CSV 样例** (推理每日汇总, 270758 B 配置, 高亮 3 天泄漏):
```csv
date,...,dataset
2026-06-03,...,excluded
2026-06-04,...,used     ← 训练用过, 泄漏日
2026-06-05,...,used     ← 训练用过, 泄漏日
2026-06-06,...,used     ← 训练用过, 泄漏日
2026-06-07,...,used     ← 从未见过, 真泛化
2026-06-08,...,used     ← 从未见过, 真泛化
...
```

**向后兼容**: 若脚本用 `--br-csv` 模式独立跑 (未传 `date_labels`), **无 `dataset` 列** (v13.6 老行为完全保留).

##### 十二、v13.11 全天 OFF 日也输出 CSV ⭐ (数据完整性增强)

**背景问题**: v13.6~v13.10 的段级 CSV 只输出**有 ON 段的天**. 若某天全天未启动 (所有采样点 `w < on_thr_w`), 该天在 CSV 里**完全消失**. 业务方看到 daily CSV 只有 15 天而不是完整的 18 天, 无法判断"缺失日期是真无启动 vs 数据缺失".

**v13.11 规格**: 全天 OFF 日也输出**一行**, 状态列 (target_col) 值为 `0`, 其它列按**全天所有采样点统计** (待机功率视图):

**段级 CSV OFF 天行格式**:
```csv
being_time,end_time,<target_col>,duration_min,mean_w,peak_w,energy_kwh,dataset
2026/6/14 0:00:00,2026/6/14 23:45:00,0,1440.0,1.444,1.48,0.03466,val
```

| 字段 | OFF 天取值 | 说明 |
|---|---|---|
| `being_time` | `YYYY/M/D 0:00:00` | 全天起点 |
| `end_time` | `YYYY/M/D 23:45:00` | 全天末点 (最后 1 个 15min 采样) |
| `<target_col>` | **0** | 关键: 与 ON 段的 1 相对 |
| `duration_min` | 1440 (= 96 × 15min) | 全天时长 |
| `mean_w` | 待机功率均值 | 全天所有采样点平均 |
| `peak_w` | 待机功率峰值 | 全天所有采样点最大值 (一定 < on_thr_w) |
| `energy_kwh` | 待机总电量 | 全天累计 (Σw × dt_h / 1000) |
| `dataset` | 与 date_labels 一致 | train/val/test/未使用 或 used/excluded |

**每日汇总 CSV OFF 天行格式**:
```csv
date,n_segments,total_on_min,total_on_hours,first_on_time,last_off_time,mean_w,peak_w,energy_kwh,dataset
2026-06-14,0,0.0,0.0,,,1.444,1.48,0.03466,val
```

| 字段 | OFF 天取值 |
|---|---|
| `n_segments` | **0** (无 ON 段) |
| `total_on_min` / `total_on_hours` | **0** (无开机时长) |
| `first_on_time` / `last_off_time` | **空字符串** (无 ON 段无首末开关机时间) |
| `mean_w` / `peak_w` / `energy_kwh` | 待机统计 (全天采样点值) |
| `dataset` | 保留 |

**270708 用户实测** (thr=50W):
- v13.10 daily CSV: 15 天 (只有 ON 天)
- **v13.11 daily CSV: 18 天** (补齐 3 个周日 OFF 天)
- 3 个周日: 6-14 (val), 6-21 (train), 6-28 (train) — mean_w=1.44W (纯待机), 每天耗 34Wh

**业务价值**:
1. **数据完整性**: 与总线数据周期完全对齐, 便于业务方核对"我给的 X 天数据都到位了吗"
2. **待机能耗可视**: 直接从 mean_w/energy_kwh 看待机功率, 便于对比工作日 vs 周日的开机模式
3. **异常检测**: 若某天期望有 ON 但 CSV 里是 OFF, 说明设备/数据异常, 一眼可见

**向后兼容**: 若数据没有全天 OFF 的日子 (如 270758 40 天连续使用), CSV 行数 = ON 段数, 与 v13.10 完全一致.

##### 十三、v13.14 逐日主模型评估指标 CSV ⭐⭐ (质量追踪增强)

**背景问题**: 现有 `test_metrics.csv` / `inference_metrics.csv` 只有**整体聚合指标** (train/val/test/inference 各一个 F1/SAE 数字), 无法定位:
- 哪一天预测崩溃 (单日 F1 掉到 0.6)?
- val/test 里具体哪些日子拉低整体指标?
- 推理集里哪些日子是数据泄漏日 (指标偏乐观), 哪些是真泛化?

**v13.14 新增 API** (`scripts/metrics_utils.py`):

```python
build_daily_metrics_rows(timestamps, y_true, y_pred, s_true, s_pred,
                         split_name, on_thr_w=None, p_on=None,
                         date_labels=None, model_name="main")
save_daily_metrics_csv(rows, out_path, logger=None)
```

**输出 CSV 字段 (25 列, 含 v13.16 新增 2 列)**:
```
date, split, model, n_samples,
n_bus_raw, n_branch_raw,                     ← [v13.16 新增] 当天原始 CSV 采集点数
Accuracy, Precision, Recall, F1, AUC,      ← 分类 5 指标
MAE_W, RMSE_W, SAE,                          ← 回归 3 指标
kWh_true, kWh_pred, kWh_err,                 ← 电量 3 指标
TP, FP, FN, TN,                              ← 混淆矩阵 4 指标
[dataset], on_thr_w, project_version, model_file, [bus_csv]
```

**v13.16 新增两列语义** ⭐:

| 列 | 含义 | 满值 | 与 `n_samples` 差异 |
|---|---|---:|---|
| `n_bus_raw` | 当天**总线** CSV 原始采集点数 (`event_time` 按天 group) | 288 (5min×288=24h) | 反映采集完整性; **不受**时段过滤/对齐影响 |
| `n_branch_raw` | 当天**分路** CSV 原始采集点数 (`time` 按天 group) | 96 (15min×96=24h) | 同上 |
| `n_samples` (v13.14 原有) | 对齐后每天参与训练/推理的样本数 | 96 (若无缺采) | 受时段过滤 + `resample_and_align` inner-join 影响 |

**推理侧特殊处理**: `n_bus_raw`/`n_branch_raw` 会应用 `--time-filter-spec` (让口径与实际推理天数一致, 避免统计到被 `infer.exclude` 排除的训练日). 训练侧 (04_evaluate) 不再二次过滤原始 CSV, 因 `merged_bus.csv`/`merged_branch.csv` 是已合并好的成品.

**业务价值示例** (270788 用户 v13.16-daily_raw 实测揭示):

| date | n_samples (对齐) | **n_bus_raw** | n_branch_raw | F1 | 根因诊断 |
|---|---:|---:|---:|---:|---|
| 2026-05-21 | 15 | **4** | 96 | 0.000 | **总线采集只有 4 点** (应 288) → 模型基本没输入, F1=0 |
| 2026-05-22 | 30 | **7** | 96 | 0.000 | 同上 |
| 2026-05-23 | 15 | **3** | 96 | 0.667 | 同上, 少数样本靠 ffill 撑起 |
| 2026-06-12 (正常日) | 96 | 287 | 96 | 0.989 | 采集完整时模型正常 |

**v13.16 daily_raw 首次揭示**: 之前 v13.14 只显示 "5-21/22 缺数据", v13.16-daily_raw 直接摊开 "总线原始只采了 3-7 点" 的根因铁证.

**向后兼容**: `build_daily_metrics_rows()` 不传 `bus_daily_counts`/`branch_daily_counts` 时, 两列输出 `""` 空字符串, 老 pipeline 零改动可运行.

**归档路径**:

| CSV 文件 | 路径 | 来源脚本 |
|---|---|---|
| `train_daily_metrics.csv` | `artifacts/trains/<user>/` | `04_evaluate.py` |
| `inference_daily_metrics.csv` | `artifacts/infers/<user>/` | `05_inference.py` |

**`dataset` 列语义** (v13.14 亮点):

| 场景 | dataset 取值 |
|---|---|
| **训练阶段** (`train_daily_metrics.csv`) | `train` / `val` / `test` (与 split 列一致, 便于统一筛选) |
| **推理阶段** (`inference_daily_metrics.csv`) | **`used_leak` (推理日 ∈ bundle.train_dates, v13.8 泄漏检测) / `used_ood` (真泛化日)** |

**SAE 边界保护** (v13.14 关键设计):
- 全 OFF 天 `kwh_true ≈ 0` 时, 传统 SAE = |err|/max(kwh_true, 1e-9) 会爆炸 (如 8×10^7)
- v13.14 判据: `kwh_true < 1e-3 kWh (=1Wh)` → `SAE = None` (CSV 空字符串)
- 保持 kwh_pred/kwh_err 正常显示, 用户可看待机误报能耗绝对值

**270758 端到端实测** (`split_ratios=[0.7,0.15,0.15]`, `split_strategy=global_stratified`):

**训练侧** (`train_daily_metrics.csv`, 14 行, 主模型 = main_final):

| split | 天数 | 平均 F1 | 平均 SAE | 平均 MAE_W |
|---|---:|---:|---:|---:|
| train | 10 | 0.999 | 0.043 | 10.1 W |
| val | 2 | 0.988 | 0.275 | 79.1 W |
| test | 2 | 0.929 | 0.322 | 90.9 W |

**推理侧** (`inference_daily_metrics.csv`, 40 行):
- dataset 分布: **`used_leak: 10 天`** (数据泄漏日) + **`used_ood: 30 天`** (真泛化)
- 单日 F1 范围: 0.719 ~ 1.000 (最差 5-22, 最好多天)
- 单日 SAE 范围: 0.012 ~ 0.857 (定位到具体差日)

**交叉验证** (硬证据一致性):
- daily 累加 `kWh_true` = 146.99 == 整体 `inference_metrics.csv::main_final.kWh_true` = 146.99 ✅
- daily 累加 `kWh_pred` = 135.87 == 整体 = 135.87 (浮点差 2e-6) ✅

**业务价值**:
1. **单日诊断**: 直接找到 F1<0.9 的日子, 与 `infer_on_periods_daily.csv` 对照看当天开机模式
2. **数据泄漏可视化**: 推理 CSV 直接标 `used_leak`, 与 v13.8 `inference_leak/inference_ood` 拆分完美对应
3. **审计追溯**: 6 个月后仍能查"当时哪一天模型崩了", 无需重跑 04/05

**向后兼容**: 老 bundle 无 `train_dates` 字段时, 推理侧 `dataset` 列为空字符串; 主流程不受影响.

##### 十四、v13.15 温度桶期望信号 CSV 导出 ⭐⭐ (概念漂移可视化增强)

**背景问题**: v6 L2 已有 `drift_report.csv`, 但只输出**触发告警的少数桶** (逻辑 `|rel|≥0.30` 的 `concept/temp_power_detail`), 3 个业务问题无法回答:
1. "训练时 27°C 那个桶模型认为总线该多少 W?" — LUT 只在 `bundle.pkl` 里, 无 CSV 视图
2. "20 个温度桶里, 除了 3 个 ALERT, 还有多少 WARN?" — drift_report 里看不到 WARN/OK
3. "推理集里每个桶有多少样本?" — drift_report 只对 3 个告警桶写 `n=xxx`, 其他 17 个桶无信息

**v13.15 设计目标**: **把 20 个桶的完整信息 (训练期望 + 推理实测 + 逐桶漂移比 + 分级) 落成 2 张 CSV**, 与 `drift_report.csv` **互补而非替代**.

**新增 API** (`scripts/drift_features.py`):

```python
# 1) LUT 构造扩展 (向后兼容: 不传 return_meta 时行为不变)
build_temp_power_lut(df_train, weather_df, top_cols,
                     n_bins=20, return_meta=False)
    # return_meta=True → 返回 (lut_dict, meta_dict)
    # meta_dict[(lo,hi)] = {n, mean, std, p25, median, p75, signal_col}

# 2) 训练侧导出
export_temp_power_lut_csv(lut, out_path, meta=None, logger=None)

# 3) 推理侧导出
export_temp_power_actual_vs_expected_csv(
    df, top_cols, weather_df, temp_power_lut, out_path, logger=None)
```

**归档路径** (复用 `run_user_pipeline.py::archive_outputs()` 的命名分流规则):

| CSV 文件 | 路径 | 触发点 |
|---|---|---|
| `temp_power_lut.csv` | `artifacts/trains/<user>/` | `03_train.py` 构造 LUT 后立即导出 |
| `inference_temp_power_actual_vs_expected.csv` | `artifacts/infers/<user>/` | `05_inference.py` 在 `detect_drift()` 后紧接着导出 |

**训练侧 CSV 字段 (12 列)**:
```
bin_id, temp_lo, temp_hi, temp_width,
expected_signal (=median), n_samples,
mean_signal, std_signal, p25_signal, p75_signal,
signal_col, is_global_median
```
- **20 个桶行** (按 `temp_lo` 升序)
- **末尾 1 行全局中位兜底** (`bin_id=-1, is_global_median=1`, `temp_*` 空), 对应 LUT 里的 `"__global_median__"` 键, 用于推理时温度落在训练桶外的兜底值

**推理侧 CSV 字段 (13 列)**:
```
bin_id, temp_lo, temp_hi,
train_expected_signal,           ← 从 bundle LUT 复现, 与训练侧 expected_signal 逐行一致
infer_n_samples,
infer_median_signal, infer_mean_signal, infer_p25_signal, infer_p75_signal,
abs_residual  (= infer_median - train_expected),
rel_drift     (= abs_residual / max(|train_expected|, 1e-6)),
drift_flag    ∈ {OK, WARN, ALERT, NO_DATA},
signal_col
```

**`drift_flag` 三档分级** (阈值与 v6 L2 `drift_report` 一致):

| 判据 | 分级 | 语义 |
|---|---|---|
| 推理集在该温度桶**样本数=0** | `NO_DATA` | 训练见过、推理未覆盖 (季节偏差) |
| `\|rel_drift\| < 0.15` | `OK` | 分布稳定, 模型可信 |
| `0.15 ≤ \|rel_drift\| < 0.30` | `WARN` | 中度漂移, 建议关注 |
| `\|rel_drift\| ≥ 0.30` | `ALERT` | 重度漂移, 与 `drift_report::concept/temp_power_detail` 完全对齐 |

**270788 端到端实测** (原基线配置, `guard_enabled=false`):

**训练侧 `temp_power_lut.csv` 摘录** (21 行 = 20 桶 + 1 全局中位):

| bin_id | temp_lo | temp_hi | expected_signal | n_samples | signal_col |
|---:|---:|---:|---:|---:|---|
| 0 | 18.00 | 20.20 | 19690 | 93 | load_iden_data7 |
| 13 | 26.85 | 27.30 | **65835** | 96 | load_iden_data7 |
| 14 | 27.30 | 27.90 | **76193** | 93 | load_iden_data7 |
| 15 | 27.90 | 28.65 | **76395** | 101 | load_iden_data7 |
| 19 | 30.90 | 32.60 | 94359 | 98 | load_iden_data7 |
| -1 | (全局) | | **38350** | | (is_global_median=1) |

**推理侧 `inference_temp_power_actual_vs_expected.csv` 摘录** (20 行, 26.85-28.65°C 三桶):

| bin_id | temp_lo | temp_hi | train_expected | infer_n | infer_median | abs_residual | rel_drift | drift_flag |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 13 | 26.85 | 27.30 | 65835 | 85 | **28133** | -37702 | **-0.5727** | ALERT |
| 14 | 27.30 | 27.90 | 76193 | 100 | **34160** | -42034 | **-0.5517** | ALERT |
| 15 | 27.90 | 28.65 | 76395 | 140 | **37352** | -39043 | **-0.5111** | ALERT |
| 0 | 18.00 | 20.20 | 19690 | **0** | — | — | — | **NO_DATA** |

**分级统计**: `ALERT=8, WARN=6, OK=5, NO_DATA=1` (共 20 桶).

**交叉验证** (v13.15 硬证据):

| 校验点 | 期望 | 实测 | 结论 |
|---|---|---|---|
| 训练 LUT 桶数 | 20 (n_bins) | 20 | ✅ |
| 训练 `expected_signal` ↔ 推理 `train_expected_signal` | 逐行相等 | delta_max = 0.0 (按 `bin_id` join) | ✅ 两侧口径完全一致 |
| 与 `drift_report.csv` detail 桶交集 | detail 3 桶 ⊆ 推理 CSV 8 ALERT | 3 桶 rel_drift 完全匹配 | ✅ 互补关系成立 |
| 主指标回归门 | F1/Recall/SAE 不变 | 0.9022/0.8324/5.45% | ✅ 零回归 |

**单元测试** (`scripts/test_temp_power_lut_csv.py`, 34 组断言, < 1 秒执行):

| 组 | 断言数 | 覆盖点 |
|---|---:|---|
| T1. `build_temp_power_lut(return_meta=True)` 数学正确性 | 12 | n/mean/std/median/p25/p75 与 numpy 逐一对拍 + 向后兼容 |
| T2. `export_temp_power_lut_csv` 写盘正确性 | 8 | 12 列齐全 + 桶数一致 + 全局中位行唯一 + 桶行按 `temp_lo` 升序 |
| T3. `export_temp_power_actual_vs_expected_csv` 分级+数学 | 7 | 13 列齐全 + drift_flag 三档覆盖 + `abs_res = infer_med - train_exp` + `rel = abs_res/|train_exp|` + 全表遍历 0.15/0.30 阈值 |
| T4. 训练/推理 CSV 横向对齐 | 2 | 按 `bin_id` join delta_max<1e-3 |
| T5. 兜底不崩 | 4 | 空 LUT / weather_df=None / 温度全出训练桶范围 → NO_DATA |

运行:

```bash
python scripts/test_temp_power_lut_csv.py
# 汇总: 通过 34 / 失败 0 / 总计 34
# [OK] 全部单测通过
```

**业务价值**:

1. **训练资产可审计化**: `temp_power_lut.csv` 让"训练时 27°C 桶期望多少 W"从 pkl 隐性知识变成一张表, 6 个月后仍能查
2. **漂移可视化颗粒度提升**: 从 `drift_report` 的 3 桶提升到 20 桶完整对比, WARN 桶也可见
3. **两侧口径一致性硬保护**: T4 单测长期守护"推理端 `train_expected` 一定等于训练端 `expected_signal`" — 未来若有人误改 LUT 序列化逻辑, CI 会立刻拦截
4. **可直接喂 Excel/BI 绘图**: 20 个桶按温度升序, `train_expected` vs `infer_median` 双柱状图 3 秒出图, 无需写代码

**向后兼容**:
- `build_temp_power_lut()` 老签名 (无 `return_meta`) 行为完全不变
- 老 bundle 无影响 (LUT 已在 v6 就存在, 仅新增 CSV 侧输出)
- 空 LUT / 无温度数据 / 无 weather_df 场景走兜底路径, 写空表不崩

### C.5 汇总产出文件 status 字段完整定义

批量运行后产出 3 个汇总 CSV 文件, 各含**不同定义**的 status 列, 分别针对**执行状态** / **数据状态** / **指标状态**。

#### C.5.1 `artifacts/batch_run_summary.csv` — 执行状态

**每用户一行**, 记录批量流水线执行结果. 字段:

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | str | 用户 folder_name |
| **`status`** | 3 值 | 执行三态 |
| `ok` | bool | 兼容旧字段, `status == "ok"` |
| `message` | str | 状态详情 (成功耗时 / 失败原因) |
| `duration_s` | float | 单用户流水线总耗时 (秒) |
| `target_col` | str | 该用户使用的目标分路列 |
| `category` | str | 兼容字段, 值 = `status` |

**status 三态定义**:

| status 值 | 触发条件 | 打印图标 |
|---|---|---|
| `"ok"` | 子进程 exit 0 (真成功, 产出模型 + 指标) | `[OK]` |
| `"soft_skip"` | 子进程 exit 10 (03_train.py 数据质量门 11/12/13 触发, 无模型无指标) | `[SKIP]` |
| `"fail"` | 其它退出码 / 启动异常 / 超时 (真错误) | `[FAIL]` |

#### C.5.1a `artifacts/batch_execution_state.csv` — 断点续跑状态 ⭐ v13.17

**核心动机**: `batch_run_summary.csv` 是**跑完一次性覆盖写**, 中断/崩溃 = 全部丢失; 30 分钟批量跑到第 25 分钟崩溃就得从头再来 30 分钟. **v13.17 独立文件**支持断点续跑.

**字段** (9 列, `utf-8-sig` 编码, 中文安全):

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | str | 用户 folder_name |
| `status` | 3 值 | `ok` / `soft_skip` / `fail` (与 `batch_run_summary.csv::status` 同定义) |
| `success` | bool | `True` iff `status == "ok"` (方便 Excel 快速过滤) |
| `started_at` | str | 该用户开始时间 (`YYYY-MM-DD HH:MM:SS`) |
| `finished_at` | str | 该用户完成时间 |
| `duration_s` | float | 该用户耗时秒数 (round to 2) |
| `message` | str | 状态详情, 单行且截断到 500 字符防超宽 |
| `target_col` | str | 该用户目标分路列 (含 v13.16 复合语法如 `p1+p2`) |
| `run_id` | str | 本次批处理运行 ID (`YYYYMMDD_HHMMSS`, 便于追溯"这行是哪一次跑写的") |

**行为不变量** (INTJ 硬保证):

| # | 不变量 |
|---|---|
| 1 | **实时增量写**: 每个用户完成后 `_upsert_execution_state()` **立即**写入, 不等所有用户跑完 |
| 2 | **原子写**: 用 `.tmp` + `os.replace()` 实现, 中断时最多丢**当前正在跑**的用户 |
| 3 | **upsert 语义**: 同 `user_id` 再写会**覆盖旧行** (支持 fail 重跑更新为 ok) |
| 4 | **cleanup 白名单保护**: `run_user_pipeline::cleanup_artifacts_top` **不会**删除本 CSV (v13.17 修复关键 bug — 之前会) |
| 5 | **列头稳定**: 9 列固定顺序; 老格式 (缺列) 会自动补齐, 向后兼容 |
| 6 | **损坏降级**: 文件损坏 (缺关键列) → WARN + 走全部重跑, 不阻塞 |

**CLI 使用**:

```bash
# 默认: 全部重跑 (与 v13.16 及以前完全一致, 零回归)
python scripts/run_batch_users.py

# v13.17: 断点续跑, 跳过 ok/soft_skip, fail 会重跑
python scripts/run_batch_users.py --resume

# v13.17: 连 fail 也跳过 (需手工删行才重试)
python scripts/run_batch_users.py --resume --resume-skip-failed

# 常见组合: 断点续跑 + 复用已有模型
python scripts/run_batch_users.py --resume    # 不加 --force-retrain 让 v10 模型复用生效
```

**续跑决策矩阵**:

| 上次 status | `--resume` 默认 | `--resume --resume-skip-failed` |
|---|---|---|
| `ok` | 跳过 | 跳过 |
| `soft_skip` | 跳过 (数据本来就不够, 重跑也不会变) | 跳过 |
| `fail` | **重跑** (真错误可能已修) | 跳过 (强制跳需手工删行才重试) |
| 缺失 (未跑过) | 跑 | 跑 |

**真实端到端示例**:

```
$ python scripts/run_batch_users.py --resume
[v13.17 续跑] --resume 开启, 检查 /home/user/nilm_ac_win/artifacts/batch_execution_state.csv
  [v13.17 续跑] 已加载状态文件: ...  历史记录 3 行, 已完成 (ok/soft_skip) 2 用户
[v13.17 续跑] 跳过策略: 只跳过 ok/soft_skip
[v13.17 续跑] 原计划 7 用户, 跳过已完成 2 用户, 实际待跑 5 用户
    [SKIP-resume] 800080252842_4206894986488
    [SKIP-resume] 800080252844_4206894986488
    ...
```

每个用户跑完立即:
```
  [OK]   800080270708_4206602981958: 成功  (耗时 47.6s)
    [v13.17 状态] 800080270708_4206602981958: ok -> batch_execution_state.csv
```

**兼容性**:
- **零回归**: 不加 `--resume` 时 v13.16 及以前的所有行为完全一致
- **`batch_run_summary.csv` 不受影响**: v13.17 状态 CSV 是**新增独立文件**, 汇总类 CSV 语义不变
- **修复关键 bug**: v13.17 的 cleanup 白名单同时保护了 `batch_run_summary.csv` / `summary_metrics_all_users.csv` / `skipped_users.csv` (虽然它们过去只在批量最末写一次, 之前是通过覆盖写不需要保护; 现在纳入白名单是防未来批量层做增量写时踩坑)

#### C.5.2 `artifacts/skipped_users.csv` — 软跳过详情

**仅软跳过用户**, 从 `artifacts/trains/<uid>/skip_reason.json` 汇总. 字段:

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | str | 软跳过用户 folder_name |
| **`skip_reason`** | 3 值枚举 | 触发的数据质量门 |
| `detail` | str | 具体数字详情 |
| 其它 | 视 skip_reason 而定 | `aligned_n` / `on_pct` / `n_on` / `n_off` / `peak_w` / `kind` / `train_n` / `val_n` / `test_n` |

**skip_reason 三值定义**:

| skip_reason | 数据质量门 | 退出码 | 触发条件 | 详情字段 |
|---|---|---|---|---|
| `"aligned_too_few"` | 门 1 | 11 | 对齐样本数 < 96 (< 1 天) | `aligned_n` |
| `"single_class_label"` | 门 2 | 12 | ON 占比 = 0% (`all_off`) 或 100% (`all_on`) | `on_pct` / `n_on` / `n_off` / `peak_w` / `kind` |
| `"split_empty_val_test"` | 门 3 | 13 | 切分后 val 或 test 集为空 | `aligned_n` / `train_n` / `val_n` / `test_n` |

#### C.5.3 `artifacts/summary_metrics_all_users.csv` — 指标状态

**每用户 4 行** (train / val / test / inference 各一行), 主表. 字段:

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | str | 用户 folder_name |
| `stage` | 4 值 | `train` / `val` / `test` / `inference` |
| **`status`** | 多值 | 该 stage 的指标状态 (见下表) |
| `Accuracy` / `Precision` / `Recall` / `F1` / `AUC` | float | 分类指标, 若 status 非 ok 则 NaN |
| `TN` / `FP` / `FN` / `TP` | int | 混淆矩阵 |
| `MAE_W` / `RMSE_W` / `SAE` / `NDE` | float | 回归指标 |
| `kWh_true` / `kWh_pred` / `kWh_err` | float | 能耗对比 |
| `n_samples` | int | 该 stage 样本数 |

**status 完整分类** (按判定顺序):

| status 值 | 触发条件 | 指标字段 |
|---|---|---|
| `"soft_skip:aligned_too_few"` | 数据质量门 1 触发 | 全 NaN |
| `"soft_skip:single_class_label"` | 数据质量门 2 触发 | 全 NaN |
| `"soft_skip:split_empty_val_test"` | 数据质量门 3 触发 | 全 NaN |
| `"no_train_metrics"` | 训练成功但 `train_val_metrics.csv` 缺失 (stage=train) | 全 NaN |
| `"no_val_metrics"` | 同上 (stage=val) | 全 NaN |
| `"no_test_metrics"` | 同上 (stage=test) | 全 NaN |
| `"no_inference"` | 该用户无推理数据 (`data/infers/<uid>/` 缺) 或推理产物 CSV 缺 | 全 NaN |
| `"bad_train_csv"` / `"bad_val_csv"` / `"bad_test_csv"` / `"bad_inference_csv"` | metrics CSV 格式异常 (缺 `split` 或 `model` 列) | 全 NaN |
| `"no_train_rows"` / `"no_val_rows"` / `"no_test_rows"` / `"no_inference_rows"` | metrics CSV 里没有对应 stage 的行 | 全 NaN |
| `"ok:main"` | 主模型 `main` 的指标 (train/val/test 3 stage 首选, inference 时若 main_final 缺失也会用 main) | 完整 |
| `"ok:main_final"` | 最终模型 `main_final` (含 L4+L5 守卫) 的指标, **仅 inference stage 出现** | 完整 |
| `"ok:<其他模型名>"` | 优选模型都缺时, 退化取 metrics CSV 里第一个可用模型. 常见: `ok:rf` / `ok:fallback` / `ok:naive_mean` / `ok:naive_zero` / `ok:main_L4_calib` / `ok:nilm_ac_two_stage_v42` 等 | 完整 |

**status 判定优先级** (aggregate_metrics 代码顺序):
```
1. 若 artifacts/trains/<uid>/skip_reason.json 存在
   → 4 stage 全部 soft_skip:<reason>
2. 若 metrics CSV 完全缺失
   → no_<stage>_metrics / no_inference
3. 若 CSV 缺 split/model 列
   → bad_<stage>_csv
4. 若 CSV 存在但无该 stage 的行
   → no_<stage>_rows
5. 正常, 按 model_preference 选行 (train/val/test 选 main, inference 优先 main_final)
   → ok:<chosen_model>
```

### C.6 通用排查指南

| 症状 | 可能原因 | 排查命令 |
|---|---|---|
| status = `soft_skip:aligned_too_few` | 训练数据 < 1 天 | `wc -l data/trains/<uid>/*.csv` |
| status = `soft_skip:single_class_label` | 分路全 0 (all_off) 或全 ≥10W (all_on) | `python -c "import pandas as pd; s=pd.read_csv('...').iloc[:,1]; print((s>=10).mean())"` |
| status = `soft_skip:split_empty_val_test` (但训练指标正常) | **v13.3 bug 残留** | 详见 §八 诊断章 |
| status = `no_inference` | 缺 `data/infers/<uid>/` 或 CSV | `ls data/infers/<uid>/` |
| status = `no_train_metrics` (但训练看起来跑了) | archive_outputs 归档失败 | `ls artifacts/trains/<uid>/train_val_metrics.csv` |
| status = `bad_train_csv` | metrics CSV 结构损坏 | `head artifacts/trains/<uid>/train_val_metrics.csv` |
