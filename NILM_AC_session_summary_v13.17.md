# NILM AC 项目 — 精简会话摘要 v13.17

> **用途**: 承接下一次新会话的**最小化上下文**. 完整历史见 `NILM_AC_session_complete.md`.
> **本文档规模**: ~310 行, 覆盖当前状态 + 关键决策链 + INTJ 教训 + 下一步 TODO.

---

## 1. 项目基础

**项目路径**: `/home/user/nilm_ac_win/`
**当前版本**: `v6.12.6+v6.15.0-graceful-v13.17`
**技术栈**: Python 3.13 + sklearn + pandas + XGBoost, Windows/Linux 双平台
**任务**: 从总线电参量 (76 路 load_iden_data*) 分解出空调分路功率, 15min 采样.

## 2. 角色与工作风格 (INTJ)

- **中文回复**, 代码注释中文
- **杜绝主观臆断, 必须以代码硬证据/数据查证支持结论** (最重要)
- **详细命令输出 + 关键指标对比表 + Bug 根因诊断**
- **提供选项时用 ask_user 工具**
- **严谨对待清理操作: 清理前必须确认可恢复性**
- **Windows GBK 编码坑**: 用 ASCII `[OK]/[SKIP]/[FAIL]`, 别用 ⏭ U+23ED

## 3. 关键代码位置

| 文件 | 作用 | 关键函数 |
|---|---|---|
| `scripts/common.py` | 全局常量 (L96 `ON_THR_W=10.0`, L164 `SPLIT_STRATEGY`, L172 `SPLIT_RATIOS`) | - |
| `scripts/03_train.py` | 训练主流程 | 训练开头 L118-160 (v13.5 env vars 覆盖), L385 (v13.8 保存 train_dates), L295 (state=y>=ON_THR_W) |
| `scripts/04_evaluate.py` | test 集评估 | v13.5-fix `on_thr_eval = float(bundle.get("ON_THR"))`; **v13.14 生成 train_daily_metrics.csv** |
| `scripts/05_inference.py` | 推理 (含 L4/L5 防御) | **v13.14 生成 inference_daily_metrics.csv (used_leak/used_ood 自动标记)** |
| `scripts/split_utils.py` | 数据集切分 | **v13.12 stratified_day round 修复, v13.13 新增 global_stratified** |
| `scripts/analyze_on_periods.py` | 分路开机时段分析 (v13.6 新增) | `compute_on_periods()` + `compute_daily_summary()` (v13.11 支持 OFF 天, v13.10 dataset 归属) |
| `scripts/metrics_utils.py` | 指标计算 | **v13.14 `build_daily_metrics_rows()` + `save_daily_metrics_csv()`** |
| `scripts/feature_utils.py` | 特征工程 | v13.7 `assert_no_nan_features()` NaN 硬检测 |
| `scripts/drift_features.py` | 漂移感知特征 + 温度桶 LUT | v6 `build_temp_power_lut()` + `build_drift_features()`; **v13.15 `return_meta=True` 扩展 + `export_temp_power_lut_csv()` + `export_temp_power_actual_vs_expected_csv()`** |
| `scripts/feature_utils.py` | 数据加载与特征工程 | v13.7 `assert_no_nan_features()`; **v13.16 `load_branch_csv(path, target_col=None)` 复合列物化 (含 '+' 时新增列 = 分量 sum(skipna=False))** |
| `scripts/02_align_and_feat.py` | 时间对齐 + 特征相关性 | v12 时段过滤; **v13.16 传 TARGET_COL 到 load_branch_csv + 日志显性化提示** |
| `scripts/time_filter_utils.py` | JSON 配置解析 (v12/v13/v13.5) | `get_user_common_overrides()` (9 项 common 覆盖), v13.13 VALID 加 global_stratified, **v13.16 `get_user_target_col` 正则放宽 + 复合防呆** |
| `scripts/run_user_pipeline.py` | 单用户端到端流水线 | `run_analyze_step()` (v13.6), v13.10 训练后补跑 + 推理前 date_labels 注入, **v13.16 `_validate_target_col` 正则放宽** |
| `scripts/run_batch_users.py` | 多用户批量入口 | 从 `--time-filter-config` 读 JSON, **v13.16 复合列分量完整性校验** |
| `scripts/analyze_on_periods.py` | 分路开机时段分析 | v13.6/9/10/11 一系列增强, **v13.16 `_RE_PN_COMPOSITE` + `_materialize_composite_target()` 幂等物化** |
| `scripts/test_temp_power_lut_csv.py` | v13.15 单测 | 34 组断言 5 用例 (T1-T5), 独立可跑不依赖 pkl/bundle |
| `scripts/test_composite_target_col.py` | **v13.16 新增单测** | **47 组断言 6 用例 (T1-T6): 正则/归一化/防呆 (17) + 一致性 (11) + 用户示例逐值 [32,16,16,24] 对齐 (6) + 边界+兼容 (4) + analyze 端到端 (7) + resample_and_align 集成 (2)** |
| `scripts/test_min_w_column.py` | **v13.16-min_w 新增单测** | **25 组断言 7 用例 (T1-T7): 段级 min_w 位置+值 (6) + 多段独立 (3) + daily ON 天 min=min(各段) (4) + OFF 天 min=全天 (4) + 复合列 (4) + 空 df 兜底 (2) + 老输入向后兼容 (2)** |
| `scripts/test_daily_raw_counts.py` | **v13.16-daily_raw 新增单测** | **28 组断言 8 用例 (T1-T8): 字段+位置 (4) + 值正确 (4) + 向后兼容 (2) + raw_counts 基础/过滤 (7) + 兜底 (2) + 端到端 (5) + 落盘读回 (4)** |
| `scripts/test_batch_execution_state.py` | **v13.17 新增单测** | **37 组断言 12 用例 (T1-T12): 加载 4 (不存在/损坏/老格式/正常) + `_get_completed_users` retry 分支 5 + `_upsert` 首写+覆盖+原子 11 + 端到端崩溃恢复 3 + fail→ok 更新 3 + CSV 格式 4 + 中文 utf-8-sig 兼容 2** |
| `scripts/metrics_utils.py` | 指标 API | **v13.16-daily_raw 新增 `compute_raw_daily_counts(csv, time_col, time_filter_spec=None) -> {date:int}` 工具函数 + `build_daily_metrics_rows(..., bus_daily_counts=None, branch_daily_counts=None)` 双参数扩展** |
| `scripts/run_batch_users.py` | 批量入口 | v13.16 复合列校验 + **v13.17 断点续跑: `_load_execution_state()` / `_get_completed_users()` / `_upsert_execution_state()` 三 API + `--resume` / `--resume-skip-failed` CLI + 每用户跑完 upsert 状态 (原子写, 崩溃时最多丢当前正跑的)** |
| `scripts/run_user_pipeline.py` | 单用户流水线 | v13.16 `_validate_target_col` 正则放宽 + **v13.17 关键 bug 修复: `cleanup_artifacts_top` 加白名单保护 batch_execution_state.csv 等 5 类批量层持久化文件 (之前每次单用户结束都会 unlink → 状态 CSV 只剩最后 1 行)** |

## 4. 完整版本演进 (v11 → v13.14)

| 版本 | 关键改动 | 一句话 |
|---|---|---|
| v11 | D87_ADAPTIVE_GUARD_ENABLED 总开关 (默认 False) | 变频空调 (d87 尖峰退化) 需要关掉守卫 |
| v12 | 时段过滤 (train/infer 独立 include+exclude) | 支持任意时段配置 |
| v13.1 | 用户级 `guard_enabled` + 自动检测降级 (判据 A/B) | 270708 F1 0.887→0.996 |
| v13.2 | per-split 时段过滤 (train/val/test 独立) | 4 步语义: 原切分→include 硬锚定→严格保持形状→exclude |
| v13.3 | 修 archive_outputs 旧 skip_reason.json bug | 沙箱端到端塞入残留复现 |
| v13.4 | `target_col` 配置化 (最高优先级) | 270708 双通道 p1/p2 覆盖 |
| v13.4-fix | target_col 从 [p1-p4] 放宽到通用 pN | 支持 p0/p5/p10/p128 |
| **v13.5** ⭐⭐⭐ | **9 项 common 常量用户级覆盖** | on_thr_w/split_ratios/split_strategy/post_min_on/post_fill_short_off/weather_lat/lon/use_weather/use_temp_season |
| v13.5-fix | ON_THR_W 3 处硬编码 bug 修复 | 04_evaluate + baseline_utils + run_user_pipeline 全部从 bundle 读 |
| v13.6 | 分路开机时段分析工具 + 流水线集成 | `analyze_on_periods.py`, 训练前+推理前各自动跑 |
| v13.7 | 特征矩阵 NaN 硬检测 (fail-fast) | 270758 Windows `Input X contains NaN` 崩溃根治 |
| **v13.8** ⭐⭐⭐ | **train/infer 数据泄漏自动检测 + 拆分指标** | 270758 用户追问驱动, 拆 leak/ood 两组指标 |
| v13.8-fix1 | leak/ood 拆分覆盖 3 主模型 | main / main_L4_calib / main_final, 定量证明 L4 层防过拟合价值 |
| v13.9 | analyze_on_periods 4 字段计算 6 层审计 | 主流 100% 正确, 边界 2 bug 已 WARN 显性化 |
| v13.10 | 分路 CSV dataset 归属列 | 训练标 train/val/test/未使用, 推理标 used/excluded |
| v13.11 | 全天 OFF 日也输出 CSV | 270708 3 个周日 OFF 精准捕获 (待机 1.44W) |
| v13.12 | stratified_day round 精度修复 | 每月单独 round 累积舍入 bug, 主目标 train 优先 |
| **v13.13** ⭐⭐ | **新增 `global_stratified` 切分策略** | 270758 用户 14 天 (5-21~6-03) 从 9/3/2 → 10/2/2 完美 |
| **v13.14** ⭐⭐ | **逐日主模型评估指标 CSV** | train_daily_metrics.csv + inference_daily_metrics.csv (dataset 自动标 used_leak/used_ood) |
| **v13.15** ⭐⭐ | **温度桶期望信号 CSV 导出 (概念漂移可视化)** | 训练侧 temp_power_lut.csv (12 列 20 桶 + 全局中位) + 推理侧 inference_temp_power_actual_vs_expected.csv (13 列 20 桶, drift_flag 三档 OK/WARN/ALERT/NO_DATA); 与 drift_report.csv 互补 (后者只写触发告警的 3 桶); 34 组单测全通过; 270788 实测 8 ALERT + 6 WARN + 5 OK + 1 NO_DATA, 主指标零回归 |
| **v13.16** ⭐⭐ | **`target_col` 支持复合列语义 (`p1+p2` / `p1+p2+p3`)** | 场景: 同一空调多分路合并作为总标签; 语法 JSON 配置 `"target_col": "p1+p2"` 或 `"p1+p2+p3"`; 物化位置 `feature_utils.load_branch_csv` 加载 CSV 后新增列 `p1+p2` (值 = `parts_df.sum(axis=1, skipna=False)` NaN 传播不静默补 0); 归一化 `" P1 + p2 "` -> `"p1+p2"`; 复合防呆拒绝 `p1+p1` 重复分量; 3 处正则统一放宽 `^p\d+(\+p\d+)*$`; 47 组单测全通过 (含用户示例 [32,16,16,24] 逐值硬对齐); 270788 双验证 = 单列 p1 零回归 + 复合 p1+p2 F1=0.9395 kWh≈p1×2 物理合理 |
| **v13.16-min_w** ⭐ | **`analyze_on_periods` 段级/daily 追加 `min_w` 列 (最小瞬时功率)** | 已有 `mean_w`/`peak_w` 独缺 min, 变频空调判"是否短暂低谷"必需; 4 处物化点全覆盖 (段级正常/跨日/OFF 天/daily); daily ON 天 = min(各段 min), OFF 天沿用段行; 位置 `duration_min` 之后 `mean_w` 之前; 老段级 CSV 无 min_w 输入优雅降级 daily.min_w=空; 25 组单测全通过; 270788 硬证据: 40 天段级 min_w 中位 36W (变频低档), 全部满足 min≤mean≤peak ✅ |
| **v13.16-daily_raw** ⭐⭐ | **`train/inference_daily_metrics.csv` 追加 `n_bus_raw`/`n_branch_raw` 2 列** | 已有 `n_samples` 是**对齐后**样本数, 独缺原始采集完整性视角; `metrics_utils` 新增 `compute_raw_daily_counts()` 工具函数 + `build_daily_metrics_rows()` 加 2 可选参数; 位置 `n_samples` 后 `Accuracy` 前; 推理侧应用 `--time-filter-spec` 保口径一致; CSV 从 23→25 列; **270788 首发揭示**: 5-21/22/23/25 四天 F1=0 根因是**总线原始只采了 3-7 点** (应 288, 分路全 96 完整), 铁证摊开; 28 组单测全通过; 向后兼容: 不传 counts 时新 2 列 = "" |
| **v13.17** ⭐⭐ | **`run_batch_users.py` 断点续跑 (`--resume`) + 实时增量状态 CSV** | `batch_run_summary.csv` 一次性覆盖写中断=全丢 → 无法断点续跑; **v13.17 独立新文件** `artifacts/batch_execution_state.csv` (9 列 upsert schema, 每用户跑完立即原子写 `.tmp + os.replace`); `--resume` 默认关闭零回归; 续跑决策: ok/soft_skip 跳, fail 重跑 (可 `--resume-skip-failed` 也跳); **关键 bug 修复**: `cleanup_artifacts_top` 加白名单保护 5 类批量层持久化文件 (之前每次单用户结束都 unlink 顶层 → 状态 CSV 只剩最后 1 行, 续跑失效); 37 组单测全通过; 真实端到端: 270708+270848 首跑 2 ok → `--resume` 加载 2 已完成 → 0 待跑 <1s ✅ |

## 5. 服役用户 (7 + 1 深度分析主角)

| user_id | 数据 | target | 特点 | 触发的关键版本 |
|---|---|---|---|---|
| 800080252842_4206894986488 | 116 天大变频 | p1 | 用于 V2 分析 | v11 |
| 800080252844_4206894986488 | 定频大 892W | p1 | 基线用户 | v11 |
| 800080270708_4206602981958 | 18 天变频小 (peak 235W) | p1 | 双通道, 周日不开机 | v13.1 / v13.4 / v13.5-fix / **v13.11** |
| 800080270737_4206680982373 | 40 天变频小 p4 (待机 16-24W 污染) | p4 | 6 大工况维度案例 | **v13.5 (催生)** |
| **800080270758_4206918577333** | 40 天变频小 | p2 | 数据泄漏演示主角 | **v13.7/v13.8/v13.10/v13.12/v13.13/v13.14** |
| 800080270825_4206911115606 | 定频大 858W | p1 | 基线用户 | v11 |
| 800080270848_4206671776099 | 定频小 137W | p1 | 基线用户 | v11 |
| **800080270788_4206701750448** | **40 天变频小 peak 342W** | **p1** | **本轮深度分析主角**: 三层根因链 (数据分布 50% + 温度漂移 30% + 硬阈值 20%); 6-24 稀疏 ON 唯一日 (N=1 案例); 6-27 单日 15min FN 诊断 | **v13.15 (催生温度桶 CSV)** |

## 6. 关键 tar.gz 打包 (可复现)

| 版本 | 路径 | 大小 | MD5 |
|---|---|---|---|
| v13 (旧基线) | `/home/user/nilm_ac_win-v6.12.6+v6.15.0-graceful-v13.tar.gz` | 5.1 MB | `a9291c70b6cfbf86922998c1944de104` |
| v13.8 (中间) | `/home/user/nilm_ac_win-v6.12.6+v6.15.0-graceful-v13.8.tar.gz` | 7.5 MB | `669b44deae8634e296cbababcff7e07b` |
| v13.14 (上一版) | `/home/user/nilm_ac_win-v6.12.6+v6.15.0-graceful-v13.14.tar.gz` | 7.9 MB | `9b401fdd20abbe76f8e2e8585b3f6702` |
| v13.15 (待打包) | 尚未生成 tar.gz | — | — |
| **v13.16 (最新, 待打包)** | 尚未生成 tar.gz | — | — |
| docs 精简版 (v13.14) | `/home/user/docs_only_v13.14.tar.gz` | 205 KB | `e654c63ebb91bce78ca94161ce9cdde6` |
| v13.14 打包前备份 | `/home/user/nilm_ac_win-pre-v1314-pack-backup-20260714_161223.tar.gz` | 7.6 MB | — |

## 7. INTJ 累计核心教训 (14 条)

**架构/设计**:
1. **默认参数在极端场景失效** — v11 全局关守卫 → 变频 OK 但定频回归; v13.1 用户级+自动检测才治本
2. **自动检测胜过手动配置** — 判据 B (覆盖率 <30%) 用**数据本身**说话, 不依赖用户懂 d87
3. **反推逻辑的不可预期性需要配置逃生舱** — v13.4 让业务方能明确指定 p2
4. **物理约束不可穿透, 必须显性化权衡** (v13.13) — 单月 5 天无法同时满足精准 70% 和 val/test 至少 1 天, 唯一选项是给用户 2 种策略主动选

**代码质量**:
5. **归档逻辑必须清理旧标记** (v13.3) — 潜伏了 v9→v12 期间, 因为很少测试"失败→修复→重跑"
6. **配置系统的全链路口径一致性必须端到端测试** (v13.5-fix) — 只测 03 层不够, 必须测 04/05/run_pipeline 4 层
7. **跨平台鲁棒性 bug 必须 fail-fast 显性化, 不能静默兜底** (v13.7) — Windows 环境 NaN 崩溃, INTJ 选择 raise + 精准定位, 而非 fillna 掩盖

**业务洞察**:
8. **训练前健全性检查工具的 ROI 是最高的** (v13.6) — 训练开跑前先看每天开机, 直接避免"盲改 on_thr_w → SAE 灾难"
9. **数据泄漏是"整体加权 SAE"的最大骗子** (v13.8) — 270758 整体 SAE 2.63% 实际是 leak 3.33% + ood 7.39% 加权; 真实泛化被掩盖 2.8 倍
10. **每个防御层的价值必须能被定量拆分证明** (v13.8-fix1) — L4 校正把 leak/ood SAE 差异从 22 倍压平到 ≈0
11. **主动审计比被动等 bug 更值** (v13.9) — 用户没报 bug, 只是要求核对 4 字段, 硬做 6 层验证反而挖出 2 个边界 bug
12. **反直觉的数据必须溯源到物理机制** (v13.10) — SAE U 形曲线 (1 天暖启动最优, 3 天变差), 溯源到暖启动日 vs 目标日的分布偏移

**可视化**:
13. **数据可视化能暴露算法层的隐藏 bug** (v13.11+v13.12) — v13.10 dataset 列本意可视化归属, 意外让用户从 dataset 分布对不上比例发现 v13.12 round bug
14. **聚合指标是分析的起点, 不是终点** (v13.14) — daily 视图打开单日诊断, 结合 v13.10 dataset + v13.8 泄漏检测形成完整链
15. **黑盒 pkl 里的关键资产必须显性化为 CSV** (v13.15) — `temp_power_lut` 在 v6 就存在于 `bundle.pkl` 里, 但 6 个月内没人能回答"训练时 27°C 桶期望多少 W". 显式导出 CSV 后, 训练资产变可审计, 且与推理侧对比直接可用. 教训: **模型资产的透明度决定长期可维护性**, 隐性知识必须显性化
16. **稀有工况样本的 3 档决策矩阵** (v13.15 270788 案例) — N=1: 不放训练也不放推理评估 (单样本估不准指标); N=2-3: 手工按 include 硬锚定 (1 入 train + 1 入 val); N≥4: 走 stratified_day 自动分层. **物理硬证据**: 270788 实验 A 把 6-24 稀疏日 (N=1, max_p=342W) 强塞入 train → 推理 Recall 从 0.832 崩到 0.670. N=1 的分布外推靠算法无解, 只能业务侧补数据
17. **DSL 扩展应"贴近用户表达, 远离下游侵入"** (v13.16) — 复合 `target_col="p1+p2"` 语义放在**数据加载层一次物化**, 下游 20+ 处业务代码 (resample_and_align/label_cleaner/analyze_on_periods/03_train/04_evaluate/05_inference/metrics_utils) 完全无感, 把它当普通列名. 反面案例: 若在每个使用点解析 '+' 求和, 侵入点 20+ 处, 极易漏改+回归. 教训: **配置语法的表达能力应扩展在解析层, 而非蔓延到所有使用点**. 单点改动=单点回归责任, 多点改动=多点隐患.
18. **NaN 传播 vs 静默补 0 是数据一致性的分水岭** (v13.16) — 复合列求和用 `sum(skipna=False)`, 任一分量为空则整行 NaN, 下游 `dropna(subset=["y_ac"])` 显式剔除. 若用 `skipna=True` 会静默把缺失当 0, 电量估计**偏低**且用户无感知. 教训: 涉及能量/kWh 的聚合, **优先让缺失显性化, 不做善意的填充**.

## 8. 当前累计统计

- **代码增量**: ~3810 行 (相对 v12, v13.17 新增 ~360 行 = run_batch_users +125 + run_user_pipeline +10 + 单测 +310 + 文档 +140)
- **单元测试**: **277 组** (v13.1-v13.16-daily_raw 累计 240 + **v13.17 37**), 全部一次通过
- **子版本链**: v12 → v13 (5 子) → v13.5 (fix) → v13.6/7/8 (fix1) → v13.9/10/11/12/13/14/15/16 (含 -min_w / -daily_raw 追加) / **17**
- **文档**: README_WIN.md (~3230 行), REPORT.md (~3140 行), session_export/NILM_AC_session_complete.md (5730 行, 附录 J-daily_raw), session_summary_v13.17 (本文档)

## 9. 未完成事项 (TODO 供下次会话)

**已知未修**:
- ⚪ 诊断脚本硬编码 10W (`diag_new_data.py` / `diag_inference.py` 用 `ON_THR_BUSINESS_W`, 优先级低)
- ⚪ `stratified_day_bucket` 策略未恢复 (v11 有过后回退, 对稀疏用户如 270737 可能有价值)
- ⚪ v13.14 只输出 `main_final` 一个模型的 daily, 未来可扩展 `--extra-model main_L4_calib main` 参数
- ⚪ v13.14 dataset 只覆盖训练侧 (train/val/test) 和推理侧 (used_leak/used_ood), 若同时想加入 test/val 泄漏检测 (即 val_leak/test_leak) 可扩展
- ⚪ **v13.15 / v13.16 / v13.17 未打包 tar.gz** — 用户尚未选择打包, 待选后生成 v13.17 完整包 + docs_only_v13.17 精简包 + MD5
- ⚪ **v13.17 待做扩展**: (a) `--resume` 场景下的 `batch_run_summary.csv` 只含本次跑的用户, 若想合并成"完整历史视图"可加 `--merge-history` 参数; (b) 若需按 `run_id` 保留多次执行历史 (append 而非 upsert), 可加 `--state-append` 模式 (目前是 upsert 覆盖)
- ⚪ **v13.15 CSV 可视化配套图** — 目前只输出 CSV, 若加自动生成"训练期望 vs 推理实测"双柱状图 PNG 更直观 (matplotlib 1 页)
- ⚪ **v13.16 分量权重扩展** — 当前复合列是等权求和 `sum`, 若未来场景需要 "0.8*p1+0.2*p2" 加权合并, 可扩展语法 (风险: 语法复杂度上升, 需评估必要性)

**v13.9 已识别但未修 (选项 2/3)**:
- ⚪ analyze_on_periods 非均匀采样 → dur/energy 内部口径不一致 (仅 --br-csv 模式偶发, 已 WARN)
- ⚪ analyze_on_periods 时间断裂被合并 → duration 严重高估 (仅 --br-csv 模式偶发, 已 WARN)

**v13.10 上下文边界效应可选深度修复**:
- ⚪ 给 `05_inference.py` 加 `--warmup-days N` 参数 (默认 7, 自动前拓推理窗口)
- ⚪ `run_user_pipeline.py` 里做暖启动分布检查, 若差 >20% WARN

**270788 深度分析遗留 (v13.15 上一轮)**:
- ⚪ **改善建议 1: 降阈值 0.930→0.7** — 91 FN 中 20 个 (22%) p_on 0.5-0.93 可救回, 预估 Recall 0.832 → ≥0.87
- ⚪ **改善建议 2: 加低负荷稳态日到 train.include** — 从 5-21/5-22/5-23 挑选 (60-90W 稳态段), 与失败的实验 A (加稀疏日 6-24) 性质不同
- ⚪ **建议 3: L4 校正应用于低置信度 ON** (需算法层改动)
- ⚪ **建议 4: 温度自适应重训 (长期)** — 需新数据触发

## 10. 承接下一次会话的最小指引 (给未来的 agent 用)

```
本项目 = NILM 空调分解, Python + sklearn + XGBoost
当前版本 = v6.12.6+v6.15.0-graceful-v13.17
项目路径 = /home/user/nilm_ac_win/
完整历史 = /home/user/nilm_ac_win/session_export/NILM_AC_session_complete.md (5363 行)
本摘要 = /home/user/nilm_ac_win/session_export/NILM_AC_session_summary_v13.16.md (本文件)

必读:
- §2 (角色与工作风格 INTJ 硬证据链)
- §3 (关键代码位置, 含 v13.16 复合列相关文件)
- §5 (服役用户特征, 含 270788 深度分析主角)
- §7 (18 条 INTJ 教训, 避免踩坑)
- §9 (TODO 未完成事项, 含 v13.15/v13.16 未打包 + 270788 4 项改善建议)

深读 (需要时):
- 完整历史 md 附录 F/G/H/I (v11~v13.16 各版本详细设计与验证)
- README §附 C.4.2 (v13.16 复合 target_col 详解) / §C.4.8 §十三 (v13.14 daily) / §十四 (v13.15 温度桶)
- REPORT §14.5 (v13→v13.16 完整交付清单) 与 §14.7 (INTJ 反思, 累计 18 条)

打包路径 (若需回滚):
- v13.14 tar.gz: 7.9 MB, MD5 9b401fdd20abbe76f8e2e8585b3f6702
- v13.14 备份: /home/user/nilm_ac_win-pre-v1314-pack-backup-20260714_161223.tar.gz
- v13.15 / v13.16 tar.gz: 尚未打包 (用户下一步待选)

v13.15 新增 CSV 交付物:
- artifacts/trains/<user>/temp_power_lut.csv (12 列, 20 桶 + 1 全局中位)
- artifacts/infers/<user>/inference_temp_power_actual_vs_expected.csv (13 列 20 桶, drift_flag ∈ {OK,WARN,ALERT,NO_DATA})
- 单测: python scripts/test_temp_power_lut_csv.py → 34/34 通过 (<1s)

v13.16 新增能力 (复合 target_col):
- JSON: "target_col": "p1+p2" 或 "p1+p2+p3"
- 语义: 加载分路 CSV 时自动新增列 "p1+p2" = 逐行 p1+p2 (sum skipna=False)
- 归一化: " P1 + p2 " → "p1+p2"
- 防呆: p1+p1 拒绝 (语义无意义)
- 单测: python scripts/test_composite_target_col.py → 47/47 通过 (<2s)

v13.16 关键 API 签名:
- feature_utils.load_branch_csv(path, target_col=None)
  # 含 '+' 时自动物化: df["p1+p2"] = df[["p1","p2"]].sum(axis=1, skipna=False)
- time_filter_utils.get_user_target_col(config, user_id) -> "pN" 或 "pA+pB..." 或 None
- analyze_on_periods._materialize_composite_target(df, target_col) -> df (幂等)

v13.16-min_w 输出增强:
- 段级 CSV 列: being_time, end_time, <target>, duration_min, **min_w**, mean_w, peak_w, energy_kwh, dataset
- daily CSV 列: date, n_segments, total_on_min, total_on_hours, first_on_time, last_off_time, **min_w**, mean_w, peak_w, energy_kwh, dataset
- daily min_w 语义: ON 天 = min(各 ON 段 min_w) (开机期间最低瞬时功率); OFF 天 = 全天最小 (待机下限)
- 物理不变量: min_w ≤ mean_w ≤ peak_w (270788 40 天 all pass)

v13.16-daily_raw 输出增强 (daily metrics 加原始采集完整性视角):
- train_daily_metrics.csv / inference_daily_metrics.csv 追加 2 列 (位置在 n_samples 之后):
  * n_bus_raw    = 当天总线 CSV 原始采集点数 (5min 满 288)
  * n_branch_raw = 当天分路 CSV 原始采集点数 (15min 满 96)
- 与 n_samples 的差异: n_samples 是对齐后 (受时段过滤+inner-join); n_bus_raw/n_branch_raw 是原始 CSV group_by 计数
- 关键 API: metrics_utils.compute_raw_daily_counts(csv, time_col, time_filter_spec=None) -> {date: int}
- 推理侧应用 --time-filter-spec 让口径与实际推理天数一致
- 270788 首发揭示: 5-21/22/23/25 F1=0 根因 = 总线原始只采 3-7 点 (不是缺分路)

v13.17 断点续跑 + 状态 CSV:
- 新文件 artifacts/batch_execution_state.csv (9 列, utf-8-sig): user_id, status, success,
  started_at, finished_at, duration_s, message, target_col, run_id
- 每用户跑完立即 _upsert_execution_state() (原子写 .tmp + os.replace)
- CLI: --resume 启用续跑 (默认关闭零回归), --resume-skip-failed 连 fail 也跳过
- 续跑决策: ok/soft_skip → 跳; fail → 重跑 (默认) 或跳过 (--resume-skip-failed)
- 关键修复: run_user_pipeline::cleanup_artifacts_top 加白名单保护 5 类批量层持久化文件
  (batch_execution_state.csv, batch_execution_state.csv.tmp, batch_run_summary.csv,
   summary_metrics_all_users.csv, skipped_users.csv)
- 老代码 bug: 之前每次单用户 pipeline 结束都 unlink artifacts 顶层所有文件 → 状态 CSV 只剩 1 行

270788 v13.16 端到端验证:
- 单列 p1: F1=0.9022 / Recall=0.8324 / SAE=5.45% (与 v13.15 完全一致, 零回归)
- 复合 p1+p2: F1=0.9395 / Recall=0.9006 / SAE=4.10% / kWh_true=32.95 ≈ p1×2 (物理合理)
- min_w 统计: 40 段/40 天中位=36W, max=174W, 揭示变频低档存在硬凭证
- daily_raw 统计: 训练 39 天 min=3/max=291, 推理 20 天 min=3/max=291, 4 天严重不足直接铁证解释 F1=0
```

---

*v13.14 摘要生成于 2026-07-14 (v13.14 打包 + session_export 同步完成后)*
*v13.15 摘要更新于 2026-07-15 (温度桶 CSV 导出 + 34 单测全通过, 尚未打包)*
*v13.16 摘要更新于 2026-07-15 (复合 target_col p1+p2 支持 + 47 单测全通过, 尚未打包)*
*v13.16-min_w 摘要更新于 2026-07-15 (analyze_on_periods 追加 min_w 列 + 25 单测全通过, 主指标零回归)*
*v13.16-daily_raw 摘要更新于 2026-07-15 (daily_metrics 追加 n_bus_raw/n_branch_raw 2 列 + 28 单测 + 270788 首发揭示 5-21/22/23/25 F1=0 根因是总线只采 3-7 点)*
*v13.17 摘要更新于 2026-07-23 (run_batch_users 断点续跑 --resume + 状态 CSV 实时增量写 + 37 单测 + 关键 bug 修复 cleanup_artifacts_top 加白名单 + 270708+270848 真实端到端验证 0 待跑 <1s)*
*完整历史见 NILM_AC_session_complete.md*
