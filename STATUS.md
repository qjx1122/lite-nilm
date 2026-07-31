# STATUS.md

> 会话续接状态文件。每次 session 开局读它恢复上下文，收尾更新它固化上下文（见 BOOTSTRAP.md）。
> 更完整的历史见 `NILM_AC_session_summary_v13.17.md` + `V14_UPGRADE_REPORT.md`。

## 当前目标
- 已完成（2026-07-31 session 2）：**5 用户 v14 批量重跑 + 与上版对比 + 逐用户逐日问题详析** → 交付 `V14_RERUN_ANALYSIS.md`
- 已完成（2026-07-31/08-01 session 3）：**P0–P2 整改路线图执行 + 全量重验** → 交付 `V14_REMEDIATION_REPORT.md`
- 已完成（2026-08-01 session 4）：**推理集扩充 6 月训练窗外 OOD 验证 + 逐日详析 + 共性梳理** → 交付 `V14_JUNE_EXT_ANALYSIS.md`
- 下一会话：待指派（候选：P-CE1 泛化守卫 / P-CE5 guard 滑窗锚 / P-CE6 日报 coverage 列 / 8 月数据回流）

## 已完成
- [x] v14 收尾（session 1）：93 组 v14 单测全过 + symmetry 测试可移植性修复 + 烟测/工具链实测
- [x] **批量重跑**：`run_batch_users.py --time-filter-config data/time_filters.json --force-retrain`，5/5 ok，542s，泄漏检测过，装 lightgbm 4.7.0 对齐上版模型形态
- [x] **三版对比**（base/上版/本次，仅7月口径）：均值 F1 0.962（=上版, base+0.007）、MAE 121.2W、SAE 24.2%
- [x] **U2844 bus_guard v14.1.1 修复验证**：7/1–7/4 OFF 日 FP 44→0，7/5 FN 36→22，F1 0.897→0.962
- [x] **U842 改口径适配**：本次推理含 6 月验证段（n=1340 F1=0.893），仅7月口径重切（F1=0.932 vs 上版 0.991，窗口改制代价+风险暴露）
- [x] **U842 7/6 整日崩塌根因实锤（P8 天气先验失配）**：p_on=0.05 / 全维扫描 Top 偏差全为天气族（diurnal 1.87σ/humidity 1.24σ）/ 梅雨组 p_on 0.684 vs 晴热组 0.990 / 6·21(26.2°C RH86)同签名复现
- [x] 逐用户逐日问题表（75+ 天全部标注 P1–P8）+ 整改路线图 → `V14_RERUN_ANALYSIS.md`

## 已完成（session 3 整改）
- [x] v14.2 整改模块落地（功率温桶标定 lift-only / 时段先验抑制 / 日报 4 新列）+ 47 组单测
- [x] **P0-1 温桶标定启用 3 用户**：U0778 SAE 38.3%→19.0%（达标）、U0789 20.8%→16.7%（达标）、U0800 36.5%→35.4%（部分愈，7 月高温无桶+档位上行 80% 未达标）
- [x] **P0-3 U0800 扩窗证伪回退**（F1 0.9566→0.9269），替代方案 **v14.4 解耦统计窗**（分类器原窗 + LUT/先验用 [5/21,6/30] 全量训练侧日期重建，3 用户统一政策）
- [x] **P0-2 U842 梅雨修复三实验全部证伪**：E1 去天气（F1 0.7199）/ E2 代表日回流（0.8882,FN 增倍）/ E3 去 dow（0.6418），7/6 均依然全灭；线性探针同败+bus/标签排除 → 联合分布外，标记未愈需业务补样本
- [x] **P1-1 U2844 对称标定 cap=p90 落地实测空转**（上抬 0 下压 0），残余 +5.7% 为整体平移非温桶问题，bus_guard 已担
- [x] **P1-2 U2844 fill_short_off {3,5} 网格**（val 5/28-6/4 段内推理，零泄漏）两档逐点一致 → 保守保留 3
- [x] **P2-1 time_prior 证据否决**：U0789 双低可压 1/42、U842 0/20，FP 集中在开关机过渡段非低先验时段
- [x] v14.3 新机制：exclude_features 黑名单 + align_features_to_bundle 对齐哨兵（04/05/06 三处）+ 对称标定 + 一致性测试动态发现（21+11 组新单测）
- [x] **终验 5 用户全量重跑 546.5s 全 ok**：F1/P/R/TP/FP/FN 与基线逐位一致（零回归实证）
- [x] 交付 `V14_REMEDIATION_REPORT.md`（逐项「做了什么/参数出处/验证数字/未愈声明」）

## 已完成（session 4 扩充验证）
- [x] **推理集扩充**: 5 用户 6 月训练窗外日全部入推理集 (U842 至6/30; U2844/U0789 6/5-30; U0778 6/9-30; U0800 6/2-30), 全量重跑 485.2s 5/5 ok
- [x] **回归闸**: 7 月段 U842/U0800 与 v14.3 终验逐位一致; U0778/U0789 ±0.3~0.9pp (尾窗特征公差); U2844 漂移已查实=P-CE5
- [x] **核心新发现**: P-CE1 电路归属错位虚块 (U0789 11d/630步, U0800 17d/716步恒定40步, U0778 2d/95步; bus 实测中午块 0.6~1.5kW>ON 日水位, 非分类器错误); P-CE2 梅雨崩塌新增 U842 6/8、U2844 6/22(guard误杀); P-CE3 6月高估↔7月低估双向档位漂移直接印证
- [x] **P-CE5 查实**: bus_guard pwr_scale 全窗池化, June 入窗后 0.953→1.000, U2844 7月 SAE 8.3→13.2% (7/5 反改善 48%→17.4%)
- [x] 交付 `V14_JUNE_EXT_ANALYSIS.md` (口径声明/回归闸/5用户逐日表/P-CE1~CE8 共性+修复路径)

## 进行中
- （无）

## 下一步（TODO，按优先级）
1. **P-CE1 泛化 bus_guard 到三小用户**（需先落 P-CE8 防误杀门: force-off 前"分路连续0+bus块功率≈目标档位"双重门）
2. **P-CE5 guard 自适应锚改逐日/滑窗**（K 由训练侧 val 选型）→ 修复 U2844 扩窗回归
3. **P-CE6 日报加 coverage 列**（n_bus_raw/n_branch_raw 对 96 步比对，轻量）
4. **8 月数据回流**后复测: P-CE2 梅雨样本、CE3 LUT 基准重构、U0800 尾部高温
5. **U0789 分路/双开状态建模**（独立专题；CE1 确认后优先级升）

## 决策记录 / 踩坑
- **⚠️ venv 不跨 session 持久**：bash 快照剔除 `.venv`，每个新 session 开头必须 `python3 -m venv .venv && pip install -r requirements.txt`（~25s）；需 lightgbm 时另装（~3s）
- **⚠️ 本地 git 历史可能被重置到 base commit（dd5e87c）而工作区文件保留**：远端分支才是真相——开局先 `git fetch origin arena/019fb816-lite-nilm:refs/remotes/origin/arena/019fb816-lite-nilm` 对比 HEAD，若本地落后/分叉则 rebase 到远端再干活（session 2 已踩并修复：本地 commit 曾错挂 dd5e87c 上，rebase 到 86ce7ad 解决）
- bundle.pkl 反序列化需 `sys.path.insert(0,"scripts")`（EnsembleClf 在 v14_enhancements 模块内）；本仓库 bundle 键为 `feat_cols/feat_names/clf/scaler/best_thr/branch_temp_power_lut/hourly_on_prior/...`
- **venv 补装注意**：requirements.txt 不含 pyyaml/requests，批跑前需 `pip install pyyaml requests`（session 3 踩坑修复）
- **批跑 --users 是空格分隔**（nargs="*"），逗号会被当成单个用户 ID 导致"发现 0 用户"
- **分路/总线 CSV 名易混**：总线 = `e241_...-Ch1-...csv`（event_time + load_iden_data*），分路 = `<meter_id>-...csv`（time + p1..p4）；对齐后分路列重命名 `y_ac`
- **resample_and_align keep_cols=None 可能出 0 行**（全 load_iden 列含全 NaN 列时），训推一致必须用 bundle feat_cols
- artifacts/models/logs 被 .gitignore 排除不入库（快照内仍保留在本地工作区），报告引用其相对路径与仓库惯例一致
- U842 与上版对比必须**按 ≥2026-07-01 重切**（本次推理 n=2780=6月验证段1340+7月1440，上版表 3.1 仅 7 月；真值 kWh 138.6 两侧一致口径自洽）
- U842 7/6 排查路径：排除数据事故（原始 d73/分路/气象缓存正常）→ p_on 崩塌在分类层 → 168 维全维扫描定位天气族 → 湿度 Middling 陷阱（ON 窗湿度 93 vs 日均 88；日均口径否决"湿度独因"假设，转向温差/cooling_degree 联合签名）→ 6/21 同签名复现确认 P8
- 5σ 异常/focal gamma 测试设计踩坑记录见 git log（session 1 commit）

## 关键文件路径
- `STATUS.md` — 本文件 | `V14_JUNE_EXT_ANALYSIS.md` — **最新主交付**（6 月扩段 OOD+共性梳理）
- `V14_REMEDIATION_REPORT.md` — P0-P2 整改验证 | `V14_RERUN_ANALYSIS.md` — v14 基线对比+逐日详析+路线图 | `V14_BATCH_COMPARISON.md` / `V14_DAILY_METRICS_ANALYSIS.md` — 上版对比基线
- `data/time_filters.json` — 5 用户最终配置（U2844 bus_guard+direction=both；3 用户 power_temp_calib+calib_stats_include）
- `scripts/power_temp_calib.py` — 温桶标定/时段先验（含 direction=both cap=p90 对称模式）
- `results_v6_15_0/` — 一致性测试符号链接农场（已 .gitignore，供 test_train_infer_symmetry 实跑）
- `artifacts/infers/<user>/inference_daily_metrics.csv` / `inference_result.csv` — 逐日/逐点
- `logs/_batch/batch_run_20260731_151530.log` — 批跑日志（bus_guard 统计在此）
- `models/<user>/nilm_ac_two_stage.pkl` — v14 bundle（clf=EnsembleClf）
- `scripts/test_v14_enhancements.py` — v14 单测 93 断言（session 1）
