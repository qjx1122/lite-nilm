# STATUS.md

> 会话续接状态文件。每次 session 开局读它恢复上下文，收尾更新它固化上下文（见 BOOTSTRAP.md）。
> 更完整的历史见 `NILM_AC_session_summary_v13.17.md` + `V14_UPGRADE_REPORT.md`。

## 当前目标
- 已完成（2026-07-31 session 2）：**5 用户 v14 批量重跑 + 与上版对比 + 逐用户逐日问题详析** → 交付 `V14_RERUN_ANALYSIS.md`
- 下一会话：待指派（候选：按报告 §6 P0 项实施——U842 梅雨先验修复 / 功率温度条件标定 / U0800 扩窗）

## 已完成
- [x] v14 收尾（session 1）：93 组 v14 单测全过 + symmetry 测试可移植性修复 + 烟测/工具链实测
- [x] **批量重跑**：`run_batch_users.py --time-filter-config data/time_filters.json --force-retrain`，5/5 ok，542s，泄漏检测过，装 lightgbm 4.7.0 对齐上版模型形态
- [x] **三版对比**（base/上版/本次，仅7月口径）：均值 F1 0.962（=上版, base+0.007）、MAE 121.2W、SAE 24.2%
- [x] **U2844 bus_guard v14.1.1 修复验证**：7/1–7/4 OFF 日 FP 44→0，7/5 FN 36→22，F1 0.897→0.962
- [x] **U842 改口径适配**：本次推理含 6 月验证段（n=1340 F1=0.893），仅7月口径重切（F1=0.932 vs 上版 0.991，窗口改制代价+风险暴露）
- [x] **U842 7/6 整日崩塌根因实锤（P8 天气先验失配）**：p_on=0.05 / 全维扫描 Top 偏差全为天气族（diurnal 1.87σ/humidity 1.24σ）/ 梅雨组 p_on 0.684 vs 晴热组 0.990 / 6·21(26.2°C RH86)同签名复现
- [x] 逐用户逐日问题表（75+ 天全部标注 P1–P8）+ 整改路线图 → `V14_RERUN_ANALYSIS.md`

## 进行中
- （无）

## 下一步（TODO，按优先级）
1. **P0 U842 梅雨先验修复**：天气特征去独断（限分裂收益/剔除 1-2 个）+ 6/19、6/21 类代表日回流训练（<7/1）→ 重跑验证 7/6
2. **P0 功率温度条件标定**（residual×温桶）：U0778（62%）、U0800（62%）、U0789（78%）电量不可用
3. **P0 U0800 训练窗 6 天→6 月底**（数据 5/21 起有 40 天）
4. **P1 U2844 标定方向修正**（该用户 7 月为高估 +6%，现 fixed_scale=0.85 按低估设计）+ 中档偏弱 ON 增强（7/5 类）
5. **P1 U0789 分路/双开建模**；日报指标改版（ON/OFF 拆分、false_on_kWh）
6. （长期）V14 报告 §九 P1-P3 roadmap

## 决策记录 / 踩坑
- **⚠️ venv 不跨 session 持久**：bash 快照剔除 `.venv`，每个新 session 开头必须 `python3 -m venv .venv && pip install -r requirements.txt`（~25s）；需 lightgbm 时另装（~3s）
- **⚠️ 本地 git 历史可能被重置到 base commit（dd5e87c）而工作区文件保留**：远端分支才是真相——开局先 `git fetch origin arena/019fb816-lite-nilm:refs/remotes/origin/arena/019fb816-lite-nilm` 对比 HEAD，若本地落后/分叉则 rebase 到远端再干活（session 2 已踩并修复：本地 commit 曾错挂 dd5e87c 上，rebase 到 86ce7ad 解决）
- bundle.pkl 反序列化需 `sys.path.insert(0,"scripts")`（EnsembleClf 在 v14_enhancements 模块内）；本仓库 bundle 键为 `feat_cols/feat_names/clf/scaler/best_thr/...`
- artifacts/models/logs 被 .gitignore 排除不入库（快照内仍保留在本地工作区），报告引用其相对路径与仓库惯例一致
- U842 与上版对比必须**按 ≥2026-07-01 重切**（本次推理 n=2780=6月验证段1340+7月1440，上版表 3.1 仅 7 月；真值 kWh 138.6 两侧一致口径自洽）
- U842 7/6 排查路径：排除数据事故（原始 d73/分路/气象缓存正常）→ p_on 崩塌在分类层 → 168 维全维扫描定位天气族 → 湿度 Middling 陷阱（ON 窗湿度 93 vs 日均 88；日均口径否决"湿度独因"假设，转向温差/cooling_degree 联合签名）→ 6/21 同签名复现确认 P8
- 5σ 异常/focal gamma 测试设计踩坑记录见 git log（session 1 commit）

## 关键文件路径
- `STATUS.md` — 本文件 | `V14_RERUN_ANALYSIS.md` — **本次主交付**（对比+逐日详析）
- `V14_BATCH_COMPARISON.md` / `V14_DAILY_METRICS_ANALYSIS.md` — 上版对比基线
- `data/time_filters.json` — 5 用户配置（U2844 含 bus_guard v14.1.1；U842 为 P0 v3 窗口+6月验证段）
- `artifacts/infers/<user>/inference_daily_metrics.csv` / `inference_result.csv` — 逐日/逐点
- `logs/_batch/batch_run_20260731_151530.log` — 批跑日志（bus_guard 统计在此）
- `models/<user>/nilm_ac_two_stage.pkl` — v14 bundle（clf=EnsembleClf）
- `scripts/test_v14_enhancements.py` — v14 单测 93 断言（session 1）
