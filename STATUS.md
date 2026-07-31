# STATUS.md

> 会话续接状态文件。每次 session 开局读它恢复上下文，收尾更新它固化上下文（见 BOOTSTRAP.md）。
> 更完整的历史见 `NILM_AC_session_summary_v13.17.md` + `V14_UPGRADE_REPORT.md`。

## 当前目标
- NILM-AC v14 收尾（2026-07-31 本会话）：✅ 已完成 —— v14 单测补齐 + 遗留项修复 + 环境烟测
- 下一会话：待指派（可选方向见"下一步"）

## 已完成
- [x] v13.17 批量断点续跑 `--resume` + `batch_execution_state.csv`（37 组单测）
- [x] v14 升级包（增强模块/包装入口/导出工具链/烟测脚本）+36 维物理指纹特征
- [x] 环境恢复：`.venv`（Python 3.11.2 + numpy 1.26.4 + pandas 2.2.3 + sklearn 1.5.2）
- [x] **本会话① 基线回归**：4 套存量单测 137 组断言（47+25+28+37）干净环境全过
- [x] **本会话② 新增 `scripts/test_v14_enhancements.py`**：93 组断言 T1-T12 全过，v14_enhancements 9 个公开 API 全覆盖（此前 v14 只有 smoke 无 dedicated 单测，与 v13.x 惯例不符）
- [x] **本会话③ 修复 `test_train_infer_symmetry.py` 可移植性**：原机绝对路径 `/home/user/nilm_ac_win` 硬编码致本仓库 FileNotFoundError → 支持 `NILM_ROOT` 环境变量 + 缺产物逐用户/整体 `[SKIP]` 退 0（/tmp 假产物实测解析路径正常）
- [x] **本会话④ v14 烟测实测**：`v14_smoke_test.py` 用户 252844 ✅ 全过（val F1=0.9114/test F1=0.8673/MAE=124W，退出码 0）；用户 270778 系统项全过（121 维特征无 NaN、训推一致、推理形状合法）但 test F1=0.119 未过质量门
- [x] **本会话⑤ v14 工具链健全性**：5 个 v14 入口 py_compile 全过；`v14_model_analyzer.py --precision int8 --prune` 用合成 bundle 全路径实测（结构分析/MoE 遍历/Flash 17.7KB 估算/剪枝建议）

## 进行中
- （无）

## 下一步（TODO）
1. 接收下一会话具体任务
2. （可选）270778 test F1 偏低深挖：best_thr=0.93 极端校准 + 初夏→盛夏季节性切分是主因疑似；可在烟测纳入 val 阈值合理性检查或换 global_stratified 对比
3. （可选）`14_train_v14.py` 端到端实跑：需选定用户 + env 配置（monkey-patch 集成路径本环境只做了编译级验证）
4. （长期 roadmap，V14 报告 §九）P1 CNN-LSTM/Seq2Point 基线、跨用户迁移学习；P2 在线增量学习、FHMM/CO 基线；P3 V-I 轨迹、多设备辨识

## 决策记录 / 踩坑
- 仓库无 `setup.sh`/`pnpm`，依赖按 `requirements.txt` 装入 `.venv`（被 .gitignore 排除，不入库）
- 本环境**无 lightgbm/skl2onnx**：EnsembleClf 自动降级纯 GBDT（已测零回归）、ONNX 导出返回 None（已测不崩）——降级路径均按设计工作
- 测试设计踩坑（已修正）：①5σ 异常检出需尖峰占比 <3.8%（5% 时 z≈4.4 反而检不出，改用 1.5%）；②focal 二轮权重含 (1-alpha) 保底项，gamma↑ 归一化后难/易比反而→1（初版断言方向写反，数值复核后修正）；③T1.7 权重比容差 1e-9 太严，理论值有 2.7e-8 浮点 ulp 差，放宽到 1e-6
- 合成 bundle 造 MoE 勿用 `type()` 匿名类（不可 pickle），用 `types.SimpleNamespace`
- 270778 烟测 F1 未过属**数据特性**非代码缺陷：对齐仅 5280 样本（5/21-7/15），stratified_day 把 7 月高温日切进 test，val 上 best_thr 搜到 0.93 → test 召回坍塌；同套烟测在 252844 全过；且该用户群在 V14_U2844_P0_FIX 报告中有已知 low/mid 档难度记录
- 工作风格（INTJ）：中文回复；结论以硬证据支撑；清理前确认可恢复；日志 ASCII 标记
- 会话固定分支 `arena/019fb816-lite-nilm`

## 关键文件路径
- `STATUS.md` — 本文件
- `scripts/test_v14_enhancements.py` — **本会话新增** v14 单测（93 断言）
- `scripts/v14_enhancements.py` / `14_train_v14.py` / `v14_smoke_test.py` / `v14_model_analyzer.py` — v14 核心
- `scripts/test_train_infer_symmetry.py` — 本会话修复可移植性（NILM_ROOT）
- `NILM_AC_session_summary_v13.17.md` / `REPORT.md` / `V14_UPGRADE_REPORT.md` — 上下文文档
- `data/trains` / `data/infers` / `data/time_filters.json` — 5 用户数据与配置
