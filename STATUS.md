# STATUS.md

> 会话续接状态文件。每次 session 开局读它恢复上下文，收尾更新它固化上下文（见 BOOTSTRAP.md）。
> 更完整的历史见 `NILM_AC_session_summary_v13.17.md` + `V14_UPGRADE_REPORT.md`。

## 当前目标
- NILM-AC：已从 v13.17 演进到 **v14**（方向②精度鲁棒性 / ③漂移小样本低算力 / ④算法架构 / ⑦工程化流水线 / ⑧特征工程，全部非侵入式、开关可关）
- 本会话（2026-07-31）：开局恢复现场完成，等待本 session 具体任务

## 已完成
- [x] v13.17 批量断点续跑 `--resume` + `batch_execution_state.csv` 原子写 + cleanup 白名单修复（37 组单测过）
- [x] v14 升级包：`scripts/v14_enhancements.py`（EnsembleClf / focal 加权 / 校准 / 小样本自动超参 / RunningStats / 数据质量诊断 / 训健度报告）、`14_train_v14.py` 包装入口、ONNX/INT8/m2cgen 导出 + 模型卡、烟测脚本
- [x] v14 方向⑧特征工程深挖：+36 维物理指纹特征默认启用；d87 推理路径特征注入 bug 修复
- [x] 环境恢复：`.venv`（Python 3.11.2 + numpy 1.26.4 + pandas 2.2.3 + sklearn 1.5.2，`requirements.txt` 全量装好）

## 进行中
- （无，等待任务指派）

## 下一步（TODO）
1. 接收本会话具体任务（可能方向：v14 单测补齐 / V14 报告遗留项 / 新用户数据接入）
2. 若涉及训练/评估，先跑 `scripts/v14_smoke_test.py` 烟测验证环境可用

## 决策记录 / 踩坑
- 仓库无 `setup.sh`/`pnpm`，依赖按 `requirements.txt` 装入 `.venv`（被 .gitignore 排除，不入库）
- 工作风格（INTJ）：中文回复；杜绝主观臆断，一切结论以代码/数据硬证据支撑；清理操作前先确认可恢复性；Windows GBK 环境下日志用 ASCII `[OK]/[SKIP]/[FAIL]`
- 会话固定分支 `arena/019fb816-lite-nilm`，不开新分支
- 项目原始路径 `/home/user/nilm_ac_win/`（见 v13.17 摘要）；本仓库为精简版 lite-nilm，数据在 `data/`（trains/infers/time_filters*.json）

## 关键文件路径
- `STATUS.md` — 本文件
- `NILM_AC_session_summary_v13.17.md` — v13.17 全量上下文摘要
- `REPORT.md` / `V14_UPGRADE_REPORT.md` — 项目总报告 / v14 升级报告
- `scripts/03_train.py` / `04_evaluate.py` / `05_inference.py` — 训/评/推主流程
- `scripts/14_train_v14.py` / `v14_enhancements.py` / `v14_smoke_test.py` — v14 增强
- `scripts/run_user_pipeline.py` / `run_batch_users.py` — 单用户/批量流水线
- `data/trains` / `data/infers` / `data/time_filters.json` — 数据与配置
