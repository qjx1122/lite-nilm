# STATUS.md

> 会话续接状态文件。每次 session 开局读它恢复上下文，收尾更新它固化上下文（见 BOOTSTRAP.md）。
> 更完整的历史见 `NILM_AC_session_summary_v13.17.md` + `V14_UPGRADE_REPORT.md`。

## 当前目标
- 已完成（2026-07-31 session 2）：**5 用户 v14 批量重跑 + 与上版对比 + 逐用户逐日问题详析** → 交付 `V14_RERUN_ANALYSIS.md`
- 已完成（2026-07-31/08-01 session 3）：**P0–P2 整改路线图执行 + 全量重验** → 交付 `V14_REMEDIATION_REPORT.md`
- 已完成（2026-08-01 session 4）：**推理集扩充 6 月训练窗外 OOD 验证 + 逐日详析 + 共性梳理** → 交付 `V14_JUNE_EXT_ANALYSIS.md`
- 已完成（2026-08-03 session 5）：**5 用户逐日训练+推理评估指标 × 数据质量合并视图 + 逐用户详析** → 交付 `V14_TRAIN_INFER_DAILY_ANALYSIS.md` + `artifacts/daily_train_infer_metrics_view.md`；途中破获 **v14.6 陈旧 pyc 竞态事故**（见下）
- 已完成（2026-08-03 session 5 续）：**5 用户汇总整体指标视图（分类×回归，训练×推理 micro-pooled）** → `scripts/build_overall_metrics_view.py` + `artifacts/overall_metrics_view.md`；池化口径与官方 `inference_metrics.csv` 逐位一致（Acc/F1 偏差 0.00000 交叉验证）；核心数字：分类 训练合并 F1=0.986 → 6月 0.744（P=0.625，FP=1646 虚块步）→ 7月 0.964；回归 训练 MAE 27.0W/SAE −3.8% → 6月 MAE 162.5W/+50.0% → 7月 124.1W/−15.1%
- 已完成（2026-08-03 session 7）：**U842 Windows/沙盒 LightGBM active 但 best_thr=0.74 vs 0.57 跨平台差异定位 + v14.7 阈值稳定化修复** → `postprocess.search_best_threshold` val-only 近似同分容忍带 + Recall/Precision 定向 tie-break；LightGBM deterministic 单线程/col-wise；新增 `scripts/test_postprocess_threshold_stability.py`。
- 已完成（2026-08-03 session 8）：**U842 训练/推理逐日 F1<90% 与 SAE>20% 异常日详析** → `U842_DAILY_FAILURE_ANALYSIS.md`；结论：训练侧无 ON 日 F1<90%，18 个 F1=0 均为 perfect-off 口径；训练 SAE>20% 3 天均为功率层；推理 ON 日 F1<90% 5 天（6/8、6/9、6/21、7/5、7/6），推理 SAE>20% 14 天，拆为分类漏检型/高功率低估型/低功率高估型。
- 已完成（2026-08-03 session 9）：**U842 P1/P2/P3 三方案 train+val-only 离线优化验证** → `scripts/experiment_u842_p1_p2_p3.py` + `U842_P1_P2_P3_OPTIMIZATION_REPORT.md`；P1 高湿 recall guard (`RH>=88%, p_on>=0.45, 09-22h`) OOD F1 0.9144→0.9172、FN 202→195、SAE 14.60%→14.07%，主要改善 7/3、7/4；P2 daily power mode scale OOD SAE 14.60%→14.07% 但低功率高估日略恶化；P3 温湿桶 OOD SAE 14.60%→14.51% 收益过小。三者均未减少异常日数量，P1 可灰度，P2/P3 暂不建议生产上线。
- 已完成（2026-08-03 session 10）：**按用户要求重做 P2 为 `mode_classifier + per-mode regressor` 并验证** → `scripts/experiment_u842_p1_p2_p3.py` 更新 + `U842_P2_MODE_MODEL_REPORT.md`；模式阈值来自 train+val 真 ON 三分位 564.29/747.70W，RF classifier + per-mode RF regressor；test MAE 27.67→24.15W、SAE 3.62%→3.09%；inference MAE 108.07→103.46W、SAE 14.60%→14.07%、kWh_pred 192.09→193.27，但 SAE>20% 天数 14→15，6/8/7/6 分类整日漏检无效，6/19 低功率长时高估仍严重。结论：方向优于 daily scale，但需日级 loss-aware 目标 + 总线/段级模式特征后再作为生产候选。
- 已完成（2026-08-03 session 11）：**P2 加入日级 loss-aware 目标 + 增强模式判别特征 + 与 P1 分离验证** → `U842_P2_LOSSAWARE_MODE_REPORT.md`；新增日内分段/低置信计数/连续ON段/rolling/预测区间等特征，候选用 train-only fit + val objective、最终 train+val 重训；train_val MAE 11.01→5.46W、test MAE 27.67→24.82W，但 inference OOD 退化：MAE 108.07→114.92W、SAE 14.60%→18.42%、SAE>20%天 14→17、kWh_pred 192.09→183.49。结论：loss-aware 版本验证失败，不能上线。
- 已完成（2026-08-03 session 12）：**P2 引入真正原始总线/段级特征验证** → `U842_P2_RAWBUS_SEGMENT_REPORT.md`；使用 raw `load_iden_data73/1/74/2/79/7/75/5/77/3/76/4` 15min值 + diff/rolling + 日级raw统计 + baseline预测ON段内raw统计，P2仍与P1分离且不改分类；test MAE 27.67→21.95W、SAE 3.62%→2.49%；inference MAE 108.07→93.22W、SAE 14.60%→7.49%、kWh_pred 192.09→208.07、SAE>20%天 14→11。风险：6/19/6/27 低功率长时ON高估加重，6/8/7/6分类整日漏检仍无效。
- 已完成（2026-08-03 session 13）：**P2 rawbus + train/val-only 安全闸 + P1 recall guard 组合灰度验证** → `U842_P2_RAWBUS_SAFETY_P1_REPORT.md`；安全闸为 train+val 日级 RF 判断 rawbus 是否优于 baseline（label 35/15, 阈值0.5），仅用预测/天气/raw73日级与ON段统计；P2_rawbus_safety inference SAE 7.49%→7.99%、SAE>20天 11→10（拦住7/11但损失部分总量收益）；组合 `P1_plus_P2_rawbus_safety`：F1 0.9144→0.9172、Recall 0.8560→0.8610、FN 202→195、SAE 14.60%→7.45%、MAE 108.07→91.67W、SAE>20天 14→9。
- 已完成（2026-08-03 session 14）：**低功率长时风险闸 + partial-day coverage闸 + P1低概率梅雨guard 验证** → `U842_P2_RISK_COVERAGE_P1_LOWPROB_REPORT.md`；coverage闸回退 coverage<0.9 且rawbus上抬日（修6/5）；低功率长时风险闸回退 rawbus上抬且0.45≤p_on_q50≤0.60、RH≤85、pred_on_n≥40（拦6/19/6/27 rawbus恶化但baseline高估仍在）；P1低概率梅雨guard 两支路（低温RH≥80/temp≤22/p_ge02≥20/raw73_core≥1800用train+val ON p01=196W；暖湿RH≥85/temp≥25/p_ge02≥35/raw73_core≥2300用train+val median=681.6W）修复6/8/7/6整日漏检。最终 `P1_lowprob_plus_P2_risk` inference：F1 0.9144→0.9569、Recall 0.8560→0.9330、FN 202→94、SAE 14.60%→3.10%、MAE 108.07→78.86W、SAE>20天 14→8；仍不建议全量上线，需灰度监控6/8功率高估、6/19低功率高估。
- 已完成（2026-08-04 session 15）：**A/B0 baseline vs A/B1 P2_rawbus_safety_cov_lowrisk vs A/B2 P1_lowprob_plus_P2_risk 灰度监控验证** → `U842_AB_GRAY_MONITOR_REPORT.md`；test 上 A/B1=A/B2 SAE 3.62%→2.97%、MAE 27.67→22.70W；inference 上 A/B1 只改功率 SAE 14.60%→8.60%、MAE 108.07→91.60W、SAE>20天 14→10；A/B2 同时修分类+功率 F1 0.9144→0.9569、Recall 0.8560→0.9330、FN 202→94、SAE 14.60%→3.10%、MAE 108.07→78.86W、SAE>20天 14→8。四类监控结论：低温梅雨低功率 6/8 分类修复但功率仍高估；低功率长时 6/19/6/27 风险闸回退rawbus但baseline高估仍在；coverage<0.9 partial-day 6/5被回退避免恶化；全OFF高湿6/23无误触发。
- 已完成（2026-08-04 session 16）：**当前优化版本 A/B2 在 U842 上的全阶段/每日验证测试与每日指标详析** → `U842_CURRENT_OPTIMIZED_DAILY_VALIDATION_REPORT.md`；先强制重训 U842 (`batch_run_20260804_013059`) 再重跑实验。A/B2 当前 inference F1=0.9569、Precision=0.9820、Recall=0.9330、SAE=3.10%、MAE=78.86W、kWh_pred=217.95、FN=94；全阶段每日表已列 train/val/test/inference 每天 F1/P/R/SAE/kWh/TP/FP/FN/TN。重点日：6/8 分类修复但 SAE 92.1%，6/19/6/27 风险闸回退rawbus但baseline高估仍在，6/23 全OFF高湿无误触发。
- 已完成（2026-08-04 session 17）：**按用户要求排除 6/5、6/25，聚焦 U842 四个异常日 6/8、6/9、6/21、7/5 做专项归因与方案验证** → `U842_TARGET_DAYS_0608_0609_0621_0705_ANALYSIS.md`；结论：6/8=低温梅雨低功率，A/B2 分类已修但 p01=196W 功率锚过高，建议低温锚改 train+val ON p005=113W（diagnostic SAE 92.1%→10.6%）；6/9=低功率边界FN，当前能耗已好 SAE=2.7%，若 F1 硬门槛可低功率边界guard（diagnostic F1 0.837→1.0, SAE 18.8%）；6/21=暖湿高功率 partial ON，需暖湿partial guard（diagnostic F1 0.438→1.0, SAE 73.3%→15.0%）；7/5=晚间尾段FN，暖湿tail guard（diagnostic F1 0.860→0.990, SAE 14.4%→11.8%）。声明 diagnostic candidate 仅量化上限，生产参数仍需 train/val-only 选型。
- 已完成（2026-08-04 session 18）：**按 P0→P1→P2→P3 优先级依次修复并重新验证四个重点日** → `scripts/experiment_u842_priority_fixes.py` + `U842_PRIORITY_FIXES_VALIDATION_REPORT.md`；P0 低温锚 p005=113W 修 6/8（F1 0→0.990, SAE 100%→10.6%）；P1 暖湿 partial 修 6/21（F1 0.438→0.990, SAE 79.8%→13.4%）；P2 warm tail 修 7/5（F1 0.860→0.980, SAE 18.5%→13.8%）；P3 lowpower edge 修 6/9（F1 0.837→0.990, SAE 31.7%→20.0%边界）。最终 inference F1=0.9794、Recall=0.9815、SAE=0.735%、MAE=66.58W、FN=26；test 不退化（F1=0.9885, SAE=2.97%）。注意：这是 focused sequential 修复上限验证，P1/P2/P3 guard 生产化仍需 train/val-only 泛化选型。
- 已完成（2026-08-04 session 19）：**使用当前 P0→P3 优化版本重新验证 U842 所有配置数据并逐日分析** → `U842_OPTIMIZED_ALL_DATA_VALIDATION_REPORT.md`；覆盖 train/val/test/inference 全阶段日级表。整体：train F1=1.000/SAE=0.30%、val F1=0.987/SAE=0.17%、test F1=0.9885/SAE=2.97%、inference F1=0.9794/Recall=0.9815/SAE=0.735%/MAE=66.58W/kWh_pred=226.57/FN=26。异常：train/val ON 日无 F1<90 或 SAE>20；test 仍 4/24、6/2 两个 SAE>20；inference 仍 6/19、6/27、7/1 等 5 个 SAE>20，核心未愈为低功率长时 baseline 高估与 6/9 SAE 19.99% 边界。
- 已完成（2026-08-04 session 20）：**U842 2026-06-19 低功率长时 ON 高估专项分析** → `U842_20260619_LOWPOWER_LONG_ON_ANALYSIS.md`；硬证：分类几乎完美 F1=0.9901、TP/FP/FN/TN=50/0/1/45，但 true_on_mean≈203W、pred_on_mean≈763W、kWh 2.60→9.54、SAE=266.6%。raw73_on_mean≈2692 导致 rawbus 也倾向上抬，风险闸回退 baseline 正确但 baseline 自身高估。train+val 低功率长时样本仅 6/4(196W,RH80)、6/10(292W,RH59)、6/11(279W,RH58)，无“raw bus 高+baseline 高+true≈200W”同型强样本。建议下一步：低功率长时专用 cap/anchor，RH>=80 用≈200W，RH<80 用≈300~325W；或 long_low_power_mode classifier + per-mode regressor，需补样本/灰度验证。
- 下一会话：待指派（候选：P0 no-positive day 报表口径 / 将 P0-P3 guards 纳入 train-val-only 泛化选型 / P1 lowprob 梅雨guard 灰度配置化 / P2 低功率长时ON cap/anchor 修6/19&6/27 / P3 补桶样本后重估 / P-CE1 泛化守卫 / P-CE5 guard 滑窗锚 / P-CE6 日报 coverage 列 / 8 月数据回流）

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

## 已完成（session 5 合并视图 + pyc 事故破获）
- [x] **v14.6 陈旧 pyc 竞态破获（重大）**：04:14 批跑 U2844 指标异常（7月 kWh_true 138.56 vs 锚 84.16）一度误判为"分路标签数据版本翻转"；逐层硬证（原始 CSV 四列逐日核算 → 锚点=**p2 列指纹** 84.16/6.05/2.47/5.62 全等 → 配置 `target_col=p2` → 批日志对齐段统计 峰值892W/零样本72.3% = **p1 指纹**（p2 应为 899W/77.2%））坐实为 **CPython .pyc 校验只看 (mtime 秒, size)**：run_user_pipeline.main() 在 patch_common 前先 `from common import ON_THR_W`（无 --common-overrides 用户必经），同秒同尺寸改写 common.py 致 02 子进程吃陈旧 pyc。U842(p1→p1 无害)、U0778/U0789/U0800(有 overrides 或尺寸变更多幸免) 逐一对号入座。**无任何数据文件翻转**。
- [x] **修复**：`patch_common`/`restore_common` 写入后 `_purge_common_pyc()` 删 `__pycache__/common.*.pyc`；另修 run_batch_users 未传 `--time-filter-config` 时 `_power_temp_calib_json` 的 UnboundLocalError（变量初始化前置）
- [x] **回归测试**：`scripts/test_patch_common_pyc_fix.py` 9 断言全过（含同秒碰撞构造复现：修复前子进程读 p1/修复后读 p2）
- [x] **修复后全量复跑**（475.2s 5/5 ok）：U2844 7月 kWh_true=84.16、7/5=6.045、F1=0.9721 全归锚；5/5 用户与 v14.3 终验链逐位一致 → **既有 V14 三份报告数字无需勘误**（它们本就走 p2/正确链）
- [x] **主交付**：5 用户 train/val/test + 推理(6月扩段+7月) 逐日合并视图生成器 `scripts/build_daily_train_infer_view.py`（SAE 比率口径修正 + 满采/缺口质量卡）；`V14_TRAIN_INFER_DAILY_ANALYSIS.md` 逐用户详析（训练侧/推理6月/推理7月分段 + 数据质量交叉 + 共性 CE1+/CE3+/NEW-Q1~Q4）
- [x] **新证据链**：口径错位 FP 训练侧首发（U2844 val 6/4 FP12、test 5/22 FP6 ↔ 推理 6/28 FP38）；训练侧 SAE 离群日=推理档位失败前哨（U842 6/2→7/1-4；U0789 5/28→7/13-15）；U2844 8 个"全零无信息日"需单独归类；训练标签结构（无 OFF 日×3 用户/U0800 仅 6 天）决定推理失败模式

## 已完成（session 6 U842 跨环境指标差异诊断）
- [x] **用户截图（外来运行）U842 四行汇总 vs 本地 0479b12 链差异根因实锤 = 对方环境缺 LightGBM**：N 逐格全等（4490/1536/1440/2780 → 数据/窗口/target 完全一致）；受控复现 `pip uninstall lightgbm` 后单跑 U842 → test F1 0.9835748792 与截图 0.9835740 七位一致（=理论值 2·509/(2·509+2+15)），inference TN/FP=1364/13 逐格一致，FN 403 vs 421（±18 为库版本噪声），kWh_pred 159.8 vs 157.4；机理链：EnsembleClf 在无 lightgbm 时优雅退化为单 GBDT（lgb_weight=0.4 集成 → 单模型），best_thr 0.57→0.75，样本内几乎不动（train 仅 1 点），OOD 推理 Recall 0.856→0.70（FN 202→421 量级），kWh_pred 192→157（−18%）
- [x] **meta 显形闸**：`model_meta.json` 新增 `clf_class` / `ensemble_lgb_active`（03_train.py L1205 附近，3 行零行为变化）＋ test_v14_enhancements T4.9/T4.y 2 断言；排查口诀：对比两边 model_meta.json 该字段即知是否同型
- [x] 附注：截图 inference 行 status=`ok:main`（本地 `ok:main_final`）提示对方汇总组表器/代码版本亦偏旧；不影响根因判定
- [x] 受控复现后已无损恢复：reinstall lightgbm + /tmp 备份还原 U842 artifacts/models，锚复核 7月 138.56/0.9322 [OK]

## 已完成（session 7 U842 Windows/沙盒 best_thr 跨平台稳定化）
- [x] **新证据修正**：用户 Windows 更新至 `95274bd` 后 `clf_class=EnsembleClf`、`ensemble_lgb_active=True`、`pickle_lgb_active=True`，但 `best_thr=0.74`；三重 hash 诊断显示关键数据/脚本均为 `[OK] EOL_ONLY_CRLF`，即语义内容与 HEAD 一致，raw hash 差异仅 Windows CRLF。故根因从"LightGBM 未激活"细化为：**同代码/数据下 Windows/Conda/LightGBM/OpenMP 概率微漂移触发 val 阈值阶梯函数跳档**。
- [x] **直接量化**：沙盒 LightGBM active 链 `best_thr=0.57`；若仅把阈值抬到 0.74，U842 inference 从 `TN/FP/FN/TP=1354/23/202/1201, F1=0.9144, kWh_pred=192.09` 退化到约 `1364/13/416/987, F1≈0.821, kWh_pred≈158`，与用户 Windows `1364/13/421/982, F1=0.8190, kWh_pred=157.37` 对齐；推理集中 `0.57<=p_on<0.74` 约 220 点，其中真实 ON 约 211 点，是 OOD 放大器。
- [x] **v14.7 修复**：`scripts/postprocess.py::search_best_threshold` 增加 val-only 稳定化：raw 最优 F_beta 下方 `tol=min(0.002,max(1e-4,2/n_val))` 视为有限验证集近似同分；`beta>=1` 在候选内优先 Recall、再 Precision、再低阈值，`beta<1` 维持 Precision 优先。该规则只依赖 val 样本数和 val 曲线，不读取推理集/7月标签，符合参数纪律。
- [x] **复现闸**：`scripts/v14_enhancements.py::EnsembleClf` 的 LightGBM 改为 `n_jobs=1, deterministic=True, force_col_wise=True` 并固定 seed 族，降低 Windows/Linux histogram/OpenMP 非 bitwise 概率漂移；`03_train.py` 写入 `raw_best_thr/raw_best_fbeta/selected_fbeta/threshold_stability_tol/threshold_selection_policy` 与 `runtime_versions` 便于事后排查。
- [x] **沙盒回归锚**：v14.7 后 U842 单用户强制重训 (`batch_run_20260803_104313`) 仍回到 `best_thr=0.57`、`ensemble_lgb_active=True`、inference `F1=0.9143509707 / Recall=0.8560228083 / kWh_pred=192.0877048`；新增 `threshold_curve_val.csv` 含 tn/fp/fn/tp 列，top plateau 0.57~0.62；`runtime_versions` 已写入 meta。
- [x] **单测**：新增 `scripts/test_postprocess_threshold_stability.py`（10 断言）覆盖 raw 高阈值近似同分但 Recall 更高时选低阈值、关闭稳定化保留旧行为、beta<1 Precision 优先、完全平台阶取左端、元数据自洽。

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
- **⚠️ 批跑必须带 `--time-filter-config data/time_filters.json`**：不带则 per-user target_col/过滤全失效且（v14.6 前）直接 UnboundLocalError；target_col 唯一真源是该配置的 `target_col`（U842=p1, U2844=p2, U0778=p2, U0789=p1+p2, U0800=p1）
- **⚠️ 陈旧 pyc 竞态（v14.6 已修）**：同秒同尺寸改写 scripts/common.py 会让 02/03/04/05/06 子进程吃到旧 TARGET_COL 的 .pyc；事故指纹 = 批日志对齐段「峰值/零样本占比」与配置列不符；修复 = `run_user_pipeline._purge_common_pyc()`（patch/restore 后删除 common.*.pyc）+ `test_patch_common_pyc_fix.py` 兜底
- **⚠️ U842 阈值跳档（v14.7 已修）**：Windows raw hash 与沙盒不同先判 CRLF；若三重 hash 全 `[OK] EOL_ONLY_CRLF` 且 `ensemble_lgb_active=True`，重点查 `best_thr/raw_best_thr/threshold_selection_policy`。U842 0.74 高阈值会把 `0.57<=p_on<0.74` 的 200+ 推理 ON 点压成 FN；修复 = val-only 近似同分 Recall 优先 tie-break + LightGBM deterministic 单线程。
- U842 与 U2844 共用分路表 4206894986488（两拷贝 md5 不同但重叠日一致）：U842 用 p1、U2844 用 p2；U2844 文件 6/19-6/23 斜杠日期、p1 在 6/20/6/22 与 2025-07 全段为 NaN
- U842 与上版对比必须**按 ≥2026-07-01 重切**（本次推理 n=2780=6月验证段1340+7月1440，上版表 3.1 仅 7 月；真值 kWh 138.6 两侧一致口径自洽）
- U842 7/6 排查路径：排除数据事故（原始 d73/分路/气象缓存正常）→ p_on 崩塌在分类层 → 168 维全维扫描定位天气族 → 湿度 Middling 陷阱（ON 窗湿度 93 vs 日均 88；日均口径否决"湿度独因"假设，转向温差/cooling_degree 联合签名）→ 6/21 同签名复现确认 P8
- 5σ 异常/focal gamma 测试设计踩坑记录见 git log（session 1 commit）

## 关键文件路径
- `STATUS.md` — 本文件 | `V14_TRAIN_INFER_DAILY_ANALYSIS.md` — **最新主交付**（训练+推理逐日×数据质量详析）| `artifacts/daily_train_infer_metrics_view.md` — 全量逐日合并视图 | `scripts/build_daily_train_infer_view.py` — 视图生成器 | `scripts/test_patch_common_pyc_fix.py` — pyc 竞态回归
- `V14_JUNE_EXT_ANALYSIS.md` — 6 月扩段 OOD+共性梳理
- `V14_REMEDIATION_REPORT.md` — P0-P2 整改验证 | `V14_RERUN_ANALYSIS.md` — v14 基线对比+逐日详析+路线图 | `V14_BATCH_COMPARISON.md` / `V14_DAILY_METRICS_ANALYSIS.md` — 上版对比基线
- `data/time_filters.json` — 5 用户最终配置（U2844 bus_guard+direction=both；3 用户 power_temp_calib+calib_stats_include）
- `scripts/power_temp_calib.py` — 温桶标定/时段先验（含 direction=both cap=p90 对称模式）
- `results_v6_15_0/` — 一致性测试符号链接农场（已 .gitignore，供 test_train_infer_symmetry 实跑）
- `artifacts/infers/<user>/inference_daily_metrics.csv` / `inference_result.csv` — 逐日/逐点
- `logs/_batch/batch_run_20260731_151530.log` — 批跑日志（bus_guard 统计在此）
- `models/<user>/nilm_ac_two_stage.pkl` — v14 bundle（clf=EnsembleClf）
- `scripts/test_v14_enhancements.py` — v14 单测 93 断言（session 1）
