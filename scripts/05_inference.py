# -*- coding: utf-8 -*-
"""
Step 5: 独立推理脚本 (v5 升级: 支持多基线模型对比)

功能:
    1) 从 CSV 加载总线数据 (必需) 和分路标签 (可选)
    2) 加载训练好的主模型 .pkl
    3) 可选加载一个或多个基线模型并行推理
    4) 输出推理结果 CSV (含所有模型预测)
    5) 若提供分路标签, 输出各模型评估指标 + 透视对比表
    6) 无标签时, 输出主模型 vs 基线模型的"一致性指标"

命令行示例 (Windows):
    # 基础: 单模型推理 (向后兼容 v4 行为)
    python scripts\\05_inference.py

    # v5 新增: 主模型 + 内置基线对比
    python scripts\\05_inference.py --baseline rf fallback
    python scripts\\05_inference.py --baseline rf naive_mean naive_zero

    # v5 新增: 加载外部 pkl 做对比 (如 v4.2 对照)
    python scripts\\05_inference.py --baseline models\\nilm_ac_two_stage_v42.pkl

    # 混合模式: 内置 + 外部 同时对比
    python scripts\\05_inference.py --baseline rf fallback models\\nilm_ac_two_stage_v42.pkl

    # 完整命令: 新数据 + 标签 + 多基线
    python scripts\\05_inference.py ^
        --bus     data\\new_bus.csv ^
        --branch  data\\new_branch.csv ^
        --model   models\\nilm_ac_two_stage.pkl ^
        --baseline rf fallback models\\nilm_ac_two_stage_v42.pkl ^
        --out     artifacts\\predictions\\result.csv ^
        --metric-out artifacts\\metrics\\metrics.csv

支持的基线别名:
    rf           : 单阶段 RandomForest (来自主 bundle)
    fallback     : MoE 全局兜底回归器 (无季节路由对照)
    naive_mean   : 训练集 ON 均值 (随机性下限)
    naive_zero   : 全 0 预测 (理论下限)
    <path>.pkl   : 外部模型 bundle 文件 (如 v4.2)
"""
import argparse
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from common import (INFER_BUS_CSV, INFER_BR_CSV,    # v6.12.6+v6.15.0 推理路径独立
                    BUS_CSV, BR_CSV,                # 兼容: 训练路径仍然可读
                    MODEL_PKL, PRED_DIR, METRIC_DIR,
                    ARTIFACT_DIR,
                    ON_THR_W,             # v6.12.6 单一阈值
                    TARGET_COL,           # v13.16 复合列物化需要
                    WEATHER_LATITUDE, WEATHER_LONGITUDE, WEATHER_CACHE_DIR,
                    SUMMER_TEMP_THRESHOLD, WINTER_TEMP_THRESHOLD,
                    setup_chinese_font, get_logger, Timer)
from feature_utils import (load_bus_csv, load_branch_csv,
                           resample_and_align, build_features,
                           assert_no_nan_features, align_features_to_bundle)
from postprocess import apply_postprocess
from weather_utils import get_weather_for_period
from expert_utils import assign_season
from baseline_utils import (BaselineRegistry, BaselineRunner,
                            merge_predictions, cross_model_consistency)
from residual_calibrator import ResidualCalibrator, ModelSwitcher
from metrics_utils import (compute_classification_metrics,
                           compute_regression_metrics,
                           save_predictions_csv,
                           flatten_metrics_to_rows,
                           save_metrics_csv,
                           build_comparison_table,
                           build_leak_ood_metric_rows,
                           build_daily_metrics_rows,
                           save_daily_metrics_csv,
                           compute_raw_daily_counts)  # v13.16

log = get_logger("infer")


def parse_args():
    p = argparse.ArgumentParser(description="NILM 独立推理脚本 (含多基线对比)")
    p.add_argument("--bus",    type=str, default=str(INFER_BUS_CSV),
                   help=f"总线 CSV 路径 (默认 INFER_BUS_CSV={INFER_BUS_CSV})")
    p.add_argument("--branch", type=str, default=str(INFER_BR_CSV),
                   help=f"分路标签 CSV 路径 (默认 INFER_BR_CSV={INFER_BR_CSV}, 可选)")
    p.add_argument("--model",  type=str, default=str(MODEL_PKL),
                   help="主模型 .pkl 路径")
    p.add_argument("--baseline", nargs="*", default=[],
                   help="基线模型列表 (rf/fallback/naive_mean/naive_zero/<.pkl 路径>)")
    p.add_argument("--out",    type=str,
                   default=str(PRED_DIR / "inference_result.csv"),
                   help="推理结果 CSV 输出路径")
    p.add_argument("--metric-out", type=str,
                   default=str(METRIC_DIR / "inference_metrics.csv"),
                   help="评估指标 CSV (仅当 --branch 存在)")
    p.add_argument("--no-branch", action="store_true",
                   help="强制不使用分路标签")
    p.add_argument("--plot", action="store_true",
                   help="生成多模型功率曲线对比图 (artifacts/inference_comparison.png)")
    p.add_argument("--no-calib", action="store_true",
                   help="禁用 L4 残差校正层 (即使模型 bundle 含 calibrator)")
    p.add_argument("--no-switch", action="store_true",
                   help="禁用 L5 多模型动态切换 (强制使用主模型)")
    # [v12] 新增: 推理侧时段过滤 (include/exclude 组合)
    p.add_argument("--time-filter-spec", default="",
                   help="[v12] JSON 字符串, 推理数据时段过滤规格 "
                        "{'include':[[start,end],...],'exclude':[[start,end],...]}. "
                        "闭区间, 支持任意时段 (非整天粒度)")
    return p.parse_args()


def main():
    args = parse_args()
    bus_path    = Path(args.bus)
    branch_path = Path(args.branch) if not args.no_branch else None
    model_path  = Path(args.model)
    out_path    = Path(args.out)
    metric_out  = Path(args.metric_out)
    baselines   = list(args.baseline) if args.baseline else []

    log.info("=" * 70)
    log.info("Step 5: 独立推理 (v5: 多基线对比)")
    log.info(f"  总线 CSV  : {bus_path}")
    log.info(f"  分路 CSV  : {branch_path if branch_path else '<未提供>'}")
    log.info(f"  主模型    : {model_path}")
    log.info(f"  基线模型  : {baselines if baselines else '<无, 仅主模型>'}")
    log.info(f"  结果 CSV  : {out_path}")
    log.info(f"  指标 CSV  : {metric_out if branch_path else '<跳过>'}")
    log.info("=" * 70)

    # ---------- 1. 校验文件 ----------
    if not bus_path.exists():
        log.error(f"总线 CSV 不存在: {bus_path}"); return 1
    if not model_path.exists():
        log.error(f"模型文件不存在: {model_path}"); return 1
    has_label = branch_path is not None and branch_path.exists()
    if branch_path is not None and not branch_path.exists():
        log.warning(f"分路 CSV 不存在, 跳过评估: {branch_path}")

    # ---------- 2. 加载主模型 ----------
    with Timer("加载主模型", log):
        bundle = joblib.load(model_path)
        scaler   = bundle["scaler"]
        clf      = bundle["clf"]
        reg      = bundle["reg"]
        reg_low  = bundle.get("reg_low")
        reg_high = bundle.get("reg_high")
        moe      = bundle.get("moe")
        quantile_alpha = float(bundle.get("quantile_alpha", 0.5))
        quantile_low   = float(bundle.get("quantile_low",  0.10))
        quantile_high  = float(bundle.get("quantile_high", 0.90))
        top_cols = bundle["feat_cols"]
        best_thr = float(bundle["best_thr"])
        # [一致性检查] 与 03_train.py 常量对齐:
        #   POST_MIN_ON=1 -> fallback=1 ✓ 一致
        #   POST_FILL_SHORT_OFF=3 -> fallback=0 ⚠️ 不一致 (仅当加载 v6.10 前的老 bundle 缺此键时才触发,
        #     当前生成的 bundle 一定含此键, 实际推理值为 3, 与训练一致. 保留 0 以兼容极老 bundle 行为)
        post_min_on         = int(bundle.get("post_min_on", 1))
        post_fill_short_off = int(bundle.get("post_fill_short_off", 0))
        model_ver = bundle.get("version", "v1")
        log.info(f"  模型训练时间: {bundle.get('trained_at', 'unknown')}  ({model_ver})")
        log.info(f"  Stage-1 阈值: {best_thr:.3f}")
        log.info(f"  期望特征列  : {len(top_cols)} (Top-25 原始电参量)")
        log.info(f"  后处理: min_on={post_min_on}, fill_short_off={post_fill_short_off}")

    # ---------- 3. 加载数据 ----------
    with Timer("加载总线数据", log):
        bus, all_cols = load_bus_csv(bus_path)
        log.info(f"  总线 shape={bus.shape}")
    # v6.12.a: 排除由 resample_and_align 自动生成的 5min 极值衍生列
    _DERIVED_SUFFIXES = ("_max5", "_min5", "_absmax5")
    missing = [c for c in top_cols
               if c not in bus.columns and not c.endswith(_DERIVED_SUFFIXES)]
    if missing:
        log.error(f"总线 CSV 缺少模型所需列: {missing[:5]}... (共 {len(missing)} 列)")
        return 1

    branch = None
    if has_label:
        with Timer("加载分路数据", log):
            # [v13.16] 传 TARGET_COL 让 load_branch_csv 按需物化复合列
            branch = load_branch_csv(branch_path, target_col=TARGET_COL)
            log.info(f"  分路 shape={branch.shape}")
            if TARGET_COL and "+" in TARGET_COL:
                log.info(f"  [v13.16] 已物化复合列 '{TARGET_COL}' = "
                         f"{' + '.join(TARGET_COL.split('+'))} 逐行求和")

    # ---------- 3b. [v12] 时段过滤 (在对齐前) ----------
    try:
        from time_filter_utils import cli_arg_to_spec, apply_time_filter, spec_summary
        _tf_spec = cli_arg_to_spec(args.time_filter_spec)
        if _tf_spec is not None:
            log.info(f"  [v12 推理时段过滤] 规格: {spec_summary(_tf_spec)}")
            bus = apply_time_filter(bus, "event_time", _tf_spec, "infer_bus", logger=log)
            if branch is not None:
                branch = apply_time_filter(branch, "time", _tf_spec, "infer_branch", logger=log)
            if len(bus) == 0:
                log.error("[v12 时段过滤] 过滤后总线为空, 无法推理")
                return 1
    except ImportError:
        pass

    # ---------- 4. 对齐 + 特征工程 ----------
    with Timer("重采样 + 对齐 + 特征工程", log):
        if has_label:
            df = resample_and_align(bus, branch, keep_cols=top_cols)
            log.info(f"  对齐后样本(含标签): {len(df)}")
        else:
            df = resample_and_align(bus, branch_df=None, keep_cols=top_cols)
            log.info(f"  对齐后样本(无标签): {len(df)}")
        if len(df) == 0:
            log.error("对齐后无数据, 请检查时间范围与字段")
            return 1

        # v5: 拉取气象 + 计算季节标签 (与训练一致)
        use_weather_features = bool(bundle.get("use_weather_features", False))
        use_temp_based_season = bool(bundle.get("use_temp_based_season", False))
        weather_df = None
        if use_weather_features:
            with Timer("[v5] 加载气象 (自动缓存)", log):
                weather_df = get_weather_for_period(
                    latitude=bundle.get("weather_latitude", WEATHER_LATITUDE),
                    longitude=bundle.get("weather_longitude", WEATHER_LONGITUDE),
                    start_ts=df.index.min(), end_ts=df.index.max(),
                    cache_dir=WEATHER_CACHE_DIR, logger=log,
                )
                log.info(f"  气象 shape={weather_df.shape}, "
                         f"温度范围 [{weather_df['temperature_2m'].min():.1f}, "
                         f"{weather_df['temperature_2m'].max():.1f}] °C")

        # v6: 从 bundle 取出训练时保存的 LUT, 注入漂移特征
        temp_power_lut = bundle.get("temp_power_lut")
        if temp_power_lut:
            log.info(f"  [v6] 复用训练 LUT 构造漂移特征 "
                     f"({len([k for k in temp_power_lut if isinstance(k, tuple)])} 个桶)")
        X_df = build_features(df, top_cols, weather_df=weather_df,
                              temp_power_lut=temp_power_lut)
        log.info(f"  特征 shape={X_df.shape}")

        # [v14.3] 训练/推理特征对齐: 以 bundle["feat_names"] 为唯一口径
        #   (支持训练端特征黑名单 NILM_EXCLUDE_FEATURES, 详见 feature_utils.align_features_to_bundle)
        X_df = align_features_to_bundle(X_df, bundle, logger=log, ctx="05_inference")
        
        # v6 L2: 漂移检测告警
        if temp_power_lut and weather_df is not None:
            from drift_detect import detect_drift
            drift_report = detect_drift(df, top_cols,
                                        weather_df=weather_df,
                                        temp_power_lut=temp_power_lut,
                                        logger=log)
            drift_path = METRIC_DIR / "drift_report.csv"
            drift_path.parent.mkdir(parents=True, exist_ok=True)
            drift_report.to_csv(drift_path, index=False, encoding="utf-8-sig")
            log.info(f"  [v6] 漂移检测报告 -> {drift_path}")

            # v13.15: 逐桶 "训练期望 vs 推理实测" 明细导出, 便于按桶精确定位漂移
            from drift_features import export_temp_power_actual_vs_expected_csv
            _ave_path = METRIC_DIR / \
                "inference_temp_power_actual_vs_expected.csv"
            export_temp_power_actual_vs_expected_csv(
                df, top_cols, weather_df, temp_power_lut,
                _ave_path, logger=log)

        # 季节标签
        if use_temp_based_season and weather_df is not None:
            daily_t = weather_df["temperature_2m"].resample("D").mean()
            ts_dates = pd.DatetimeIndex(df.index).normalize()
            daily_avg_temp = daily_t.reindex(ts_dates, method="ffill").values
            season_labels = assign_season(
                df.index, daily_avg_temp=daily_avg_temp,
                use_temperature=True,
                summer_th=bundle.get("summer_temp_threshold", SUMMER_TEMP_THRESHOLD),
                winter_th=bundle.get("winter_temp_threshold", WINTER_TEMP_THRESHOLD),
            )
        else:
            season_labels = assign_season(df.index, use_temperature=False)

    # ---------- 5. 主模型推理 ----------
    with Timer("主模型推理 (MoE + 后处理)", log):
        # [v13.7] NaN 硬检测: X_df 有 NaN 会让 sklearn GBM 直接崩 (270758 案例根因)
        # 检测到会 WARN 打印列名/时间戳定位 + 主动 raise, 避免掉进 sklearn 报错栈
        assert_no_nan_features(X_df, stage_name="inference", logger=log,
                               raise_on_nan=True)
        X = X_df.values.astype(np.float32)
        X_s = scaler.transform(X)
        p_on = clf.predict_proba(X_s)[:, 1]
        raw_state = (p_on >= best_thr).astype(int)

        if moe is not None:
            p_reg  = np.clip(moe.predict(X_s, season_labels, alpha=quantile_alpha), 0, None)
            y_low  = np.clip(moe.predict(X_s, season_labels, alpha=quantile_low),   0, None)
            y_high = np.clip(moe.predict(X_s, season_labels, alpha=quantile_high),  0, None)
            log.info(f"  使用季节分层 MoE 推理 ({model_ver})")
        else:
            p_reg = np.clip(reg.predict(X_s), 0, None)
            y_low  = (np.clip(reg_low.predict(X_s),  0, None)
                      if reg_low  is not None else None)
            y_high = (np.clip(reg_high.predict(X_s), 0, None)
                      if reg_high is not None else None)
            log.info(f"  使用全局回归器 (老模型)")

        state_pred, y_pred = apply_postprocess(
            raw_state, p_reg,
            min_on=post_min_on, fill_short_off=post_fill_short_off,
        )
        if y_low  is not None: y_low  = y_low  * state_pred
        if y_high is not None: y_high = y_high * state_pred

        log.info(f"  主模型推理完成, 平均功率 {y_pred.mean():.2f} W")
        log.info(f"  raw_ON={int(raw_state.sum())}, "
                 f"postproc_ON={int(state_pred.sum())} / {len(state_pred)}")

        # ---------- 5a. 日级 d87 启动签名守卫 (v6.12.2 自适应版) ----------
        # v6.12.2 改进 (替代 v6.12.1 的固定阈值):
        #   1. 用 d87.min() 不再 abs() -- 启动是负向尖峰, 区别于正向噪声
        #   2. 训练阈值 = train_off_P1 × safety_factor (训练用户基准)
        #   3. [*] 推理时按 user_d73_p95 / train_d73_p95 比例缩放,
        #      物理依据: d87 启动尖峰 ∝ 空调功率, d73_p95 反映用户电力规模
        # 实测效果 (842 + 270848):
        #   - 842  (d73_p95=2813): 缩放因子 0.956 -> 阈值 -91.8 -> 守卫 100% 准确
        #   - 270848 (d73_p95=612): 缩放因子 0.208 -> 阈值 -20.0 -> 守卫 100% 准确
        D87_COL = "load_iden_data87"
        D73_COL = "load_iden_data73"   # v6.12.2: 主功率列, 用作缩放锚点
        _d87_meta = bundle.get("d87_guard_meta", {})
        if _d87_meta.get("enabled", False):
            threshold_base = float(_d87_meta.get("threshold_base",
                                                  _d87_meta.get("threshold", -96.0)))
            train_d73_p95 = _d87_meta.get("train_d73_p95")
            adaptive_scaling = _d87_meta.get("adaptive_scaling", False)
            scale_min = float(_d87_meta.get("scale_min", 0.05))
            scale_max = float(_d87_meta.get("scale_max", 2.0))

            # v6.12.4: 显示标定方法 (用于诊断)
            calibration = _d87_meta.get("calibration", "v6.12.2_off_p1_x_sf")
            if calibration == "v6.15.0_adaptive":
                calib_brief = (
                    f"v6.15.0 自适应: ON×AF({_d87_meta['allow_factor']:.3f})={_d87_meta['on_constraint']:.1f}, "
                    f"OFF×MF({_d87_meta['margin_factor']:.3f})={_d87_meta['off_constraint']:.1f}, "
                    f"软最大w={_d87_meta.get('softmax_weight', 1.0):.2f}, "
                    f"绑定={_d87_meta['binding_constraint']}, "
                    f"概率融合={'on' if _d87_meta.get('prob_fusion_enabled') else 'off'}"
                )
            elif calibration == "v6.12.4_dual_source":
                calib_brief = (
                    f"v6.12.4 双源: ON_P10×{_d87_meta['allow_factor']}={_d87_meta['on_constraint']:.0f}, "
                    f"OFF_P99×{_d87_meta['margin_factor']}={_d87_meta['off_constraint']:.0f}, "
                    f"绑定={_d87_meta['binding_constraint']}"
                )
            else:
                calib_brief = "旧 v6.12.2 (OFF_P1×SF)"

            # 计算推理用户的 d73_p95 (自适应缩放锚点)
            if (adaptive_scaling and train_d73_p95
                    and D73_COL in bus.columns and bus[D73_COL].notna().any()):
                user_d73_p95 = float(bus[D73_COL].quantile(0.95))
                raw_scale = user_d73_p95 / train_d73_p95
                # 限制缩放因子上下界, 避免极端值 (例如全无空调用户)
                scale = float(np.clip(raw_scale, scale_min, scale_max))
                D87_GUARD_TH = threshold_base * scale
                _guard_source = (
                    f"d73 自适应缩放 (user_d73_p95={user_d73_p95:.0f} / "
                    f"train_d73_p95={train_d73_p95:.0f} = {raw_scale:.3f}, "
                    f"clip->{scale:.3f}) × {threshold_base:.0f} = {D87_GUARD_TH:.1f}; "
                    f"标定={calib_brief}"
                )
            else:
                # 退化到训练用户基准阈值
                D87_GUARD_TH = threshold_base
                _guard_source = (
                    f"训练用户基准阈值 = {D87_GUARD_TH:.0f}; 标定={calib_brief}"
                )
        else:
            # 兼容旧版 bundle 无 d87_guard_meta 字段时的退化
            D87_GUARD_TH = -50.0
            _guard_source = "退化默认 (-50, bundle 无 d87 元数据)"

        # [v11] d87 自适应守卫总开关: bundle['d87_guard_meta']['enabled']=False 时
        # 跳过整个 5a 步级状态机块 + 后续 6a 基线压制块 (mask_no_startup 未定义 -> 自动跳过)
        # 这允许训练侧通过 D87_ADAPTIVE_GUARD_ENABLED=False 完全关闭 d87 压制机制,
        # 让模型/后处理直接决定 ON/OFF (适用于变频空调 d87 尖峰退化的场景)
        _d87_guard_run = bool(_d87_meta.get("enabled", False))
        if not _d87_guard_run:
            log.info("  [v11] d87 守卫已关闭 (bundle['d87_guard_meta']['enabled']=False), "
                     "跳过 5a 步级状态机 + 6a baseline 压制")

        if _d87_guard_run and D87_COL in bus.columns and "event_time" in bus.columns:
            # v6.12.6: 步级状态机守卫 (替代 v6.12.3/4/5 的日级守卫)
            # 物理依据 (用户1 5/21 等典型样本):
            #   - 当天 d87 启动点在 10:31 出现 (|d87|=80)
            #   - 但模型在 09:15~10:15 已经预测 ~110W (错误: 真实仅 21W 非空调)
            #   - 日级守卫无法处理这种"启动前不识别"的边界
            # 设计:
            #   对每个 15min 推理步:
            #     1. 在 5min bus 中查找 ≤ ts+15min 范围内最近的启动点
            #     2. 该启动点之前的步 -> 强制 OFF (启动前不可能是空调)
            #     3. 启动点之后 MAX_ON_HOURS 小时内 -> 信任模型预测
            #     4. 超出 MAX_ON_HOURS -> 强制 OFF (典型空调一次运行 ≤ 12h)
            _bus = bus[["event_time", D87_COL]].copy()
            _bus["event_time"] = pd.to_datetime(_bus["event_time"])
            _bus = _bus.sort_values("event_time").set_index("event_time")
            # ============================================================
            # v6.15.0 概率融合守卫 (方案 C)
            # 思想: 5min 启动点判定时, 引入对应 15min 步的模型概率 p_on
            #       p_on 高 -> 局部阈值降低 -> 边界 |d87| 信号也能放行 (修复 v6.14.2 卡边界问题)
            #       p_on 低 -> 局部阈值保持 -> 严守不退让 (避免 FP)
            #       局部阈值 = D87_GUARD_TH × (1 - gamma × p_on), 但不低于 min_ratio
            # ============================================================
            prob_fusion = bool(_d87_meta.get("prob_fusion_enabled", False))
            prob_gamma  = float(_d87_meta.get("prob_gamma", 0.30))
            prob_min_ratio = float(_d87_meta.get("prob_min_ratio", 0.60))
            guard_abs_th = abs(D87_GUARD_TH)

            if prob_fusion and len(p_on) > 0:
                # 把每个 5min bus 点映射到所属 15min 步的 p_on
                # df.index 是 15min 推理步时间戳, p_on 长度 = len(df)
                # 对每个 5min ts, 找其落入的 15min 窗口 (向下取整到 15min)
                # 该步的 p_on 即为该 5min 点的模型置信度
                # 把每个 5min ts 映射到 15min 步索引 (用 datetime64 高效计算)
                df_index_arr = pd.DatetimeIndex(df.index).values.astype('datetime64[ns]')
                bus_ts_arr   = pd.DatetimeIndex(_bus.index).values.astype('datetime64[ns]')
                if len(df_index_arr) > 0:
                    df_start = df_index_arr[0]
                    step_min = 15.0  # 推理步长 (与 RESAMPLE 一致)
                    delta_min = (bus_ts_arr - df_start) / np.timedelta64(1, 'm')
                    step_idx = (delta_min // step_min).astype(int)
                    step_idx = np.clip(step_idx, 0, len(p_on) - 1)
                    p_on_at_bus = p_on[step_idx]   # 每个 5min ts 对应的 p_on
                else:
                    p_on_at_bus = np.zeros(len(_bus))
                # 局部阈值: 每个 5min ts 一个 (按 p_on 缩放)
                local_factor = np.clip(1.0 - prob_gamma * p_on_at_bus, prob_min_ratio, 1.0)
                local_th = guard_abs_th * local_factor
                startup_mask = _bus[D87_COL].abs().values >= local_th
                n_startup_p_boosted = int(((local_factor < 1.0) & startup_mask).sum())
                log.info(
                    f"  [d87 守卫 v6.15.0 概率融合] 阈值范围 "
                    f"[{local_th.min():.1f}, {local_th.max():.1f}], "
                    f"概率提升放行点数={n_startup_p_boosted}"
                )
            else:
                # 退化: 不使用概率融合 (兼容旧 bundle)
                startup_mask = (_bus[D87_COL].abs() >= guard_abs_th).values

            startup_ts = _bus.index[startup_mask].tolist()
            n_startup_total = len(startup_ts)

            # v6.12.6 步级状态机守卫 (原始版, MAX_ON_HOURS 硬编码 12h)
            # 对每个 15min 推理步, 找其窗口 [ts, ts+15min] 内最近的过去启动点:
            #   1. 启动点之前 -> 强制 OFF (启动前不可能是空调)
            #   2. 启动点之后 MAX_ON_HOURS(12h) 内 -> 信任模型预测
            #   3. 超出 MAX_ON_HOURS -> 强制 OFF (典型空调一次运行 ≤ 12h)
            MAX_ON_HOURS = float(_d87_meta.get("max_on_hours", 12.0))
            STARTUP_LOOKAHEAD_MIN = 15.0

            mask_no_startup = np.zeros(len(df), dtype=bool)
            df_ts = df.index.to_pydatetime() if hasattr(df.index, "to_pydatetime") \
                    else df.index.tolist()
            startup_arr = np.array([pd.Timestamp(s) for s in startup_ts])

            for i, ts in enumerate(df_ts):
                ts = pd.Timestamp(ts)
                window_end = ts + pd.Timedelta(minutes=STARTUP_LOOKAHEAD_MIN)
                past_startups = startup_arr[startup_arr <= window_end] \
                                if len(startup_arr) > 0 else np.array([])
                if len(past_startups) == 0:
                    mask_no_startup[i] = True
                else:
                    last_startup = past_startups[-1]
                    gap_hours = (ts - last_startup).total_seconds() / 3600.0
                    if gap_hours > MAX_ON_HOURS:
                        mask_no_startup[i] = True

            guard_dropped = int((mask_no_startup & (state_pred == 1)).sum())
            n_steps_suppressed = int(mask_no_startup.sum())
            n_steps_passed = len(mask_no_startup) - n_steps_suppressed

            log.info(
                f"  [d87 守卫 v6.12.6 步级状态机] |阈值|={guard_abs_th:.0f} ({_guard_source})"
            )
            log.info(
                f"  [d87 守卫] 5min 启动点总数: {n_startup_total}, "
                f"MAX_ON_HOURS={MAX_ON_HOURS}h"
            )
            log.info(
                f"  [d87 守卫] 步级判定: 信任模型 {n_steps_passed} 步 (放行), "
                f"强制 OFF {n_steps_suppressed} 步 (压制)"
            )
            if guard_dropped > 0:
                state_pred[mask_no_startup] = 0
                y_pred[mask_no_startup] = 0.0
                p_reg[mask_no_startup] = 0.0
                p_on[mask_no_startup] = 0.0
                if y_low is not None:
                    y_low[mask_no_startup] = 0.0
                if y_high is not None:
                    y_high[mask_no_startup] = 0.0
                log.info(
                    f"  [d87 守卫] 实际压制 ON 步={guard_dropped}, "
                    f"压制后 postproc_ON={int(state_pred.sum())}"
                )
            else:
                log.info(f"  [d87 守卫] 模型预测的所有 ON 步均在启动后 {MAX_ON_HOURS}h 内, 无需压制")
        else:
            log.warning(f"  [d87 守卫] 跳过 (缺列 {D87_COL} 或 event_time)")
            mask_no_startup = None    # 后续 baseline 守卫扩展将检查此变量

        # ---------- 5a2. 总线一致性守卫 + 功率标定 (v14.1 P0) ----------
        # 在 d87 步级守卫之后、L4 之前执行:
        #   - FP: pred ON 但总线抬升不足以支撑预测功率 -> 强制 OFF
        #   - 日级: 当日 inconsistent 比例过高 -> 整日清零 (OFF 日误报)
        #   - FN: pred OFF 但总线抬升进入空调功率带且持续 -> 强制 ON
        #   - 功率: 可信 ON 上用 lift/y_pred 中位比做有界缩放
        # 开关优先级: bundle['bus_guard_meta'].enabled > env NILM_BUS_GUARD_*
        try:
            from bus_consistency_guard import (
                apply_bus_consistency_guard,
                build_bus_power_15min_from_bus_df,
                default_meta_enabled_for_user,
                DEFAULT_META as _BG_DEFAULT,
            )
            import json as _json_bg
            import os as _os_bg

            _bg_meta = None
            # 1) bundle 内嵌
            if isinstance(bundle.get("bus_guard_meta"), dict):
                _bg_meta = dict(bundle["bus_guard_meta"])
            # 2) 环境变量 JSON 覆盖/补充
            _env_json = _os_bg.environ.get("NILM_BUS_GUARD_JSON", "").strip()
            if _env_json:
                try:
                    _env_meta = _json_bg.loads(_env_json)
                    if isinstance(_env_meta, dict):
                        if _bg_meta is None:
                            _bg_meta = dict(_BG_DEFAULT)
                            _bg_meta.update(_env_meta)
                        else:
                            _bg_meta.update(_env_meta)
                except Exception as _e_bg:
                    log.warning(f"  [bus_guard] NILM_BUS_GUARD_JSON 解析失败: {_e_bg}")
            # 3) 简易开关 env
            _env_en = _os_bg.environ.get("NILM_BUS_GUARD_ENABLED", "").strip().lower()
            if _env_en in ("1", "true", "yes", "on"):
                if _bg_meta is None:
                    _bg_meta = dict(_BG_DEFAULT)
                _bg_meta["enabled"] = True
            elif _env_en in ("0", "false", "no", "off"):
                if _bg_meta is None:
                    _bg_meta = dict(_BG_DEFAULT)
                _bg_meta["enabled"] = False

            if _bg_meta is not None and _bg_meta.get("enabled", False):
                # v14.1.1: 传原始 5min 总线 (d87 日级签名) + 回归功率基底
                # p_reg 在后处理前是完整回归输出; 此处 y 已乘 state, 用 p_reg 作 raw reg
                _reg_base = np.asarray(p_reg, dtype=float).copy()
                state_pred, y_pred, p_on, _bg_info = apply_bus_consistency_guard(
                    state_pred, y_pred, p_on,
                    bus_power_15min=None,
                    timestamps=df.index,
                    meta=_bg_meta,
                    logger=log,
                    bus_df=bus,
                    y_pred_raw_reg=_reg_base,
                )
                # 同步分位带与 p_reg
                if y_low is not None:
                    y_low = np.asarray(y_low, dtype=float) * state_pred
                if y_high is not None:
                    y_high = np.asarray(y_high, dtype=float) * state_pred
                p_reg = np.asarray(p_reg, dtype=float) * state_pred
                # 若召回步 p_reg 为 0, 用 y_pred 回填
                _miss = (state_pred == 1) & (p_reg <= 1)
                if np.any(_miss):
                    p_reg[_miss] = y_pred[_miss]
                log.info(
                    f"  [bus_guard] 完成: postproc_ON={int(state_pred.sum())}/{len(state_pred)}, "
                    f"meanW={float(y_pred.mean()):.2f}"
                )
            else:
                log.info("  [bus_guard] 未启用 (bundle/env 均未打开)")
        except Exception as _bg_exc:
            log.warning(f"  [bus_guard] 执行失败, 忽略并继续: {_bg_exc}")
            import traceback as _tb_bg
            log.debug(_tb_bg.format_exc())

    # ---------- 5b. L4 残差校正 (在主预测上加性校正) ----------
    # v6.8 重构: 三条预测轨并行保留, 便于评估 L4/L5 各自的实际收益
    #   y_pred_raw_main   : 无 L4 无 L5 (原始 MoE 预测) -> 对应 train_val 中的 main
    #   y_pred_after_L4   : 有 L4 无 L5                  -> 对应 main_L4_calib
    #   y_pred (最终)     : 有 L4 有 L5                  -> 对应 main_final (生产输出)
    y_pred_raw_main = y_pred.copy()
    y_pred_after_L4 = y_pred.copy()    # 默认 = 无校正版本 (若 L4 未启用)
    residual_calib = bundle.get("residual_calib")
    use_calib = bundle.get("use_residual_calib", False) and \
                residual_calib is not None and \
                not args.no_calib
    if use_calib and residual_calib._trained:
        log.info("-" * 70)
        log.info("[L4] 应用残差校正层")
        # 计算近期信号 (与训练时一致)
        if top_cols and top_cols[0] in df.columns:
            top1 = df[top_cols[0]].astype(float)
            recent_24h = top1.rolling(window=96, min_periods=4,
                                      closed="left").mean().bfill().values
        else:
            recent_24h = np.zeros(len(df))
        delta = residual_calib.predict_delta(
            y_pred, df.index, weather_df, recent_24h, season_labels,
            state_pred=state_pred,
        )
        y_pred_after_L4 = np.clip(y_pred + delta, 0, None)
        y_pred = y_pred_after_L4.copy()   # 继续传给后续 L5
        log.info(f"  [L4] 校正量统计: 均值={delta.mean():+.2f}W, "
                 f"中位={np.median(delta):+.2f}W, "
                 f"max={delta.max():+.2f}W, min={delta.min():+.2f}W")
        log.info(f"  [L4] 校正前后平均功率: {y_pred_raw_main.mean():.2f} -> "
                 f"{y_pred.mean():.2f} W")
    else:
        if args.no_calib:
            log.info("[L4] --no-calib 指定, 跳过残差校正")
        elif residual_calib is None or not getattr(residual_calib, "_trained", False):
            log.info("[L4] bundle 不含已训练 L4 校正器, 跳过残差校正")

    # ---------- 6. 基线模型推理 (v5 新增) ----------
    baseline_results = {}
    if baselines:
        log.info("-" * 70)
        log.info(f"【基线对比】运行 {len(baselines)} 个基线模型")
        with Timer("基线模型推理", log):
            registry = BaselineRegistry(bundle, logger=log)
            runner   = BaselineRunner(registry, logger=log)
            baseline_results = runner.run_all(
                baselines=baselines,
                df_aligned=df,
                top_cols=top_cols,
                X_main_scaled=X_s,
                state_pred_main=state_pred,
                weather_df=weather_df,
            )
        log.info(f"  实际运行成功的基线数: {len(baseline_results)}")
        # ---------- 6a. d87 守卫扩展到所有 baseline (v6.12.1) ----------
        # 主模型已在 5a 应用守卫; baseline 在此同步压制, 保证 L5 加权对齐
        # v6.12.1: 5a 段可能因缺列将 mask_no_startup 设为 None, 此处需检查
        _mask_avail = ("mask_no_startup" in dir()
                       and mask_no_startup is not None)
        if D87_COL in bus.columns and _mask_avail:
            for name, info in baseline_results.items():
                y_b = info["y_pred"]
                if len(y_b) == len(mask_no_startup):
                    n_drop = int((mask_no_startup & (y_b > 0)).sum())
                    if n_drop > 0:
                        y_b_new = y_b.copy()
                        y_b_new[mask_no_startup] = 0.0
                        info["y_pred"] = y_b_new
                        # 同步压制 state (若 baseline 有提供)
                        if "state_pred" in info and info["state_pred"] is not None \
                                and len(info["state_pred"]) == len(mask_no_startup):
                            info["state_pred"][mask_no_startup] = 0
                        log.info(f"    [{name:<20}] [d87 守卫] 压制 {n_drop} 步")
        for name, info in baseline_results.items():
            y_b = info["y_pred"]
            log.info(f"    [{name:<20}] 平均功率 {y_b.mean():>6.1f}W  "
                     f"max {y_b.max():>6.1f}W  ({info['model'].description})")

    # ---------- 6b. L5 多模型动态切换 ----------
    y_pred_pre_switch = y_pred.copy()   # 保留切换前的预测
    switch_decision = None
    if not args.no_switch:
        # 优先用 v4.2 (外部 pkl) 作为 fallback, 若用户传了
        fb_name = None
        for name in baseline_results:
            if "v42" in name or "v4_2" in name or "baseline" in name.lower():
                fb_name = name; break
        # 否则用 fallback 别名 (MoE 全局兜底)
        if fb_name is None and "fallback" in baseline_results:
            fb_name = "fallback"

        if fb_name is not None:
            log.info("-" * 70)
            log.info(f"[L5] 多模型动态切换 (备选 fallback = {fb_name})")
            # 读取漂移报告
            drift_report = None
            try:
                drift_path = METRIC_DIR / "drift_report.csv"
                if drift_path.exists():
                    drift_report = pd.read_csv(drift_path)
            except Exception:
                pass

            switcher = ModelSwitcher(main_bundle=bundle, fallback_bundle={},
                                     logger=log)
            # v6.9: 把 L4 实际启用状态传给 L5, 决定主模型权重保留多少
            calib_active = bool(use_calib and residual_calib is not None and
                                getattr(residual_calib, "_trained", False))
            switch_decision = switcher.decide(drift_report,
                                              calib_active=calib_active)

            if switch_decision["use_fallback"] and switch_decision["main_weight"] < 1.0:
                y_fb = baseline_results[fb_name]["y_pred"]
                y_pred = ModelSwitcher.blend(
                    y_pred, y_fb, switch_decision["main_weight"]
                )
                log.info(f"  [L5] 已执行切换: 主权重={switch_decision['main_weight']}, "
                         f"切换前主均值={y_pred_pre_switch.mean():.1f}W -> "
                         f"切换后均值={y_pred.mean():.1f}W")
        else:
            log.info("[L5] 未找到合适的 fallback 模型, 跳过切换")

    # ---------- 6c. [v14.2] 推理端可选标定 (时段先验抑制 + 功率温桶标定) ----------
    # 开关: env NILM_TIME_PRIOR_JSON / NILM_POWER_TEMP_CALIB_JSON
    #   (run_batch_users 从 time_filters.json 用户级配置注入, 默认关闭零回归)
    # 参数纪律: LUT/先验全部来自 bundle 内训练侧统计, 不接触任何推理期标签,
    #   保证评估集零泄漏 (V14_RERUN_ANALYSIS.md P4/P5/P3 整改项)
    try:
        from power_temp_calib import (apply_time_prior_suppress,
                                      apply_power_temp_calib)
        import json as _json_c
        import os as _os_c

        # 6c-1. 时段先验抑制 (先压 FP, 再算功率标定)
        _env_tpr = _os_c.environ.get("NILM_TIME_PRIOR_JSON", "").strip()
        if _env_tpr:
            _cfg_tpr = None
            try:
                _cfg_tpr = _json_c.loads(_env_tpr)
            except Exception as _e_tpr:
                log.warning(f"  [v14.2 time_prior] JSON 解析失败: {_e_tpr}; raw={_env_tpr[:120]}")
            if isinstance(_cfg_tpr, dict) and _cfg_tpr.get("enable", False):
                _prior = bundle.get("hourly_on_prior")
                if _prior:
                    state_pred, _info_tpr = apply_time_prior_suppress(
                        state_pred, p_on, df.index, _prior,
                        low_rate=float(_cfg_tpr.get("low_rate", 0.01)),
                        p_req=float(_cfg_tpr.get("p_req", 0.9)), logger=log)
                    if _info_tpr.get("n_suppressed", 0) > 0:
                        y_pred = np.asarray(y_pred, dtype=float) * state_pred
                        if y_low is not None:
                            y_low = np.asarray(y_low, dtype=float) * state_pred
                        if y_high is not None:
                            y_high = np.asarray(y_high, dtype=float) * state_pred
                else:
                    log.info("  [v14.2 time_prior] bundle 无 hourly_on_prior, 跳过")

        # 6c-2. 功率温桶标定 (lift-only; state 不再变化)
        _env_ptc = _os_c.environ.get("NILM_POWER_TEMP_CALIB_JSON", "").strip()
        if _env_ptc:
            _cfg_ptc = None
            try:
                _cfg_ptc = _json_c.loads(_env_ptc)
            except Exception as _e_ptc:
                log.warning(f"  [v14.2 power_temp_calib] JSON 解析失败: {_e_ptc}; raw={_env_ptc[:120]}")
            if isinstance(_cfg_ptc, dict) and _cfg_ptc.get("enable", False):
                _lut = bundle.get("branch_temp_power_lut")
                if _lut and (weather_df is not None):
                    y_pred, _info_ptc = apply_power_temp_calib(
                        y_pred, state_pred, df.index, weather_df, _lut,
                        gamma=float(_cfg_ptc.get("gamma", 0.85)),
                        stat=str(_cfg_ptc.get("stat", "p50")),
                        min_gain=float(_cfg_ptc.get("min_gain", 1.05)),
                        direction=str(_cfg_ptc.get("direction", "lift")),
                        cap_stat=str(_cfg_ptc.get("cap_stat", "p90")),
                        logger=log)
                elif not _lut:
                    log.info("  [v14.2 power_temp_calib] bundle 无 LUT (训练时无气象), 跳过")
                else:
                    log.info("  [v14.2 power_temp_calib] 推理侧无气象数据, 跳过")
    except Exception as _e_v142:
        log.warning(f"  [v14.2] 可选标定执行失败, 忽略并继续: {_e_v142}")

    # ---------- 7. 保存推理结果 CSV (含所有基线预测) ----------
    y_true = df["y_ac"].values.astype(np.float32) if has_label else None
    # v6.12.6 单口径: OOD 评估用 ON_THR_W=10W (与训练标签同口径)
    on_thr_eval = float(bundle.get("ON_THR", ON_THR_W))
    s_true = (y_true >= on_thr_eval).astype(int) if has_label else None
    if has_label:
        log.info(f"  [v6.12.6] OOD 评估 ON 阈值 = {on_thr_eval:.0f}W (单口径)")
    # v6.8: 三层预测并存写入 CSV
    #   y_pred_W_main           = 最终生产输出 (含 L4+L5, 保持原列名兼容下游)
    #   y_pred_W_main_raw       = 原始主模型预测 (无 L4 无 L5), 对照基线
    #   y_pred_W_main_L4_calib  = 仅 L4 校正后 (不含 L5), 单独评估 L4 收益
    extra_cols_dict = {
        "state_pred_main": state_pred.astype(int),
        "p_on_main": np.round(p_on, 4),
        "y_pred_W_main_raw": np.round(y_pred_raw_main, 3),
        "y_pred_W_main_L4_calib": np.round(y_pred_after_L4, 3),
        **({"state_true": s_true} if has_label else {}),
        **({"y_pred_low_W_main":  np.round(y_low,  3)} if y_low  is not None else {}),
        **({"y_pred_high_W_main": np.round(y_high, 3)} if y_high is not None else {}),
    }
    if has_label:
        extra_cols_dict["residual_W_main_raw"] = np.round(y_pred_raw_main - y_true, 3)
        extra_cols_dict["residual_W_main_L4_calib"] = np.round(y_pred_after_L4 - y_true, 3)
    out_df = merge_predictions(
        main_pred=y_pred,
        baseline_results=baseline_results,
        timestamps=df.index,
        y_true=y_true,
        extra_cols=extra_cols_dict,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  ✓ 推理结果 -> {out_path}  ({len(out_df)} 行 × {len(out_df.columns)} 列)")

    # ---------- 8. 评估指标 (有标签 -> 多模型完整指标 + 透视对比) ----------
    if has_label:
        log.info("-" * 70)
        log.info("【多模型评估】")
        all_metric_rows = []

        # v6.12.6 单口径分类指标 (用 ON_THR_W=10W, 与训练/评估同口径)
        cls_main = compute_classification_metrics(s_true, state_pred, p_on)
        extra_base = {"threshold": best_thr,
                      "bus_csv": bus_path.name,
                      "branch_csv": branch_path.name,
                      "model_file": model_path.name,
                      "model_version": model_ver}

        # ---- main: 无 L4 无 L5 (原始主模型预测) ----
        reg_main_raw = compute_regression_metrics(y_true, y_pred_raw_main)
        log.info(f"  [main]              分类: F1={cls_main['F1']:.4f}, "
                 f"Precision={cls_main['Precision']:.4f}, "
                 f"Recall={cls_main['Recall']:.4f}")
        log.info(f"  [main]              回归: MAE={reg_main_raw['MAE_W']:.2f}W, "
                 f"RMSE={reg_main_raw['RMSE_W']:.2f}W, "
                 f"SAE={reg_main_raw['SAE']*100:.2f}%, "
                 f"kWh真/预={reg_main_raw['kWh_true']:.2f}/"
                 f"{reg_main_raw['kWh_pred']:.2f}")
        all_metric_rows += flatten_metrics_to_rows(
            "inference", "main",
            cls_metrics=cls_main, reg_metrics=reg_main_raw,
            extra={**extra_base, "note": "无 L4 无 L5, 原始主模型预测"},
            source="inference",
        )

        # ---- main_L4_calib: 仅 L4 校正 (若启用), 不含 L5 ----
        if use_calib and getattr(residual_calib, "_trained", False):
            reg_main_calib = compute_regression_metrics(y_true, y_pred_after_L4)
            log.info(f"  [main_L4_calib]     回归: MAE={reg_main_calib['MAE_W']:.2f}W, "
                     f"RMSE={reg_main_calib['RMSE_W']:.2f}W, "
                     f"SAE={reg_main_calib['SAE']*100:.2f}%, "
                     f"kWh真/预={reg_main_calib['kWh_true']:.2f}/"
                     f"{reg_main_calib['kWh_pred']:.2f}")
            all_metric_rows += flatten_metrics_to_rows(
                "inference", "main_L4_calib",
                cls_metrics=cls_main, reg_metrics=reg_main_calib,
                extra={**extra_base, "note": "L4 残差校正后, 不含 L5"},
                source="inference",   # v6.11
            )

        # ---- main_final: L4 + L5 (生产实际输出, 与下游 CSV 主列一致) ----
        # 若 L5 未触发 (主权重=1) 且 L4 未启用, 则 main_final == main, 仍单独输出便于对齐
        reg_main_final = compute_regression_metrics(y_true, y_pred)
        log.info(f"  [main_final]        回归: MAE={reg_main_final['MAE_W']:.2f}W, "
                 f"RMSE={reg_main_final['RMSE_W']:.2f}W, "
                 f"SAE={reg_main_final['SAE']*100:.2f}%, "
                 f"kWh真/预={reg_main_final['kWh_true']:.2f}/"
                 f"{reg_main_final['kWh_pred']:.2f}")
        l5_note = "L5 已切换" if (switch_decision is not None
                                  and switch_decision.get("use_fallback")
                                  and switch_decision.get("main_weight", 1.0) < 1.0) \
                  else "L5 未切换 (NORMAL)"
        all_metric_rows += flatten_metrics_to_rows(
            "inference", "main_final",
            cls_metrics=cls_main, reg_metrics=reg_main_final,
            extra={**extra_base, "note": f"生产输出 (L4+L5); {l5_note}"},
            source="inference",   # v6.11
        )

        # 基线模型
        for name, info in baseline_results.items():
            y_b = info["y_pred"]
            # v6.13: 基线评估用 BUSINESS 阈值 (与 s_true 一致)
            s_b = (y_b >= on_thr_eval).astype(int)
            cls_b = compute_classification_metrics(s_true, s_b, y_b)
            reg_b = compute_regression_metrics(y_true, y_b)
            log.info(f"  [{name:<20}] 分类: F1={cls_b['F1']:.4f}, "
                     f"Precision={cls_b['Precision']:.4f}, Recall={cls_b['Recall']:.4f}")
            log.info(f"  [{name:<20}] 回归: MAE={reg_b['MAE_W']:.2f}W, "
                     f"RMSE={reg_b['RMSE_W']:.2f}W, SAE={reg_b['SAE']*100:.2f}%")
            all_metric_rows += flatten_metrics_to_rows(
                "inference", name, cls_metrics=cls_b, reg_metrics=reg_b,
                extra={"baseline_kind": info["model"].kind,
                       "description": info["model"].description},
                source="inference",   # v6.11
            )

        # [v13.8] 数据泄漏检测 + 拆分指标 (仅在有标签且 bundle 保存了 train_dates 时)
        # 场景: 用户 infer 时段配置忘记排除训练区间, 导致推理集含 train_dates 交集样本.
        # 输出: WARN 日志 + 追加两组指标行 (split=inference_leak / inference_ood),
        #       让业务方看清 "整体 SAE" 里泄漏部分和真 OOD 部分各自的表现.
        # [v13.8-fix1] extra_model_preds 覆盖 main_L4_calib + main_final, 让业务方能对比
        #       L4/L5 收益在泄漏 vs OOD 部分的差异 (通常 L5 切换在 OOD 上收益更大).
        _train_dates_from_bundle = bundle.get("train_dates", None)
        if _train_dates_from_bundle is not None and len(_train_dates_from_bundle) > 0:
            try:
                # 组装 extra_model_preds: 只加真正与 main 不同的变体
                _extra_preds = {}
                if use_calib and getattr(residual_calib, "_trained", False):
                    _extra_preds["main_L4_calib"] = y_pred_after_L4
                # main_final 恒生成 (即使 L4/L5 都关, 也要有对比行, 与主流程 5 处 flatten 对齐)
                _extra_preds["main_final"] = y_pred

                _leak_rows, _ood_rows, _leak_meta = build_leak_ood_metric_rows(
                    timestamps=df.index,
                    y_true=y_true,
                    y_pred_main=y_pred_raw_main,  # main = 原始 MoE (无 L4 无 L5)
                    s_true=s_true,
                    s_pred_main=state_pred,       # 3 个变体共用同一 state_pred
                    p_on_main=p_on,               # 主模型 ON 概率
                    train_dates_set=set(_train_dates_from_bundle),
                    extra_base={**extra_base,
                                "note_leak_source": "v13.8_leak_split"},
                    source="inference",
                    logger=log,
                    extra_model_preds=_extra_preds,
                )
                all_metric_rows += _leak_rows
                all_metric_rows += _ood_rows
            except Exception as _leak_e:
                log.warning(f"  [v13.8 泄漏检测] 失败, 忽略并继续主流程: {_leak_e}")
        else:
            log.info(f"  [v13.8 泄漏检测] 跳过 (bundle 未含 train_dates, "
                     f"可能是旧版模型未升级; 重训后自动启用)")

        # 长表保存 (v6.10: 默认 append=True, 完整保留每次推理的历史指标)
        merged_inf = save_metrics_csv(all_metric_rows, metric_out, append=True)
        log.info(f"  ✓ 评估指标长表 -> {metric_out}  "
                 f"(本次追加 {len(all_metric_rows)} 行, 文件累计 {len(merged_inf)} 行)")

        # ---------- [v13.14] 推理侧逐日主模型评估指标 ----------
        # 用主模型 main_final (y_pred = L4+L5 生产实际输出) 逐日计算 F1/SAE/kWh
        # [v13.14+] date_labels: 用 v13.8 train_dates 标记数据泄漏日 (若存在)
        #   used_leak    : 该推理日出现在训练集中 (泄漏, 指标偏乐观)
        #   used_ood     : 真 OOD 推理日 (无泄漏, 代表真实泛化)
        log.info("[v13.14] 生成推理阶段逐日主模型指标")
        try:
            _train_dates_set = set(bundle.get("train_dates", []) or [])
            _inf_date_labels = None
            if _train_dates_set:
                _inf_date_labels = {}
                _uniq_dates = pd.to_datetime(pd.Series(df.index)).dt.strftime("%Y-%m-%d").unique()
                for _d in _uniq_dates:
                    _inf_date_labels[_d] = "used_leak" if _d in _train_dates_set else "used_ood"

            # [v13.16] 预计算推理侧总线/分路每天原始采集点数.
            # 应用 --time-filter-spec 让口径与实际参与推理的天数完全一致
            # (推理常配 infer.exclude 排除训练日, 若不过滤会把训练日的点也算进来误导)
            _inf_bus_counts = compute_raw_daily_counts(
                bus_path, "event_time",
                time_filter_spec=args.time_filter_spec if args.time_filter_spec else None,
                logger=log)
            _inf_br_counts = {}
            if branch_path is not None and branch_path.exists():
                _inf_br_counts = compute_raw_daily_counts(
                    branch_path, "time",
                    time_filter_spec=args.time_filter_spec if args.time_filter_spec else None,
                    logger=log)

            _inf_daily = build_daily_metrics_rows(
                df.index, y_true, y_pred, s_true, state_pred,
                split_name="inference",
                on_thr_w=on_thr_eval,
                p_on=p_on,
                date_labels=_inf_date_labels,
                model_name="main_final",
                extra={"project_version": bundle.get("version", ""),
                       "model_file": model_path.name,
                       "bus_csv": bus_path.name},
                bus_daily_counts=_inf_bus_counts,      # [v13.16]
                branch_daily_counts=_inf_br_counts,    # [v13.16]
            )
            _inf_daily_path = METRIC_DIR / "inference_daily_metrics.csv"
            save_daily_metrics_csv(_inf_daily, _inf_daily_path, logger=log)
        except Exception as _e:
            log.warning(f"  [v13.14] 日级指标计算失败, 忽略: {_e}")
            import traceback; traceback.print_exc()

        # 透视对比表 (新增)
        if baseline_results:
            key_metrics = ["F1", "Precision", "Recall", "Accuracy",
                           "MAE_W", "RMSE_W", "SAE", "NDE",
                           "kWh_true", "kWh_pred", "kWh_err"]
            pivot = build_comparison_table(all_metric_rows,
                                           include_metrics=key_metrics)
            comp_path = metric_out.parent / (metric_out.stem + "_comparison.csv")
            pivot.to_csv(comp_path, index=False, encoding="utf-8-sig")
            log.info(f"  ✓ 多模型透视对比 -> {comp_path}")
            log.info("=" * 70)
            log.info("【多模型对比表】")
            for line in pivot.to_string(index=False).split("\n"):
                log.info(f"  {line}")
            log.info("=" * 70)
    else:
        # 无标签场景: 输出一致性指标
        if baseline_results:
            log.info("-" * 70)
            log.info("【无标签一致性分析】(主模型 vs 各基线)")
            # [v13.5 bug 修复] 传 bundle 里的 ON_THR, 避免硬编码 10W
            cons = cross_model_consistency(y_pred, baseline_results,
                                            on_thr_w=on_thr_eval)
            cons_path = metric_out.parent / "inference_consistency.csv"
            cons_path.parent.mkdir(parents=True, exist_ok=True)
            cons.to_csv(cons_path, index=False, encoding="utf-8-sig")
            for line in cons.to_string(index=False).split("\n"):
                log.info(f"  {line}")
            log.info(f"  ✓ 一致性指标 -> {cons_path}")
        else:
            log.info("未提供分路标签且无基线对比, 跳过指标计算")

    # ---------- 9. 多模型可视化 (可选) ----------
    if args.plot and baseline_results:
        try:
            _plot_comparison(df.index, y_true, y_pred, baseline_results,
                             out_path=ARTIFACT_DIR / "inference_comparison.png")
        except Exception as e:
            log.warning(f"  绘图失败: {e}")

    log.info("=" * 70)
    log.info("Step 5 推理完成")
    return 0


def _plot_comparison(timestamps, y_true, y_pred_main, baseline_results,
                     out_path: Path):
    """多模型功率曲线对比图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_chinese_font()

    n_baselines = len(baseline_results)
    fig, ax = plt.subplots(figsize=(15, 6))
    if y_true is not None:
        ax.plot(timestamps, y_true, "k-", lw=1.8, label="真实值",
                alpha=0.9, zorder=10)
    ax.plot(timestamps, y_pred_main, "r-", lw=1.4, label="主模型 (v5)",
            alpha=0.9, zorder=9)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_baselines, 3)))
    for i, (name, info) in enumerate(baseline_results.items()):
        ax.plot(timestamps, info["y_pred"], "-", lw=1.0,
                color=colors[i], alpha=0.7,
                label=f"基线: {name}")
    ax.set_xlabel("时间")
    ax.set_ylabel("功率 (W)")
    ax.set_title(f"NILM 多模型推理对比 ({n_baselines + 1} 个模型)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
