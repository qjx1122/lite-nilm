# -*- coding: utf-8 -*-
"""
Step 4: 加载已训练模型, 在 test 集上评估 + 可视化
v5 升级: 支持 --baseline 多基线对比

命令行示例:
    python scripts\\04_evaluate.py                            # 默认 + 内置 RF 对比
    python scripts\\04_evaluate.py --baseline rf fallback     # 多基线
    python scripts\\04_evaluate.py --baseline models\\nilm_ac_two_stage_v42.pkl
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (ARTIFACT_DIR, MODEL_PKL, PRED_DIR, METRIC_DIR, PROJECT_VERSION,
                    BUS_CSV, BR_CSV,     # v13.16 daily raw counts 需要
                    ON_THR_W,             # v6.12.6 单一阈值
                    setup_chinese_font,
                    WEATHER_LATITUDE, WEATHER_LONGITUDE, WEATHER_CACHE_DIR,
                    SUMMER_TEMP_THRESHOLD, WINTER_TEMP_THRESHOLD,
                    get_logger, Timer)
from feature_utils import build_features, assert_no_nan_features
from metrics_utils import (build_daily_metrics_rows, save_daily_metrics_csv,
                           compute_raw_daily_counts)  # v13.16
from postprocess import apply_postprocess
from split_utils import make_splits
from weather_utils import get_weather_for_period
from expert_utils import assign_season, SEASON_LABELS
from baseline_utils import BaselineRegistry, BaselineRunner
from metrics_utils import (compute_classification_metrics,
                           compute_regression_metrics,
                           save_predictions_csv,
                           flatten_metrics_to_rows,
                           save_metrics_csv,
                           build_comparison_table)


def parse_args():
    p = argparse.ArgumentParser(description="测试集评估 (v5: 多基线, v6.8: + L4 校正后指标)")
    p.add_argument("--model", type=str, default=str(MODEL_PKL),
                   help="主模型路径")
    p.add_argument("--baseline", nargs="*",
                   default=["rf"],   # 默认带 RF 基线对比
                   help="基线模型 (rf/fallback/naive_mean/naive_zero/<.pkl>)")
    p.add_argument("--no-calib", action="store_true",
                   help="禁用 L4 残差校正后指标计算 (即使模型 bundle 含 calibrator)")
    return p.parse_args()

log = get_logger("evaluate")
chosen_font = setup_chinese_font()
log.info(f"matplotlib 字体: {chosen_font or '系统无中文字体'}")


def main():
    args = parse_args()
    from pathlib import Path
    model_path = Path(args.model)
    baselines = list(args.baseline) if args.baseline else []

    log.info("=" * 70)
    log.info("Step 4: 测试集评估 (v5: 多基线)")
    log.info(f"  主模型    : {model_path}")
    log.info(f"  基线对比  : {baselines if baselines else '<仅主模型>'}")
    log.info("=" * 70)

    # ---------- 1. 加载模型 ----------
    with Timer(f"加载模型 {model_path.name}", log):
        if not model_path.exists():
            log.error(f"模型文件不存在: {model_path}, 请先运行 03_train.py")
            return
        bundle = joblib.load(model_path)
    scaler   = bundle["scaler"]
    clf      = bundle["clf"]
    reg      = bundle["reg"]            # fallback 全局
    rf       = bundle["rf"]
    reg_low  = bundle.get("reg_low")
    reg_high = bundle.get("reg_high")
    # v4: MoE 主对象 (优先使用)
    moe      = bundle.get("moe")
    quantile_alpha = float(bundle.get("quantile_alpha", 0.5))
    quantile_low   = float(bundle.get("quantile_low",  0.10))
    quantile_high  = float(bundle.get("quantile_high", 0.90))
    top_cols = bundle["feat_cols"]
    best_thr = float(bundle["best_thr"])
    # [一致性检查] 与 03_train.py 常量对齐:
    #   POST_MIN_ON=1 -> fallback=1 ✓ 一致
    #   POST_FILL_SHORT_OFF=3 -> fallback=0 ⚠️ 不一致 (仅当加载 v6.10 前的老 bundle 缺此键时才触发,
    #     当前生成的 bundle 一定含此键, 实际评估值为 3, 与训练一致. 保留 0 以兼容极老 bundle 行为)
    post_min_on         = int(bundle.get("post_min_on", 1))
    post_fill_short_off = int(bundle.get("post_fill_short_off", 0))
    model_ver = bundle.get("version", "v1")
    log.info(f"  模型训练于: {bundle.get('trained_at', 'unknown')}  ({model_ver})")
    log.info(f"  Stage-1 阈值: {best_thr:.3f}")
    log.info(f"  特征维度: {len(bundle['feat_names'])}")
    log.info(f"  后处理: min_on={post_min_on}, fill_short_off={post_fill_short_off}")

    # ---------- 2. 加载对齐数据 (与训练同口径) ----------
    aligned_csv = ARTIFACT_DIR / "aligned_15min.csv"
    with Timer(f"加载对齐数据 {aligned_csv.name}", log):
        df = pd.read_csv(aligned_csv, index_col=0, parse_dates=True,
                         encoding="utf-8-sig")
        log.info(f"  数据 shape={df.shape}")

    # v5: 拉取气象 + 计算季节标签 (与训练时一致)
    use_weather_features = bool(bundle.get("use_weather_features", False))
    use_temp_based_season = bool(bundle.get("use_temp_based_season", False))
    weather_df = None
    if use_weather_features:
        with Timer("[v5] 加载气象数据 (自动缓存)", log):
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

    # 季节标签
    if use_temp_based_season and weather_df is not None:
        daily_t = weather_df["temperature_2m"].resample("D").mean()
        ts_dates = pd.DatetimeIndex(df.index).normalize()
        daily_avg_temp = daily_t.reindex(ts_dates, method="ffill").values
        season_labels_all = assign_season(
            df.index, daily_avg_temp=daily_avg_temp,
            use_temperature=True,
            summer_th=bundle.get("summer_temp_threshold", SUMMER_TEMP_THRESHOLD),
            winter_th=bundle.get("winter_temp_threshold", WINTER_TEMP_THRESHOLD),
        )
    else:
        season_labels_all = assign_season(df.index, use_temperature=False)
    # [v13.7] NaN 硬检测: X_df 若含 NaN 直接 raise, 避免后续 predict_proba 崩 sklearn
    assert_no_nan_features(X_df, stage_name="evaluate", logger=log,
                           raise_on_nan=True)
    X = X_df.values.astype(np.float32)
    y = df["y_ac"].values.astype(np.float32)
    # v6.12.6 单口径: 用 ON_THR_W 定义 ON 状态 (与训练标签同口径)
    # [v13.5 bug 修复] 优先从 bundle 读, 保证 v13.5 用户级覆盖 on_thr_w 后 04/03 一致
    # 修复前: 直接用 common.ON_THR_W (import 时的 10.0), 训练用 50 时评估仍用 10, 标签不一致
    on_thr_eval = float(bundle.get("ON_THR", ON_THR_W))
    state = (y >= on_thr_eval).astype(int)
    log.info(f"  [v13.5] 评估 ON 阈值 = {on_thr_eval:.1f}W (来自 bundle, 与训练一致)")

    # ---------- 3. 提取 test 段 (与训练切分一致) ----------
    # 切分策略与比例: 优先从 bundle 读取 (与训练时一致), 否则用 common 默认
    split_strategy = bundle.get("split_strategy")
    split_ratios   = bundle.get("split_ratios")
    if split_ratios is None:
        from common import SPLIT_RATIOS
        split_ratios = SPLIT_RATIOS
    split_ratios = tuple(split_ratios)
    log.info(f"  切分策略 (与训练一致): {split_strategy}, "
             f"比例: train={split_ratios[0]:.0%} / "
             f"val={split_ratios[1]:.0%} / "
             f"test={split_ratios[2]:.0%}")
    sp = make_splits(df.index, strategy=split_strategy, ratios=split_ratios)
    idx_tr_04, idx_va_04, idx_te = sp["train"], sp["val"], sp["test"]

    # [v13 per-split time_filter] 应用 train/val/test 独立 include/exclude
    # 保持与 03_train.py 完全对称, 否则 test 集会与训练时不一致
    import os as _os_04
    _splits_filter_spec_str_04 = _os_04.environ.get("NILM_SPLITS_FILTER_SPEC", "").strip()
    if _splits_filter_spec_str_04:
        try:
            from time_filter_utils import cli_arg_to_splits_spec, apply_per_split_filter, splits_spec_summary
            _splits_spec_04 = cli_arg_to_splits_spec(_splits_filter_spec_str_04)
            if _splits_spec_04 is not None:
                log.info(f"  [v13 per-split filter] 规格: {splits_spec_summary(_splits_spec_04)}")
                idx_tr_04, idx_va_04, idx_te = apply_per_split_filter(
                    df.index, idx_tr_04, idx_va_04, idx_te,
                    _splits_spec_04, logger=log
                )
        except Exception as _e:
            log.warning(f"  [v13 per-split filter] 应用失败 ({_e}), 使用原切分")
    X_te, y_te, s_te, t_te = X[idx_te], y[idx_te], state[idx_te], df.index[idx_te]
    sea_te = season_labels_all[idx_te]   # v5 季节标签
    log.info(f"  Test 集: {len(y_te)} 条  ({t_te.min()} ~ {t_te.max()})  "
             f"ON 占比 {s_te.mean()*100:.2f}%")
    # 测试集月份分布
    mc = pd.to_datetime(t_te).to_period("M").value_counts().sort_index()
    log.info(f"  Test 月份分布: " +
             ", ".join([f"{str(k)}={v}" for k, v in mc.items()]))

    # ---------- 4. 推理 (v4: MoE 优先, 自动回退到全局 reg) ----------
    with Timer("模型推理 (test, MoE + 后处理)", log):
        X_te_s = scaler.transform(X_te)
        p_te = clf.predict_proba(X_te_s)[:, 1]
        raw_state_te = (p_te >= best_thr).astype(int)

        if moe is not None:
            # v5: 用 sea_te 季节标签取代时间戳
            p_reg     = np.clip(moe.predict(X_te_s, sea_te, alpha=quantile_alpha), 0, None)
            y_low_te  = np.clip(moe.predict(X_te_s, sea_te, alpha=quantile_low),   0, None)
            y_high_te = np.clip(moe.predict(X_te_s, sea_te, alpha=quantile_high),  0, None)
            log.info(f"  使用季节分层 MoE 推理 ({model_ver})")
        else:
            p_reg = np.clip(reg.predict(X_te_s), 0, None)
            y_low_te  = (np.clip(reg_low.predict(X_te_s),  0, None)
                         if reg_low  is not None else None)
            y_high_te = (np.clip(reg_high.predict(X_te_s), 0, None)
                         if reg_high is not None else None)
            log.info(f"  使用全局回归器 (老模型)")

        pred_state_te, y_pred_te = apply_postprocess(
            raw_state_te, p_reg,
            min_on=post_min_on, fill_short_off=post_fill_short_off,
        )
        if y_low_te  is not None: y_low_te  = y_low_te  * pred_state_te
        if y_high_te is not None: y_high_te = y_high_te * pred_state_te

        log.info(f"  raw_ON={int(raw_state_te.sum())}, "
                 f"postproc_ON={int(pred_state_te.sum())}, "
                 f"真实_ON={int(s_te.sum())}")

        # ON 残差诊断 (按季节细分)
        m = s_te == 1
        if m.any():
            res = y_pred_te[m] - y_te[m]
            log.info(f"  Test ON 整体  残差: 中位={np.median(res):+.1f}W "
                     f"均值={res.mean():+.1f}W (n={int(m.sum())})")
            for sea in SEASON_LABELS:
                mm = m & (sea_te == sea)
                if mm.sum() > 0:
                    r = y_pred_te[mm] - y_te[mm]
                    log.info(f"      + {sea:<11} 残差: 中位={np.median(r):+.1f}W "
                             f"均值={r.mean():+.1f}W (n={int(mm.sum())})")

    # ---------- 4b. [v6.8] L4 残差校正后指标 (与训练阶段同口径) ----------
    # 设计意图: train_val_metrics.csv 已含 main / main_L4_calib 两组指标,
    # 但 test 集之前缺 main_L4_calib, 导致无法横向判断 "L4 在 OOD 上是否真有用"。
    # 这里复用训练时保存到 bundle 的 residual_calib 对象, 在 test 上 apply 一次。
    y_pred_te_calib = None
    residual_calib = bundle.get("residual_calib")
    use_calib_eval = (bundle.get("use_residual_calib", False)
                      and residual_calib is not None
                      and not args.no_calib)
    if use_calib_eval and getattr(residual_calib, "_trained", False):
        log.info("-" * 70)
        log.info("[v6.8] 计算 main_L4_calib 在 Test 集上的指标")
        # 构造 recent_signal: Top-1 列近 24h 滚动均值 (与 03_train.py 同口径)
        if top_cols and top_cols[0] in df.columns:
            top1 = df[top_cols[0]].astype(float)
            recent_24h_all = top1.rolling(window=96, min_periods=4,
                                          closed="left").mean().bfill().values
        else:
            recent_24h_all = np.zeros(len(df))
        recent_te = recent_24h_all[idx_te]

        # 应用 L4 校正 (仅 ON 段生效, 内部已限幅)
        try:
            y_pred_te_calib = residual_calib.apply(
                y_pred_raw=y_pred_te,
                timestamps=t_te,
                weather_df=weather_df,
                recent_signal=recent_te,
                season_labels=sea_te,
                state_pred=pred_state_te,
            )
            mae_before = float(np.abs(y_pred_te - y_te).mean())
            mae_after  = float(np.abs(y_pred_te_calib - y_te).mean())
            log.info(f"  [L4] Test MAE: 校正前 {mae_before:.2f}W -> "
                     f"校正后 {mae_after:.2f}W  (变化 {mae_after-mae_before:+.2f}W)")
        except Exception as e:
            log.warning(f"  [L4] 校正失败, 跳过 main_L4_calib 指标: {e}")
            y_pred_te_calib = None

        # 按月残差细分 (与已知 5 月概念漂移诊断对齐, 失败不影响指标)
        if y_pred_te_calib is not None:
            try:
                t_te_pd = pd.DatetimeIndex(t_te)
                month_str = t_te_pd.strftime("%Y-%m")
                on_mask = (s_te == 1)
                for month in sorted(set(month_str[on_mask])):
                    sel = on_mask & (month_str == month)
                    if sel.sum() == 0:
                        continue
                    r_raw   = y_pred_te[sel]       - y_te[sel]
                    r_calib = y_pred_te_calib[sel] - y_te[sel]
                    log.info(f"      + {month}  n={int(sel.sum()):>3d}  "
                             f"残差中位: 校正前 {np.median(r_raw):+7.1f}W -> "
                             f"校正后 {np.median(r_calib):+7.1f}W  "
                             f"| MAE: 校正前 {np.abs(r_raw).mean():.1f}W -> "
                             f"校正后 {np.abs(r_calib).mean():.1f}W")
            except Exception as e:
                log.warning(f"  [L4] 按月细分失败 (不影响 main_L4_calib 指标): {e}")
    else:
        if args.no_calib:
            log.info("[v6.8] --no-calib 指定, 跳过 main_L4_calib 指标")
        elif residual_calib is None or not getattr(residual_calib, "_trained", False):
            log.info("[v6.8] bundle 不含已训练 L4 校正器, 跳过 main_L4_calib 指标")

    # ---------- 5. 基线模型推理 (v5 统一框架) ----------
    # 截取 test 段的对齐 df 子集, 供基线 runner 使用
    df_te = df.iloc[idx_te].copy()

    baseline_results = {}
    if baselines:
        log.info("-" * 70)
        log.info(f"【基线对比】运行 {len(baselines)} 个基线模型")
        registry = BaselineRegistry(bundle, logger=log)
        runner   = BaselineRunner(registry, logger=log)
        baseline_results = runner.run_all(
            baselines=baselines,
            df_aligned=df_te,
            top_cols=top_cols,
            X_main_scaled=X_te_s,
            state_pred_main=pred_state_te,
            weather_df=weather_df.loc[(weather_df.index >= df_te.index.min()) &
                                      (weather_df.index <= df_te.index.max() +
                                       pd.Timedelta(hours=1))]
                       if weather_df is not None else None,
        )
        for name, info in baseline_results.items():
            y_b = info["y_pred"]
            log.info(f"  [{name:<20}] 平均功率 {y_b.mean():>6.1f}W")

    # ---------- 6. 指标 ----------
    log.info("-" * 70)
    log.info("【Test 集多模型评估】")
    all_metric_rows = []

    # v6.12.6 单口径: 用 ON_THR_W=10W 评估 test (与训练标签同口径)
    cls_te = compute_classification_metrics(s_te, pred_state_te, p_te)
    reg_te = compute_regression_metrics(y_te, y_pred_te)
    log.info(f"  [main] 分类: F1={cls_te['F1']:.4f}, "
             f"Precision={cls_te['Precision']:.4f}, Recall={cls_te['Recall']:.4f}")
    log.info(f"  [main] 回归: MAE={reg_te['MAE_W']:.2f}W, "
             f"SAE={reg_te['SAE']*100:.2f}%, "
             f"kWh真/预={reg_te['kWh_true']:.2f}/{reg_te['kWh_pred']:.2f}")
    all_metric_rows += flatten_metrics_to_rows(
        "test", "main", cls_metrics=cls_te, reg_metrics=reg_te,
        extra={"threshold": best_thr, "model_version": model_ver},
        source="evaluate",
    )

    # [v6.8] main_L4_calib 指标 (若 L4 校正成功)
    # 分类指标与主模型完全相同 (L4 不改变状态, 仅修正功率), 直接复用 cls_te
    if y_pred_te_calib is not None:
        reg_te_calib = compute_regression_metrics(y_te, y_pred_te_calib)
        log.info(f"  [main_L4_calib] 回归: MAE={reg_te_calib['MAE_W']:.2f}W, "
                 f"SAE={reg_te_calib['SAE']*100:.2f}%, "
                 f"kWh真/预={reg_te_calib['kWh_true']:.2f}/"
                 f"{reg_te_calib['kWh_pred']:.2f}")
        all_metric_rows += flatten_metrics_to_rows(
            "test", "main_L4_calib",
            cls_metrics=cls_te, reg_metrics=reg_te_calib,
            extra={"threshold": best_thr, "model_version": model_ver,
                   "note": "L4 残差校正后 (test 集 OOD 验证)"},
            source="evaluate",   # v6.11
        )

    # 各基线指标
    for name, info in baseline_results.items():
        y_b = info["y_pred"]
        # v6.12.6 基线评估用 ON_THR (单口径)
        # [v13.5 bug 修复] 从 bundle 读, 保持与 s_te (test state) 同阈值口径
        s_b = (y_b >= on_thr_eval).astype(int)
        cls_b = compute_classification_metrics(s_te, s_b, y_b)
        reg_b = compute_regression_metrics(y_te, y_b)
        log.info(f"  [{name:<20}] 分类: F1={cls_b['F1']:.4f}, "
                 f"Precision={cls_b['Precision']:.4f}, Recall={cls_b['Recall']:.4f}")
        log.info(f"  [{name:<20}] 回归: MAE={reg_b['MAE_W']:.2f}W, "
                 f"SAE={reg_b['SAE']*100:.2f}%")
        all_metric_rows += flatten_metrics_to_rows(
            "test", name, cls_metrics=cls_b, reg_metrics=reg_b,
            extra={"baseline_kind": info["model"].kind,
                   "description": info["model"].description},
            source="evaluate",   # v6.11
        )

    # ---------- 7. 保存预测 + 指标 CSV ----------
    save_predictions_csv(t_te, y_te, y_pred_te,
                         s_te, pred_state_te, p_te,
                         y_pred_low=y_low_te, y_pred_high=y_high_te,
                         out_path=PRED_DIR / "test_pred.csv")
    # [v6.8] 追加 main_L4_calib 列到 test_pred.csv 末尾, 便于 BI/Excel 横向对比
    if y_pred_te_calib is not None:
        _tp = pd.read_csv(PRED_DIR / "test_pred.csv", encoding="utf-8-sig")
        _tp["y_pred_main_L4_calib_W"] = np.round(y_pred_te_calib, 3)
        _tp["residual_main_L4_calib_W"] = np.round(y_pred_te_calib - y_te, 3)
        _tp.to_csv(PRED_DIR / "test_pred.csv", index=False, encoding="utf-8-sig")
    log.info(f"  ✓ {PRED_DIR / 'test_pred.csv'}")

    # 每个基线单独存预测明细 (便于精细分析)
    for name, info in baseline_results.items():
        save_predictions_csv(t_te, y_te, info["y_pred"],
                             out_path=PRED_DIR / f"test_pred_{name}.csv")
        log.info(f"  ✓ {PRED_DIR / f'test_pred_{name}.csv'}")

    metric_path = METRIC_DIR / "test_metrics.csv"
    # v6.10: 默认 append=True, 历史评估指标完整保留 (含 project_version 字段区分版本)
    merged_te = save_metrics_csv(all_metric_rows, metric_path, append=True)
    log.info(f"  ✓ {metric_path}  (本次追加 {len(all_metric_rows)} 行, "
             f"文件累计 {len(merged_te)} 行)")

    # ---------- [v13.14] 逐日主模型评估指标 (train + val + test 三集合并) ----------
    # 动机: 现有指标是整体聚合, 无法定位单日异常. 加 daily 视图 (每天 1 行 × 3 splits)
    # 便于业务方: (1) 找出预测崩溃的具体日期; (2) 结合 dataset 归属看 val/test 里哪些日子拉低指标;
    #             (3) 与 analyze_on_periods.py 的 daily 汇总配对分析
    # [v13.14+] 每行 dataset 列 = split_name 本身 (train/val/test), 与 v13.10 归属一致
    log.info("-" * 70)
    log.info("[v13.14] 生成逐日主模型评估指标 (train + val + test)")
    try:
        _all_daily_rows = []
        _common_extra = {"project_version": PROJECT_VERSION,
                         "model_file": MODEL_PKL.name if hasattr(MODEL_PKL, "name") else str(MODEL_PKL)}

        # [v13.16] 预计算总线/分路每天原始采集点数, train/val/test 共用同一份原始 CSV
        # 注: 训练侧不应用 time_filter (04_evaluate 阶段没 CLI 传入 spec, 且训练已按 spec 切过);
        # 这里统计的是原始 CSV 完整性, 与 pipeline 处理无关
        _bus_daily_counts = compute_raw_daily_counts(BUS_CSV, "event_time",
                                                      logger=log)
        _br_daily_counts  = compute_raw_daily_counts(BR_CSV,  "time",
                                                      logger=log)

        # test 集 (本脚本刚计算) - dataset 恒 = "test"
        _te_dates = pd.to_datetime(pd.Series(t_te)).dt.strftime("%Y-%m-%d").unique()
        _te_labels = {d: "test" for d in _te_dates}
        _all_daily_rows += build_daily_metrics_rows(
            t_te, y_te, y_pred_te, s_te, pred_state_te,
            split_name="test", on_thr_w=on_thr_eval, p_on=p_te,
            date_labels=_te_labels,
            model_name="main_final", extra=_common_extra,
            bus_daily_counts=_bus_daily_counts,           # [v13.16]
            branch_daily_counts=_br_daily_counts,         # [v13.16]
        )
        # train + val 集 (从 03_train.py 写的 pred CSV 读回)
        for _split, _csv_name in [("train", "train_pred.csv"), ("val", "val_pred.csv")]:
            _pred_csv = PRED_DIR / _csv_name
            if not _pred_csv.exists():
                log.warning(f"  [v13.14] {_pred_csv} 不存在, 跳过 {_split} 日汇")
                continue
            _dp = pd.read_csv(_pred_csv, encoding="utf-8-sig")
            _dp_ts = pd.to_datetime(_dp["time"])
            _dp_dates = _dp_ts.dt.strftime("%Y-%m-%d").unique()
            _dp_labels = {d: _split for d in _dp_dates}
            _all_daily_rows += build_daily_metrics_rows(
                _dp_ts,
                _dp["y_true_W"].values,
                _dp["y_pred_W"].values,
                _dp["state_true"].values if "state_true" in _dp.columns else (_dp["y_true_W"].values >= on_thr_eval).astype(int),
                _dp["state_pred"].values if "state_pred" in _dp.columns else (_dp["y_pred_W"].values >= on_thr_eval).astype(int),
                split_name=_split, on_thr_w=on_thr_eval,
                p_on=_dp["p_on"].values if "p_on" in _dp.columns else None,
                date_labels=_dp_labels,
                model_name="main_final", extra=_common_extra,
                bus_daily_counts=_bus_daily_counts,       # [v13.16]
                branch_daily_counts=_br_daily_counts,     # [v13.16]
            )
        _daily_path = METRIC_DIR / "train_daily_metrics.csv"
        save_daily_metrics_csv(_all_daily_rows, _daily_path, logger=log)
    except Exception as _e:
        log.warning(f"  [v13.14] 日级指标计算失败, 忽略: {_e}")
        import traceback; traceback.print_exc()

    # 多模型透视对比表 (新增)
    if baseline_results:
        key_metrics = ["F1", "Precision", "Recall", "Accuracy",
                       "MAE_W", "RMSE_W", "SAE", "NDE",
                       "kWh_true", "kWh_pred", "kWh_err"]
        pivot = build_comparison_table(all_metric_rows,
                                       include_metrics=key_metrics)
        comp_path = METRIC_DIR / "test_metrics_comparison.csv"
        pivot.to_csv(comp_path, index=False, encoding="utf-8-sig")
        log.info(f"  ✓ 测试集多模型对比 -> {comp_path}")
        log.info("=" * 70)
        log.info("【Test 多模型对比表】")
        for line in pivot.to_string(index=False).split("\n"):
            log.info(f"  {line}")
        log.info("=" * 70)

    # 兼容: 保留 y_rf_te 给后续可视化用
    y_rf_te = baseline_results.get("rf", {}).get("y_pred")
    if y_rf_te is None:
        y_rf_te = np.clip(rf.predict(X_te_s), 0, None)

    # ---------- 8. 汇总 train/val/test 指标到一张表 (v6.11 修复版) ----------
    # train_val_metrics 与 test_metrics 是 append 累积历史 (含 timestamp + project_version
    # + source v6.11 字段)。下面 all_metrics_summary 和 metrics_pivot 是 "派生快照",
    # 每次评估覆盖写, 仅展示最新版本的最新一次结果。
    #
    # v6.11 重要修复:
    #   (1) 旧 bug: drop_duplicates(keep='last') 对相同 (split, model) 用 timestamp 排序去重,
    #            但主模型 03_train 的 fallback/rf 行与 v42 03b 的 fallback/rf 行同名,
    #            v42 训练在后, "last" 行覆盖主模型行, 导致 fallback/rf 指标错误地被
    #            v42 训练时的指标覆盖, 真正主模型的 fallback/rf 指标永久丢失。
    #   (2) 修复: 增加 source 字段 (main_train / v42_baseline / evaluate / inference),
    #            优先保留与主模型源 (main_train) 一致的行, 确保 main / main_L4_calib /
    #            fallback / rf 都来自同一次主模型训练。v42_baseline 独立一行。
    tv_csv = METRIC_DIR / "train_val_metrics.csv"
    if tv_csv.exists():
        all_df = pd.concat([pd.read_csv(tv_csv, encoding="utf-8-sig"),
                            pd.read_csv(metric_path, encoding="utf-8-sig")],
                           ignore_index=True)

        # 仅取最新版本
        snapshot_df = all_df
        if "project_version" in all_df.columns:
            latest_ver = all_df.dropna(subset=["project_version"])["project_version"].iloc[-1]
            snapshot_df = all_df[all_df["project_version"] == latest_ver].copy()
            log.info(f"  [v6.11] 快照口径: 最新版本={latest_ver}, "
                     f"全量历史 {len(all_df)} 行 -> 同版本 {len(snapshot_df)} 行")

        # v6.11 核心修复: 用 source 字段区分指标来源
        # 主模型训练 (source='main_train') 写的 fallback/rf 与
        # v42 训练 (source='v42_baseline') 写的 fallback/rf 不再混淆
        if "source" in snapshot_df.columns and "timestamp" in snapshot_df.columns:
            # 去重 key 含 source: 同 (split, model, source, metric_type, metric) 才算重复
            # 然后按 timestamp 取最新 (允许多次跑同源时只保留最近一次)
            n_before = len(snapshot_df)
            snapshot_df = snapshot_df.sort_values("timestamp") \
                .drop_duplicates(subset=["split", "model", "source",
                                         "metric_type", "metric"],
                                 keep="last")
            log.info(f"  [v6.11] source-aware 去重: {n_before} -> {len(snapshot_df)} 行")
            log.info(f"  [v6.11] 各源行数: " +
                     ", ".join([f"{s}={n}" for s, n in
                                snapshot_df["source"].value_counts().items()]))
        elif "timestamp" in snapshot_df.columns:
            # 旧 bundle 兼容路径 (没有 source 字段)
            snapshot_df = snapshot_df.sort_values("timestamp") \
                .drop_duplicates(subset=["split", "model", "metric_type", "metric"],
                                 keep="last")
            log.warning("  [v6.11] ⚠ 数据无 source 字段, 退回旧去重逻辑 (可能出现 v42 覆盖主模型 bug)")

        all_path = METRIC_DIR / "all_metrics_summary.csv"
        snapshot_df.to_csv(all_path, index=False, encoding="utf-8-sig")
        log.info(f"  ✓ 汇总 train+val+test (最新版本快照) -> {all_path}")

        # 透视一张人类友好的对比表 (基于最新版本快照)
        # 注: 这里 source 列被聚合, 但因为 model 名已经区分 (v42_baseline 独立),
        # 主模型组 (main/main_L4_calib/fallback/rf) 与 v42 组在 pivot 中不会冲突
        pivot = snapshot_df.pivot_table(
            index=["model", "metric_type", "metric"],
            columns="split", values="value", aggfunc="first",
        ).reset_index()
        pivot_path = METRIC_DIR / "metrics_pivot.csv"
        pivot.to_csv(pivot_path, index=False, encoding="utf-8-sig")
        log.info(f"  ✓ 透视对比表 (最新版本快照) -> {pivot_path}")

    # ---------- 9. 可视化 ----------
    with Timer("生成测试集可视化图", log):
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(t_te, y_te, "k-", lw=1.2, label="真实值 (空调实测)")
        axes[0].plot(t_te, y_pred_te, "r-", lw=1.2, alpha=0.85,
                     label="预测 (中位数 P50)")
        if y_low_te is not None and y_high_te is not None:
            axes[0].fill_between(t_te, y_low_te, y_high_te,
                                 color="r", alpha=0.18,
                                 label="P10~P90 置信区间")
        axes[0].fill_between(t_te, 0, y_te, color="k", alpha=0.08)
        axes[0].set_ylabel("功率 (W)")
        axes[0].set_title("测试集 - NILM 空调功率分解 (v3: 分位回归+样本加权)")
        axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)

        axes[1].plot(t_te, y_te, "k-", lw=1.2, label="真实值")
        axes[1].plot(t_te, y_rf_te, "b-", lw=1.2, alpha=0.7,
                     label="预测 RF基线")
        axes[1].set_ylabel("功率 (W)")
        axes[1].set_title("基线对照 - 单阶段 RandomForest")
        axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)

        axes[2].step(t_te, s_te, "k-", lw=1.2, where="post", label="真实状态")
        axes[2].step(t_te, pred_state_te, "r--", lw=1.2, where="post",
                     alpha=0.85, label=f"预测状态 (thr={best_thr:.2f})")
        axes[2].set_ylabel("开/关"); axes[2].set_xlabel("时间")
        axes[2].set_title("阶段一 状态分类结果")
        axes[2].legend(loc="upper right"); axes[2].grid(alpha=0.3)
        axes[2].set_ylim(-0.1, 1.1)
        plt.tight_layout()
        fig_path = ARTIFACT_DIR / "test_prediction.png"
        plt.savefig(fig_path, dpi=120)
        log.info(f"  ✓ {fig_path}")

        # 特征重要性
        imp = pd.Series(reg.feature_importances_,
                        index=bundle["feat_names"]).sort_values().tail(20)
        plt.figure(figsize=(8, 7))
        imp.plot.barh(color="#3a7ca5")
        plt.title("Top-20 特征重要性 (回归阶段)")
        plt.xlabel("Importance"); plt.tight_layout()
        fi_path = ARTIFACT_DIR / "feat_importance.png"
        plt.savefig(fi_path, dpi=120)
        log.info(f"  ✓ {fi_path}")

    log.info("=" * 70)
    log.info("Step 4 评估完成")


if __name__ == "__main__":
    main()
