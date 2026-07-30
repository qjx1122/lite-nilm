# -*- coding: utf-8 -*-
"""
Step 6: 带偏置校正的推理脚本 (v4.1 修订, 零重训)

机制 (基于诊断结果, 针对 transition expert 系统性低估):
    1) 加载训练好的 v4 模型 (无需重训)
    2) 推理后, 对 transition 季节的预测做分段线性校正
    3) 校正系数从 val 集自动学习 (Isotonic Regression)

用法 (Windows):
    python scripts\06_inference_with_calib.py
    python scripts\06_inference_with_calib.py --bus xxx.csv --branch yyy.csv

校正前/校正后的指标会双份输出, 便于直观对比改善幅度。
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

from common import (INFER_BUS_CSV, INFER_BR_CSV,    # v6.12.6+v6.15.0 推理路径独立
                    BUS_CSV, BR_CSV,                # 兼容: 训练路径仍然可读
                    MODEL_PKL, PRED_DIR, METRIC_DIR,
                    ARTIFACT_DIR, ON_THR_W, ON_THR_BUSINESS_W,    # 别名 = ON_THR_W
                    WEATHER_LATITUDE, WEATHER_LONGITUDE, WEATHER_CACHE_DIR,
                    SUMMER_TEMP_THRESHOLD, WINTER_TEMP_THRESHOLD,
                    get_logger, Timer)
from feature_utils import (load_bus_csv, load_branch_csv,
                           resample_and_align, build_features)
from postprocess import apply_postprocess
from expert_utils import assign_season, SEASON_LABELS
from weather_utils import get_weather_for_period
from metrics_utils import (compute_classification_metrics,
                           compute_regression_metrics,
                           save_predictions_csv,
                           flatten_metrics_to_rows,
                           save_metrics_csv)

log = get_logger("infer_calib")


def parse_args():
    p = argparse.ArgumentParser(description="带校正的推理 (v4.1)")
    p.add_argument("--bus",    type=str, default=str(INFER_BUS_CSV),
                   help=f"总线 CSV 路径 (默认 INFER_BUS_CSV={INFER_BUS_CSV})")
    p.add_argument("--branch", type=str, default=str(INFER_BR_CSV),
                   help=f"分路 CSV 路径 (默认 INFER_BR_CSV={INFER_BR_CSV}, 可选)")
    p.add_argument("--model",  type=str, default=str(MODEL_PKL))
    p.add_argument("--out",    type=str,
                   default=str(PRED_DIR / "inference_result_calib.csv"))
    p.add_argument("--metric-out", type=str,
                   default=str(METRIC_DIR / "inference_metrics_calib.csv"))
    p.add_argument("--no-branch", action="store_true")
    p.add_argument("--calib-method", choices=["isotonic", "linear", "none"],
                   default="isotonic", help="校正方法")
    return p.parse_args()


def learn_calibrator_from_val(model_pkl: Path, method: str = "isotonic"):
    """
    从 val 集学习 transition expert 的偏置校正函数
    返回: dict[season -> callable]  对每个季节学一个校正映射
    """
    val_csv = PRED_DIR / "val_pred.csv"
    if not val_csv.exists():
        log.warning(f"{val_csv} 不存在, 无法学习校正, 使用恒等映射")
        return {sea: (lambda x: x) for sea in SEASON_LABELS}

    df = pd.read_csv(val_csv, parse_dates=["time"], encoding="utf-8-sig")
    seasons = assign_season(df["time"])
    calibrators = {}

    for sea in SEASON_LABELS:
        # 只用该季节中真实 ON 的样本学校正
        m = (seasons == sea) & (df["state_true"] == 1) & (df["y_pred_W"] > 0)
        n = int(m.sum())
        if n < 30:
            log.info(f"  [Calib/{sea:<11}] val 样本仅 {n} (<30), 使用恒等映射")
            calibrators[sea] = lambda x: x
            continue
        x = df.loc[m, "y_pred_W"].values.astype(float)
        y = df.loc[m, "y_true_W"].values.astype(float)
        if method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
            ir.fit(x, y)
            calibrators[sea] = ir.predict
            # 输出关键映射点供检视
            for p in [400, 500, 600, 700, 800]:
                log.info(f"  [Calib/{sea:<11}] {p:>4}W -> {ir.predict([p])[0]:>6.1f}W")
        elif method == "linear":
            # y = a + b*x
            b, a = np.polyfit(x, y, 1)[::-1]
            calibrators[sea] = lambda xx, _a=a, _b=b: _a + _b * xx
            log.info(f"  [Calib/{sea:<11}] 线性: y = {a:+.2f} + {b:.3f} * x  (n={n})")
        else:
            calibrators[sea] = lambda x: x
        log.info(f"  [Calib/{sea:<11}] 学习完成, 训练样本 n={n}, "
                 f"训练残差中位 {np.median(x - y):+.1f}W")
    return calibrators


def apply_seasonal_calibration(y_pred, timestamps, calibrators,
                               state_pred=None):
    """对每个时间点按其季节路由到对应的 calibrator"""
    out = np.array(y_pred, dtype=float).copy()
    seasons = assign_season(timestamps)
    for sea in SEASON_LABELS:
        m = (seasons == sea)
        if state_pred is not None:
            m = m & (state_pred == 1)
        if m.any() and sea in calibrators:
            out[m] = np.clip(calibrators[sea](out[m]), 0, None)
    return out


def main():
    args = parse_args()
    bus_path    = Path(args.bus)
    branch_path = Path(args.branch) if not args.no_branch else None
    model_path  = Path(args.model)
    out_path    = Path(args.out)
    metric_out  = Path(args.metric_out)

    log.info("=" * 72)
    log.info("Step 6: 带偏置校正的推理 (v4.1)")
    log.info(f"  总线 CSV  : {bus_path}")
    log.info(f"  分路 CSV  : {branch_path if branch_path else '<未提供>'}")
    log.info(f"  模型文件  : {model_path}")
    log.info(f"  校正方法  : {args.calib_method}")
    log.info(f"  结果 CSV  : {out_path}")
    log.info(f"  指标 CSV  : {metric_out}")
    log.info("=" * 72)

    # ---------- 1. 校验 ----------
    if not bus_path.exists():
        log.error(f"总线 CSV 不存在: {bus_path}"); return 1
    if not model_path.exists():
        log.error(f"模型文件不存在: {model_path}"); return 1
    has_label = branch_path is not None and branch_path.exists()

    # ---------- 2. 加载模型 ----------
    with Timer("加载模型", log):
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
        post_min_on         = int(bundle.get("post_min_on", 1))
        post_fill_short_off = int(bundle.get("post_fill_short_off", 0))
        log.info(f"  模型版本: {bundle.get('version', 'v1')}")

    # ---------- 3. 学习校正器 (基于 val 集预测) ----------
    log.info("-" * 72)
    log.info("[Calib] 从 val 集学习季节偏置校正器")
    calibrators = learn_calibrator_from_val(model_path, method=args.calib_method)

    # ---------- 4. 加载数据 ----------
    with Timer("加载总线数据", log):
        bus, all_cols = load_bus_csv(bus_path)
    missing = [c for c in top_cols if c not in bus.columns]
    if missing:
        log.error(f"总线 CSV 缺列: {missing[:5]}..."); return 1

    branch = None
    if has_label:
        with Timer("加载分路数据", log):
            branch = load_branch_csv(branch_path)

    with Timer("重采样 + 对齐 + 特征工程", log):
        df = resample_and_align(bus, branch, keep_cols=top_cols)
        if len(df) == 0:
            log.error("对齐后无数据"); return 1

        # v5: 加载气象 (与训练一致)
        use_weather_features = bool(bundle.get("use_weather_features", False))
        use_temp_based_season = bool(bundle.get("use_temp_based_season", False))
        weather_df = None
        if use_weather_features:
            weather_df = get_weather_for_period(
                latitude=bundle.get("weather_latitude", WEATHER_LATITUDE),
                longitude=bundle.get("weather_longitude", WEATHER_LONGITUDE),
                start_ts=df.index.min(), end_ts=df.index.max(),
                cache_dir=WEATHER_CACHE_DIR, logger=log,
            )

        X_df = build_features(df, top_cols, weather_df=weather_df,
                              temp_power_lut=bundle.get("temp_power_lut"))
        log.info(f"  对齐 + 特征 shape={X_df.shape}")

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

    # ---------- 5. 推理 ----------
    with Timer("模型推理", log):
        X = X_df.values.astype(np.float32)
        X_s = scaler.transform(X)
        p_on = clf.predict_proba(X_s)[:, 1]
        raw_state = (p_on >= best_thr).astype(int)

        timestamps = df.index
        if moe is not None:
            # v5: 用 season_labels 取代 timestamps
            p_reg_raw  = np.clip(moe.predict(X_s, season_labels, alpha=quantile_alpha), 0, None)
            y_low_raw  = np.clip(moe.predict(X_s, season_labels, alpha=quantile_low),   0, None)
            y_high_raw = np.clip(moe.predict(X_s, season_labels, alpha=quantile_high),  0, None)
        else:
            p_reg_raw  = np.clip(reg.predict(X_s), 0, None)
            y_low_raw  = np.clip(reg_low.predict(X_s), 0, None) if reg_low is not None else None
            y_high_raw = np.clip(reg_high.predict(X_s), 0, None) if reg_high is not None else None

        state_pred, y_pred_raw = apply_postprocess(
            raw_state, p_reg_raw,
            min_on=post_min_on, fill_short_off=post_fill_short_off,
        )

    # ---------- 6. 应用校正 ----------
    with Timer("应用季节偏置校正", log):
        y_pred_calib = apply_seasonal_calibration(
            y_pred_raw, timestamps, calibrators, state_pred=state_pred,
        )
        if y_low_raw is not None:
            y_low_calib = apply_seasonal_calibration(
                y_low_raw * state_pred, timestamps, calibrators, state_pred=state_pred,
            )
            y_high_calib = apply_seasonal_calibration(
                y_high_raw * state_pred, timestamps, calibrators, state_pred=state_pred,
            )
        else:
            y_low_calib = y_high_calib = None

        # 季节诊断
        seasons = assign_season(timestamps)
        for sea in SEASON_LABELS:
            m = (seasons == sea) & (state_pred == 1)
            if m.any():
                log.info(f"  [Apply/{sea:<11}] n={int(m.sum()):<5} "
                         f"校正前均值={y_pred_raw[m].mean():>6.1f}W -> "
                         f"校正后均值={y_pred_calib[m].mean():>6.1f}W "
                         f"(平均提升 {y_pred_calib[m].mean()-y_pred_raw[m].mean():+.1f}W)")

    # ---------- 7. 保存预测 ----------
    if has_label:
        y_true = df["y_ac"].values.astype(np.float32)
        # v6.13: 评估用 BUSINESS 阈值
        s_true = (y_true >= ON_THR_BUSINESS_W).astype(int)
        # 保存校正后版本
        out_df = pd.DataFrame({
            "time": df.index,
            "y_true_W":         np.round(y_true, 3),
            "y_pred_W_raw":     np.round(y_pred_raw, 3),
            "y_pred_W_calib":   np.round(y_pred_calib, 3),
            "residual_W_raw":   np.round(y_pred_raw - y_true, 3),
            "residual_W_calib": np.round(y_pred_calib - y_true, 3),
            "state_true":       s_true.astype(int),
            "state_pred":       state_pred.astype(int),
            "p_on":             np.round(p_on, 4),
        })
        if y_low_calib is not None:
            out_df["y_pred_low_W_calib"]  = np.round(y_low_calib, 3)
            out_df["y_pred_high_W_calib"] = np.round(y_high_calib, 3)
    else:
        out_df = pd.DataFrame({
            "time": df.index,
            "y_pred_W_raw":   np.round(y_pred_raw, 3),
            "y_pred_W_calib": np.round(y_pred_calib, 3),
            "state_pred":     state_pred.astype(int),
            "p_on":           np.round(p_on, 4),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"  ✓ 推理结果 -> {out_path}  ({len(out_df)} 行)")

    # ---------- 8. 双份指标对比 ----------
    if has_label:
        log.info("-" * 72)
        log.info("【校正前 vs 校正后 对比】")
        cls = compute_classification_metrics(s_true, state_pred, p_on)
        reg_raw   = compute_regression_metrics(y_true, y_pred_raw)
        reg_calib = compute_regression_metrics(y_true, y_pred_calib)

        log.info(f"  分类指标(无校正): F1={cls['F1']:.4f}, "
                 f"Precision={cls['Precision']:.4f}, Recall={cls['Recall']:.4f}")
        log.info(f"")
        log.info(f"  回归指标 校正前: MAE={reg_raw['MAE_W']:>6.2f}W  "
                 f"RMSE={reg_raw['RMSE_W']:>7.2f}W  SAE={reg_raw['SAE']*100:>5.2f}%  "
                 f"kWh真/预={reg_raw['kWh_true']:.2f}/{reg_raw['kWh_pred']:.2f}")
        log.info(f"  回归指标 校正后: MAE={reg_calib['MAE_W']:>6.2f}W  "
                 f"RMSE={reg_calib['RMSE_W']:>7.2f}W  SAE={reg_calib['SAE']*100:>5.2f}%  "
                 f"kWh真/预={reg_calib['kWh_true']:.2f}/{reg_calib['kWh_pred']:.2f}")
        log.info(f"")
        log.info(f"  改善幅度: MAE {reg_raw['MAE_W']-reg_calib['MAE_W']:+.2f}W, "
                 f"SAE {(reg_raw['SAE']-reg_calib['SAE'])*100:+.2f}%, "
                 f"kWh_err {reg_raw['kWh_err']-reg_calib['kWh_err']:+.2f}")

        # 保存两份指标
        extra_raw   = {"calibration": "none",
                       "bus_csv": bus_path.name, "branch_csv": branch_path.name}
        extra_calib = {"calibration": args.calib_method,
                       "bus_csv": bus_path.name, "branch_csv": branch_path.name}
        rows = []
        rows += flatten_metrics_to_rows("inference_raw", "v4_moe_no_calib",
                                        cls_metrics=cls, reg_metrics=reg_raw,
                                        extra=extra_raw)
        rows += flatten_metrics_to_rows("inference_calib", "v4_moe_with_calib",
                                        cls_metrics=cls, reg_metrics=reg_calib,
                                        extra=extra_calib)
        # v6.10: append 模式, 完整保留历史
        merged = save_metrics_csv(rows, metric_out, append=True)
        log.info(f"  ✓ 指标 CSV -> {metric_out}  "
                 f"(本次追加 {len(rows)} 行, 文件累计 {len(merged)} 行)")

    log.info("=" * 72)
    log.info("Step 6 完成")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
